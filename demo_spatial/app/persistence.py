"""Crash-safe persistence for :class:`app.spatial_memory.SpatialMemory` (design D14, section 9).

What is stored
--------------
Vector-less :class:`ObservationRecord` dicts: everything needed to rebuild a
``WorldState`` except the embedding, which is recomputed from
``metadata.label`` with whatever embedder is active when the memory restores
(D15: swapping the embedder is a non-event).

Files (all inside ``data_dir``)
-------------------------------
``memory.snapshot.json``      full state, atomic (tmp -> fsync -> os.replace)
``memory.snapshot.bak.json``  the previous snapshot (fallback when the current one is corrupt)
``memory.journal.jsonl``      ``{"op":"upsert","rec":{...}}`` / ``{"op":"delete","id":"..."}``
                              lines appended since the last snapshot, fsync'd at most once per
                              ``journal_fsync_s``
``*.corrupt-<ts>.json``        an unparsable snapshot or ``.bak``, moved aside by ``load()``
``*.failed-<ts>.*``            snapshot, ``.bak`` and journal of a restore that could not insert
                              them, moved aside by ``set_aside()`` so the empty session that
                              follows cannot overwrite them; never loaded automatically

Threading
---------
Writers (the camera thread via ``SpatialMemory.observe``) only *enqueue*; a single
``persistence-writer`` thread appends journal lines, fsyncs and takes snapshots.  Snapshots
requested while the writer runs are executed *by the writer, in queue order*, so a snapshot
always reflects every journal line written before it and the journal can be truncated right
after the snapshot without losing an enqueued record (see ``_handle_control``).

Session protocol
----------------
``load()`` -> caller inserts the records into a fresh client and obtains new ids ->
``begin_session(header, records_with_new_ids, provider=...)`` writes a synchronous snapshot,
truncates the journal and starts the writer.  Journal ids therefore always refer to ids that
exist in the snapshot they follow.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

__all__ = [
    "BAK_NAME",
    "JOURNAL_NAME",
    "SNAPSHOT_FORMAT",
    "SNAPSHOT_NAME",
    "MemoryPersistence",
    "ObservationRecord",
    "PersistenceWriter",
    "RestoreReport",
    "SnapshotProvider",
    "coerce_record",
    "copy_record",
]

SNAPSHOT_FORMAT = 2
SNAPSHOT_NAME = "memory.snapshot.json"
BAK_NAME = "memory.snapshot.bak.json"
JOURNAL_NAME = "memory.journal.jsonl"

_FRAMES = ("image", "camera", "room")
_WRITER_TICK_S = 0.1
_DRAIN_BATCH = 2000


class ObservationRecord(TypedDict):
    """One live point of the memory, without its vector."""

    id: str
    scene_id: str
    label: str
    room: str | None
    frame: str
    x: float
    y: float
    z: float
    timestamp_ms: int
    confidence: float
    prev_state_id: str | None
    metadata: dict[str, Any]


RECORD_KEYS: tuple[str, ...] = (
    "id",
    "scene_id",
    "label",
    "room",
    "frame",
    "x",
    "y",
    "z",
    "timestamp_ms",
    "confidence",
    "prev_state_id",
    "metadata",
)

SnapshotProvider = Callable[[], tuple[dict, list[ObservationRecord]]]
"""Returns ``(header, records)`` for a snapshot; called on the writer thread."""


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback: numpy scalars/arrays, sets, paths."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _unit(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or f < 0.0 or f > 1.0:
        return None
    return f


def coerce_record(obj: Any) -> ObservationRecord | None:
    """Validate a decoded JSON object as an :class:`ObservationRecord`.

    Returns ``None`` for anything malformed (missing keys, out-of-range coordinates,
    wrong types) so a torn or hand-edited line never crashes a restore.
    """
    if not isinstance(obj, Mapping):
        return None
    try:
        rid = obj["id"]
        scene_id = obj["scene_id"]
        label = obj["label"]
        frame = obj.get("frame", "image")
        x, y, z = _unit(obj["x"]), _unit(obj["y"]), _unit(obj["z"])
        ts = int(obj["timestamp_ms"])
        conf = _unit(obj.get("confidence", 1.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not (isinstance(rid, str) and rid) or not isinstance(scene_id, str):
        return None
    if not isinstance(label, str) or frame not in _FRAMES or ts < 0:
        return None
    if x is None or y is None or z is None or conf is None:
        return None
    room = obj.get("room")
    if room is not None and not isinstance(room, str):
        return None
    prev = obj.get("prev_state_id")
    if prev is not None and not isinstance(prev, str):
        prev = None
    meta = obj.get("metadata")
    meta = dict(meta) if isinstance(meta, Mapping) else {}
    return ObservationRecord(
        id=rid,
        scene_id=scene_id,
        label=label,
        room=room,
        frame=frame,
        x=x,
        y=y,
        z=z,
        timestamp_ms=ts,
        confidence=conf,
        prev_state_id=prev,
        metadata=meta,
    )


def copy_record(rec: ObservationRecord) -> ObservationRecord:
    """Shallow copy with its own metadata dict (records are mutable dicts)."""
    out = dict(rec)
    out["metadata"] = dict(rec.get("metadata") or {})
    return out  # type: ignore[return-value]


@dataclass
class RestoreReport:
    """What :meth:`MemoryPersistence.load` found on disk."""

    records: int
    rooms: list[str]
    newest_ms: int | None
    bad_lines: int
    source: str  # "snapshot" | "snapshot+journal" | "bak" | "empty" | "corrupt" | "disabled"
    snapshot_embedder: str | None = None
    journal_ops: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "records": int(self.records),
            "rooms": list(self.rooms),
            "newest_ms": self.newest_ms,
            "bad_lines": int(self.bad_lines),
            "source": self.source,
            "snapshot_embedder": self.snapshot_embedder,
            "journal_ops": int(self.journal_ops),
            "detail": self.detail,
        }


@dataclass
class _Control:
    """Marker travelling through the write queue (handled in order by the writer)."""

    kind: str  # "snapshot" | "clear" | "stop"
    done: threading.Event
    error: BaseException | None = None


class PersistenceWriter(threading.Thread):
    """Daemon thread that owns every journal/snapshot write of one ``MemoryPersistence``."""

    def __init__(self, owner: MemoryPersistence) -> None:
        super().__init__(name="persistence-writer", daemon=True)
        self._owner = owner

    def run(self) -> None:  # pragma: no cover - exercised through MemoryPersistence
        try:
            self._owner._writer_loop()
        except Exception:  # never let the thread die silently
            logger.exception("persistence writer crashed")
            self._owner._errors += 1


class MemoryPersistence:
    """Snapshot + journal persistence with a dedicated writer thread (see module docs)."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        *,
        snapshot_interval_s: float = 20.0,
        journal_fsync_s: float = 1.0,
        max_records: int = 100_000,
        enabled: bool = True,
    ) -> None:
        self._data_dir = Path(data_dir).expanduser()
        self.snapshot_interval_s = float(snapshot_interval_s)
        self.journal_fsync_s = float(journal_fsync_s)
        self.max_records = int(max_records)
        self._enabled = bool(enabled)

        self.snapshot_path = self._data_dir / SNAPSHOT_NAME
        self.bak_path = self._data_dir / BAK_NAME
        self.journal_path = self._data_dir / JOURNAL_NAME

        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._io_lock = threading.RLock()
        self._provider: SnapshotProvider | None = None
        self._thread: PersistenceWriter | None = None
        self._closed = False
        self._final_snapshot = True

        self._journal_fh: Any = None
        self._journal_lines = 0
        self._journal_bytes = 0
        self._unsynced = False
        self._last_fsync_t = 0.0
        self._dirty = False
        self._last_snapshot_ms: int | None = None
        self._last_snapshot_t = time.monotonic()
        self._snapshots = 0
        self._errors = 0
        self._last_report: RestoreReport | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(
        cls,
        data_dir: str | os.PathLike[str] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> MemoryPersistence:
        """Build from ``LOCI_DATA_DIR`` / ``LOCI_PERSIST`` / ``LOCI_SNAPSHOT_INTERVAL_S`` /
        ``LOCI_JOURNAL_FSYNC_S`` / ``LOCI_MAX_RECORDS`` (section 4.12 defaults)."""
        e = os.environ if env is None else env
        d = data_dir or e.get("LOCI_DATA_DIR") or "~/.local/share/loci-handheld"
        persist = e.get("LOCI_PERSIST", "1").strip().lower() not in ("0", "false", "no", "off", "")
        return cls(
            d,
            snapshot_interval_s=float(e.get("LOCI_SNAPSHOT_INTERVAL_S", "20") or 20),
            journal_fsync_s=float(e.get("LOCI_JOURNAL_FSYNC_S", "1") or 1),
            max_records=int(e.get("LOCI_MAX_RECORDS", "100000") or 100_000),
            enabled=persist,
        )

    # ------------------------------------------------------------------ properties
    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_report(self) -> RestoreReport | None:
        return self._last_report

    @property
    def provider(self) -> SnapshotProvider | None:
        return self._provider

    # ------------------------------------------------------------------ load
    def load(self) -> tuple[list[ObservationRecord], dict, RestoreReport]:
        """Snapshot (or ``.bak``) + journal overlay -> ``(records, header, report)``.

        Never raises for bad data: a corrupt snapshot is quarantined as
        ``memory.snapshot.corrupt-<ts>.json`` and the ``.bak`` is tried; unparsable
        journal lines are skipped and counted.
        """
        if not self._enabled:
            report = RestoreReport(0, [], None, 0, "disabled")
            self._last_report = report
            return [], {}, report

        records, header, source = self._load_snapshot_chain()
        by_id: dict[str, ObservationRecord] = {r["id"]: r for r in records}
        applied, bad = self._apply_journal(by_id)
        if applied and source == "snapshot":
            source = "snapshot+journal"

        recs = sorted(by_id.values(), key=lambda r: (r["timestamp_ms"], r["id"]))
        if self.max_records > 0 and len(recs) > self.max_records:
            recs = recs[-self.max_records :]

        rooms = sorted({r["room"] for r in recs if r.get("room")})
        emb = header.get("embedder") if isinstance(header, dict) else None
        report = RestoreReport(
            records=len(recs),
            rooms=rooms,
            newest_ms=max((r["timestamp_ms"] for r in recs), default=None),
            bad_lines=bad,
            source=source,
            snapshot_embedder=emb.get("name") if isinstance(emb, dict) else None,
            journal_ops=applied,
        )
        self._last_report = report
        return recs, header, report

    def _load_snapshot_chain(self) -> tuple[list[ObservationRecord], dict, str]:
        corrupt = False
        for path, label in ((self.snapshot_path, "snapshot"), (self.bak_path, "bak")):
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict) or not isinstance(data.get("records"), list):
                    raise ValueError("snapshot is not a {..., 'records': [...]} object")
                records = [r for r in map(coerce_record, data["records"]) if r is not None]
                header = {k: v for k, v in data.items() if k != "records"}
                return records, header, label
            except (OSError, ValueError, TypeError) as exc:
                corrupt = True
                self._errors += 1
                self._quarantine(path, label, exc)
        return [], {}, "corrupt" if corrupt else "empty"

    def _quarantine(self, path: Path, label: str, exc: Exception) -> None:
        stem = "memory.snapshot" if label == "snapshot" else "memory.snapshot.bak"
        target = self._data_dir / f"{stem}.corrupt-{int(time.time() * 1000)}.json"
        try:
            os.replace(path, target)
            logger.error("corrupt %s %s (%s); quarantined as %s", label, path, exc, target.name)
        except OSError as ren_exc:  # pragma: no cover - permissions
            logger.error("corrupt %s %s (%s); could not quarantine: %s", label, path, exc, ren_exc)

    def set_aside(self, reason: str) -> list[str]:
        """Rename the snapshot, its ``.bak`` and the journal to ``*.failed-<ts>.*``.

        For a restore that read the files but could not insert them: the session then starts
        empty, its first snapshot would push the good one to ``.bak`` and truncate the
        journal, and the next periodic snapshot would overwrite the ``.bak`` as well, so the
        memory (and through the mirror sync the cloud copy) was gone within about 40 s.
        Returns the new file names, which nothing loads automatically.
        """
        stamp = int(time.time() * 1000)
        moved: list[str] = []
        for path in (self.snapshot_path, self.bak_path, self.journal_path):
            if not path.exists():
                continue
            target = path.with_name(f"{path.stem}.failed-{stamp}{path.suffix}")
            try:
                os.replace(path, target)
                moved.append(target.name)
            except OSError as exc:  # pragma: no cover - permissions
                self._errors += 1
                logger.error("could not set %s aside after a failed restore: %s", path.name, exc)
        if moved:
            logger.error("restore failed (%s); kept %s for recovery", reason, ", ".join(moved))
        return moved

    def _apply_journal(self, by_id: dict[str, ObservationRecord]) -> tuple[int, int]:
        if not self.journal_path.exists():
            return 0, 0
        applied = 0
        bad = 0
        try:
            with open(self.journal_path, "rb") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    if not isinstance(obj, dict):
                        bad += 1
                        continue
                    op = obj.get("op")
                    if op == "upsert":
                        rec = coerce_record(obj.get("rec"))
                        if rec is None:
                            bad += 1
                            continue
                        by_id[rec["id"]] = rec
                        applied += 1
                    elif op == "delete":
                        rid = obj.get("id")
                        if not isinstance(rid, str):
                            bad += 1
                            continue
                        by_id.pop(rid, None)
                        applied += 1
                    else:
                        bad += 1
        except OSError as exc:
            logger.error("cannot read journal %s: %s", self.journal_path, exc)
            self._errors += 1
        if bad:
            logger.warning("journal %s: skipped %d unparsable line(s)", self.journal_path, bad)
        return applied, bad

    # ------------------------------------------------------------------ session
    def begin_session(
        self,
        header: dict,
        records: Iterable[ObservationRecord],
        *,
        provider: SnapshotProvider | None = None,
    ) -> None:
        """Synchronous snapshot + journal truncation, then start the writer thread.

        ``provider`` is what the writer calls for periodic / requested / final snapshots
        (``SpatialMemory`` registers its ``export_records``-based payload).
        """
        if provider is not None:
            self._provider = provider
        if not self._enabled:
            return
        self._closed = False
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self.snapshot_now(header, records)
        except OSError as exc:
            # An unwritable LOCI_DATA_DIR must not leave the session without a writer:
            # the journal queue would grow unbounded and nothing would ever retry.
            with self._io_lock:
                self._errors += 1
            self._dirty = True
            logger.error(
                "initial snapshot in %s failed (%s); starting the writer anyway, it retries",
                self._data_dir,
                exc,
            )
        if not self.running:
            self._thread = PersistenceWriter(self)
            self._thread.start()

    # ------------------------------------------------------------------ enqueue
    def append_upsert(self, rec: ObservationRecord) -> None:
        if not self._enabled:
            return
        self._queue.put({"op": "upsert", "rec": copy_record(rec)})
        self._dirty = True

    def append_delete(self, state_id: str) -> None:
        if not self._enabled:
            return
        self._queue.put({"op": "delete", "id": str(state_id)})
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def request_snapshot(self, *, wait: bool = False, timeout_s: float = 10.0) -> threading.Event:
        """Snapshot through the writer (queue order).  Returns the completion event.

        Without a running writer the snapshot is taken synchronously from the provider.
        """
        done = threading.Event()
        if not self._enabled or self._provider is None:
            done.set()
            return done
        if self.running:
            self._queue.put(_Control("snapshot", done))
            if wait:
                done.wait(timeout_s)
            return done
        try:
            self._snapshot_from_provider()
        except Exception as exc:
            self._errors += 1
            logger.error("synchronous snapshot failed: %s", exc)
        done.set()
        return done

    def clear(self, *, wait: bool = True, timeout_s: float = 10.0) -> threading.Event:
        """Delete snapshot, backup and journal (in queue order when the writer runs)."""
        done = threading.Event()
        if not self._enabled:
            done.set()
            return done
        if self.running:
            self._queue.put(_Control("clear", done))
            if wait:
                done.wait(timeout_s)
            return done
        self._clear_files()
        done.set()
        return done

    def close(self, timeout_s: float = 10.0, *, final_snapshot: bool = True) -> None:
        """Flush the queue, take a final snapshot and join the writer.  Idempotent.

        ``final_snapshot=False`` only fsyncs the journal (tests use it to simulate a
        crash: the next ``load()`` must rebuild the state from snapshot + journal).
        """
        if self._closed:
            return
        self._closed = True
        self._final_snapshot = bool(final_snapshot)
        if self.running:
            done = threading.Event()
            self._queue.put(_Control("stop", done))
            done.wait(timeout_s)
            if self._thread is not None:
                self._thread.join(timeout_s)
        else:
            with self._io_lock:
                self._close_journal()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        try:
            pending = self._queue.qsize()
        except NotImplementedError:  # pragma: no cover - macOS SimpleQueue has qsize
            pending = -1
        return {
            "enabled": self._enabled,
            "data_dir": str(self._data_dir),
            "running": self.running,
            "last_snapshot_ms": self._last_snapshot_ms,
            "snapshots": self._snapshots,
            "journal_lines": self._journal_lines,
            "journal_bytes": self._journal_bytes,
            "unsynced": self._unsynced,
            "queue": pending,
            "errors": self._errors,
            "dirty": self._dirty,
            "last_restore": self._last_report.to_dict() if self._last_report else None,
            "snapshot_interval_s": self.snapshot_interval_s,
            "journal_fsync_s": self.journal_fsync_s,
            "max_records": self.max_records,
        }

    # ------------------------------------------------------------------ snapshot IO
    def snapshot_now(self, header: dict, records: Iterable[ObservationRecord]) -> None:
        """Atomic snapshot: tmp -> fsync -> ``os.replace``; previous -> ``.bak``; journal
        truncated afterwards.  Raises ``OSError`` on IO failure (the writer logs it)."""
        if not self._enabled:
            return
        recs = sorted(records, key=lambda r: (int(r["timestamp_ms"]), str(r["id"])))
        if self.max_records > 0 and len(recs) > self.max_records:
            recs = recs[-self.max_records :]
        now_ms = int(time.time() * 1000)
        payload: dict[str, Any] = {"format": SNAPSHOT_FORMAT, "saved_at_ms": now_ms}
        payload.update({k: v for k, v in dict(header).items() if k != "records"})
        payload["format"] = SNAPSHOT_FORMAT
        payload["saved_at_ms"] = now_ms
        payload["record_count"] = len(recs)
        payload["records"] = recs

        with self._io_lock:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._data_dir / f"{SNAPSHOT_NAME}.tmp-{os.getpid()}-{threading.get_ident()}"
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, separators=(",", ":"), default=_json_default)
                    fh.flush()
                    os.fsync(fh.fileno())
                if self.snapshot_path.exists():
                    os.replace(self.snapshot_path, self.bak_path)
                os.replace(tmp, self.snapshot_path)
            except BaseException:
                self._errors += 1
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:  # pragma: no cover
                    pass
                raise
            self._fsync_dir()
            self._truncate_journal()
            self._dirty = False
            self._last_snapshot_ms = now_ms
            self._last_snapshot_t = time.monotonic()
            self._snapshots += 1

    def _snapshot_from_provider(self) -> None:
        if self._provider is None:
            return
        header, records = self._provider()
        self.snapshot_now(header, records)

    def _fsync_dir(self) -> None:
        try:
            fd = os.open(self._data_dir, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform without dir fsync
            return
        try:
            os.fsync(fd)
        except OSError:  # pragma: no cover
            pass
        finally:
            os.close(fd)

    # ------------------------------------------------------------------ journal IO
    def _open_journal(self) -> Any:
        if self._journal_fh is None:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._journal_fh = open(self.journal_path, "a", encoding="utf-8")  # noqa: SIM115
            try:
                self._journal_bytes = self._journal_fh.tell()
            except OSError:  # pragma: no cover
                self._journal_bytes = 0
        return self._journal_fh

    def _close_journal(self) -> None:
        fh = self._journal_fh
        self._journal_fh = None
        if fh is None:
            return
        try:
            if self._unsynced:
                fh.flush()
                os.fsync(fh.fileno())
                self._unsynced = False
        except OSError as exc:  # pragma: no cover
            self._errors += 1
            logger.error("journal fsync failed: %s", exc)
        finally:
            fh.close()

    def _truncate_journal(self) -> None:
        self._close_journal()
        with open(self.journal_path, "w", encoding="utf-8") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        self._journal_lines = 0
        self._journal_bytes = 0
        self._unsynced = False

    def _write_line(self, obj: dict) -> None:
        line = json.dumps(obj, separators=(",", ":"), default=_json_default) + "\n"
        fh = self._open_journal()
        fh.write(line)
        self._journal_lines += 1
        self._journal_bytes += len(line.encode("utf-8"))
        self._unsynced = True

    def _fsync_journal(self, force: bool = False) -> None:
        if not self._unsynced or self._journal_fh is None:
            return
        now = time.monotonic()
        if not force and now - self._last_fsync_t < self.journal_fsync_s:
            return
        self._journal_fh.flush()
        os.fsync(self._journal_fh.fileno())
        self._unsynced = False
        self._last_fsync_t = now

    def _clear_files(self) -> None:
        with self._io_lock:
            self._close_journal()
            for p in (self.snapshot_path, self.bak_path, self.journal_path):
                with contextlib.suppress(FileNotFoundError):
                    p.unlink()
            self._journal_lines = 0
            self._journal_bytes = 0
            self._unsynced = False
            self._dirty = False

    # ------------------------------------------------------------------ writer thread
    def _writer_loop(self) -> None:
        stop = False
        while not stop:
            try:
                first = self._queue.get(timeout=_WRITER_TICK_S)
            except queue.Empty:
                first = None
            if first is not None:
                stop = self._drain([first])
            if stop:
                break
            with self._io_lock:
                self._tick_io()
        with self._io_lock:
            self._close_journal()

    def _drain(self, items: list[Any]) -> bool:
        """Write queued lines in order; handle control markers.  Returns True on stop."""
        while len(items) < _DRAIN_BATCH:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        for item in items:
            if isinstance(item, _Control):
                stopped = self._handle_control(item)
                if stopped:
                    return True
                continue
            with self._io_lock:
                try:
                    self._write_line(item)
                except OSError as exc:
                    self._errors += 1
                    logger.error("journal append failed: %s", exc)
        return False

    def _tick_io(self) -> None:
        try:
            self._fsync_journal()
        except OSError as exc:
            self._errors += 1
            logger.error("journal fsync failed: %s", exc)
        if (
            self._provider is not None
            and self._dirty
            and time.monotonic() - self._last_snapshot_t >= self.snapshot_interval_s
        ):
            try:
                self._snapshot_from_provider()
            except Exception as exc:
                self._errors += 1
                logger.error("periodic snapshot failed: %s", exc)

    def _handle_control(self, ctrl: _Control) -> bool:
        stopped = False
        try:
            with self._io_lock:
                if ctrl.kind == "snapshot":
                    self._fsync_journal(force=True)
                    self._snapshot_from_provider()
                elif ctrl.kind == "clear":
                    self._clear_files()
                elif ctrl.kind == "stop":
                    stopped = True
                    # Drain whatever raced in after the stop marker before the final snapshot.
                    while True:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(item, _Control):
                            item.done.set()
                            continue
                        self._write_line(item)
                    self._fsync_journal(force=True)
                    if self._provider is not None and self._final_snapshot:
                        self._snapshot_from_provider()
                    self._close_journal()
        except Exception as exc:
            ctrl.error = exc
            self._errors += 1
            logger.error("persistence %s failed: %s", ctrl.kind, exc)
        finally:
            ctrl.done.set()
        return stopped
