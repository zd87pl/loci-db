"""Handheld runtime wiring for the Pi 5 + OAK-D Lite demo (design §4.11, task T7).

This module owns everything between a :class:`DetectionSource` and the FastAPI app:

* :class:`HandheldConfig` — every ``LOCI_*`` knob of design §4.12 read once from an
  environment mapping (``from_env``) and exported without secrets (``to_dict``).
* :class:`HandheldRuntime` — the single ``handheld-runtime`` thread of decision D17: it
  pulls ``source.events()``, feeds tag frames to the :class:`RoomLocalizer`, turns each
  :class:`SpatialDetection` into a metric :class:`Detection` (room metres when the
  camera pose is known), hands the frame to ``SceneIngestion.process_detections`` and
  keeps the 1 Hz presence log (D24) in ``SpatialMemory``.  ``run_until_done`` runs the
  same loop synchronously for tests and the simulator.
* :func:`build_handheld` — constructs source, localizer, recorder and runtime from a
  config (``None`` runtime when ``LOCI_DETECTION_SOURCE=none``).
* :func:`answer_text` — the synchronous text path shared by ``POST /api/query`` and the
  headset loop (``answer_fn``), returning ``answer`` / ``intent`` / ``resolution`` /
  ``results`` / ``changes`` / ``latency_ms``.
* :func:`start_voice_loop` — starts the T11 :class:`HeadsetVoiceLoop` on a daemon thread
  when ``LOCI_VOICE=headset`` (``headset_voice`` is imported lazily).

No ``depthai``, ``cv2`` or audio package is imported here; the hardware sources import
their dependencies lazily on their side.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .detection_source import (
    CameraPose,
    DetectionFrame,
    DetectionSource,
    JsonlRecorder,
    SimDetectionSource,
    SourceEvent,
    SpatialDetection,
    TagFrame,
    env_flag,
    make_detection_source,
)
from .room_anchor import PoseHold, RoomLocalizer, RoomMap, make_tag_detector
from .scene_ingestion import Detection, SceneIngestion

if TYPE_CHECKING:
    from .label_embedding import LabelEmbedder
    from .spatial_memory import SpatialMemory
    from .voice_pipeline import VoicePipeline

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOMS_FILE = REPO_ROOT / "deploy" / "handheld" / "config" / "rooms.example.json"
DEFAULT_DATA_DIR = Path("~/.local/share/loci-handheld")
PRESENCE_INTERVAL_MS = 1000  # memory.note_presence at <= 1 Hz per room (D24)
# A held pose (D8: the last tag solve kept for LOCI_POSE_HOLD_S) is stored as *anchored*
# only while its confidence (1 - age / hold_s) is still this high; below it, sightings keep
# the room-frame position but are flagged unanchored so answers carry the "roughly" hedge
# (D13: never a confident wrong-room answer right after walking out of a room).
STORE_MIN_ANCHOR_CONFIDENCE = 0.7
_VOICE_OFF = frozenset({"", "none", "0", "off", "false", "no"})


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _env_str(env: Mapping[str, str], key: str, default: str) -> str:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", key, raw, default)
        return default


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %s", key, raw, default)
        return default


def _env_path(env: Mapping[str, str], key: str) -> Path | None:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return None
    return Path(raw.strip()).expanduser()


def camera_only_pose() -> CameraPose:
    """The pose reported when no localizer exists or no tag was ever accepted."""
    return CameraPose(None, "camera", None, False, 0.0, (), math.inf)


# ---------------------------------------------------------------------------
# Configuration (§4.11 / §4.12)
# ---------------------------------------------------------------------------


@dataclass
class HandheldConfig:
    """All handheld knobs, read once from the environment (design §4.12 defaults)."""

    detection_source: str = "none"
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR.expanduser())
    persist: bool = True
    snapshot_interval_s: float = 20.0
    journal_fsync_s: float = 1.0
    max_records: int = 100_000
    rooms_file: Path | None = None
    tag_size_m: float = 0.16
    tag_size_set: bool = False  # True when LOCI_TAG_SIZE_M overrides rooms.json
    tag_detector: str = "auto"
    pose_hold_s: float = 10.0
    pose_stale_s: float = 60.0
    consensus_refire_ms: int = 2000
    consensus_radius_m: float = 0.3
    dedup_radius_m: float = 0.35
    moved_threshold_m: float = 0.5
    ws_push_hz: float = 5.0
    sim_scenario: str = "mug_two_rooms"
    sim_realtime: bool = True
    sim_oracle_pose: bool = False
    sim_loop: bool = False
    sim_start_offset_s: float = 0.0  # shift simulated timestamps into the past (tests/demos)
    label_embedder: str = "hash"
    record_path: Path | None = None
    replay_path: Path | None = None
    voice: str = "none"
    tls: bool = False
    port: int = 8001

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HandheldConfig:
        e: Mapping[str, str] = os.environ if env is None else env
        rooms_file = _env_path(e, "LOCI_ROOMS_FILE")
        if rooms_file is None and DEFAULT_ROOMS_FILE.is_file():
            rooms_file = DEFAULT_ROOMS_FILE
        data_dir = _env_path(e, "LOCI_DATA_DIR") or DEFAULT_DATA_DIR.expanduser()
        tag_size_raw = e.get("LOCI_TAG_SIZE_M", "").strip()
        return cls(
            detection_source=_env_str(e, "LOCI_DETECTION_SOURCE", "none").lower(),
            data_dir=data_dir,
            persist=env_flag(e, "LOCI_PERSIST", True),
            snapshot_interval_s=_env_float(e, "LOCI_SNAPSHOT_INTERVAL_S", 20.0),
            journal_fsync_s=_env_float(e, "LOCI_JOURNAL_FSYNC_S", 1.0),
            max_records=_env_int(e, "LOCI_MAX_RECORDS", 100_000),
            rooms_file=rooms_file,
            tag_size_m=_env_float(e, "LOCI_TAG_SIZE_M", 0.16),
            tag_size_set=bool(tag_size_raw),
            tag_detector=_env_str(e, "LOCI_TAG_DETECTOR", "auto").lower(),
            pose_hold_s=_env_float(e, "LOCI_POSE_HOLD_S", 10.0),
            pose_stale_s=_env_float(e, "LOCI_POSE_STALE_S", 60.0),
            consensus_refire_ms=_env_int(e, "LOCI_CONSENSUS_REFIRE_MS", 2000),
            consensus_radius_m=_env_float(e, "LOCI_CONSENSUS_RADIUS_M", 0.3),
            dedup_radius_m=_env_float(e, "LOCI_DEDUP_RADIUS_M", 0.35),
            moved_threshold_m=_env_float(e, "LOCI_MOVED_THRESHOLD_M", 0.5),
            ws_push_hz=_env_float(e, "LOCI_WS_PUSH_HZ", 5.0),
            sim_scenario=_env_str(e, "LOCI_SIM_SCENARIO", "mug_two_rooms"),
            sim_realtime=env_flag(e, "LOCI_SIM_REALTIME", True),
            sim_oracle_pose=env_flag(e, "LOCI_SIM_ORACLE_POSE", False),
            sim_loop=env_flag(e, "LOCI_SIM_LOOP", False),
            sim_start_offset_s=_env_float(e, "LOCI_SIM_START_OFFSET_S", 0.0),
            label_embedder=_env_str(e, "LOCI_LABEL_EMBEDDER", "hash").lower(),
            record_path=_env_path(e, "LOCI_RECORD_PATH"),
            replay_path=_env_path(e, "LOCI_REPLAY_PATH"),
            voice=_env_str(e, "LOCI_VOICE", "none").lower(),
            tls=env_flag(e, "LOCI_TLS", False),
            port=_env_int(e, "PORT", 8001),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe, secret-free view for ``/api/handheld/status``."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else v
        return out


def load_rooms(cfg: HandheldConfig, *, env: Mapping[str, str] | None = None) -> RoomMap:
    """``rooms.json`` from the config, else the built-in kitchen/living_room map.

    Never raises: a missing or malformed file is logged and the default map is used.
    ``LOCI_TAG_SIZE_M`` overrides the file only when the operator set it.
    """
    override = cfg.tag_size_m if cfg.tag_size_set else None
    if cfg.rooms_file is not None:
        try:
            rooms = RoomMap.load(cfg.rooms_file, tag_size_m=override, env=env)
            logger.info("rooms: %s from %s", ", ".join(rooms.names()), cfg.rooms_file)
            return rooms
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.error("could not load %s (%s); using the default rooms", cfg.rooms_file, exc)
    rooms = RoomMap.default()
    if override is not None:
        rooms.tag_size_m = float(override)
    return rooms


def make_embedder_safe(
    cfg: HandheldConfig, *, env: Mapping[str, str] | None = None
) -> tuple[LabelEmbedder, str | None]:
    """``make_embedder(cfg.label_embedder)`` with a loud fallback to the hash embedder.

    ``LOCI_LABEL_EMBEDDER=static`` without the model files must not take the server
    down (the typed path is the committed path, D27); the error is returned so the
    status endpoint can show it.
    """
    from .label_embedding import HashLabelEmbedder, make_embedder

    try:
        return make_embedder(cfg.label_embedder, env=env), None
    except (ImportError, OSError, KeyError, ValueError) as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "label embedder %r unavailable (%s); falling back to the hash embedder",
            cfg.label_embedder,
            msg,
        )
        return HashLabelEmbedder(), msg


# ---------------------------------------------------------------------------
# Runtime thread (D17)
# ---------------------------------------------------------------------------


class HandheldRuntime:
    """Consumes a :class:`DetectionSource` on one daemon thread (``handheld-runtime``).

    ``step`` is the per-event body (also called synchronously by tests).  State read by
    the API/WebSocket side (``latest_frame``, ``latest_pose``, ``frames`` ...) is
    plain attribute assignment of immutable objects, so no lock is needed on the
    reader side; ``status()`` snapshots everything in one call.
    """

    def __init__(
        self,
        source: DetectionSource,
        localizer: RoomLocalizer | None,
        ingestion: SceneIngestion,
        memory: SpatialMemory,
        cfg: HandheldConfig,
        recorder: JsonlRecorder | None = None,
    ) -> None:
        self.source = source
        self.localizer = localizer
        self.ingestion = ingestion
        self.memory = memory
        self.cfg = cfg
        self.recorder = recorder

        self.latest_frame: DetectionFrame | None = None
        self.latest_pose: CameraPose | None = None
        self.latest_detections: list[Detection] = []
        self.seq = 0
        self.frames = 0
        self.tag_frames = 0
        self.errors = 0
        self.error: str | None = None
        self.done = False
        self.dropped_no_depth = 0
        self.last_timestamp_ms: int | None = None
        self.started_at: float | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_presence: tuple[str, int] | None = None
        self._recorder_failed = False
        self._vocab_applied: list[str] | None = None
        self.vocabulary_updates = 0

    # -- lifecycle ------------------------------------------------------------------

    @property
    def thread_alive(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> None:
        """Start the ``handheld-runtime`` daemon thread (idempotent while alive)."""
        if self.thread_alive:
            return
        self._stop.clear()
        self.done = False
        self._thread = threading.Thread(target=self._run, name="handheld-runtime", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Ask the source to stop, join the thread and close the recorder."""
        self._stop.set()
        try:
            self.source.stop()
        except Exception:  # a broken source must not block shutdown
            logger.exception("source.stop() failed")
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout_s)
            if t.is_alive():
                logger.warning("handheld-runtime thread did not stop within %.1fs", timeout_s)
        if self.recorder is not None:
            self.recorder.close()

    def run_until_done(
        self, max_events: int | None = None, *, timeout_s: float | None = None
    ) -> int:
        """Run the event loop to completion and return the detection frames processed.

        When ``start()`` already launched the thread this waits for it (``timeout_s``),
        which is what the API tests do after the lifespan started the runtime;
        otherwise the loop runs synchronously in the calling thread (``max_events``
        bounds it).  The count is the runtime total, so a single call on a fresh
        runtime returns the frames of that run.
        """
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout_s)
            return self.frames
        if self.done and t is not None:
            return self.frames
        self._stop.clear()
        self.done = False
        self._run(max_events)
        return self.frames

    def _run(self, max_events: int | None = None) -> None:
        n = 0
        self.started_at = time.monotonic()
        try:
            if self._stop.is_set():
                return  # stop() landed before the thread body ran: never open the camera
            self.source.start()
            if self._stop.is_set():
                self.source.stop()
                return
            self._sync_vocabulary()
            for ev in self.source.events():
                if self._stop.is_set():
                    break
                try:
                    self.step(ev)
                except Exception as exc:
                    self.errors += 1
                    self.error = f"{type(exc).__name__}: {exc}"
                    logger.exception("handheld step failed: %s", exc)
                # Recorded *after* step(): the localizer fills ``TagFrame.tags`` with what
                # it detected on the (never serialised) luma plane, so a replay of a real
                # camera recording anchors exactly like the live run (design §4.14).
                self._record(ev)
                n += 1
                if max_events is not None and n >= max_events:
                    break
        except Exception as exc:
            self.errors += 1
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("handheld runtime stopped on error: %s", exc)
        finally:
            self.done = True
            if self.recorder is not None:
                try:
                    self.recorder.flush()
                except Exception:
                    logger.debug("recorder flush failed", exc_info=True)

    def _sync_vocabulary(self) -> None:
        """Push ``source.classes`` into the resolver vocabulary whenever it changes.

        The OAK source only learns the model's labels once the device is open (and
        again after a reconnect), long after the lifespan's one-time
        ``set_vocabulary`` call; the sim/replay sources know them at construction.
        """
        classes = getattr(self.source, "classes", None)
        setter = getattr(self.memory, "set_vocabulary", None)
        if classes is None or setter is None:
            return
        current = [str(c) for c in classes]
        if not current or current == self._vocab_applied:
            return
        try:
            setter(current)
        except Exception as exc:
            logger.warning("set_vocabulary failed: %s", exc)
            return
        self._vocab_applied = current
        self.vocabulary_updates += 1

    def _record(self, ev: SourceEvent) -> None:
        if self.recorder is None or self._recorder_failed:
            return
        try:
            self.recorder.write(ev)
        except Exception as exc:
            self._recorder_failed = True
            logger.error("recording stopped: %s", exc)

    # -- per-event body -------------------------------------------------------------

    def step(self, ev: SourceEvent) -> None:
        """Tag frames localise; detection frames become metric detections and are
        pushed through the consensus buffer (synchronously, on this thread)."""
        if isinstance(ev, TagFrame):
            self.tag_frames += 1
            if self.localizer is not None:
                self.localizer.update(ev)
            return
        if not isinstance(ev, DetectionFrame):
            logger.debug("ignoring unknown event %r", type(ev).__name__)
            return

        self._sync_vocabulary()
        pose = ev.oracle_pose
        if pose is not None and self.localizer is not None:
            self.localizer.set_oracle(pose)
        if pose is None:
            pose = self.localizer.current() if self.localizer is not None else camera_only_pose()

        dets: list[Detection] = []
        metric: list[Detection] = []
        for sd in ev.detections:
            det = self.to_detection(sd, pose, ev.source)
            dets.append(det)
            if det.xyz_cam_m is None:
                # No stereo depth: shown on the live strip, never stored as a 0 m
                # position (hardware facts: all-zero spatialCoordinates = no depth).
                self.dropped_no_depth += 1
                continue
            metric.append(det)

        self.ingestion.process_detections(metric, ev.timestamp_ms)
        self._note_presence(pose, ev.timestamp_ms)

        self.latest_detections = dets
        self.latest_pose = pose
        self.latest_frame = ev
        self.last_timestamp_ms = ev.timestamp_ms
        self.seq = ev.seq
        self.frames += 1

    @staticmethod
    def pose_trusted(pose: CameraPose) -> bool:
        """Anchored *and* not decayed below :data:`STORE_MIN_ANCHOR_CONFIDENCE`."""
        return bool(pose.anchored) and float(pose.anchor_confidence) >= STORE_MIN_ANCHOR_CONFIDENCE

    @staticmethod
    def to_detection(sd: SpatialDetection, pose: CameraPose, source: str) -> Detection:
        """One :class:`SpatialDetection` + the current pose -> metric :class:`Detection`."""
        in_room = pose.frame == "room" and pose.T_room_cam is not None and bool(pose.room)
        pos_room = (
            RoomLocalizer.to_room(pose, sd.xyz_cam_m)
            if in_room and sd.xyz_cam_m is not None
            else None
        )
        return Detection(
            label=sd.label,
            cx=sd.cx,
            cy=sd.cy,
            width=sd.width,
            height=sd.height,
            confidence=sd.confidence,
            source=source,
            depth_m=sd.depth_m,
            xyz_cam_m=sd.xyz_cam_m,
            pos_room_m=pos_room,
            room=pose.room,
            frame="room" if in_room else "camera",
            anchored=bool(in_room and HandheldRuntime.pose_trusted(pose)),
            anchor_confidence=float(pose.anchor_confidence) if in_room else 0.0,
        )

    def _note_presence(self, pose: CameraPose, ts_ms: int) -> None:
        if not (pose.frame == "room" and pose.room and self.pose_trusted(pose)):
            return
        last = self._last_presence
        if last is not None and last[0] == pose.room and ts_ms - last[1] < PRESENCE_INTERVAL_MS:
            return
        note = getattr(self.memory, "note_presence", None)
        if note is None:
            return
        note(pose.room, int(ts_ms))
        self._last_presence = (pose.room, int(ts_ms))

    # -- read side ------------------------------------------------------------------

    def current_pose(self) -> CameraPose:
        if self.localizer is not None:
            return self.localizer.current()
        return self.latest_pose if self.latest_pose is not None else camera_only_pose()

    def live_detections(self) -> list[dict[str, Any]]:
        """``Detection.to_dict()`` of the latest frame, always with ``distance_m``."""
        out = []
        for d in self.latest_detections:
            dd = d.to_dict()
            dd.setdefault("distance_m", None)
            out.append(dd)
        return out

    def source_status(self) -> dict[str, Any]:
        try:
            return self.source.status().to_dict()
        except Exception as exc:  # a status call must never break the API
            return {"state": "error", "error": f"{type(exc).__name__}: {exc}"}

    def status(self) -> dict[str, Any]:
        pose = self.latest_pose
        rec = self.recorder
        return {
            "source": self.source_status(),
            "localizer": self.localizer.status() if self.localizer is not None else None,
            "frames": self.frames,
            "tag_frames": self.tag_frames,
            "errors": self.errors,
            "error": self.error,
            "done": self.done,
            "thread_alive": self.thread_alive,
            "ingestion": self.ingestion.consensus_stats,
            "seq": self.seq,
            "last_timestamp_ms": self.last_timestamp_ms,
            "pose": pose.to_dict() if pose is not None else None,
            "dropped_no_depth": self.dropped_no_depth,
            "vocabulary_updates": self.vocabulary_updates,
            "recorder": None if rec is None else {"path": str(rec.path), "lines": rec.lines},
            "uptime_s": None
            if self.started_at is None
            else round(time.monotonic() - self.started_at, 1),
        }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def build_handheld(
    memory: SpatialMemory,
    ingestion: SceneIngestion,
    cfg: HandheldConfig,
    *,
    env: Mapping[str, str] | None = None,
    source: DetectionSource | None = None,
) -> tuple[HandheldRuntime | None, RoomLocalizer | None]:
    """Source + localizer + recorder + runtime from ``cfg`` (``(None, None)`` for ``none``).

    ``source`` injects a ready source (tests); otherwise ``make_detection_source`` reads
    ``cfg.detection_source`` with ``env``.  Configuration errors (unknown source,
    missing replay path) propagate as ``ValueError`` so the caller can report them.
    """
    e: Mapping[str, str] = os.environ if env is None else env
    if source is None:
        source = make_detection_source(cfg.detection_source, env=e)
    if source is None:
        return None, None
    if isinstance(source, SimDetectionSource) and cfg.sim_start_offset_s:
        source.start_ms = int(time.time() * 1000 - cfg.sim_start_offset_s * 1000.0)

    rooms = memory.rooms
    detector = None
    try:
        detector = make_tag_detector(cfg.tag_detector, family=rooms.tag_family, env=e)
    except (ImportError, OSError, ValueError) as exc:
        logger.warning(
            "tag detector %r unavailable (%s); pose from oracle/tags only", cfg.tag_detector, exc
        )
    localizer = RoomLocalizer(
        rooms,
        detector,
        hold=PoseHold(hold_s=cfg.pose_hold_s),
        stale_s=cfg.pose_stale_s,
        tags_path=(cfg.data_dir / "tags.json") if cfg.persist else None,
    )
    recorder = None
    if cfg.record_path is not None:
        try:
            recorder = JsonlRecorder(cfg.record_path)
            logger.info("recording source events to %s", cfg.record_path)
        except OSError as exc:
            logger.error("cannot open LOCI_RECORD_PATH %s: %s", cfg.record_path, exc)
    runtime = HandheldRuntime(source, localizer, ingestion, memory, cfg, recorder=recorder)
    return runtime, localizer


# ---------------------------------------------------------------------------
# Text answers (shared by POST /api/query and the headset loop)
# ---------------------------------------------------------------------------


def _as_dict(obj: Any) -> Any:
    to_dict = getattr(obj, "to_dict", None)
    return to_dict() if callable(to_dict) else obj


def answer_text(
    voice: VoicePipeline,
    memory: SpatialMemory,
    text: str,
    *,
    room: str | None = None,
) -> dict[str, Any]:
    """Parse -> resolve -> query -> phrase, synchronously and without TTS.

    ``room`` fills the intent's room when the sentence did not name one (the phone
    page's room selector).  Returns ``answer``, ``intent``, ``resolution``, ``results``,
    ``changes`` and ``latency_ms``; the headset loop uses ``["answer"]``.
    """
    from .voice_pipeline import answer_query

    t0 = time.perf_counter()
    intent = voice.parse(text)
    if room and not intent.room:
        intent = dataclasses.replace(intent, room=str(room))
    bundle = answer_query(intent, memory, current_pose=voice.current_pose())
    return {
        "answer": bundle.text,
        "intent": intent.to_dict(),
        "resolution": _as_dict(bundle.resolution) if bundle.resolution is not None else None,
        "results": [_as_dict(o) for o in bundle.results],
        "changes": [_as_dict(c) for c in bundle.changes],
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


# ---------------------------------------------------------------------------
# Headset voice loop (T11 contract)
# ---------------------------------------------------------------------------


class VoiceLoopHandle:
    """A running :class:`HeadsetVoiceLoop` plus its thread and stop event."""

    def __init__(self, loop: Any, thread: threading.Thread, stop: threading.Event) -> None:
        self.loop = loop
        self.thread = thread
        self._stop = stop

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def status(self) -> dict[str, Any]:
        st = dict(self.loop.status())
        st["thread_alive"] = self.alive
        return st

    def last_events(self, n: int = 20) -> list[dict]:
        return list(self.loop.last_events(n))

    def stop(self, timeout_s: float = 2.0) -> None:
        """Signal the loop and join; an in-progress recording may hold the thread for
        up to ``max_seconds``, hence the bounded join."""
        self._stop.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout_s)


def voice_enabled(env: Mapping[str, str] | None = None) -> bool:
    e: Mapping[str, str] = os.environ if env is None else env
    return (e.get("LOCI_VOICE") or "none").strip().lower() not in _VOICE_OFF


def start_voice_loop(
    answer_fn: Callable[[str], str],
    stt: Any,
    tts: Any,
    *,
    env: Mapping[str, str] | None = None,
    audio: Any = None,
    ptt: Any = None,
    on_event: Callable[[dict], None] | None = None,
) -> VoiceLoopHandle | None:
    """Build the headset loop from the environment and run it on a daemon thread.

    Returns ``None`` when ``LOCI_VOICE`` is not ``headset``.  ``audio``/``ptt`` inject
    fakes (tests).  Misconfiguration raises (the caller logs and keeps serving).
    """
    e: Mapping[str, str] = os.environ if env is None else env
    if not voice_enabled(e):
        return None
    from .headset_voice import make_voice_loop_from_env, start_voice_thread

    loop = make_voice_loop_from_env(
        answer_fn, stt, tts, env=e, audio=audio, ptt=ptt, on_event=on_event
    )
    if loop is None:
        return None
    stop = threading.Event()
    thread = start_voice_thread(loop, stop)
    logger.info("headset voice loop started (ptt=%s)", loop.status().get("ptt"))
    return VoiceLoopHandle(loop, thread, stop)


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_ROOMS_FILE",
    "PRESENCE_INTERVAL_MS",
    "REPO_ROOT",
    "HandheldConfig",
    "HandheldRuntime",
    "VoiceLoopHandle",
    "answer_text",
    "build_handheld",
    "camera_only_pose",
    "load_rooms",
    "make_embedder_safe",
    "start_voice_loop",
    "voice_enabled",
]
