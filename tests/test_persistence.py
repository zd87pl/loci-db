"""Tests for demo_spatial/app/persistence.py and SpatialMemory.restore/flush/close (T4, D14).

Everything runs against ``tmp_path``; the writer thread is real but its timers are
shortened (``journal_fsync_s`` 0.02 s, no periodic snapshot unless a test asks).
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app import loci_bridge  # noqa: E402
from app.label_embedding import hash_label_embedding  # noqa: E402
from app.persistence import (  # noqa: E402
    BAK_NAME,
    JOURNAL_NAME,
    SNAPSHOT_NAME,
    MemoryPersistence,
    RestoreReport,
    coerce_record,
)
from app.room_anchor import RoomMap  # noqa: E402
from app.spatial_memory import SpatialMemory  # noqa: E402

NOW_MS = 1_760_000_000_000


class FakeEmbedder:
    name = "fake:8"
    dim = 8

    def embed(self, text: str) -> list[float]:
        return list(hash_label_embedding(text, 8))

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


def _persistence(tmp_path, **kw) -> MemoryPersistence:
    kw.setdefault("snapshot_interval_s", 1e9)
    kw.setdefault("journal_fsync_s", 0.02)
    return MemoryPersistence(tmp_path, **kw)


def _mem(tmp_path, **kw) -> SpatialMemory:
    pkw = {k: kw.pop(k) for k in list(kw) if k in ("snapshot_interval_s", "max_records")}
    kw.setdefault("rooms", RoomMap.default())
    return SpatialMemory(epoch_size_ms=5000, persistence=_persistence(tmp_path, **pkw), **kw)


def _room_obs(mem, label, pos, ts, room="kitchen", conf=0.9):
    return mem.observe(
        label,
        cx=0.5,
        cy=0.5,
        confidence=conf,
        timestamp_ms=ts,
        min_confidence=0.0,
        xyz_cam_m=(pos[0], 1.4 - pos[1], 2.5 - pos[2]),
        pos_room_m=pos,
        room=room,
        anchored=True,
        anchor_confidence=1.0,
        source="sim",
    )


def _wait_journal(p: MemoryPersistence, lines: int, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = p.status()
        if st["queue"] == 0 and st["journal_lines"] >= lines and not st["unsynced"]:
            return
        time.sleep(0.01)
    raise AssertionError(f"journal never reached {lines} lines: {p.status()}")


def _journal_ops(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _obs_tuple(o):
    return (
        o.label,
        o.state_id,
        o.timestamp_ms,
        o.confidence,
        o.room,
        o.frame,
        o.pos_m,
        o.xyz_cam_m,
        o.cx,
        o.cy,
        o.depth_m,
        o.anchored,
        o.anchor_confidence,
        o.source,
        o.first_seen_ms,
        o.merge_count,
        o.scene_id,
    )


# ---------------------------------------------------------------------------
# MemoryPersistence on its own
# ---------------------------------------------------------------------------


def test_load_empty_dir_and_disabled(tmp_path) -> None:
    p = _persistence(tmp_path / "fresh")
    recs, header, report = p.load()
    assert recs == [] and header == {}
    assert report.source == "empty" and report.records == 0
    assert report.to_dict()["source"] == "empty"

    off = MemoryPersistence(tmp_path / "off", enabled=False)
    assert off.load()[2].source == "disabled"
    off.begin_session({"format": 2}, [])
    off.append_upsert(_record("a", 1))
    assert not (tmp_path / "off").exists()
    assert off.status()["enabled"] is False
    off.request_snapshot(wait=True)
    off.clear()
    off.close()


def _record(rid: str, ts: int, scene="kitchen/cup", label="cup", room="kitchen") -> dict:
    return {
        "id": rid,
        "scene_id": scene,
        "label": label,
        "room": room,
        "frame": "room",
        "x": 0.5,
        "y": 0.3,
        "z": 0.2,
        "timestamp_ms": ts,
        "confidence": 0.9,
        "prev_state_id": None,
        "metadata": {"label": label, "pos_m": [0.0, 0.75, 1.2], "first_seen_ms": ts},
    }


def test_coerce_record_rejects_malformed() -> None:
    good = coerce_record(_record("a", 5))
    assert good is not None and good["id"] == "a" and good["frame"] == "room"
    assert coerce_record(None) is None
    assert coerce_record({"id": "a"}) is None
    bad_x = _record("a", 5)
    bad_x["x"] = 1.5
    assert coerce_record(bad_x) is None
    bad_frame = _record("a", 5)
    bad_frame["frame"] = "orbit"
    assert coerce_record(bad_frame) is None
    neg = _record("a", -1)
    assert coerce_record(neg) is None
    no_meta = _record("a", 5)
    no_meta["metadata"] = "nope"
    assert coerce_record(no_meta)["metadata"] == {}


def test_snapshot_now_is_atomic_and_rotates_bak(tmp_path) -> None:
    p = _persistence(tmp_path)
    p.snapshot_now({"format": 2, "embedder": {"name": "hash:128", "dim": 128}}, [_record("a", 1)])
    assert (tmp_path / SNAPSHOT_NAME).exists()
    assert not (tmp_path / BAK_NAME).exists()
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]
    p.snapshot_now({"format": 2}, [_record("b", 2), _record("a", 1)])
    data = json.loads((tmp_path / SNAPSHOT_NAME).read_text())
    assert [r["id"] for r in data["records"]] == ["a", "b"]  # newest last
    assert data["format"] == 2 and "saved_at_ms" in data
    bak = json.loads((tmp_path / BAK_NAME).read_text())
    assert [r["id"] for r in bak["records"]] == ["a"]
    assert (tmp_path / JOURNAL_NAME).exists()
    assert (tmp_path / JOURNAL_NAME).stat().st_size == 0
    assert p.status()["snapshots"] == 2
    p.close()


def test_journal_overlay_and_torn_tail(tmp_path) -> None:
    p = _persistence(tmp_path)
    p.begin_session({"format": 2}, [_record("a", 1), _record("b", 2)])
    p.append_upsert(_record("c", 3))
    p.append_delete("a")
    newer_b = _record("b", 4)
    p.append_upsert(newer_b)
    _wait_journal(p, 3)
    p.close(final_snapshot=False)
    assert not p.running
    with open(tmp_path / JOURNAL_NAME, "ab") as fh:
        fh.write(b'{"op":"upsert","rec":{"id":"z","scene')  # torn tail, no newline
    with open(tmp_path / JOURNAL_NAME, "ab") as fh:
        fh.write(b"\n{garbage\n")
    with open(tmp_path / JOURNAL_NAME, "ab") as fh:
        fh.write(b'{"op":"explode","id":"q"}\n')

    p2 = _persistence(tmp_path)
    recs, header, report = p2.load()
    ids = {r["id"]: r for r in recs}
    assert set(ids) == {"b", "c"}
    assert ids["b"]["timestamp_ms"] == 4
    assert report.bad_lines == 3
    assert report.journal_ops == 3
    assert report.source == "snapshot+journal"
    assert report.rooms == ["kitchen"]
    assert report.newest_ms == 4
    assert header["format"] == 2


def test_corrupt_snapshot_is_quarantined_and_bak_used(tmp_path) -> None:
    p = _persistence(tmp_path)
    p.snapshot_now({"format": 2}, [_record("a", 1)])
    p.snapshot_now({"format": 2}, [_record("a", 1), _record("b", 2)])
    (tmp_path / SNAPSHOT_NAME).write_text("{not json")
    p2 = _persistence(tmp_path)
    recs, _, report = p2.load()
    assert [r["id"] for r in recs] == ["a"]
    assert report.source == "bak"
    quarantined = [f for f in os.listdir(tmp_path) if f.startswith("memory.snapshot.corrupt-")]
    assert len(quarantined) == 1
    assert not (tmp_path / SNAPSHOT_NAME).exists()
    assert p2.status()["errors"] == 1
    # both corrupt -> "corrupt" with zero records, nothing raised
    (tmp_path / SNAPSHOT_NAME).write_text("[]")
    (tmp_path / BAK_NAME).write_text("nope")
    p3 = _persistence(tmp_path)
    recs3, _, report3 = p3.load()
    assert recs3 == [] and report3.source == "corrupt"


def test_max_records_prune_on_load_and_snapshot(tmp_path) -> None:
    p = _persistence(tmp_path, max_records=3)
    p.snapshot_now({"format": 2}, [_record(str(i), i) for i in range(6)])
    data = json.loads((tmp_path / SNAPSHOT_NAME).read_text())
    assert [r["id"] for r in data["records"]] == ["3", "4", "5"]
    p.begin_session({"format": 2}, [_record(str(i), i) for i in range(3, 6)])
    p.append_upsert(_record("9", 9))
    p.append_upsert(_record("0", 0))  # older than everything -> pruned on load
    _wait_journal(p, 2)
    p.close(final_snapshot=False)
    recs, _, report = _persistence(tmp_path, max_records=3).load()
    assert [r["id"] for r in recs] == ["4", "5", "9"]
    assert report.records == 3


def test_writer_thread_lifecycle_and_status(tmp_path) -> None:
    calls: list[int] = []

    def provider():
        calls.append(1)
        return {"format": 2, "embedder": {"name": "x", "dim": 1}}, [_record("a", 1)]

    p = _persistence(tmp_path, snapshot_interval_s=0.05)
    p.begin_session({"format": 2}, [], provider=provider)
    assert p.running
    st = p.status()
    for key in (
        "enabled",
        "data_dir",
        "last_snapshot_ms",
        "journal_lines",
        "journal_bytes",
        "queue",
        "errors",
        "last_restore",
    ):
        assert key in st
    p.append_upsert(_record("b", 2))
    _wait_journal(p, 1)
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls, "periodic snapshot never ran"
    ev = p.request_snapshot(wait=True)
    assert ev.is_set()
    assert p.status()["journal_lines"] == 0  # truncated after the snapshot
    p.close()
    assert not p.running
    assert p._thread is not None and not p._thread.is_alive()
    p.close()  # idempotent
    data = json.loads((tmp_path / SNAPSHOT_NAME).read_text())
    assert [r["id"] for r in data["records"]] == ["a"]


def test_clear_removes_files(tmp_path) -> None:
    p = _persistence(tmp_path)
    p.begin_session({"format": 2}, [_record("a", 1)], provider=lambda: ({"format": 2}, []))
    p.append_upsert(_record("b", 2))
    _wait_journal(p, 1)
    p.clear(wait=True)
    assert not (tmp_path / SNAPSHOT_NAME).exists()
    assert not (tmp_path / JOURNAL_NAME).exists()
    p.append_upsert(_record("c", 3))
    _wait_journal(p, 1)
    p.close(final_snapshot=False)
    recs, _, report = _persistence(tmp_path).load()
    assert [r["id"] for r in recs] == ["c"] and report.source == "empty"


def test_from_env(tmp_path) -> None:
    env = {
        "LOCI_DATA_DIR": str(tmp_path / "d"),
        "LOCI_PERSIST": "1",
        "LOCI_SNAPSHOT_INTERVAL_S": "7",
        "LOCI_JOURNAL_FSYNC_S": "0.5",
        "LOCI_MAX_RECORDS": "42",
    }
    p = MemoryPersistence.from_env(env=env)
    assert p.data_dir == tmp_path / "d"
    assert p.enabled and p.snapshot_interval_s == 7 and p.journal_fsync_s == 0.5
    assert p.max_records == 42
    assert MemoryPersistence.from_env(env={**env, "LOCI_PERSIST": "0"}).enabled is False


# ---------------------------------------------------------------------------
# SpatialMemory + persistence
# ---------------------------------------------------------------------------


def test_observe_flush_restore_identical_results(tmp_path) -> None:
    mem = _mem(tmp_path)
    report0 = mem.restore()
    assert report0.source == "empty" and report0.records == 0
    t0 = NOW_MS
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0)
    _room_obs(mem, "cup", (1.1, 0.8, 1.0), t0 + 500)  # merge
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), t0 + 1000)
    _room_obs(mem, "laptop", (-0.8, 0.7, 1.0), t0 + 60_000, room="living_room")
    mem.observe("book", cx=0.2, cy=0.3, confidence=0.8, timestamp_ms=t0 + 70_000, min_confidence=0)
    mem.observe(
        "remote",
        cx=0.5,
        cy=0.5,
        confidence=0.8,
        timestamp_ms=t0 + 80_000,
        min_confidence=0,
        xyz_cam_m=(0.2, -0.1, 2.0),
    )
    before = {
        lab: [_obs_tuple(o) for o in mem.where_is(lab, limit=10)]
        for lab in ("cup", "bottle", "laptop", "book", "remote")
    }
    assert all(not o.restored for lab in before for o in mem.where_is(lab))
    stats_before = mem.stats()
    mem.flush()
    assert (tmp_path / SNAPSHOT_NAME).exists()
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]
    mem.close()
    assert not mem.persistence.running

    mem2 = _mem(tmp_path)
    report = mem2.restore()
    assert report.records == 5
    assert report.source == "snapshot"
    assert sorted(report.rooms) == ["kitchen", "living_room"]
    assert report.newest_ms == t0 + 80_000
    assert report.snapshot_embedder == "hash:128"
    after = {
        lab: [_obs_tuple(o) for o in mem2.where_is(lab, limit=10)]
        for lab in ("cup", "bottle", "laptop", "book", "remote")
    }
    # ids differ (fresh insert_batch); everything else identical
    strip = lambda rows: [t[:1] + t[2:] for t in rows]  # noqa: E731
    for lab in before:
        assert strip(after[lab]) == strip(before[lab]), lab
        assert all(o.restored for o in mem2.where_is(lab, limit=10))
    assert mem2.where_is("cup")[0].to_dict()["restored"] is True
    assert mem2.observation_count == stats_before["observation_count"]
    assert sorted(mem2.tracked_labels) == sorted(mem.tracked_labels)
    assert mem2.rooms_seen == ["kitchen", "living_room"]
    assert mem2.stats()["frames"] == {"image": 1, "camera": 1, "room": 3}
    assert mem2.stats()["persistence"]["last_restore"]["records"] == 5
    assert loci_bridge.point_count(mem2._client) == 5
    assert mem2.restore_report is report
    assert mem2.restore() is report  # idempotent
    # a new sighting after restore is not "restored" and merges with the restored one
    new_id = _room_obs(mem2, "cup", (1.05, 0.8, 1.0), t0 + 90_000 - 85_000)
    assert new_id is not None
    cup = mem2.where_is("cup")
    assert len(cup) == 1 and not cup[0].restored and cup[0].merge_count == 2
    mem2.close()


def test_journal_overlay_applied_on_restore_after_crash(tmp_path) -> None:
    mem = _mem(tmp_path)
    mem.restore()
    t0 = NOW_MS
    first = _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0)
    mem.flush()  # snapshot holds `first`
    assert mem.persistence.status()["journal_lines"] == 0
    second = _room_obs(
        mem, "cup", (1.1, 0.8, 1.0), t0 + 500
    )  # merge -> delete first, upsert second
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), t0 + 1000)
    _wait_journal(mem.persistence, 3)
    ops = _journal_ops(tmp_path / JOURNAL_NAME)
    assert [o["op"] for o in ops] == ["delete", "upsert", "upsert"]
    assert ops[0]["id"] == first and ops[1]["rec"]["id"] == second
    mem.persistence.close(final_snapshot=False)  # "crash": no final snapshot

    mem2 = _mem(tmp_path)
    report = mem2.restore()
    assert report.source == "snapshot+journal"
    assert report.records == 2
    assert report.journal_ops == 3
    cups = mem2.where_is("cup", limit=10)
    assert len(cups) == 1
    assert cups[0].merge_count == 1 and cups[0].first_seen_ms == t0
    assert cups[0].pos_m[0] == pytest.approx(1.05)
    assert mem2.where_is("bottle")[0].room == "kitchen"
    assert loci_bridge.point_count(mem2._client) == 2
    mem2.close()


def test_begin_session_truncates_journal_before_first_write(tmp_path) -> None:
    mem = _mem(tmp_path)
    mem.restore()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), NOW_MS + 1000)
    _wait_journal(mem.persistence, 2)
    mem.persistence.close(final_snapshot=False)
    assert len(_journal_ops(tmp_path / JOURNAL_NAME)) == 2

    mem2 = _mem(tmp_path)
    mem2.restore()
    # synchronous snapshot with the live (new) ids, journal empty
    assert (tmp_path / JOURNAL_NAME).stat().st_size == 0
    assert mem2.persistence.status()["journal_lines"] == 0
    snap = json.loads((tmp_path / SNAPSHOT_NAME).read_text())
    live_ids = {r["id"] for r in mem2.export_records()}
    assert {r["id"] for r in snap["records"]} == live_ids
    assert snap["observation_count"] == 2
    assert snap["embedder"] == {"name": "hash:128", "dim": 128}
    assert snap["epoch_size_ms"] == 5000
    # the first post-restore write lands in the fresh journal and refers to a live id
    rid = _room_obs(mem2, "laptop", (0.0, 0.7, 1.0), NOW_MS + 5000)
    _wait_journal(mem2.persistence, 1)
    ops = _journal_ops(tmp_path / JOURNAL_NAME)
    assert ops[0]["op"] == "upsert" and ops[0]["rec"]["id"] == rid
    mem2.close()
    # clean close -> final snapshot, journal truncated again
    assert (tmp_path / JOURNAL_NAME).stat().st_size == 0
    assert len(json.loads((tmp_path / SNAPSHOT_NAME).read_text())["records"]) == 3


def test_first_observe_without_restore_starts_the_session(tmp_path) -> None:
    mem = _mem(tmp_path)
    mem.restore()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    mem.close()

    mem2 = _mem(tmp_path)  # nobody calls restore()
    _room_obs(mem2, "bottle", (-0.6, 0.9, 1.2), NOW_MS + 1000)
    assert mem2.restore_report is not None and mem2.restore_report.records == 1
    assert {o.label for o in mem2.current_objects()} == {"cup", "bottle"}
    assert mem2.where_is("cup")[0].restored and not mem2.where_is("bottle")[0].restored
    mem2.close()
    recs, _, _ = _persistence(tmp_path).load()
    assert {r["label"] for r in recs} == {"cup", "bottle"}


def test_restore_with_different_embedder_re_embeds(tmp_path) -> None:
    mem = _mem(tmp_path, embedder=FakeEmbedder())
    mem.restore()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    mem.observe("keys", cx=0.4, cy=0.4, confidence=0.9, timestamp_ms=NOW_MS, min_confidence=0)
    mem.close()
    snap = json.loads((tmp_path / SNAPSHOT_NAME).read_text())
    assert snap["embedder"]["name"] == "fake:8"
    assert all("vector" not in r for r in snap["records"])

    mem2 = _mem(tmp_path)  # default hash embedder, 128-d client
    report = mem2.restore()
    assert report.records == 2 and report.snapshot_embedder == "fake:8"
    assert mem2.where_is("cup")[0].pos_m == pytest.approx((1.0, 0.8, 1.0))
    assert mem2.where_is("keys")[0].cx == pytest.approx(0.4)
    rec = next(r for r in mem2.export_records() if r["label"] == "cup")
    assert rec["metadata"]["embedder"] == "hash:128"
    hits = mem2._client.query(vector=hash_label_embedding("keys", 128), limit=5)
    assert any(h.scene_id == "keys" for h in hits)
    mem2.close()


def test_max_records_prune_through_memory(tmp_path) -> None:
    mem = _mem(tmp_path, max_records=3)
    mem.restore()
    for i, lab in enumerate(["cup", "bottle", "laptop", "book", "remote"]):
        _room_obs(mem, lab, (0.2 * i, 0.8, 1.0), NOW_MS + i * 10_000)
    mem.close()
    mem2 = _mem(tmp_path, max_records=3)
    report = mem2.restore()
    assert report.records == 3
    assert sorted(mem2.tracked_labels) == ["book", "laptop", "remote"]
    mem2.close()


def test_clear_through_memory_removes_files_and_future_restores_empty(tmp_path) -> None:
    mem = _mem(tmp_path)
    mem.restore()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    mem.flush()
    mem.clear()
    assert not (tmp_path / SNAPSHOT_NAME).exists()
    assert mem.where_is("cup") == []
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), NOW_MS + 1000)
    mem.close()
    mem2 = _mem(tmp_path)
    report = mem2.restore()
    assert report.records == 1 and mem2.tracked_labels == ["bottle"]
    mem2.close()


def test_periodic_snapshot_from_writer_thread(tmp_path) -> None:
    mem = _mem(tmp_path, snapshot_interval_s=0.05)
    mem.restore()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        st = mem.persistence.status()
        if st["snapshots"] >= 2 and st["journal_lines"] == 0:
            break
        time.sleep(0.01)
    else:
        raise AssertionError(mem.persistence.status())
    snap = json.loads((tmp_path / SNAPSHOT_NAME).read_text())
    assert [r["label"] for r in snap["records"]] == ["cup"]
    mem.close()


def test_restore_report_dataclass() -> None:
    r = RestoreReport(3, ["kitchen"], 10, 1, "snapshot")
    d = r.to_dict()
    assert d["records"] == 3 and d["rooms"] == ["kitchen"] and d["source"] == "snapshot"
    assert d["bad_lines"] == 1 and d["newest_ms"] == 10


def test_failed_initial_snapshot_still_starts_the_writer(tmp_path) -> None:
    """An unwritable LOCI_DATA_DIR must not leave the session without a writer thread
    (the queue would grow unbounded and nothing would ever retry); the failure is
    counted and visible in status()."""
    blocker = tmp_path / "data"
    blocker.write_text("not a directory")
    p = MemoryPersistence(blocker, snapshot_interval_s=1e9, journal_fsync_s=0.02)
    mem = SpatialMemory(epoch_size_ms=5000, rooms=RoomMap.default(), persistence=p)
    report = mem.restore()
    assert report.records == 0 and p.running is True
    assert p.status()["errors"] >= 1 and p.status()["running"] is True
    assert _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS) is not None
    assert mem.where_is("cup")[0].label == "cup"  # memory keeps working without disk
    mem.close()
    assert p.running is False


def test_failed_restore_keeps_the_files_it_could_not_load(tmp_path, monkeypatch) -> None:
    mem = _mem(tmp_path)
    mem.restore()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), NOW_MS + 1000)
    mem.flush()
    _room_obs(mem, "laptop", (-0.8, 0.7, 1.0), NOW_MS + 2000)
    mem.close()  # final snapshot: cup, bottle, laptop

    mem2 = _mem(tmp_path)

    def boom(_records):
        raise RuntimeError("index build failed")

    monkeypatch.setattr(mem2, "_insert_restored", boom)
    report = mem2.restore()
    assert report.source == "error" and "kept memory.snapshot.failed-" in report.detail
    _room_obs(mem2, "book", (0.0, 0.8, 1.0), NOW_MS + 3000)
    mem2.flush()
    mem2.flush()  # a second snapshot would have overwritten the .bak too
    mem2.close()
    kept = sorted(f for f in os.listdir(tmp_path) if ".failed-" in f)
    assert {f.split(".failed-")[0] for f in kept} == {
        "memory.journal",
        "memory.snapshot",
        "memory.snapshot.bak",
    }

    def labels(name):
        with open(tmp_path / name, encoding="utf-8") as fh:
            return sorted(r["label"] for r in json.load(fh)["records"])

    assert labels(next(f for f in kept if f.startswith("memory.snapshot.failed-"))) == [
        "bottle",
        "cup",
        "laptop",
    ]
    assert labels(SNAPSHOT_NAME) == ["book"]  # the new session's own memory


def test_restore_failure_reports_error_and_keeps_persisting(tmp_path, monkeypatch) -> None:
    p = _persistence(tmp_path)
    mem = SpatialMemory(epoch_size_ms=5000, rooms=RoomMap.default(), persistence=p)
    monkeypatch.setattr(p, "load", lambda: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    report = mem.restore()
    assert report.source == "error" and "disk on fire" in report.detail
    assert mem.restore() is report  # idempotent, no second attempt
    assert p.running is True
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    mem.flush()
    assert p.snapshot_path.exists()
    mem.close()
