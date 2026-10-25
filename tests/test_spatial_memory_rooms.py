"""Tests for the handheld SpatialMemory (T4): room/camera/image frames, scene ids,
keyed merge, exact-recency ``where_is``, ``changes()``, label resolution, locking.

Fakes only: no persistence on disk (see ``test_persistence.py``), hash embedder.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app import loci_bridge  # noqa: E402
from app.label_embedding import HashLabelEmbedder, hash_label_embedding  # noqa: E402
from app.room_anchor import CAMERA_BOX, RoomMap, RoomSpec, room_to_unit  # noqa: E402
from app.spatial_memory import (  # noqa: E402
    EMBED_DIM,
    Change,
    ObjectObservation,
    SpatialMemory,
    _label_embedding,
    scene_key,
    split_scene_key,
)

NOW_MS = 1_760_000_000_000  # a fixed "now" far from the epoch boundary logic
H = 3_600_000


class FakeEmbedder:
    """Deterministic tiny embedder (not 'static:' so the resolver stays lexical)."""

    name = "fake:8"
    dim = 8

    def embed(self, text: str) -> list[float]:
        vec = hash_label_embedding(text, 8)
        return list(vec)

    def embed_many(self, texts):
        return [self.embed(t) for t in texts]


def _mem(**kw) -> SpatialMemory:
    kw.setdefault("rooms", RoomMap.default())
    return SpatialMemory(epoch_size_ms=5000, **kw)


def _room_obs(mem, label, pos, ts, room="kitchen", conf=0.9, anchored=True, cam=None, **kw):
    cam = cam if cam is not None else (pos[0], 1.4 - pos[1], 2.5 - pos[2])
    return mem.observe(
        label,
        cx=0.5,
        cy=0.5,
        confidence=conf,
        timestamp_ms=ts,
        min_confidence=0.0,
        xyz_cam_m=cam,
        pos_room_m=pos,
        room=room,
        anchored=anchored,
        anchor_confidence=1.0 if anchored else 0.0,
        source="sim",
        **kw,
    )


# ---------------------------------------------------------------------------
# Scene keys and compatibility exports
# ---------------------------------------------------------------------------


def test_scene_key_layout() -> None:
    assert scene_key("cup", "image", None) == "cup"
    assert scene_key("cup", "camera", "kitchen") == "unanchored/cup"
    assert scene_key("cup", "room", "Living Room") == "living_room/cup"
    assert split_scene_key("cup") == (None, "cup")
    assert split_scene_key("unanchored/cup") == ("unanchored", "cup")
    assert split_scene_key("living_room/cell phone") == ("living_room", "cell phone")
    with pytest.raises(ValueError):
        scene_key("cup", "room", None)
    with pytest.raises(ValueError):
        scene_key("cup", "bogus", None)


def test_legacy_embedding_exports() -> None:
    assert EMBED_DIM == 128
    vec = _label_embedding("keys")
    assert len(vec) == 128
    assert vec == hash_label_embedding("keys", 128)
    assert vec == HashLabelEmbedder().embed("keys")


def test_observe_frames_produce_scene_ids() -> None:
    mem = _mem()
    ts = NOW_MS
    i_id = mem.observe("cup", cx=0.5, cy=0.5, confidence=0.9, timestamp_ms=ts, min_confidence=0)
    c_id = mem.observe(
        "cup",
        cx=0.5,
        cy=0.5,
        confidence=0.9,
        timestamp_ms=ts + 10_000,
        min_confidence=0,
        xyz_cam_m=(0.1, 0.2, 1.5),
    )
    r_id = _room_obs(mem, "cup", (1.0, 0.8, 1.0), ts + 20_000)
    assert len({i_id, c_id, r_id}) == 3
    recs = {r["id"]: r for r in mem.export_records()}
    assert recs[i_id]["scene_id"] == "cup" and recs[i_id]["frame"] == "image"
    assert recs[c_id]["scene_id"] == "unanchored/cup" and recs[c_id]["frame"] == "camera"
    assert recs[r_id]["scene_id"] == "kitchen/cup" and recs[r_id]["frame"] == "room"
    # the payload stored in LOCI carries the same scene_id and frame
    payload = loci_bridge.get_payload(mem._client, r_id)
    assert payload is not None
    assert payload["scene_id"] == "kitchen/cup"
    assert payload["metadata"]["frame"] == "room"
    assert mem.stats()["frames"] == {"image": 1, "camera": 1, "room": 1}


def test_room_frame_unit_coords_and_clipping() -> None:
    mem = _mem()
    pos = (1.0, 0.8, 1.0)
    rid = _room_obs(mem, "cup", pos, NOW_MS)
    rec = {r["id"]: r for r in mem.export_records()}[rid]
    (ux, uy, uz), clipped = room_to_unit(pos, RoomMap.default().extents("kitchen"))
    assert (rec["x"], rec["y"], rec["z"]) == pytest.approx((ux, uy, uz))
    assert rec["metadata"]["clipped"] is False and clipped is False

    far = _room_obs(mem, "bottle", (9.0, 0.5, 1.0), NOW_MS)  # x outside [-3, 3]
    rec_far = {r["id"]: r for r in mem.export_records()}[far]
    assert rec_far["x"] == 1.0
    assert rec_far["metadata"]["clipped"] is True
    assert rec_far["metadata"]["pos_m"] == [9.0, 0.5, 1.0]  # raw metres always kept

    cam_id = mem.observe(
        "book",
        cx=0.5,
        cy=0.5,
        confidence=0.9,
        timestamp_ms=NOW_MS,
        min_confidence=0,
        xyz_cam_m=(0.0, 0.0, 3.0),
    )
    rec_cam = {r["id"]: r for r in mem.export_records()}[cam_id]
    (cx_u, cy_u, cz_u), _ = room_to_unit((0.0, 0.0, 3.0), CAMERA_BOX)
    assert (rec_cam["x"], rec_cam["y"], rec_cam["z"]) == pytest.approx((cx_u, cy_u, cz_u))


def test_metadata_round_trip_through_where_is() -> None:
    mem = _mem()
    rid = _room_obs(mem, "Cup", (1.0, 0.8, 1.0), NOW_MS, cam=(0.3, 0.6, 1.5))
    hits = mem.where_is("cup")
    assert len(hits) == 1
    o = hits[0]
    assert o.state_id == rid
    assert o.label == "cup"
    assert o.room == "kitchen"
    assert o.frame == "room"
    assert o.pos_m == pytest.approx((1.0, 0.8, 1.0))
    assert o.xyz_cam_m == pytest.approx((0.3, 0.6, 1.5))
    assert o.depth_m == pytest.approx(1.5)
    assert o.anchored is True
    assert o.anchor_confidence == pytest.approx(1.0)
    assert o.source == "sim"
    assert o.first_seen_ms == NOW_MS
    assert o.merge_count == 0
    assert o.restored is False
    assert o.scene_id == "kitchen/cup"
    d = o.to_dict()
    for key in (
        "label",
        "cx",
        "cy",
        "confidence",
        "timestamp_ms",
        "age_seconds",
        "last_seen_at",
        "last_seen_time",
        "position_description",
        "state_id",
        "depth_m",
        "room",
        "frame",
        "x_m",
        "y_m",
        "z_m",
        "xyz_cam_m",
        "anchored",
        "anchor_confidence",
        "source",
        "first_seen_ms",
        "restored",
        "scene_id",
    ):
        assert key in d, key
    assert (d["x_m"], d["y_m"], d["z_m"]) == (1.0, 0.8, 1.0)
    assert d["xyz_cam_m"] == [0.3, 0.6, 1.5]
    # legacy image-frame dict has x_m None and the legacy wording
    mem.observe("book", cx=0.1, cy=0.2, confidence=0.9, timestamp_ms=NOW_MS, min_confidence=0)
    legacy = mem.where_is("book")[0].to_dict()
    assert legacy["x_m"] is None and legacy["frame"] == "image"
    assert legacy["position_description"] == "on the left, toward the back"
    assert "depth_m" not in legacy


def test_position_description_metric_has_fallback_wording() -> None:
    o = ObjectObservation(
        label="cup",
        cx=0.5,
        cy=0.5,
        confidence=0.9,
        timestamp_ms=NOW_MS,
        state_id="x",
        room="kitchen",
        frame="room",
        pos_m=(1.0, 0.8, 2.0),
    )
    text = o.position_description
    assert isinstance(text, str) and text
    assert "kitchen" in text or "m" in text
    cam = ObjectObservation(
        label="cup",
        cx=0.5,
        cy=0.5,
        confidence=0.9,
        timestamp_ms=NOW_MS,
        state_id="y",
        frame="camera",
        xyz_cam_m=(0.0, 0.0, 2.0),
    )
    assert cam.position_description


# ---------------------------------------------------------------------------
# where_is across rooms / recency
# ---------------------------------------------------------------------------


def test_where_is_across_rooms_and_room_filter() -> None:
    mem = _mem()
    t0 = NOW_MS
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0, room="kitchen")
    _room_obs(mem, "cup", (-0.8, 0.7, 1.0), t0 + 60_000, room="living_room")
    mem.observe("cup", cx=0.5, cy=0.5, confidence=0.9, timestamp_ms=t0 + 120_000, min_confidence=0)
    mem.observe(
        "cup",
        cx=0.5,
        cy=0.5,
        confidence=0.9,
        timestamp_ms=t0 + 180_000,
        min_confidence=0,
        xyz_cam_m=(0.0, 0.0, 2.0),
    )

    all_hits = mem.where_is("cup", limit=10)
    assert [h.frame for h in all_hits] == ["camera", "image", "room", "room"]
    assert [h.timestamp_ms for h in all_hits] == sorted(
        [h.timestamp_ms for h in all_hits], reverse=True
    )
    assert mem.where_is("cup", limit=1)[0].frame == "camera"

    kitchen = mem.where_is("cup", room="kitchen")
    assert len(kitchen) == 1 and kitchen[0].room == "kitchen"
    living = mem.where_is("cup", room="Living Room")
    assert len(living) == 1 and living[0].room == "living_room"
    assert mem.where_is("cup", room="bedroom") == []
    assert mem.where_is("laptop") == []
    assert sorted(mem.rooms_seen) == ["kitchen", "living_room"]


def test_where_is_newest_first_with_hours_old_points() -> None:
    """Decay must not hide the newest sighting behind older ones (D11 regression)."""
    mem = _mem()
    now = int(time.time() * 1000)
    stamps = [now - 30 * H, now - 3 * H, now - 2 * H - 60_000, now - 10_000, now]
    for i, ts in enumerate(stamps):
        mem.observe(
            "cup", cx=0.1 + 0.2 * i, cy=0.5, confidence=0.9, timestamp_ms=ts, min_confidence=0
        )
    hits = mem.where_is("cup", limit=1)
    assert len(hits) == 1 and hits[0].timestamp_ms == now
    hits3 = mem.where_is("cup", limit=3)
    assert [h.timestamp_ms for h in hits3] == sorted(stamps, reverse=True)[:3]
    hits_all = mem.where_is("cup", limit=20)
    assert [h.timestamp_ms for h in hits_all] == sorted(stamps, reverse=True)


def test_where_is_more_than_twenty_points_of_one_label() -> None:
    """argpartition truncation regression: 40 points with tied vectors, newest must win."""
    mem = _mem(dedup_window_ms=1000)
    t0 = NOW_MS
    stamps = [t0 + i * 10_000 for i in range(40)]
    for i, ts in enumerate(stamps):
        mem.observe(
            "book",
            cx=(i % 10) / 10,
            cy=(i // 10) / 4,
            confidence=0.9,
            timestamp_ms=ts,
            min_confidence=0,
        )
    assert loci_bridge.point_count(mem._client) == 40
    top5 = mem.where_is("book", limit=5)
    assert [h.timestamp_ms for h in top5] == stamps[-1:-6:-1]
    everything = mem.where_is("book", limit=40)
    assert len(everything) == 40
    assert [h.timestamp_ms for h in everything] == stamps[::-1]
    assert mem.history("book", limit=3)[0].timestamp_ms == stamps[-1]


def test_where_is_min_confidence_uses_qualifying_window() -> None:
    mem = _mem()
    mem.observe("cup", cx=0.1, cy=0.1, confidence=0.95, timestamp_ms=NOW_MS, min_confidence=0)
    mem.observe(
        "cup", cx=0.9, cy=0.9, confidence=0.4, timestamp_ms=NOW_MS + 60_000, min_confidence=0
    )
    assert len(mem.where_is("cup")) == 2
    strong = mem.where_is("cup", min_confidence=0.7)
    assert len(strong) == 1 and strong[0].confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_merge_is_keyed_and_never_queries_the_client(monkeypatch) -> None:
    mem = _mem()
    t0 = NOW_MS
    first = _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0)

    def boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("merge must not run a vector query")

    monkeypatch.setattr(mem._client, "query", boom)
    second = _room_obs(mem, "cup", (1.1, 0.8, 1.05), t0 + 500, conf=0.7)
    assert second is not None and second != first
    monkeypatch.undo()
    hits = mem.where_is("cup", limit=10)
    assert len(hits) == 1
    assert hits[0].state_id == second
    assert loci_bridge.point_count(mem._client) == 1


def test_metric_merge_within_radius_keeps_first_seen_and_counts() -> None:
    mem = _mem(dedup_radius_m=0.35)
    t0 = NOW_MS
    first = _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0, conf=0.6)
    second = _room_obs(mem, "cup", (1.2, 0.8, 1.0), t0 + 1000, conf=0.9)  # 0.2 m away
    third = _room_obs(mem, "cup", (1.25, 0.85, 1.0), t0 + 2000, conf=0.8)
    assert len({first, second, third}) == 3
    hits = mem.where_is("cup", limit=10)
    assert len(hits) == 1
    o = hits[0]
    assert o.state_id == third
    assert o.first_seen_ms == t0
    assert o.merge_count == 2
    assert o.timestamp_ms == t0 + 2000
    assert o.confidence == pytest.approx(0.9)
    # confidence weighted mean of the chain
    x_after_second = (0.6 * 1.0 + 0.9 * 1.2) / 1.5
    x_expected = (0.9 * x_after_second + 0.8 * 1.25) / 1.7
    assert o.pos_m[0] == pytest.approx(x_expected, abs=1e-6)
    assert loci_bridge.point_count(mem._client) == 1
    assert loci_bridge.get_payload(mem._client, first) is None
    assert loci_bridge.get_payload(mem._client, second) is None
    assert loci_bridge.get_payload(mem._client, third) is not None
    assert mem.observation_count == 3


def test_metric_merge_beyond_radius_inserts_new_record() -> None:
    mem = _mem(dedup_radius_m=0.35)
    a = _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    b = _room_obs(mem, "cup", (1.5, 0.8, 1.0), NOW_MS + 1000)  # 0.5 m away
    assert a != b
    assert len(mem.where_is("cup", limit=10)) == 2


def test_no_merge_across_anchored_states_or_rooms() -> None:
    mem = _mem()
    t0 = NOW_MS
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0, anchored=True)
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0 + 500, anchored=False)  # same spot, held pose
    assert len(mem.where_is("cup", limit=10)) == 2

    mem2 = _mem()
    _room_obs(mem2, "cup", (1.0, 0.8, 1.0), t0, room="kitchen")
    _room_obs(mem2, "cup", (1.0, 0.8, 1.0), t0 + 500, room="living_room")
    assert len(mem2.where_is("cup", limit=10)) == 2

    mem3 = _mem()
    mem3.observe("cup", cx=0.5, cy=0.5, confidence=0.9, timestamp_ms=t0, min_confidence=0,
                 xyz_cam_m=(0.0, 0.0, 1.0))  # fmt: skip
    _room_obs(mem3, "cup", (0.0, 1.4, 1.5), t0 + 500, cam=(0.0, 0.0, 1.0))
    assert len(mem3.where_is("cup", limit=10)) == 2


def test_merge_moves_point_in_region_queries() -> None:
    mem = _mem()
    t0 = NOW_MS
    old_id = _room_obs(mem, "cup", (-2.0, 0.8, 1.0), t0)
    new_id = _room_obs(mem, "cup", (-1.8, 0.8, 1.0), t0 + 1000)
    assert new_id != old_id
    ext = RoomMap.default().extents("kitchen")
    merged = mem.where_is("cup")[0]
    assert merged.state_id == new_id
    assert merged.pos_m[0] == pytest.approx(-1.9)  # equal confidences -> plain mean
    (ux, uy, _), _ = room_to_unit(merged.pos_m, ext)
    found = mem.objects_in_region(ux - 0.005, ux + 0.005, uy - 0.05, uy + 0.05, room="kitchen")
    assert [o.state_id for o in found] == [new_id]
    (ox, oy, _), _ = room_to_unit((-2.0, 0.8, 1.0), ext)  # 0.1 m = 0.0167 unit away
    stale = mem.objects_in_region(ox - 0.005, ox + 0.005, oy - 0.05, oy + 0.05, room="kitchen")
    assert stale == []
    # the same box without a room filter also finds only the merged point
    assert [o.state_id for o in mem.objects_in_region(ux - 0.005, ux + 0.005, 0, 1)] == [new_id]
    assert mem.objects_in_region(0, 1, 0, 1, room="living_room") == []


def test_image_frame_merge_still_uses_iou_proxy() -> None:
    mem = _mem(dedup_iou_threshold=0.5)
    a = mem.observe("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=NOW_MS, min_confidence=0)
    b = mem.observe(
        "cup", cx=0.51, cy=0.51, confidence=0.7, timestamp_ms=NOW_MS + 100, min_confidence=0
    )
    assert a != b
    hits = mem.where_is("cup")
    assert len(hits) == 1 and hits[0].merge_count == 1 and hits[0].first_seen_ms == NOW_MS


# ---------------------------------------------------------------------------
# current_objects / recent_changes / changes
# ---------------------------------------------------------------------------


def test_current_objects_per_room() -> None:
    mem = _mem()
    t0 = NOW_MS
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0, room="kitchen")
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), t0 + 1, room="kitchen")
    _room_obs(mem, "cup", (-0.8, 0.7, 1.0), t0 + 60_000, room="living_room")
    allobjs = mem.current_objects()
    assert {o.label for o in allobjs} == {"cup", "bottle"}
    assert allobjs[0].label == "cup" and allobjs[0].room == "living_room"
    kitchen = mem.current_objects(room="kitchen")
    assert {o.label for o in kitchen} == {"cup", "bottle"}
    assert all(o.room == "kitchen" for o in kitchen)
    assert [o.label for o in mem.current_objects(room="living_room")] == ["cup"]
    assert mem.current_objects(room="bedroom") == []
    assert sorted(mem.tracked_labels) == ["bottle", "cup"]


def test_recent_changes_window() -> None:
    mem = _mem()
    now = int(time.time() * 1000)
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), now - 120_000)
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), now - 1000)
    recent = mem.recent_changes(window_seconds=30)
    assert [o.label for o in recent] == ["bottle"]
    assert {o.label for o in mem.recent_changes(window_seconds=300)} == {"cup", "bottle"}


def test_changes_new_moved_seen_gone_with_presence() -> None:
    mem = _mem(moved_threshold_m=0.5)
    t0 = NOW_MS
    # cup: seen at A long ago, then at B (2 m away) inside the window -> moved
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0)
    _room_obs(mem, "cup", (-1.2, 0.8, 1.5), t0 + 700_000)
    # bottle: first sighting inside the window -> new
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), t0 + 650_000)
    # book: seen long ago and again at the same place inside the window -> seen
    _room_obs(mem, "book", (0.5, 0.75, 2.0), t0)
    _room_obs(mem, "book", (0.55, 0.75, 2.0), t0 + 690_000)
    # laptop: last seen long ago; camera was in the kitchen for 8 s recently -> gone
    _room_obs(mem, "laptop", (2.0, 0.75, 2.0), t0 + 10_000)
    # remote: last seen long ago in the living room, no presence there -> not reported
    _room_obs(mem, "remote", (0.0, 0.5, 1.0), t0 + 10_000, room="living_room")
    now = t0 + 710_000
    for s in range(9):
        mem.note_presence("kitchen", now - 9_000 + s * 1000)

    changes = mem.changes("kitchen", 600, now_ms=now)
    by_label = {c.label: c for c in changes}
    assert [c.kind for c in changes] == ["moved", "new", "gone", "seen"]
    assert by_label["cup"].kind == "moved"
    assert by_label["cup"].pos_before_m == pytest.approx((1.0, 0.8, 1.0))
    assert by_label["cup"].pos_m == pytest.approx((-1.2, 0.8, 1.5))
    assert by_label["cup"].delta_m == pytest.approx(math.dist((1.0, 0.8, 1.0), (-1.2, 0.8, 1.5)))
    assert by_label["bottle"].kind == "new"
    assert by_label["book"].kind == "seen"
    assert by_label["laptop"].kind == "gone"
    assert by_label["laptop"].last_seen_ms == t0 + 10_000
    assert "remote" not in by_label
    assert all(c.room == "kitchen" for c in changes)
    d = by_label["cup"].to_dict()
    assert set(d) == {
        "kind",
        "label",
        "room",
        "last_seen_ms",
        "first_seen_ms",
        "pos_m",
        "pos_before_m",
        "delta_m",
    }

    everything = mem.changes(None, 600, now_ms=now)
    assert {c.label for c in everything} == {"cup", "bottle", "book", "laptop"}
    # without presence nothing is "gone"
    mem2 = _mem()
    _room_obs(mem2, "laptop", (2.0, 0.75, 2.0), t0)
    assert mem2.changes("kitchen", 600, now_ms=t0 + 700_000) == []
    assert isinstance(Change("new", "cup", None, 0, None, None, None, None), Change)


def test_changes_reports_moved_within_a_fresh_run():
    """A cup first seen 20 s ago and displaced 10 s ago is "moved", not "new" (demo story)."""
    mem = _mem(moved_threshold_m=0.5)
    t0 = NOW_MS
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), t0 + 10_000)
    _room_obs(mem, "cup", (-1.2, 0.8, 1.5), t0 + 20_000)
    # bottle: two sightings inside the window at (nearly) the same place -> new
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.2), t0 + 10_000)
    _room_obs(mem, "bottle", (-0.6, 0.9, 1.6), t0 + 20_000)
    changes = mem.changes("kitchen", 600, now_ms=t0 + 30_000)
    by_label = {c.label: c for c in changes}
    assert [c.kind for c in changes] == ["moved", "new"]
    assert by_label["cup"].pos_before_m == pytest.approx((1.0, 0.8, 1.0))
    assert by_label["cup"].delta_m == pytest.approx(math.dist((1.0, 0.8, 1.0), (-1.2, 0.8, 1.5)))
    assert by_label["cup"].first_seen_ms == t0 + 10_000
    assert by_label["bottle"].kind == "new"


def test_note_presence_merges_intervals() -> None:
    mem = _mem()
    mem.note_presence("kitchen", 1000)
    mem.note_presence("kitchen", 2000)
    mem.note_presence("kitchen", 9000)  # gap >= 5 s -> new interval
    assert mem._presence["kitchen"] == [[1000, 2000], [9000, 9000]]
    assert mem._presence_ms("kitchen", 0, 10_000) == 1000
    assert mem._presence_ms("kitchen", 1500, 1800) == 300


# ---------------------------------------------------------------------------
# objects_in_region / resolve_label / stats
# ---------------------------------------------------------------------------


def test_objects_in_region_dedupes_newest_first_and_legacy_path() -> None:
    mem = _mem()
    mem.observe("cup", cx=0.2, cy=0.2, confidence=0.9, timestamp_ms=NOW_MS, min_confidence=0)
    mem.observe(
        "cup", cx=0.25, cy=0.6, confidence=0.9, timestamp_ms=NOW_MS + 60_000, min_confidence=0
    )
    mem.observe("bottle", cx=0.3, cy=0.3, confidence=0.9, timestamp_ms=NOW_MS, min_confidence=0)
    mem.observe("phone", cx=0.9, cy=0.9, confidence=0.9, timestamp_ms=NOW_MS, min_confidence=0)
    found = mem.objects_in_region(0.0, 0.5, 0.0, 1.0)
    labels = [o.label for o in found]
    assert sorted(labels) == ["bottle", "cup"]
    cup = next(o for o in found if o.label == "cup")
    assert cup.timestamp_ms == NOW_MS + 60_000  # newest of the duplicates
    assert mem.objects_in_region(0.8, 1.0, 0.8, 1.0)[0].label == "phone"


def test_resolve_label_uses_detector_vocabulary_and_tracked_labels() -> None:
    mem = _mem()
    mem.set_vocabulary(["cup", "bottle", "cell phone"])
    assert mem.resolve_label("my mug").label == "cup"
    assert mem.resolve_label("mug").how == "alias"
    assert mem.resolve_label("keys").label is None
    mem.observe("keys", cx=0.5, cy=0.5, confidence=0.9, timestamp_ms=NOW_MS, min_confidence=0)
    res = mem.resolve_label("my keys")
    assert res.label == "keys" and res.how == "exact"
    assert res.to_dict()["label"] == "keys"
    assert mem.resolve_label("phone").label == "cell phone"


def test_stats_keys() -> None:
    mem = _mem()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    mem.where_is("cup")
    s = mem.stats()
    for key in (
        "observation_count",
        "tracked_objects",
        "labels",
        "last_query",
        "rooms_seen",
        "frames",
        "persistence",
        "embedder",
        "version",
    ):
        assert key in s, key
    assert s["rooms_seen"] == ["kitchen"]
    assert s["frames"]["room"] == 1
    assert s["persistence"] is None
    assert s["embedder"] == "hash:128"
    assert s["version"] >= 1
    assert s["last_query"]["shards_searched"] >= 1
    assert mem.rooms is not None and mem.rooms.get("kitchen") is not None
    assert mem.started_at_ms > 0


def test_custom_embedder_sizes_the_client_and_rooms_file() -> None:
    rooms = RoomMap(
        [
            RoomSpec(
                name="Study Room", anchor_tag=7, extents_m={"x": (-2, 2), "y": (0, 3), "z": (0, 4)}
            )
        ]
    )
    mem = SpatialMemory(embedder=FakeEmbedder(), rooms=rooms)
    rid = _room_obs(mem, "laptop", (1.0, 1.5, 2.0), NOW_MS, room="Study Room")
    rec = {r["id"]: r for r in mem.export_records()}[rid]
    assert rec["scene_id"] == "study_room/laptop"
    assert (rec["x"], rec["y"], rec["z"]) == pytest.approx((0.75, 0.5, 0.5))
    assert rec["metadata"]["embedder"] == "fake:8"
    hits = mem.where_is("laptop", room="study room")
    assert len(hits) == 1 and hits[0].room == "Study Room"
    assert mem.stats()["embedding_dim"] == 8
    with pytest.raises(ValueError):
        mem._client.query(vector=_label_embedding("laptop"), limit=1)  # 128-d vs 8-d client


def test_unknown_room_uses_default_room_extents() -> None:
    mem = _mem()
    rid = _room_obs(mem, "cup", (0.0, 2.0, 3.0), NOW_MS, room="garage")  # y=2 m: above CAMERA_BOX
    rec = {r["id"]: r for r in mem.export_records()}[rid]
    assert rec["y"] == pytest.approx(0.8)  # 2.0 / 2.5 with the default room extents
    assert rec["metadata"]["clipped"] is False
    assert mem.rooms_seen == ["garage"]


def test_confidence_gate_and_label_normalisation() -> None:
    mem = _mem(label_confidence_overrides={"Cell Phone": 0.9})
    assert mem.observe("cell  phone", cx=0.5, cy=0.5, confidence=0.8, min_confidence=0.0) is None
    sid = mem.observe("Cell Phone", cx=0.5, cy=0.5, confidence=0.95, min_confidence=0.0)
    assert sid is not None
    assert mem.where_is("cell phone")[0].label == "cell phone"
    assert mem.observe("", cx=0.5, cy=0.5, confidence=0.95, min_confidence=0.0) is None
    assert mem.observe("cup", cx=0.5, cy=0.5, confidence=0.3) is None


def test_clear_resets_everything() -> None:
    mem = _mem()
    _room_obs(mem, "cup", (1.0, 0.8, 1.0), NOW_MS)
    v = mem.version
    mem.clear()
    assert mem.where_is("cup") == []
    assert mem.current_objects() == []
    assert mem.observation_count == 0
    assert loci_bridge.point_count(mem._client) == 0
    assert mem.version > v
    assert mem.restore().source == "disabled"
    mem.flush()
    mem.close()


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------


def test_observe_from_thread_while_querying() -> None:
    mem = _mem()
    errors: list[BaseException] = []
    t0 = NOW_MS

    def writer() -> None:
        try:
            for i in range(300):
                _room_obs(mem, "cup", (1.0 + (i % 7) * 0.01, 0.8, 1.0), t0 + i * 200)
                _room_obs(mem, "bottle", (-0.6, 0.9, 1.2 + (i % 5) * 0.5), t0 + i * 200)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    th = threading.Thread(target=writer)
    th.start()
    try:
        deadline = time.monotonic() + 10
        while th.is_alive() and time.monotonic() < deadline:
            mem.where_is("cup", limit=3)
            mem.changes("kitchen", 600, now_ms=t0 + 100_000)
            mem.current_objects(room="kitchen")
            mem.objects_in_region(0, 1, 0, 1, room="kitchen")
            mem.stats()
    finally:
        th.join(10)
    assert not th.is_alive()
    assert errors == []
    assert mem.observation_count == 600
    assert mem.where_is("cup", limit=1)[0].timestamp_ms == t0 + 299 * 200
    # every live record is mirrored by exactly one LOCI point
    assert loci_bridge.point_count(mem._client) == len(mem.export_records())


def test_changes_moved_is_not_masked_by_an_intermediate_sighting() -> None:
    """The first consensus fire after re-entering a room comes from a single, not yet
    smoothed tag pose and lands between the old and the new place; "moved" must be
    decided against the settled old position, not only the previous slot."""
    mem = _mem(moved_threshold_m=0.5)
    t0 = NOW_MS
    old, mid, new = (1.02, 0.76, 0.92), (-0.78, 0.93, 1.76), (-1.13, 0.76, 1.52)
    _room_obs(mem, "cup", old, t0 + 4_000)
    _room_obs(mem, "cup", mid, t0 + 15_000)
    _room_obs(mem, "cup", new, t0 + 17_000)
    assert math.dist(mid, new) < 0.5 < math.dist(old, new)
    changes = mem.changes("kitchen", 600, now_ms=t0 + 20_000)
    assert [c.kind for c in changes] == ["moved"]
    assert changes[0].pos_before_m == pytest.approx(old)
    assert changes[0].pos_m == pytest.approx(new)
    assert changes[0].delta_m == pytest.approx(math.dist(old, new))
