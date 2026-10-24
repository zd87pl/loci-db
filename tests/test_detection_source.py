"""Tests for ``demo_spatial/app/detection_source.py`` (T2): numpy only, no hardware."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.detection_source import (  # noqa: E402
    AXIS_FIX,
    COCO80,
    CameraPose,
    DetectionFrame,
    DetectionSource,
    HostClock,
    JsonlRecorder,
    JsonlReplaySource,
    Scenario,
    SimDetectionSource,
    SpatialDetection,
    TagFrame,
    apply_T,
    camera_pose_T,
    depthai_xyz_to_cam,
    invert_T,
    make_detection_source,
    yaw_deg_of,
)

# ---------------------------------------------------------------------------
# Axis mapping
# ---------------------------------------------------------------------------


def test_axis_fix_tables() -> None:
    assert np.array_equal(AXIS_FIX["rdf"], np.eye(3))
    assert np.array_equal(AXIS_FIX["ruf"], np.diag([1.0, -1.0, 1.0]))


def test_depthai_xyz_to_cam_rdf_is_identity_in_metres() -> None:
    assert depthai_xyz_to_cam((120.0, -50.0, 1840.0)) == pytest.approx((0.12, -0.05, 1.84))


def test_depthai_xyz_to_cam_ruf_flips_y_only() -> None:
    assert depthai_xyz_to_cam((120.0, -50.0, 1840.0), "ruf") == pytest.approx((0.12, 0.05, 1.84))


@pytest.mark.parametrize("xyz", [(0.0, 0.0, 0.0), (10.0, 20.0, 0.0), (10.0, 20.0, -5.0)])
def test_depthai_xyz_to_cam_no_depth_is_none(xyz: tuple[float, float, float]) -> None:
    assert depthai_xyz_to_cam(xyz) is None
    assert depthai_xyz_to_cam(xyz, "ruf") is None


def test_depthai_xyz_to_cam_unknown_axes() -> None:
    with pytest.raises(ValueError, match="axes"):
        depthai_xyz_to_cam((1.0, 1.0, 1000.0), "xyz")


# ---------------------------------------------------------------------------
# HostClock
# ---------------------------------------------------------------------------


def test_host_clock_uses_min_latency_offset() -> None:
    clock = HostClock(max_skew_ms=1000, warmup=3)
    # device steady clock vs arrival: latencies 30, 10, 50 ms -> offset = 1_000_000 + 10
    assert clock.to_unix_ms(1000, 1_001_030) == 1_001_030  # first sample: arrival itself
    assert clock.to_unix_ms(1100, 1_001_110) == 1_001_110  # new min (10 ms) -> exact arrival
    assert clock.to_unix_ms(1200, 1_001_250) == 1_001_210  # 50 ms late -> corrected by min offset
    assert clock.synced
    assert clock.offset_ms == 1_000_010
    # after warm-up the offset is frozen even when a lower-latency sample arrives
    assert clock.to_unix_ms(1300, 1_001_305) == 1_001_310
    assert clock.fallbacks == 0


def test_host_clock_fallback_counts_and_resyncs() -> None:
    clock = HostClock(max_skew_ms=1000, warmup=2)
    clock.to_unix_ms(1000, 5_000_000)
    clock.to_unix_ms(1100, 5_000_100)
    assert clock.synced
    # device rebooted: its clock restarted at ~0 while arrival advanced far beyond max_skew
    assert clock.to_unix_ms(50, 5_020_000) == 5_020_000
    assert clock.fallbacks == 1
    assert not clock.synced
    # and the clock re-arms its warm-up against the new device clock
    assert clock.to_unix_ms(150, 5_020_100) == 5_020_100  # first sample after reset
    assert clock.to_unix_ms(250, 5_020_230) == 5_020_200  # 30 ms late -> corrected by new min
    assert clock.offset_ms == 5_019_950
    assert clock.synced


# ---------------------------------------------------------------------------
# SE(3) helpers (conventions of design §4.1 / §4.4)
# ---------------------------------------------------------------------------


def test_camera_pose_T_known_chain() -> None:
    T = camera_pose_T((0.0, 1.4, 2.5), 0.0)
    assert apply_T(T, (0.0, 0.0, 1.0)) == pytest.approx((0.0, 1.4, 1.5))
    assert apply_T(T, (0.5, 0.0, 1.0)) == pytest.approx((0.5, 1.4, 1.5))  # viewer's right -> +x
    assert apply_T(T, (0.0, 0.3, 1.0)) == pytest.approx((0.0, 1.1, 1.5))  # camera down -> room down
    assert np.allclose(T[:3, :3], np.diag([1.0, -1.0, -1.0]))


@pytest.mark.parametrize("yaw", [0.0, 45.0, 90.0, -30.0, 170.0])
def test_yaw_round_trip_and_heading_definition(yaw: float) -> None:
    T = camera_pose_T((1.0, 1.2, 2.0), yaw)
    assert yaw_deg_of(T) == pytest.approx(yaw)
    inv = invert_T(T)
    assert np.allclose(inv @ T, np.eye(4))
    if yaw == 90.0:
        assert T[:3, 2] == pytest.approx((1.0, 0.0, 0.0))  # +90 looks toward +x


def test_camera_pose_to_dict_flattens_T() -> None:
    pose = CameraPose("kitchen", "room", camera_pose_T((0.0, 1.4, 2.5), 0.0), True, 1.0, (0,), 0.0)
    d = pose.to_dict()
    assert d["camera_xyz_m"] == pytest.approx([0.0, 1.4, 2.5])
    assert d["heading_deg"] == pytest.approx(0.0)
    assert d["tag_ids"] == [0] and d["age_s"] == 0.0 and d["anchored"] is True
    none = CameraPose(None, "camera", None, False, 0.0, (), float("inf")).to_dict()
    assert none["camera_xyz_m"] is None and none["heading_deg"] is None and none["age_s"] is None


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def _run(source: SimDetectionSource) -> list[DetectionFrame | TagFrame]:
    source.start()
    return list(source.events())


def _ev_dicts(events: list[DetectionFrame | TagFrame]) -> list[dict]:
    return [e.to_dict() for e in events]


def test_sim_is_deterministic_for_a_seed() -> None:
    a = _run(SimDetectionSource(seed=3, start_ms=1_700_000_000_000))
    b = _run(SimDetectionSource(seed=3, start_ms=1_700_000_000_000))
    c = _run(SimDetectionSource(seed=4, start_ms=1_700_000_000_000))
    assert _ev_dicts(a) == _ev_dicts(b)
    assert _ev_dicts(a) != _ev_dicts(c)


def test_sim_timeline_and_segment_counts() -> None:
    src = SimDetectionSource(fps=10.0, start_ms=0)
    assert src.frames_total == 180
    events = _run(src)
    dets = [e for e in events if isinstance(e, DetectionFrame)]
    tags = [e for e in events if isinstance(e, TagFrame)]
    assert len(dets) == 180
    assert [d.seq for d in dets] == list(range(180))
    assert [d.timestamp_ms for d in dets] == [i * 100 for i in range(180)]
    assert all(d.source == "sim" for d in dets)

    def in_window(ts: int, lo_s: float, hi_s: float) -> bool:
        return lo_s * 1000 <= ts < hi_s * 1000

    kitchen1 = [d for d in dets if in_window(d.timestamp_ms, 0, 6)]
    hallway = [d for d in dets if in_window(d.timestamp_ms, 6, 9)]
    living = [d for d in dets if in_window(d.timestamp_ms, 9, 13)]
    kitchen2 = [d for d in dets if in_window(d.timestamp_ms, 13, 18)]
    assert (len(kitchen1), len(hallway), len(living), len(kitchen2)) == (60, 30, 40, 50)
    assert all(d.detections == [] for d in hallway)
    assert all({x.label for x in d.detections} == {"cup", "bottle"} for d in kitchen1)
    assert all({x.label for x in d.detections} == {"laptop", "couch"} for d in living)
    assert all({x.label for x in d.detections} == {"cup", "bottle"} for d in kitchen2)

    # tag frames at 5 Hz only in the tag segments: 6 s + 4 s + 5 s = 15 s -> 75 frames
    assert len(tags) == 75
    assert not any(in_window(t.timestamp_ms, 6, 9) for t in tags)
    assert all(t.gray is None and t.tags is not None for t in tags)
    assert {t.tags[0].tag_id for t in tags if in_window(t.timestamp_ms, 0, 6)} == {0}
    assert {t.tags[0].tag_id for t in tags if in_window(t.timestamp_ms, 9, 13)} == {1}
    # a tag frame precedes the detection frame of the same instant
    first_tag = events.index(tags[0])
    assert isinstance(events[first_tag + 1], DetectionFrame)
    assert events[first_tag + 1].timestamp_ms == tags[0].timestamp_ms
    assert src.status().state == "done"
    assert src.status().frames == 180


def test_sim_boxes_and_positions_are_consistent() -> None:
    src = SimDetectionSource(oracle_pose=True, start_ms=0, seed=1)
    events = _run(src)
    assert all(isinstance(e, DetectionFrame) for e in events), "oracle mode emits no tag frames"
    scenario = Scenario.mug_two_rooms()
    for fr in events:
        assert fr.oracle_pose is not None
        for det in fr.detections:
            assert 0.0 <= det.xmin < det.xmax <= 1.0
            assert 0.0 <= det.ymin < det.ymax <= 1.0
            assert det.label in COCO80 and det.label_id == COCO80.index(det.label)
            assert 0.5 <= det.confidence <= 0.99
            assert det.xyz_cam_m is not None and det.depth_m is not None and det.depth_m > 0.3
            pose = fr.oracle_pose
            assert pose.frame == "room" and pose.T_room_cam is not None
            p_room = apply_T(pose.T_room_cam, det.xyz_cam_m)
            seg = scenario.segment_at(fr.timestamp_ms / 1000.0)
            assert seg is not None
            truth = next(o.pos_room_m for o in seg[1].objects if o.label == det.label)
            assert np.linalg.norm(p_room - np.array(truth)) < 0.2
    # the moved cup is observed in the last segment at its new place
    last = [e for e in events if e.timestamp_ms >= 13_000]
    cups = [d for e in last for d in e.detections if d.label == "cup"]
    assert len(cups) == 50
    p = apply_T(last[0].oracle_pose.T_room_cam, cups[0].xyz_cam_m)
    assert np.linalg.norm(p - np.array([-1.2, 0.8, 1.5])) < 0.2


def test_sim_oracle_pose_rooms_and_hallway() -> None:
    events = _run(SimDetectionSource(oracle_pose=True, start_ms=0))
    kitchen = events[0].oracle_pose
    assert kitchen.room == "kitchen" and kitchen.anchored and kitchen.tag_ids == (0,)
    assert kitchen.to_dict()["camera_xyz_m"] == pytest.approx([0.0, 1.4, 2.5])
    assert kitchen.to_dict()["heading_deg"] == pytest.approx(0.0)
    hallway = next(e for e in events if 6_000 <= e.timestamp_ms < 9_000).oracle_pose
    assert hallway.frame == "camera" and hallway.room is None and hallway.T_room_cam is None
    assert not hallway.anchored and hallway.anchor_confidence == 0.0
    living = next(e for e in events if 9_000 <= e.timestamp_ms < 13_000).oracle_pose
    assert living.room == "living_room" and living.tag_ids == (1,)


def test_sim_tag_corners_projected_in_image_order() -> None:
    events = _run(SimDetectionSource(start_ms=0, noise_px=0.0))
    tf = next(e for e in events if isinstance(e, TagFrame))
    assert (tf.width, tf.height) == (1280, 800)
    assert tf.K.shape == (3, 3) and tf.dist is None
    [tag] = tf.tags
    c = tag.corners_px
    assert c.shape == (4, 2) and tag.hamming == 0
    # camera at (0,1.4,2.5) facing a 16 cm tag at (0,1.4,0): side = 918.3*0.16/2.5 ~ 58.8 px
    assert np.linalg.norm(c[1] - c[0]) == pytest.approx(918.3 * 0.16 / 2.5, rel=1e-3)
    tl, tr, br, bl = c
    assert tl[0] < tr[0] and bl[0] < br[0]  # left corners left of right corners
    assert tl[1] < bl[1] and tr[1] < br[1]  # top corners above bottom corners
    assert c.mean(axis=0) == pytest.approx((645.8, 417.4), abs=1e-6)


def test_sim_no_tags_flag_and_emit_tags() -> None:
    assert not any(isinstance(e, TagFrame) for e in _run(SimDetectionSource(emit_tags=False)))


def test_sim_loop_continues_timestamps() -> None:
    src = SimDetectionSource(fps=10.0, start_ms=0, loop=True)
    src.start()
    out: list[DetectionFrame] = []
    for ev in src.events():
        if isinstance(ev, DetectionFrame):
            out.append(ev)
        if len(out) >= 200:
            src.stop()
    assert len(out) == 200
    assert out[180].timestamp_ms == 18_000 and out[199].timestamp_ms == 19_900
    assert out[199].seq == 199
    assert src.status().state == "stopped"


def test_sim_realtime_stop_from_another_thread() -> None:
    src = SimDetectionSource(fps=10.0, realtime=True)
    src.start()
    seen: list[int] = []

    def consume() -> None:
        for ev in src.events():
            if isinstance(ev, DetectionFrame):
                seen.append(ev.seq)

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    time.sleep(0.35)
    src.stop()
    th.join(timeout=2.0)
    assert not th.is_alive()
    assert 1 <= len(seen) <= 8, seen  # ~10 fps for ~0.35 s, never the whole 180
    assert src.status().state == "stopped"
    assert src.status().fps > 0


def test_sim_implements_protocol_and_classes() -> None:
    src = SimDetectionSource()
    assert isinstance(src, DetectionSource)
    assert src.name == "sim" and src.classes == COCO80 and src.classes is not COCO80
    st = src.status().to_dict()
    assert set(st) == {
        "state", "fps", "frames", "reconnects", "last_frame_age_s", "usb_speed", "device", "error"
    }  # fmt: skip
    assert st["state"] == "starting"


def test_scenario_json_round_trip_gives_identical_events() -> None:
    sc = Scenario.mug_two_rooms()
    txt = sc.to_json()
    back = Scenario.from_json(txt)
    assert back.name == sc.name and back.duration_s == sc.duration_s == 18.0
    assert back.rooms == sc.rooms and np.allclose(back.K_tag, sc.K_tag)
    assert json.loads(back.to_json()) == json.loads(txt)
    a = _run(SimDetectionSource(sc, start_ms=0))
    b = _run(SimDetectionSource(back, start_ms=0))
    assert _ev_dicts(a) == _ev_dicts(b)


def test_scenario_from_json_file_and_unknown_name(tmp_path: Path) -> None:
    p = tmp_path / "custom.json"
    p.write_text(Scenario.mug_two_rooms().to_json())
    assert SimDetectionSource(str(p)).scenario.name == "mug_two_rooms"
    with pytest.raises(ValueError, match="unknown scenario"):
        SimDetectionSource("no_such_scenario")


def test_sim_object_behind_or_outside_is_dropped() -> None:
    from app.detection_source import Segment, SimObject

    rooms = {"r": {"anchor_tag": 0, "anchor_height_m": 1.4}}
    objs = [
        SimObject("cup", (0.0, 1.4, 3.5)),  # behind the camera (camera z=2.5 looks toward -z)
        SimObject("bottle", (4.0, 1.4, 2.0)),  # far to the side, outside the FOV
        SimObject("laptop", (0.0, 1.0, 1.0)),  # in view
    ]
    sc = Scenario("t", [Segment(1.0, "r", (0.0, 1.4, 2.5), 0.0, objs, (0,))], rooms=rooms)
    events = _run(SimDetectionSource(sc, start_ms=0, oracle_pose=True))
    labels = {d.label for e in events for d in e.detections}
    assert labels == {"laptop"}


# ---------------------------------------------------------------------------
# JSONL record / replay
# ---------------------------------------------------------------------------


def _strip_source(d: dict) -> dict:
    d = dict(d)
    d.pop("source", None)
    return d


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rec" / "events.jsonl"
    src = SimDetectionSource(start_ms=1_700_000_000_000, seed=2)
    events = _run(src)
    with JsonlRecorder(path) as rec:
        for ev in events:
            rec.write(ev)
    assert rec.lines == len(events)
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    assert first["t"] == "tag" and set(first) == {
        "t",
        "timestamp_ms",
        "width",
        "height",
        "K",
        "dist",
        "tags",
    }
    det_line = json.loads(next(ln for ln in lines if '"t":"det"' in ln))
    assert set(det_line) == {"t", "timestamp_ms", "seq", "source", "device_ts_ms", "detections"}
    assert set(det_line["detections"][0]) == {
        "label", "label_id", "confidence", "xmin", "ymin", "xmax", "ymax", "xyz_cam_m"
    }  # fmt: skip
    assert "gray" not in first

    replay = JsonlReplaySource(path)
    assert isinstance(replay, DetectionSource) and replay.name == "replay"
    back = _run(replay)
    assert len(back) == len(events)
    assert [type(e) for e in back] == [type(e) for e in events]
    assert [_strip_source(e.to_dict()) for e in back] == [
        _strip_source(e.to_dict()) for e in events
    ]
    assert all(e.source == "replay" for e in back if isinstance(e, DetectionFrame))
    assert replay.status().state == "done" and replay.status().frames == 180


def test_jsonl_round_trip_keeps_oracle_pose_and_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "oracle.jsonl"
    events = _run(SimDetectionSource(oracle_pose=True, start_ms=0))
    rec = JsonlRecorder(path)
    for ev in events[:5]:
        rec.write(ev)
    rec.close()
    with path.open("a") as fh:
        fh.write('{"t": "det", "timestamp_ms": 1, "seq"\n')  # torn tail
    back = _run(JsonlReplaySource(path))
    assert len(back) == 5
    assert back[0].oracle_pose is not None
    assert np.allclose(back[0].oracle_pose.T_room_cam, events[0].oracle_pose.T_room_cam)
    assert back[0].oracle_pose.room == "kitchen"
    st = JsonlReplaySource(path)
    _run(st)
    assert st.status().error == "1 unparsable lines"


def test_jsonl_replay_loop_and_stop(tmp_path: Path) -> None:
    path = tmp_path / "loop.jsonl"
    events = _run(SimDetectionSource(oracle_pose=True, start_ms=0))
    with JsonlRecorder(path) as rec:
        for ev in events[:10]:
            rec.write(ev)
    src = JsonlReplaySource(path, loop=True)
    src.start()
    got: list[DetectionFrame] = []
    for ev in src.events():
        got.append(ev)
        if len(got) == 25:
            src.stop()
    assert len(got) == 25
    assert got[10].timestamp_ms > got[9].timestamp_ms  # second lap continues in time
    assert src.status().state == "stopped"


def test_jsonl_replay_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    src = JsonlReplaySource(path)
    assert _run(src) == []
    assert src.status().state == "done"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_detection_source_dispatch(tmp_path: Path) -> None:
    assert make_detection_source(env={}) is None
    assert make_detection_source("none", env={"LOCI_DETECTION_SOURCE": "sim"}) is None

    env = {
        "LOCI_DETECTION_SOURCE": "sim",
        "LOCI_SIM_REALTIME": "0",
        "LOCI_SIM_ORACLE_POSE": "1",
        "LOCI_SIM_LOOP": "1",
        "LOCI_SIM_SCENARIO": "mug_two_rooms",
        "LOCI_SIM_FPS": "5",
        "LOCI_SIM_SEED": "7",
    }
    sim = make_detection_source(env=env)
    assert isinstance(sim, SimDetectionSource)
    assert sim.realtime is False and sim.oracle_pose is True and sim.loop is True
    assert sim.fps == 5.0 and sim.seed == 7 and sim.frames_total == 90

    default_sim = make_detection_source("sim", env={})
    assert isinstance(default_sim, SimDetectionSource)
    assert default_sim.realtime is True and default_sim.oracle_pose is False

    rec = tmp_path / "r.jsonl"
    rec.write_text("")
    replay = make_detection_source(
        env={"LOCI_DETECTION_SOURCE": "replay", "LOCI_REPLAY_PATH": str(rec)}
    )
    assert isinstance(replay, JsonlReplaySource) and replay.path == rec
    with pytest.raises(ValueError, match="LOCI_REPLAY_PATH"):
        make_detection_source("replay", env={})
    with pytest.raises(ValueError, match="unknown detection source"):
        make_detection_source("webcam", env={})


def test_make_detection_source_oak_does_not_open_a_device() -> None:
    src = make_detection_source("oak", env={"LOCI_OAK_FPS": "15", "LOCI_TAG_STREAM": "off"})
    assert src is not None and src.name == "oak"
    assert src.cfg.fps == 15.0 and src.cfg.tag_stream is None  # type: ignore[attr-defined]
    assert src.status().state == "starting"


# ---------------------------------------------------------------------------
# Small dataclass behaviour
# ---------------------------------------------------------------------------


def test_spatial_detection_properties_and_dict() -> None:
    d = SpatialDetection("cup", 41, 0.9, 0.2, 0.4, 0.4, 0.8, (0.1, -0.05, 1.84))
    assert (d.cx, d.cy, d.width, d.height) == pytest.approx((0.3, 0.6, 0.2, 0.4))
    assert d.depth_m == 1.84
    assert SpatialDetection.from_dict(d.to_dict()) == d
    nd = SpatialDetection("cup", 41, 0.9, 0.2, 0.4, 0.4, 0.8, None)
    assert nd.depth_m is None and nd.to_dict()["xyz_cam_m"] is None
