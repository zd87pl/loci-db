"""Tests for the handheld runtime wiring (design §4.11 / §13 T7): ``HandheldConfig``,
``HandheldRuntime.step`` with simulated oracle-pose events, the presence log,
``run_until_done``, thread start/stop, recording, ``build_handheld``, ``answer_text``
and the headset voice-loop handle.  Simulator and fakes only: no camera, no audio.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app import headset_voice as hv  # noqa: E402
from app.detection_source import (  # noqa: E402
    CameraPose,
    DetectionFrame,
    JsonlReplaySource,
    SimDetectionSource,
    SpatialDetection,
    TagFrame,
    camera_pose_T,
)
from app.handheld import (  # noqa: E402
    DEFAULT_ROOMS_FILE,
    HandheldConfig,
    HandheldRuntime,
    answer_text,
    build_handheld,
    camera_only_pose,
    load_rooms,
    make_embedder_safe,
    start_voice_loop,
)
from app.room_anchor import RoomLocalizer, RoomMap  # noqa: E402
from app.scene_ingestion import Detection, SceneIngestion  # noqa: E402
from app.spatial_memory import SpatialMemory  # noqa: E402
from app.voice_pipeline import VoicePipeline  # noqa: E402

SIM_ENV = {
    "LOCI_DETECTION_SOURCE": "sim",
    "LOCI_SIM_REALTIME": "0",
    "LOCI_SIM_ORACLE_POSE": "1",
    "LOCI_PERSIST": "0",
    "LOCI_LABEL_EMBEDDER": "hash",
}
NOW_MS = 1_760_000_000_000


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeIngestion:
    """Records ``process_detections`` calls; mirrors the attributes the runtime reads."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[tuple[list[Detection], int | None]] = []
        self.frame_count = 0
        self.last_detections: list[Detection] = []
        self.classes = ["cup", "bottle"]
        self._fail_first = fail_first

    def process_detections(self, dets, timestamp_ms=None):
        if self._fail_first:
            self._fail_first = False
            raise RuntimeError("boom")
        self.calls.append((list(dets), timestamp_ms))
        self.frame_count += 1
        self.last_detections = list(dets)
        return dets

    @property
    def consensus_stats(self) -> dict:
        return {"buffer_entries": 0, "fires": len(self.calls), "suppressed_by_holdoff": 0}


class ListSource:
    """DetectionSource over a fixed event list."""

    name = "list"
    classes = ["cup", "bottle"]

    def __init__(self, events) -> None:
        self._events = list(events)
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def events(self):
        yield from self._events

    def status(self):
        from app.detection_source import SourceStatus

        return SourceStatus("done", 0.0, len(self._events), 0, None, None, "list", None)


def _det(label: str, xyz, conf: float = 0.9) -> SpatialDetection:
    return SpatialDetection(label, 0, conf, 0.4, 0.4, 0.6, 0.6, xyz)


def _kitchen_pose(anchored: bool = True) -> CameraPose:
    return CameraPose(
        "kitchen", "room", camera_pose_T((0.0, 1.4, 2.5), 0.0), anchored, 1.0, (0,), 0.0
    )


def _frame(dets, ts_ms: int, pose: CameraPose | None, seq: int = 0) -> DetectionFrame:
    return DetectionFrame(ts_ms, seq, list(dets), "sim", None, pose)


def _stack(env: dict | None = None, **cfg_overrides):
    """Real memory + ingestion + runtime + localizer around the non-realtime oracle sim."""
    env = {**SIM_ENV, **(env or {})}
    cfg = HandheldConfig.from_env(env)
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    memory = SpatialMemory(
        epoch_size_ms=5000,
        rooms=load_rooms(cfg, env=env),
        dedup_radius_m=cfg.dedup_radius_m,
        moved_threshold_m=cfg.moved_threshold_m,
    )
    ingestion = SceneIngestion(
        memory,
        None,
        use_vlm_fallback=False,
        consensus_refire_ms=cfg.consensus_refire_ms,
        consensus_radius_m=cfg.consensus_radius_m,
    )
    runtime, localizer = build_handheld(memory, ingestion, cfg, env=env)
    assert runtime is not None and localizer is not None
    memory.set_vocabulary(runtime.source.classes)
    return memory, ingestion, cfg, runtime, localizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_defaults_match_design(tmp_path: Path) -> None:
    cfg = HandheldConfig.from_env({})
    assert cfg.detection_source == "none"
    assert cfg.persist is True
    assert cfg.snapshot_interval_s == 20.0
    assert cfg.tag_size_m == 0.16 and cfg.tag_size_set is False
    assert cfg.tag_detector == "auto"
    assert cfg.pose_hold_s == 10.0 and cfg.pose_stale_s == 60.0
    assert cfg.consensus_refire_ms == 2000 and cfg.consensus_radius_m == 0.3
    assert cfg.dedup_radius_m == 0.35 and cfg.moved_threshold_m == 0.5
    assert cfg.ws_push_hz == 5.0
    assert cfg.sim_realtime is True and cfg.sim_oracle_pose is False
    assert cfg.label_embedder == "hash"
    assert cfg.data_dir == Path("~/.local/share/loci-handheld").expanduser()
    # the example rooms file ships with the repo and is the default
    assert cfg.rooms_file == DEFAULT_ROOMS_FILE and DEFAULT_ROOMS_FILE.is_file()
    assert cfg.voice == "none" and cfg.record_path is None


def test_config_parses_environment_types(tmp_path: Path) -> None:
    env = {
        "LOCI_DETECTION_SOURCE": "OAK",
        "LOCI_DATA_DIR": str(tmp_path / "d"),
        "LOCI_PERSIST": "0",
        "LOCI_SNAPSHOT_INTERVAL_S": "5",
        "LOCI_JOURNAL_FSYNC_S": "0.5",
        "LOCI_MAX_RECORDS": "123",
        "LOCI_ROOMS_FILE": str(tmp_path / "rooms.json"),
        "LOCI_TAG_SIZE_M": "0.2",
        "LOCI_TAG_DETECTOR": "aruco",
        "LOCI_POSE_HOLD_S": "7",
        "LOCI_POSE_STALE_S": "30",
        "LOCI_CONSENSUS_REFIRE_MS": "1500",
        "LOCI_CONSENSUS_RADIUS_M": "0.25",
        "LOCI_DEDUP_RADIUS_M": "0.4",
        "LOCI_MOVED_THRESHOLD_M": "0.6",
        "LOCI_WS_PUSH_HZ": "2",
        "LOCI_SIM_REALTIME": "0",
        "LOCI_SIM_ORACLE_POSE": "1",
        "LOCI_SIM_LOOP": "yes",
        "LOCI_SIM_START_OFFSET_S": "600",
        "LOCI_LABEL_EMBEDDER": "Static",
        "LOCI_RECORD_PATH": str(tmp_path / "rec.jsonl"),
        "LOCI_VOICE": "headset",
        "LOCI_TLS": "1",
        "PORT": "9000",
        "LOCI_MAX_RECORDS_BAD": "x",
    }
    cfg = HandheldConfig.from_env(env)
    assert cfg.detection_source == "oak"
    assert cfg.data_dir == tmp_path / "d"
    assert cfg.persist is False
    assert cfg.snapshot_interval_s == 5.0 and cfg.journal_fsync_s == 0.5 and cfg.max_records == 123
    assert cfg.rooms_file == tmp_path / "rooms.json"
    assert cfg.tag_size_m == 0.2 and cfg.tag_size_set is True
    assert cfg.tag_detector == "aruco"
    assert cfg.pose_hold_s == 7.0 and cfg.pose_stale_s == 30.0
    assert cfg.consensus_refire_ms == 1500 and cfg.consensus_radius_m == 0.25
    assert cfg.dedup_radius_m == 0.4 and cfg.moved_threshold_m == 0.6
    assert cfg.ws_push_hz == 2.0
    assert cfg.sim_realtime is False and cfg.sim_oracle_pose is True and cfg.sim_loop is True
    assert cfg.sim_start_offset_s == 600.0
    assert cfg.label_embedder == "static"
    assert cfg.record_path == tmp_path / "rec.jsonl"
    assert cfg.voice == "headset" and cfg.tls is True and cfg.port == 9000
    # bad numbers fall back to the default instead of raising
    assert HandheldConfig.from_env({"LOCI_WS_PUSH_HZ": "fast"}).ws_push_hz == 5.0


def test_config_to_dict_is_json_safe_and_secret_free(tmp_path: Path) -> None:
    cfg = HandheldConfig.from_env({"OPENAI_API_KEY": "sk-secret", "LOCI_DATA_DIR": str(tmp_path)})
    d = cfg.to_dict()
    text = json.dumps(d)
    assert "sk-secret" not in text
    assert d["data_dir"] == str(tmp_path)
    assert set(d) >= {"detection_source", "persist", "consensus_refire_ms", "label_embedder"}


def test_load_rooms_file_default_and_overrides(tmp_path: Path) -> None:
    cfg = HandheldConfig.from_env({})
    rooms = load_rooms(cfg, env={})
    assert rooms.names() == ["kitchen", "living_room"]
    assert rooms.get("kitchen").landmarks == {"table": (1.0, 0.75, 1.2)}
    # a broken file falls back to the built-in map; LOCI_TAG_SIZE_M still applies
    bad = tmp_path / "rooms.json"
    bad.write_text("{not json")
    cfg2 = HandheldConfig.from_env({"LOCI_ROOMS_FILE": str(bad), "LOCI_TAG_SIZE_M": "0.24"})
    rooms2 = load_rooms(cfg2, env={})
    assert rooms2.names() == ["kitchen", "living_room"]
    assert rooms2.tag_size_m == pytest.approx(0.24)
    # a missing file also falls back
    cfg3 = HandheldConfig.from_env({"LOCI_ROOMS_FILE": str(tmp_path / "nope.json")})
    assert load_rooms(cfg3, env={}).names() == ["kitchen", "living_room"]


def test_make_embedder_safe_falls_back_to_hash(tmp_path: Path) -> None:
    cfg = HandheldConfig.from_env({"LOCI_LABEL_EMBEDDER": "static"})
    emb, err = make_embedder_safe(cfg, env={"LOCI_EMBED_MODEL_DIR": str(tmp_path / "missing")})
    assert emb.name == "hash:128" and emb.dim == 128
    assert err is not None
    emb2, err2 = make_embedder_safe(HandheldConfig.from_env({}), env={})
    assert emb2.name == "hash:128" and err2 is None


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_fills_room_frame_fields_from_oracle_pose() -> None:
    sim = SimDetectionSource(realtime=False, oracle_pose=True, start_ms=NOW_MS)
    ingestion = FakeIngestion()
    memory = MagicMock()
    rt = HandheldRuntime(sim, None, ingestion, memory, HandheldConfig())
    sim.start()
    first = next(sim.events())
    assert isinstance(first, DetectionFrame) and first.oracle_pose is not None
    rt.step(first)

    assert len(ingestion.calls) == 1
    dets, ts = ingestion.calls[0]
    assert ts == NOW_MS
    by_label = {d.label: d for d in dets}
    assert set(by_label) == {"cup", "bottle"}
    cup = by_label["cup"]
    assert cup.frame == "room" and cup.room == "kitchen" and cup.anchored is True
    assert cup.anchor_confidence == pytest.approx(1.0)
    assert cup.source == "sim" and cup.xyz_cam_m is not None and cup.depth_m is not None
    assert cup.pos_room_m is not None
    # scripted kitchen cup at (1.0, 0.8, 1.0) room metres (+ 2-4 cm sim noise)
    assert cup.pos_room_m == pytest.approx((1.0, 0.8, 1.0), abs=0.15)
    assert by_label["bottle"].pos_room_m == pytest.approx((-0.6, 0.9, 1.2), abs=0.15)
    assert rt.frames == 1 and rt.seq == 0 and rt.latest_frame is first
    assert rt.latest_pose is not None and rt.latest_pose.room == "kitchen"
    assert rt.last_timestamp_ms == NOW_MS
    memory.note_presence.assert_called_once_with("kitchen", NOW_MS)


def test_step_camera_only_pose_yields_camera_frame() -> None:
    ingestion = FakeIngestion()
    rt = HandheldRuntime(ListSource([]), None, ingestion, MagicMock(), HandheldConfig())
    pose = camera_only_pose()
    rt.step(_frame([_det("cup", (0.1, 0.2, 1.5))], NOW_MS, pose))
    ((dets, _),) = ingestion.calls
    d = dets[0]
    assert d.frame == "camera" and d.room is None and d.pos_room_m is None
    assert d.anchored is False and d.anchor_confidence == 0.0
    assert d.xyz_cam_m == pytest.approx((0.1, 0.2, 1.5))
    # a held (not anchored) room pose keeps room metres but anchored=False
    held = _kitchen_pose(anchored=False)
    held.anchor_confidence = 0.0
    rt.step(_frame([_det("cup", (0.0, 0.0, 1.0))], NOW_MS + 100, held, seq=1))
    d2 = ingestion.calls[1][0][0]
    assert d2.frame == "room" and d2.room == "kitchen" and d2.anchored is False
    assert d2.pos_room_m == pytest.approx((0.0, 1.4, 1.5), abs=1e-6)


def test_step_drops_no_depth_from_memory_but_shows_it_live() -> None:
    ingestion = FakeIngestion()
    rt = HandheldRuntime(ListSource([]), None, ingestion, MagicMock(), HandheldConfig())
    rt.step(_frame([_det("cup", None), _det("bottle", (0.2, 0.1, 2.0))], NOW_MS, _kitchen_pose()))
    ((dets, _),) = ingestion.calls
    assert [d.label for d in dets] == ["bottle"]
    live = rt.live_detections()
    assert [d["label"] for d in live] == ["cup", "bottle"]
    assert live[0]["distance_m"] is None and live[1]["distance_m"] == pytest.approx(2.012, abs=0.01)
    assert rt.dropped_no_depth == 1


def test_tag_frame_goes_to_localizer_and_counts() -> None:
    loc = MagicMock()
    loc.current.return_value = camera_only_pose()
    rt = HandheldRuntime(ListSource([]), loc, FakeIngestion(), MagicMock(), HandheldConfig())
    import numpy as np

    tf = TagFrame(NOW_MS, 1280, 800, np.eye(3), None, None, [])
    rt.step(tf)
    loc.update.assert_called_once_with(tf)
    assert rt.tag_frames == 1 and rt.frames == 0
    # a detection frame without an oracle pose asks the localizer for the pose
    rt.step(_frame([_det("cup", (0.0, 0.0, 1.0))], NOW_MS, None))
    loc.current.assert_called_once()
    assert rt.latest_pose is not None and rt.latest_pose.frame == "camera"


def test_oracle_pose_is_injected_into_a_real_localizer() -> None:
    loc = RoomLocalizer(RoomMap.default(), None)
    rt = HandheldRuntime(ListSource([]), loc, FakeIngestion(), MagicMock(), HandheldConfig())
    assert loc.current().frame == "camera"
    rt.step(_frame([_det("cup", (0.0, 0.0, 1.0))], NOW_MS, _kitchen_pose()))
    cur = loc.current()
    assert cur.room == "kitchen" and cur.frame == "room" and cur.anchored is True
    assert cur.to_dict()["camera_xyz_m"] == pytest.approx([0.0, 1.4, 2.5])
    assert rt.current_pose().room == "kitchen"


def test_presence_is_noted_at_most_once_per_second_per_room() -> None:
    sim = SimDetectionSource(realtime=False, oracle_pose=True, start_ms=NOW_MS)
    memory = MagicMock()
    rt = HandheldRuntime(sim, None, FakeIngestion(), memory, HandheldConfig())
    assert rt.run_until_done() == 180
    calls = [c.args for c in memory.note_presence.call_args_list]
    assert calls, "presence log never fed"
    # kitchen 0-6 s, hallway (no presence), living_room 9-13 s, kitchen 13-18 s at 10 fps
    rooms = [r for r, _ in calls]
    assert set(rooms) == {"kitchen", "living_room"}
    assert 10 <= len(calls) <= 18
    for (r0, t0), (r1, t1) in zip(calls, calls[1:], strict=False):
        if r0 == r1:
            assert t1 - t0 >= 1000
    # the hallway has no anchored pose: no presence between 6 s and 9 s
    assert not [t for _, t in calls if NOW_MS + 6000 <= t < NOW_MS + 9000]


def test_step_errors_are_counted_and_the_loop_continues() -> None:
    ev = [
        _frame([_det("cup", (0.0, 0.0, 1.0))], NOW_MS, _kitchen_pose(), seq=0),
        _frame([_det("cup", (0.0, 0.0, 1.0))], NOW_MS + 100, _kitchen_pose(), seq=1),
    ]
    src = ListSource(ev)
    ingestion = FakeIngestion(fail_first=True)
    rt = HandheldRuntime(src, None, ingestion, MagicMock(), HandheldConfig())
    assert rt.run_until_done() == 1  # the failing frame is not counted as processed
    assert rt.errors == 1 and "boom" in (rt.error or "")
    assert rt.done is True and src.started == 1
    st = rt.status()
    assert st["errors"] == 1 and st["frames"] == 1 and st["done"] is True


# ---------------------------------------------------------------------------
# run_until_done / threads / status
# ---------------------------------------------------------------------------


def test_run_until_done_feeds_real_memory_and_status_shape() -> None:
    memory, ingestion, _cfg, rt, loc = _stack()
    assert rt.run_until_done() == 180
    assert rt.done is True and rt.thread_alive is False
    st = rt.status()
    assert set(st) >= {
        "source",
        "localizer",
        "frames",
        "errors",
        "done",
        "thread_alive",
        "ingestion",
        "seq",
        "pose",
    }
    assert st["source"]["state"] == "done" and st["source"]["frames"] == 180
    assert st["frames"] == 180 and st["errors"] == 0
    assert st["ingestion"]["fires"] > 0 and set(st["ingestion"]) == {
        "buffer_entries",
        "fires",
        "suppressed_by_holdoff",
    }
    assert st["localizer"]["room"] == "kitchen"
    assert st["pose"]["room"] == "kitchen" and st["pose"]["frame"] == "room"
    json.dumps(st)  # JSON-serialisable for /api/handheld/status

    kitchen = {o.label: o for o in memory.current_objects("kitchen")}
    assert {"cup", "bottle"} <= set(kitchen)
    assert kitchen["cup"].pos_m == pytest.approx((-1.2, 0.8, 1.5), abs=0.3)
    assert kitchen["cup"].frame == "room" and kitchen["cup"].anchored is True
    living = {o.label for o in memory.current_objects("living_room")}
    assert {"laptop", "couch"} <= living
    assert sorted(memory.rooms_seen) == ["kitchen", "living_room"]
    assert loc.current().room == "kitchen"


def test_run_until_done_waits_for_a_started_thread() -> None:
    _memory, _ingestion, _cfg, rt, _loc = _stack()
    rt.start()
    assert rt.run_until_done(timeout_s=30) == 180
    assert rt.thread_alive is False and rt.done is True
    # a second call on a finished runtime is a no-op returning the same total
    assert rt.run_until_done() == 180


def test_stop_joins_a_running_thread() -> None:
    sim = SimDetectionSource(realtime=True, oracle_pose=True, loop=True)
    env = dict(SIM_ENV)
    cfg = HandheldConfig.from_env(env)
    memory = SpatialMemory(epoch_size_ms=5000, rooms=RoomMap.default())
    ingestion = SceneIngestion(memory, None, use_vlm_fallback=False)
    rt, _loc = build_handheld(memory, ingestion, cfg, env=env, source=sim)
    assert rt is not None
    rt.start()
    assert rt.thread_alive
    deadline = time.monotonic() + 5.0
    while rt.frames < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert rt.frames >= 1
    t0 = time.monotonic()
    rt.stop(timeout_s=5.0)
    assert time.monotonic() - t0 < 5.0
    assert rt.thread_alive is False and rt.done is True
    assert sim.status().state == "stopped"
    assert [t.name for t in threading.enumerate() if t.name == "handheld-runtime"] == []


def test_recording_is_replayable(tmp_path: Path) -> None:
    rec = tmp_path / "rec.jsonl"
    memory, ingestion, cfg, rt, _loc = _stack({"LOCI_RECORD_PATH": str(rec)})
    assert rt.recorder is not None and rt.recorder.path == rec
    assert rt.run_until_done() == 180
    rt.stop()
    assert rt.recorder.lines == 180
    assert rt.status()["recorder"] == {"path": str(rec), "lines": 180}
    lines = rec.read_text().splitlines()
    assert len(lines) == 180 and json.loads(lines[0])["t"] == "det"
    # the recording drives a second runtime (oracle poses are kept in the file)
    replay = JsonlReplaySource(rec, realtime=False)
    mem2 = SpatialMemory(epoch_size_ms=5000, rooms=RoomMap.default())
    ing2 = SceneIngestion(mem2, None, use_vlm_fallback=False, consensus_refire_ms=2000)
    rt2, _ = build_handheld(mem2, ing2, cfg, env={}, source=replay)
    assert rt2 is not None and rt2.run_until_done() == 180
    cup = {o.label: o for o in mem2.current_objects("kitchen")}["cup"]
    assert cup.pos_m == pytest.approx((-1.2, 0.8, 1.5), abs=0.3)
    assert cup.source == "replay"


def test_build_handheld_none_offset_and_bad_source(tmp_path: Path) -> None:
    memory = SpatialMemory(epoch_size_ms=5000, rooms=RoomMap.default())
    ingestion = SceneIngestion(memory, None, use_vlm_fallback=False)
    assert build_handheld(memory, ingestion, HandheldConfig(), env={}) == (None, None)
    cfg = HandheldConfig.from_env({**SIM_ENV, "LOCI_SIM_START_OFFSET_S": "600"})
    rt, loc = build_handheld(memory, ingestion, cfg, env=SIM_ENV)
    assert rt is not None and loc is not None
    assert isinstance(rt.source, SimDetectionSource)
    assert rt.source.start_ms == pytest.approx(time.time() * 1000 - 600_000, abs=5_000)
    assert rt.localizer is loc and rt.recorder is None
    with pytest.raises(ValueError):
        build_handheld(memory, ingestion, HandheldConfig(detection_source="bogus"), env={})
    with pytest.raises(ValueError):  # replay without a path is a configuration error
        build_handheld(memory, ingestion, HandheldConfig(detection_source="replay"), env={})


# ---------------------------------------------------------------------------
# answer_text (shared by POST /api/query and the headset loop)
# ---------------------------------------------------------------------------


def test_answer_text_shapes_and_room_override() -> None:
    memory, _ingestion, _cfg, rt, loc = _stack()
    rt.run_until_done()
    voice = VoicePipeline(memory, None, pose_provider=loc.current)

    r = answer_text(voice, memory, "where is my mug")
    assert set(r) == {"answer", "intent", "resolution", "results", "changes", "latency_ms"}
    assert r["resolution"] == {"label": "cup", "how": "alias", "score": 1.0, "suggestions": []}
    assert "kitchen" in r["answer"].lower()
    assert r["intent"] == {"kind": "where_is", "object_name": "mug", "room": ""}
    first = r["results"][0]
    assert (first["x_m"], first["y_m"], first["z_m"]) == pytest.approx((-1.2, 0.8, 1.5), abs=0.3)
    assert r["latency_ms"] < 300

    r2 = answer_text(voice, memory, "what changed", room="kitchen")
    assert r2["intent"]["kind"] == "changes" and r2["intent"]["room"] == "kitchen"
    assert {c["label"] for c in r2["changes"]} >= {"cup", "bottle"}
    assert all(c["room"] == "kitchen" for c in r2["changes"])
    # an explicit room in the sentence wins over the override
    r3 = answer_text(voice, memory, "what's in the living room", room="kitchen")
    assert r3["intent"]["room"] == "living_room"

    r4 = answer_text(voice, memory, "where are my keys")
    assert r4["resolution"]["label"] is None and r4["results"] == []
    assert "keys" in r4["answer"]

    json.dumps(r)  # the whole thing is JSON-safe


# ---------------------------------------------------------------------------
# Headset voice loop handle (T11 contract)
# ---------------------------------------------------------------------------


class FakeAudio:
    name = "fake-audio"

    def __init__(self) -> None:
        self.played: list[bytes] = []

    def record(self, max_seconds, stop=None):
        return hv.silence_wav(0.2)

    def play(self, wav_bytes):
        self.played.append(bytes(wav_bytes))

    def list_devices(self):
        return []


class OnePressPTT:
    name = "fake-ptt"

    def __init__(self) -> None:
        self.presses = 0

    def wait_for_press(self, stop):
        if self.presses == 0:
            self.presses += 1
            return True
        stop.wait(10.0)
        return False

    def wait_for_release(self, timeout_s):
        return None


class FakeSTT:
    def transcribe(self, wav, language="en"):
        return "where is my mug"


class FakeTTS:
    def synthesize(self, text):
        return b"RIFF\x00\x00\x00\x00WAVEfmt " + text.encode()


def test_start_voice_loop_disabled_and_fake_turn() -> None:
    answers: list[str] = []

    def answer_fn(text: str) -> str:
        answers.append(text)
        return "Your mug is in the kitchen."

    assert start_voice_loop(answer_fn, FakeSTT(), FakeTTS(), env={"LOCI_VOICE": "none"}) is None
    assert start_voice_loop(answer_fn, FakeSTT(), FakeTTS(), env={}) is None

    audio, ptt = FakeAudio(), OnePressPTT()
    handle = start_voice_loop(
        answer_fn,
        FakeSTT(),
        FakeTTS(),
        env={"LOCI_VOICE": "headset", "LOCI_EARCON": "1", "LOCI_VOICE_MAX_S": "3"},
        audio=audio,
        ptt=ptt,
    )
    assert handle is not None
    try:
        deadline = time.monotonic() + 5.0
        while handle.loop.turns < 1 or not any(e["type"] == "done" for e in handle.last_events()):
            if time.monotonic() > deadline:
                pytest.fail(f"voice turn did not finish: {handle.last_events()}")
            time.sleep(0.02)
        st = handle.status()
        assert st["running"] is True and st["turns"] == 1 and st["thread_alive"] is True
        assert st["ptt"] == "fake-ptt" and st["audio"] == "fake-audio"
        types = [e["type"] for e in handle.last_events()]
        for t in ("press", "earcon", "recorded", "transcript", "answer", "tts", "played", "done"):
            assert t in types
        assert answers == ["where is my mug"]
        assert len(audio.played) == 2  # earcon + the spoken answer
        assert audio.played[1].startswith(b"RIFF")
        json.dumps(st)
    finally:
        handle.stop(timeout_s=5.0)
    assert handle.alive is False and handle.status()["running"] is False


# ---------------------------------------------------------------------------
# Review fixes: vocabulary on the camera path, tags in recordings, stop races,
# decayed held poses
# ---------------------------------------------------------------------------


class LateClassesSource(ListSource):
    """Like the OAK source: ``classes`` is only known once ``start()`` opened the device."""

    def __init__(self, events, classes_after_start) -> None:
        super().__init__(events)
        self.classes = []
        self._after = list(classes_after_start)

    def start(self) -> None:
        super().start()
        self.classes = list(self._after)


def test_vocabulary_follows_the_source_classes_after_start() -> None:
    from app.detection_source import COCO80

    memory = SpatialMemory(epoch_size_ms=5000, rooms=RoomMap.default())
    ingestion = SceneIngestion(memory, None, use_vlm_fallback=False)
    source = LateClassesSource([_frame([], NOW_MS, _kitchen_pose())], COCO80)
    rt = HandheldRuntime(source, None, ingestion, memory, HandheldConfig())
    memory.set_vocabulary(source.classes)  # what the lifespan does: still empty here
    assert memory.resolve_label("mug").label is None
    assert rt.run_until_done() == 1
    # D13: vocabulary = detector classes (COCO-80 once the device is open) ∪ tracked labels
    assert memory.resolve_label("mug").label == "cup"
    assert rt.vocabulary_updates == 1 and rt.status()["vocabulary_updates"] == 1
    voice = VoicePipeline(memory, None)
    out = answer_text(voice, memory, "where is my cup")
    assert out["resolution"]["label"] == "cup" and out["resolution"]["how"] == "exact"
    assert "haven't seen" in out["answer"] and "can't recognise" not in out["answer"]
    # a model with another label set (after a reconnect) is picked up on the next frame
    source.classes = ["cup", "keys"]
    rt.step(_frame([], NOW_MS + 100, _kitchen_pose()))
    assert memory.resolve_label("keys").label == "keys"
    assert rt.vocabulary_updates == 2
    rt.step(_frame([], NOW_MS + 200, _kitchen_pose()))
    assert rt.vocabulary_updates == 2  # unchanged list: no re-apply


class OneTagDetector:
    name = "fake-detector"

    def __init__(self, det) -> None:
        self.det = det
        self.calls = 0

    def detect(self, gray):
        self.calls += 1
        return [self.det]


def test_tag_frames_are_recorded_with_the_detected_tags(tmp_path: Path) -> None:
    """OAK tag frames carry only the luma plane (never serialised); the recording must
    keep what the localizer detected so a replay anchors like the live run (§4.14)."""
    from app.detection_source import JsonlRecorder, TagDetection, parse_event_line

    det = TagDetection(0, [[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]], 50.0, 0)
    K = [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]]
    tf = TagFrame(NOW_MS, 64, 64, K, None, gray=np.zeros((64, 64), dtype=np.uint8), tags=None)
    frame = _frame([], NOW_MS + 10, None)
    detector = OneTagDetector(det)
    localizer = RoomLocalizer(RoomMap.default(), detector)
    memory = SpatialMemory(epoch_size_ms=5000, rooms=RoomMap.default())
    ingestion = SceneIngestion(memory, None, use_vlm_fallback=False)
    rec = JsonlRecorder(tmp_path / "tags.jsonl")
    rt = HandheldRuntime(
        ListSource([tf, frame]), localizer, ingestion, memory, HandheldConfig(), rec
    )
    assert rt.run_until_done() == 1
    rt.stop()
    assert detector.calls == 1 and rt.tag_frames == 1
    lines = (tmp_path / "tags.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["t"] == "tag"
    replayed = parse_event_line(lines[0])
    assert isinstance(replayed, TagFrame) and replayed.gray is None
    assert replayed.tags is not None and [t.tag_id for t in replayed.tags] == [0]
    # the replayed frame reaches the solver with neither detector nor image
    assert [d.tag_id for d in RoomLocalizer(RoomMap.default(), None)._detect(replayed)] == [0]


def test_stop_before_the_thread_body_runs_never_starts_the_source() -> None:
    src = ListSource([_frame([], NOW_MS, _kitchen_pose())])
    rt = HandheldRuntime(src, None, FakeIngestion(), MagicMock(), HandheldConfig())
    rt._stop.set()
    rt._run()
    assert src.started == 0 and rt.done is True and rt.frames == 0

    class StopsOnStart(ListSource):
        def start(self) -> None:
            super().start()
            rt2._stop.set()  # stop() lands while start() is opening the device

    src2 = StopsOnStart([_frame([], NOW_MS, _kitchen_pose())])
    rt2 = HandheldRuntime(src2, None, FakeIngestion(), MagicMock(), HandheldConfig())
    rt2._run()
    assert src2.started == 1 and src2.stopped == 1 and rt2.frames == 0


def test_decayed_held_pose_is_stored_unanchored_and_logs_no_presence() -> None:
    from app.handheld import STORE_MIN_ANCHOR_CONFIDENCE

    T = camera_pose_T((0.0, 1.4, 2.5), 0.0)
    stale = CameraPose("kitchen", "room", T, True, STORE_MIN_ANCHOR_CONFIDENCE - 0.1, (0,), 6.0)
    fresh = CameraPose("kitchen", "room", T, True, STORE_MIN_ANCHOR_CONFIDENCE + 0.1, (0,), 1.0)
    det = HandheldRuntime.to_detection(_det("cup", (0.0, 0.0, 2.0)), stale, "sim")
    assert det.frame == "room" and det.room == "kitchen" and det.pos_room_m is not None
    assert det.anchored is False  # answered with the "roughly" hedge, never as a fact
    assert det.anchor_confidence == pytest.approx(STORE_MIN_ANCHOR_CONFIDENCE - 0.1)
    assert HandheldRuntime.to_detection(_det("cup", (0.0, 0.0, 2.0)), fresh, "sim").anchored
    memory = MagicMock()
    rt = HandheldRuntime(ListSource([]), None, FakeIngestion(), memory, HandheldConfig())
    rt._note_presence(stale, NOW_MS)
    memory.note_presence.assert_not_called()
    rt._note_presence(fresh, NOW_MS)
    memory.note_presence.assert_called_once_with("kitchen", NOW_MS)
