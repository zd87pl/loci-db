"""API tests for the handheld wiring in ``app.main`` (design §4.11 / §13 T7).

The environment is set BEFORE ``app.main`` is (re)loaded, as the module constructs its
singletons at import: simulator source (non-realtime, oracle pose), hash embedder and a
temporary data directory.  ``TestClient(app)`` runs the lifespan, which starts the
``handheld-runtime`` thread; ``runtime.run_until_done()`` waits for the scenario.
Everything is simulator-only: no camera, no audio, no network.
"""

from __future__ import annotations

import base64
import contextlib
import importlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

BASE_ENV: dict[str, str | None] = {
    "LOCI_DETECTION_SOURCE": "sim",
    "LOCI_SIM_SCENARIO": "mug_two_rooms",
    "LOCI_SIM_REALTIME": "0",
    "LOCI_SIM_ORACLE_POSE": "1",
    "LOCI_SIM_LOOP": "0",
    "LOCI_SIM_START_OFFSET_S": None,
    "LOCI_LABEL_EMBEDDER": "hash",
    "LOCI_PERSIST": "1",
    "LOCI_VOICE": "none",
    "LOCI_WS_PUSH_HZ": "20",
    "LOCI_ROOMS_FILE": None,
    "LOCI_RECORD_PATH": None,
    "LOCI_REPLAY_PATH": None,
    "TTS_ENGINE": "piper",
    "PIPER_MODEL_PATH": None,
    "WHISPER_LOCAL": "0",
    "OPENAI_API_KEY": None,
    "GOOGLE_API_KEY": None,
}


@contextlib.contextmanager
def _patched_env(overrides: dict[str, str | None]):
    saved = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_SNAPSHOT_NAMES = (
    "app",
    "cfg",
    "rooms",
    "memory",
    "ingestion",
    "runtime",
    "localizer",
    "voice",
    "_broadcast",
    "_live_push_loop",
    "_started_at_ms",
)


def _reload_main():
    import app.main as main

    return importlib.reload(main)


@contextlib.contextmanager
def booted(data_dir: Path, **extra: str | None):
    """Reload ``app.main`` under the given env and run its lifespan in a TestClient."""
    env: dict[str, str | None] = {**BASE_ENV, "LOCI_DATA_DIR": str(data_dir), **extra}
    with _patched_env(env):
        main = _reload_main()
        # importlib.reload re-executes into the SAME module object, so a later boot
        # overwrites main.* for everyone: keep this boot's singletons in a snapshot.
        snap = SimpleNamespace(
            **{name: getattr(main, name) for name in _SNAPSHOT_NAMES},
        )
        with TestClient(snap.app) as client:
            if snap.runtime is not None:
                snap.runtime.run_until_done(timeout_s=60)
            yield snap, client


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    with booted(tmp_path_factory.mktemp("sim")) as (main, client):
        yield SimpleNamespace(main=main, client=client)


def _query(client: TestClient, text: str, **extra):
    r = client.post("/api/query", json={"text": text, **extra})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Runtime state through the API
# ---------------------------------------------------------------------------


def test_anchor_reports_kitchen(sim) -> None:
    a = sim.client.get("/api/anchor").json()
    assert a["room"] == "kitchen" and a["frame"] == "room"
    assert set(a) == {
        "room",
        "frame",
        "anchored",
        "anchor_confidence",
        "camera_xyz_m",
        "heading_deg",
        "tag_ids",
        "age_s",
    }
    assert a["camera_xyz_m"] == pytest.approx([-0.4, 1.2, 3.0])
    assert a["tag_ids"] == [0]


def test_objects_filtered_by_room(sim) -> None:
    r = sim.client.get("/api/objects", params={"room": "kitchen"}).json()
    assert r["room"] == "kitchen" and set(r) >= {"objects", "count", "stats"}
    by_label = {o["label"]: o for o in r["objects"]}
    assert {"cup", "bottle"} <= set(by_label)
    cup = by_label["cup"]
    assert cup["room"] == "kitchen" and cup["frame"] == "room" and cup["anchored"] is True
    assert (cup["x_m"], cup["y_m"], cup["z_m"]) == pytest.approx((-1.2, 0.8, 1.5), abs=0.3)
    assert cup["scene_id"] == "kitchen/cup"
    living = {o["label"] for o in sim.client.get("/api/objects?room=living_room").json()["objects"]}
    assert {"laptop", "couch"} <= living and "cup" not in living
    everything = sim.client.get("/api/objects").json()
    assert everything["room"] is None and everything["count"] >= 4
    # legacy per-label endpoints still answer
    loc = sim.client.get("/api/objects/cup/location").json()
    assert loc["best_match"]["label"] == "cup"
    assert sim.client.get("/api/objects/cup/history").json()["count"] >= 1


def test_query_mug_resolves_to_cup_in_the_kitchen(sim) -> None:
    q = _query(sim.client, "where is my mug")
    assert set(q) == {
        "answer",
        "intent",
        "resolution",
        "results",
        "changes",
        "latency_ms",
        "audio_b64",
        "audio_mime",
    }
    assert q["resolution"]["label"] == "cup" and q["resolution"]["how"] == "alias"
    assert "kitchen" in q["answer"].lower()
    assert q["intent"] == {"kind": "where_is", "object_name": "mug", "room": ""}
    first = q["results"][0]
    assert (first["x_m"], first["y_m"], first["z_m"]) == pytest.approx((-1.2, 0.8, 1.5), abs=0.3)
    assert q["latency_ms"] < 1000
    assert q["audio_b64"] is None and q["audio_mime"] is None
    # room filter in the sentence
    q2 = _query(sim.client, "where is my mug in the living room")
    assert q2["intent"]["room"] == "living_room" and q2["results"] == []
    assert "living room" in q2["answer"].lower()


def test_query_room_override_and_changes(sim) -> None:
    q = _query(sim.client, "what changed", room="kitchen")
    assert q["intent"]["kind"] == "changes" and q["intent"]["room"] == "kitchen"
    assert {c["label"] for c in q["changes"]} >= {"cup", "bottle"}
    assert all(c["room"] == "kitchen" for c in q["changes"])
    q2 = _query(sim.client, "what changed in the kitchen")
    assert q2["intent"]["room"] == "kitchen"
    assert any(c["label"] == "cup" for c in q2["changes"])


def test_query_unknown_label_is_honest(sim) -> None:
    q = _query(sim.client, "where are my keys")
    assert q["resolution"]["label"] is None and q["results"] == []
    assert "keys" in q["answer"]
    assert sim.client.post("/api/query", json={"text": "   "}).status_code == 400


def test_changes_endpoint(sim) -> None:
    r = sim.client.get("/api/objects/changes", params={"room": "kitchen"}).json()
    assert set(r) == {"room", "window_seconds", "changes", "count"}
    assert r["room"] == "kitchen" and r["window_seconds"] == 600.0
    assert r["count"] == len(r["changes"]) >= 2
    assert {c["label"] for c in r["changes"]} >= {"cup", "bottle"}
    assert set(r["changes"][0]) == {
        "kind",
        "label",
        "room",
        "last_seen_ms",
        "first_seen_ms",
        "pos_m",
        "pos_before_m",
        "delta_m",
    }
    all_rooms = sim.client.get("/api/objects/changes").json()
    assert all_rooms["room"] is None and all_rooms["count"] >= 4
    # the legacy endpoint under the same prefix is untouched
    legacy = sim.client.get("/api/objects/changes/recent", params={"window_seconds": 1e6}).json()
    assert set(legacy) == {"window_seconds", "objects", "count"}


def test_rooms_endpoint(sim) -> None:
    r = sim.client.get("/api/rooms").json()
    assert r["tag_size_m"] == pytest.approx(0.16)
    names = [x["name"] for x in r["rooms"]]
    assert names == ["kitchen", "living_room"]
    kitchen = r["rooms"][0]
    assert set(kitchen) >= {
        "name",
        "anchor_tag",
        "anchor_height_m",
        "anchor_name",
        "extents_m",
        "landmarks",
        "tags_registered",
        "last_anchored_ms",
    }
    assert kitchen["anchor_tag"] == 0 and kitchen["tags_registered"] == [0]
    assert kitchen["landmarks"] == {"table": [1.0, 0.75, 1.2]}
    assert isinstance(kitchen["last_anchored_ms"], int)
    assert r["current"]["room"] == "kitchen"
    assert sorted(r["rooms_seen"]) == ["kitchen", "living_room"]


def test_handheld_status(sim) -> None:
    st = sim.client.get("/api/handheld/status").json()
    assert set(st) >= {"config", "runtime", "memory", "restore", "started_at_ms", "voice"}
    assert st["config"]["detection_source"] == "sim" and st["config"]["label_embedder"] == "hash"
    assert "OPENAI" not in str(st["config"])
    rt = st["runtime"]
    assert rt["source"]["state"] == "done" and rt["frames"] == 180 and rt["errors"] == 0
    assert set(rt["source"]) >= {"state", "fps", "frames", "reconnects", "usb_speed", "error"}
    assert set(rt["ingestion"]) == {"buffer_entries", "fires", "suppressed_by_holdoff"}
    assert rt["localizer"]["room"] == "kitchen"
    assert st["memory"]["embedder"] == "hash:128" and st["memory"]["observation_count"] > 0
    assert st["memory"]["frames"]["room"] > 0
    assert st["restore"]["source"] == "empty" and st["restore"]["records"] == 0
    assert st["voice"] is None
    assert st["errors"] == {}
    health = sim.client.get("/health").json()
    assert health["status"] == "ok"
    assert health["handheld"] == {"detection_source": "sim", "source_state": "done", "voice": False}
    stats = sim.client.get("/api/stats").json()
    assert stats["handheld"]["frames"] == 180


def test_phone_page(sim) -> None:
    r = sim.client.get("/phone")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    html = r.text
    assert 'aria-live="assertive"' in html and 'role="status"' in html
    assert "speechSynthesis" in html and "MediaRecorder" in html and "isSecureContext" in html
    assert "/api/query" in html and "/api/objects/changes" in html and "/api/rooms" in html
    assert "Where is my mug?" in html and "What changed in the kitchen?" in html
    assert "min-height: 48px" in html and 'name="viewport"' in html
    # offline: no external scripts or stylesheets
    assert "https://" not in html.split("<script>")[0].replace("https://example", "")
    assert "cdn." not in html and "<link" not in html


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


def test_ws_first_message_is_snapshot_then_live_pushes(sim) -> None:
    with sim.client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot"
        assert set(first) >= {"objects", "observation_count", "rooms_seen", "anchor"}
        assert sorted(first["rooms_seen"]) == ["kitchen", "living_room"]
        assert first["anchor"]["room"] == "kitchen"
        assert {o["label"] for o in first["objects"]} >= {"cup", "bottle"}
        seen: dict[str, dict] = {}
        for _ in range(6):
            msg = ws.receive_json()
            seen.setdefault(msg["type"], msg)
            if {"detections", "anchor"} <= set(seen):
                break
        assert {"detections", "anchor"} <= set(seen), sorted(seen)
        det = seen["detections"]
        assert det["source"] == "sim" and det["seq"] == 179
        assert set(det) >= {"timestamp_ms", "detections", "frame_count", "observation_count"}
        assert "anchor" in det
        assert det["detections"] and all("distance_m" in d for d in det["detections"])
        anchor = seen["anchor"]
        assert anchor["room"] == "kitchen" and anchor["source_state"] == "done"


def test_legacy_ingest_frame_still_broadcasts(sim) -> None:
    with sim.client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        r = sim.client.post(
            "/api/ingest/frame",
            json={"image_b64": "data:image/png;base64," + base64.b64encode(PNG_1X1).decode()},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["detections"] == []  # YOLO is not installed in the test environment
        assert set(body) == {"detections", "frame_count", "observation_count"}
        legacy = None
        for _ in range(12):
            msg = ws.receive_json()
            if msg["type"] == "detections" and "source" not in msg:
                legacy = msg
                break
        assert legacy is not None and legacy["detections"] == []
        assert legacy["frame_count"] == body["frame_count"]


# ---------------------------------------------------------------------------
# Voice endpoints (legacy shapes + additive keys)
# ---------------------------------------------------------------------------


def test_voice_text_query_keeps_shape_and_adds_resolution(sim) -> None:
    r = sim.client.post("/api/voice/text-query", json={"text": "where is my mug"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "transcription",
        "intent",
        "answer",
        "has_audio",
        "audio_b64",
        "audio_mime",
        "resolution",
        "latency_ms",
    }
    assert body["audio_b64"] is None and body["has_audio"] is False and body["audio_mime"] is None
    assert body["intent"] == {"kind": "where_is", "object_name": "mug", "room": ""}
    assert body["resolution"]["label"] == "cup"
    assert "kitchen" in body["answer"].lower()


def test_voice_tts_sniffs_media_type(sim, monkeypatch) -> None:
    async def fake_wav(text: str) -> bytes:
        return b"RIFF\x24\x00\x00\x00WAVEfmt " + text.encode()

    monkeypatch.setattr(sim.main.voice._tts, "synthesize", fake_wav)
    r = sim.client.post("/api/voice/tts", json={"text": "hello"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    assert r.content.startswith(b"RIFF")

    async def fake_mp3(text: str) -> bytes:
        return b"ID3\x04" + text.encode()

    monkeypatch.setattr(sim.main.voice._tts, "synthesize", fake_mp3)
    r = sim.client.post("/api/voice/tts", json={"text": "hello"})
    assert r.headers["content-type"].startswith("audio/mpeg")
    # /api/query with tts=true carries the same bytes inline
    q = _query(sim.client, "where is my mug", tts=True)
    assert q["audio_mime"] == "audio/mpeg"
    assert base64.b64decode(q["audio_b64"]).startswith(b"ID3")

    async def no_audio(text: str) -> bytes:
        return b""

    monkeypatch.setattr(sim.main.voice._tts, "synthesize", no_audio)
    assert sim.client.post("/api/voice/tts", json={"text": "hello"}).status_code == 503


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def test_snapshot_endpoint_and_clear_guard(sim) -> None:
    r = sim.client.post("/api/handheld/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["persistence"]) >= {"enabled", "data_dir", "journal_lines", "errors"}
    assert body["persistence"]["enabled"] is True and body["persistence"]["errors"] == 0
    assert (Path(body["persistence"]["data_dir"]) / "memory.snapshot.json").is_file()
    assert sim.client.post("/api/memory/clear", json={}).status_code == 400
    assert sim.client.post("/api/memory/clear", json={"confirm": False}).status_code == 400
    # nothing was cleared
    assert sim.client.get("/api/objects").json()["count"] >= 4


# ---------------------------------------------------------------------------
# Separate boots: aged simulation ("moved") and the simulated restart
# ---------------------------------------------------------------------------


def test_changes_report_moved_cup_when_the_walk_started_earlier(tmp_path: Path) -> None:
    # Shift the scripted walk 602 s into the past: the first kitchen sighting of the cup
    # falls before the 10-minute window, the second (moved) one inside it (§7.5).
    with booted(tmp_path / "aged", LOCI_SIM_START_OFFSET_S="602") as (main, client):
        q = _query(client, "what changed in the kitchen")
        kinds = {(c["kind"], c["label"]) for c in q["changes"]}
        assert ("moved", "cup") in kinds, kinds
        moved = next(c for c in q["changes"] if c["label"] == "cup")
        assert moved["delta_m"] == pytest.approx(2.26, abs=0.3)
        assert moved["pos_before_m"] == pytest.approx([1.0, 0.8, 1.0], abs=0.3)
        assert "moved" in q["answer"].lower() and "cup" in q["answer"].lower()
        api = client.get("/api/objects/changes", params={"room": "kitchen"}).json()
        assert ("moved", "cup") in {(c["kind"], c["label"]) for c in api["changes"]}
        assert main.runtime is not None and main.runtime.frames == 180


def test_memory_survives_a_simulated_restart(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with booted(data_dir) as (main, client):
        labels = {o["label"] for o in client.get("/api/objects").json()["objects"]}
        assert "cup" in labels
        count_before = main.memory.observation_count
        assert count_before > 0
    # the lifespan shutdown wrote the final snapshot
    assert (data_dir / "memory.snapshot.json").is_file()

    with booted(data_dir, LOCI_DETECTION_SOURCE="none") as (main2, client2):
        assert main2.runtime is None
        st = client2.get("/api/handheld/status").json()
        assert st["runtime"] is None
        assert st["restore"]["records"] > 0 and st["restore"]["source"] == "snapshot"
        assert sorted(st["restore"]["rooms"]) == ["kitchen", "living_room"]
        objs = client2.get("/api/objects", params={"room": "kitchen"}).json()["objects"]
        by_label = {o["label"]: o for o in objs}
        assert "cup" in by_label and by_label["cup"]["restored"] is True
        assert (by_label["cup"]["x_m"], by_label["cup"]["z_m"]) == pytest.approx(
            (-1.2, 1.5), abs=0.3
        )
        # Without a handheld source the resolver vocabulary is the browser demo's
        # class list (which detects "mug" itself), so ask for the stored label.
        q = _query(client2, "where is my cup")
        assert q["resolution"]["label"] == "cup" and "kitchen" in q["answer"].lower()
        assert "before i restarted" in q["answer"].lower()
        assert q["results"][0]["restored"] is True
        # no localizer: camera-only anchor, rooms still listed
        a = client2.get("/api/anchor").json()
        assert a == {
            "room": None,
            "frame": "camera",
            "anchored": False,
            "anchor_confidence": 0.0,
            "camera_xyz_m": None,
            "heading_deg": None,
            "tag_ids": [],
            "age_s": None,
        }
        rooms = client2.get("/api/rooms").json()
        assert [r["name"] for r in rooms["rooms"]] == ["kitchen", "living_room"]
        assert rooms["current"] is None
        health = client2.get("/health").json()
        assert health["handheld"]["detection_source"] == "none"
        # confirmed clear wipes memory and the persisted files
        r = client2.post("/api/memory/clear", json={"confirm": True})
        assert r.status_code == 200 and r.json()["status"] == "cleared"
        assert client2.get("/api/objects").json()["count"] == 0
        assert _query(client2, "where is my cup")["results"] == []
    with booted(data_dir, LOCI_DETECTION_SOURCE="none") as (_main3, client3):
        assert client3.get("/api/handheld/status").json()["restore"]["records"] == 0


def test_bad_source_configuration_keeps_the_api_up(tmp_path: Path) -> None:
    with booted(tmp_path / "bad", LOCI_DETECTION_SOURCE="replay", LOCI_REPLAY_PATH=None) as (
        main,
        client,
    ):
        assert main.runtime is None
        st = client.get("/api/handheld/status").json()
        assert "source" in st["errors"] and "LOCI_REPLAY_PATH" in st["errors"]["source"]
        assert client.get("/health").status_code == 200
        q = _query(client, "where is my cup")
        assert q["resolution"]["label"] == "cup" and q["results"] == []


def test_module_level_names_for_other_tasks(sim) -> None:
    """The names ``deploy/handheld`` scripts and T11's console rely on."""
    m = sim.main
    assert callable(m._broadcast) and callable(m._live_push_loop)
    assert m.cfg.detection_source == "sim"
    assert m.localizer is not None and m.runtime is not None
    assert m.voice.status()["pose_provider"] is True
    assert m.memory.rooms.names() == ["kitchen", "living_room"]
    assert isinstance(m._started_at_ms, int) and m._started_at_ms <= int(time.time() * 1000)
