"""Host side of the board's local storage: backlog frames with capture times, memory snapshot,
spoken summary and cached phrase audio. Fakes only."""

from __future__ import annotations

import io
import os
import sys
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.esp32_api import install_esp32_routes  # noqa: E402
from app.esp32_source import Esp32Config, Esp32Hub  # noqa: E402
from app.esp32_voice import Esp32VoiceBridge  # noqa: E402
from app.scene_ingestion import Detection  # noqa: E402


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (40, 80, 120)).save(buf, format="JPEG")
    return buf.getvalue()


def _tone_wav(seconds: float = 0.3, rate: int = 16000) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    raw = (np.sin(2 * np.pi * 440 * t) * 0.3 * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw)
    return buf.getvalue()


class FakeMemory:
    def __init__(self) -> None:
        self.objects = [
            SimpleNamespace(
                label="cup",
                position_description="in the center, in the middle area",
                room=None,
                timestamp_ms=1_700_000_000_000,
                age_seconds=42.0,
            ),
            SimpleNamespace(
                label="scissors",
                position_description="on the left, near the front",
                room="kitchen",
                timestamp_ms=1_699_999_990_000,
                age_seconds=52.0,
            ),
        ]
        self.observation_count = 2
        self.tracked_labels = ["cup", "scissors"]

    def current_objects(self):
        return list(self.objects)


class FakeVoice:
    tts_engine = "fake"
    stt_available = False
    _stt = SimpleNamespace(_use_local=False)

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    @property
    def _tts(self):
        voice = self

        class _T:
            async def synthesize(self, text: str) -> bytes:
                voice.synthesized.append(text)
                return _tone_wav()

        return _T()


def _build(clock_ms: int = 1_700_000_100_000):
    memory = FakeMemory()
    ingestion = MagicMock()
    ingestion.classes = ["cup"]
    ingestion.frame_count = 0
    ingestion.process_detections = MagicMock(return_value=[])
    cfg = Esp32Config(max_fps=5.0)
    hub = Esp32Hub(
        ingestion,
        cfg,
        detector=lambda _b: [Detection("cup", 0.5, 0.5, 0.1, 0.1, 0.9, source="esp32")],
        clock=lambda: clock_ms / 1000.0,
        memory=memory,
    )
    voice = FakeVoice()
    bridge = Esp32VoiceBridge(voice, rate=16000)
    app = FastAPI()
    install_esp32_routes(
        app, ingestion=ingestion, memory=memory, voice=voice, cfg=cfg, hub=hub, bridge=bridge
    )
    return SimpleNamespace(
        client=TestClient(app),
        hub=hub,
        ingestion=ingestion,
        memory=memory,
        voice=voice,
        clock_ms=clock_ms,
    )


# ---------------------------------------------------------------------------
# frames: server clock, memory revision, backlog
# ---------------------------------------------------------------------------


def test_frame_reply_carries_clock_and_memory_rev() -> None:
    t = _build()
    body = t.client.post(
        "/api/esp32/frame", content=_jpeg(), headers={"Content-Type": "image/jpeg"}
    ).json()
    assert body["now_ms"] == t.clock_ms
    assert body["memory_rev"] == t.hub.memory_rev() and body["memory_rev"] not in ("", "empty")
    assert body["backlog"] is False


def test_backlog_frame_is_stored_at_capture_time_and_skips_rate_limit() -> None:
    t = _build()
    live = t.client.post(
        "/api/esp32/frame", content=_jpeg(), headers={"Content-Type": "image/jpeg"}
    ).json()
    assert live["dropped"] is False
    captured = t.clock_ms - 5 * 60_000  # five minutes ago, while the host was unreachable
    back = t.client.post(
        "/api/esp32/frame",
        content=_jpeg(),
        headers={"Content-Type": "image/jpeg", "X-Backlog": "1", "X-Captured-Ms": str(captured)},
    ).json()
    assert (
        back["dropped"] is False and back["backlog"] is True
    )  # immediate second frame, not rate limited
    ts_live = t.ingestion.process_detections.call_args_list[0].kwargs["timestamp_ms"]
    ts_back = t.ingestion.process_detections.call_args_list[1].kwargs["timestamp_ms"]
    assert ts_live == t.clock_ms and ts_back == captured
    assert t.hub.device().backlog_frames == 1
    # a nonsense capture time (future / zero) falls back to the host clock
    weird = t.client.post(
        "/api/esp32/frame",
        content=_jpeg(),
        headers={
            "Content-Type": "image/jpeg",
            "X-Backlog": "1",
            "X-Captured-Ms": str(t.clock_ms + 10 * 60_000),
        },
    ).json()
    assert weird["backlog"] is True
    assert t.ingestion.process_detections.call_args_list[2].kwargs["timestamp_ms"] == t.clock_ms


# ---------------------------------------------------------------------------
# memory snapshot and speech
# ---------------------------------------------------------------------------


def test_memory_snapshot_and_rev_stability() -> None:
    t = _build()
    snap = t.client.get("/api/esp32/memory").json()
    assert snap["count"] == 2 and snap["objects"][0]["label"] == "cup"
    assert snap["objects"][1]["room"] == "kitchen" and snap["objects"][0]["age_s"] == 42
    assert (
        "cup in the center" in snap["summary"]
        and "scissors in the kitchen on the left" in snap["summary"]
    )
    rev = snap["rev"]
    # ageing alone does not change the revision; a move does
    t.memory.objects[0].age_seconds = 900.0
    assert t.client.get("/api/esp32/memory").json()["rev"] == rev
    t.memory.objects[0].position_description = "on the right, near the back"
    assert t.client.get("/api/esp32/memory").json()["rev"] != rev
    t.memory.objects = []
    empty = t.client.get("/api/esp32/memory").json()
    assert empty["rev"] == "empty" and "don't remember any objects" in empty["summary"]


def test_memory_pcm_and_phrase_cache() -> None:
    t = _build()
    r = t.client.get("/api/esp32/memory.pcm")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/L16")
    assert r.headers["x-memory-rev"] == t.hub.memory_rev() and len(r.content) == 9600
    assert t.voice.synthesized[-1].startswith("I can't reach the host right now.")
    n = len(t.voice.synthesized)
    t.client.get("/api/esp32/memory.pcm")  # same summary -> served from the cache
    assert len(t.voice.synthesized) == n
    p = t.client.get("/api/esp32/phrase.pcm", params={"text": "Hold the button and ask again."})
    assert p.status_code == 200 and len(p.content) == 9600
    t.client.get("/api/esp32/phrase.pcm", params={"text": "Hold the button and ask again."})
    assert t.voice.synthesized.count("Hold the button and ask again.") == 1
    assert t.client.get("/api/esp32/phrase.pcm").status_code == 422
    assert t.client.get("/api/esp32/status").json()["memory_rev"] == t.hub.memory_rev()


def test_memory_rev_ignores_the_order_of_sightings() -> None:
    t = _build()
    cup = {"label": "cup", "where": "on the left", "room": "kitchen"}
    cat = {"label": "cat", "where": "in the center", "room": None}
    assert t.hub.memory_rev([cup, cat]) == t.hub.memory_rev([cat, cup])
    moved = {**cup, "where": "on the right"}
    assert t.hub.memory_rev([moved, cat]) != t.hub.memory_rev([cup, cat])
    assert t.hub.memory_rev([]) == "empty"
