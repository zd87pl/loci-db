"""HTTP contract between an ESP32 board and the host (``/api/esp32/*``), with fakes.

No YOLO, no Whisper, no TTS engine, no hardware: the detector is a callable returning a
cup, STT is a coroutine returning fixed text, TTS returns a WAV tone. The LOCI
SpatialMemory is a MagicMock so the test only covers the ESP32 glue.
"""

from __future__ import annotations

import io
import os
import sys
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.esp32_api import install_esp32_routes  # noqa: E402
from app.esp32_source import Esp32Config, Esp32Hub  # noqa: E402
from app.esp32_voice import COULD_NOT_HEAR, Esp32VoiceBridge  # noqa: E402
from app.pcm_audio import pcm16_duration_s  # noqa: E402
from app.scene_ingestion import Detection, SceneIngestion  # noqa: E402
from app.voice_pipeline import QueryIntent, parse_intent  # noqa: E402


def _jpeg(w: int = 64, h: int = 48) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _tone_wav(rate: int = 22050, seconds: float = 0.4) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    raw = (np.sin(2 * np.pi * 440 * t) * 0.4 * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw)
    return buf.getvalue()


def _speech_pcm(seconds: float = 1.0, rate: int = 16000) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2").tobytes()


class FakeVoice:
    """Stands in for VoicePipeline: rule-based answer + WAV 'speech'."""

    def __init__(self) -> None:
        self.tts_engine = "fake-wav"
        self.stt_available = False
        self._stt = SimpleNamespace(_use_local=False)
        self.asked: list[str] = []
        self.tts_ok = True

    async def handle_text_query(self, text: str):
        self.asked.append(text)
        intent = parse_intent(text)
        answer = (
            f"Your {intent.object_name} is on the table." if intent.object_name else "I see a cup."
        )
        return SimpleNamespace(
            transcription=text,
            intent=intent,
            answer_text=answer,
            answer_audio=_tone_wav() if self.tts_ok else b"",
            latency_ms=1.0,
        )

    @property
    def _tts(self):
        voice = self

        class _T:
            async def synthesize(self, text: str) -> bytes:
                return _tone_wav() if voice.tts_ok else b""

        return _T()


class FakeWS:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    async def send_text(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("gone")
        self.sent.append(text)


async def fake_transcriber(samples: np.ndarray, language: str) -> str:
    assert samples.dtype == np.float32 and language == "en"
    return "where is my cup" if samples.size >= 8000 else ""


def _build(detector=None, voice: FakeVoice | None = None, auth=None, **cfg_kwargs):
    memory = MagicMock()
    memory.observe = MagicMock(return_value="s1")
    memory.observation_count = 0
    memory.tracked_labels = ["cup"]
    ingestion = SceneIngestion(
        memory=memory, vlm_client=None, use_vlm_fallback=False, consensus_min_count=1
    )
    cfg = Esp32Config(max_fps=cfg_kwargs.pop("max_fps", 0), **cfg_kwargs)
    detector = detector or (
        lambda _b: [Detection("cup", 0.5, 0.6, 0.2, 0.2, confidence=0.88, source="esp32")]
    )
    hub = Esp32Hub(ingestion, cfg, detector=detector)
    voice = voice or FakeVoice()
    bridge = Esp32VoiceBridge(voice, rate=cfg.audio_rate, transcriber=fake_transcriber)
    subscribers = {FakeWS(), FakeWS(fail=True)}
    app = FastAPI()
    install_esp32_routes(
        app,
        ingestion=ingestion,
        memory=memory,
        voice=voice,
        subscribers=subscribers,
        cfg=cfg,
        hub=hub,
        bridge=bridge,
        auth=auth,
    )
    client = TestClient(app)
    return SimpleNamespace(
        app=app,
        client=client,
        hub=hub,
        memory=memory,
        ingestion=ingestion,
        voice=voice,
        subscribers=subscribers,
    )


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def test_frame_post_raw_jpeg_detects_and_broadcasts() -> None:
    t = _build()
    r = t.client.post(
        "/api/esp32/frame",
        content=_jpeg(),
        headers={
            "Content-Type": "image/jpeg",
            "X-Device-Id": "pendant",
            "X-Seq": "41",
            "X-Rssi": "-55",
            "X-Device-Ms": "9000",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        body["ok"] is True
        and body["labels"] == ["cup"]
        and body["seq"] == 41
        and body["device"] == "pendant"
    )
    assert body["detections"][0]["source"] == "esp32" and body["audio_pending"] == 0
    # consensus_min_count=1 → stored straight away, and the WS set got a detections message
    t.memory.observe.assert_called_once()
    live = [ws for ws in t.subscribers if not ws.fail]
    assert len(live) == 1 and len(t.subscribers) == 1  # the failing socket was dropped
    import json

    msg = json.loads(live[0].sent[-1])
    assert msg["type"] == "detections" and msg["source"] == "esp32" and msg["device"] == "pendant"
    assert t.hub.device("pendant").rssi == -55 and t.hub.device("pendant").device_ms == 9000


def test_frame_post_multipart_and_query_params() -> None:
    t = _build()
    r = t.client.post(
        "/api/esp32/frame?device=cam2&seq=5", files={"file": ("f.jpg", _jpeg(), "image/jpeg")}
    )
    assert r.status_code == 200 and r.json()["seq"] == 5 and r.json()["device"] == "cam2"


def test_frame_post_rejects_empty_and_non_jpeg() -> None:
    t = _build()
    assert (
        t.client.post(
            "/api/esp32/frame", content=b"", headers={"Content-Type": "image/jpeg"}
        ).status_code
        == 400
    )
    r = t.client.post(
        "/api/esp32/frame",
        content=b"\x89PNG\r\n\x1a\n" + b"\0" * 32,
        headers={"Content-Type": "image/png"},
    )
    assert r.status_code == 400 and "JPEG" in r.json()["detail"]


def test_frame_jpg_and_status() -> None:
    t = _build()
    assert t.client.get("/api/esp32/frame.jpg").status_code == 404
    st = t.client.get("/api/esp32/status").json()
    assert (
        st["devices"] == {}
        and st["voice"]["stt"] == "custom"
        and st["memory"]["tracked_labels"] == ["cup"]
    )
    jpeg = _jpeg(20, 10)
    t.client.post("/api/esp32/frame", content=jpeg, headers={"Content-Type": "image/jpeg"})
    r = t.client.get("/api/esp32/frame.jpg")
    assert r.status_code == 200 and r.content == jpeg and r.headers["content-type"] == "image/jpeg"
    st = t.client.get("/api/esp32/status").json()
    dev = st["devices"]["esp32"]
    assert dev["online"] is True and dev["last_labels"] == ["cup"] and dev["last_size"] == [20, 10]
    assert st["config"]["audio_rate"] == 16000
    assert t.client.get("/esp32").status_code == 200


# ---------------------------------------------------------------------------
# audio + questions
# ---------------------------------------------------------------------------


def test_ask_queues_pcm_and_board_fetches_it() -> None:
    t = _build()
    assert t.client.get("/api/esp32/audio/next").status_code == 204
    r = t.client.post("/api/esp32/ask", json={"text": "where is my cup"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Your cup is on the table." and body["intent"]["kind"] == "where_is"
    assert body["audio_queued"] is True and body["queue_len"] == 1 and body["has_audio"] is True
    assert 0.35 < body["audio_seconds"] < 0.45
    # the board's next frame learns that audio is waiting ...
    f = t.client.post(
        "/api/esp32/frame", content=_jpeg(), headers={"Content-Type": "image/jpeg"}
    ).json()
    assert f["audio_pending"] == 1
    # ... and fetches it as raw 16 kHz PCM
    a = t.client.get("/api/esp32/audio/next?wait=0")
    assert a.status_code == 200
    assert (
        a.headers["content-type"].startswith("audio/L16") and a.headers["x-audio-rate"] == "16000"
    )
    assert a.headers["x-audio-queued"] == "0"
    assert pcm16_duration_s(a.content, 16000) == pytest.approx(0.4, abs=0.01)
    assert t.client.get("/api/esp32/audio/next").status_code == 204


def test_ask_without_speech_and_tts_failure() -> None:
    t = _build()
    r = t.client.post("/api/esp32/ask", json={"text": "where is my cup", "speak": False}).json()
    assert r["audio_queued"] is False and r["has_audio"] is False and r["answer"]
    t.voice.tts_ok = False
    r = t.client.post("/api/esp32/ask", json={"text": "where is my cup"}).json()
    assert r["audio_queued"] is False and r["tts_error"] == "tts produced no audio"
    assert t.client.post("/api/esp32/say", json={"text": "hello"}).status_code == 503
    assert t.client.post("/api/esp32/ask", json={"text": ""}).status_code == 422


def test_say_queues_for_named_device() -> None:
    t = _build()
    r = t.client.post("/api/esp32/say", json={"text": "hello there", "device": "pendant"})
    assert r.status_code == 200 and r.json()["device"] == "pendant" and r.json()["queue_len"] == 1
    assert t.client.get("/api/esp32/audio/next").status_code == 204  # default device: nothing
    assert t.client.get("/api/esp32/audio/next?device=pendant").status_code == 200


def test_voice_pcm_roundtrip() -> None:
    t = _build()
    r = t.client.post(
        "/api/esp32/voice?rate=16000",
        content=_speech_pcm(1.0),
        headers={"Content-Type": "audio/L16; rate=16000", "X-Device-Id": "pendant"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (
        body["transcription"] == "where is my cup" and body["answer"] == "Your cup is on the table."
    )
    assert (
        body["audio_queued"] is True and body["device"] == "pendant" and body["seconds_in"] == 1.0
    )
    assert t.voice.asked == ["where is my cup"]
    assert t.client.get("/api/esp32/audio/next?device=pendant").status_code == 200


def test_voice_resamples_other_rates_and_accepts_wav() -> None:
    t = _build()
    # 8 kHz PCM is resampled to 16 kHz before STT (fake STT needs >= 0.5 s at 16 kHz)
    r = t.client.post("/api/esp32/voice?rate=8000", content=_speech_pcm(1.0, rate=8000)).json()
    assert r["seconds_in"] == 1.0 and r["transcription"] == "where is my cup"
    r = t.client.post("/api/esp32/voice", content=_tone_wav(rate=22050, seconds=1.0)).json()
    assert (
        r["seconds_in"] == pytest.approx(1.0, abs=0.01) and r["transcription"] == "where is my cup"
    )


def test_voice_silence_and_short_clips_get_a_spoken_hint() -> None:
    t = _build()
    r = t.client.post("/api/esp32/voice", content=b"\x00" * 32000).json()  # 1 s of digital silence
    assert r["transcription"] == "" and r["answer"] == COULD_NOT_HEAR and r["audio_queued"] is True
    r = t.client.post("/api/esp32/voice", content=_speech_pcm(0.1)).json()  # too short
    assert r["transcription"] == "" and r["intent"]["kind"] == "unknown"
    assert t.client.post("/api/esp32/voice", content=b"").status_code == 400
    assert t.voice.asked == []


# ---------------------------------------------------------------------------
# voice bridge internals
# ---------------------------------------------------------------------------


def test_bridge_uses_local_whisper_model_directly() -> None:
    """When the pipeline holds a local faster-whisper model, PCM goes to it as float32."""
    import asyncio

    seen = {}

    class FakeModel:
        def transcribe(self, samples, language="en", beam_size=1):
            seen["samples"] = samples
            seen["language"] = language
            return [SimpleNamespace(text=" what do you see ")], {}

    stt = SimpleNamespace(_use_local=True, _local_model=None)

    def load():
        stt._local_model = FakeModel()

    stt._load_local_model = load
    voice = FakeVoice()
    voice._stt = stt
    bridge = Esp32VoiceBridge(voice, rate=16000)
    text = asyncio.run(bridge.transcribe_pcm(_speech_pcm(1.0)))
    assert text == "what do you see" and seen["language"] == "en"
    assert seen["samples"].dtype == np.float32 and seen["samples"].size == 16000
    assert bridge.status()["stt"] == "faster-whisper (local)"


def test_bridge_answer_pcm_reports_stt_latency_and_intent() -> None:
    import asyncio

    bridge = Esp32VoiceBridge(FakeVoice(), rate=16000, transcriber=fake_transcriber)
    ans = asyncio.run(bridge.answer_pcm(_speech_pcm(1.0)))
    assert ans.transcription == "where is my cup" and ans.intent_kind == "where_is"
    assert ans.pcm and ans.stt_ms >= 0 and ans.to_dict()["has_audio"] is True
    assert isinstance(parse_intent("where is my cup"), QueryIntent)
