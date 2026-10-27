"""Host speech hygiene: Whisper decode options and guards, edge-tts timeout, queued clip expiry."""

from __future__ import annotations

import asyncio
import os
import sys
import time
import types
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.esp32_source import Esp32Config, Esp32Hub  # noqa: E402
from app.esp32_voice import Esp32VoiceBridge  # noqa: E402
from app.voice_pipeline import (  # noqa: E402
    TTS,
    WhisperSTT,
    is_stt_hallucination,
    stt_prompt,
)


def _seg(text: str, no_speech: float = 0.05, logprob: float = -0.3) -> SimpleNamespace:
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob)


class FakeModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls: list[dict] = []

    def transcribe(self, audio, language="en", **opts):
        self.calls.append({"language": language, **opts})
        return iter(self.segments), SimpleNamespace(language=language)


def _stt(segments, monkeypatch) -> tuple[WhisperSTT, FakeModel]:
    monkeypatch.setenv("WHISPER_LOCAL", "1")
    monkeypatch.delenv("WHISPER_BEAM", raising=False)
    monkeypatch.delenv("WHISPER_VAD", raising=False)
    stt = WhisperSTT()
    model = FakeModel(segments)
    stt._local_model = model
    return stt, model


def test_stock_phrases_are_not_speech() -> None:
    for text in ("Thank you.", " thanks for watching! ", "You", "", "...", "Bye-bye"):
        assert is_stt_hallucination(text), text
    for text in ("Where is my cup?", "thank you, where are my keys", "cup"):
        assert not is_stt_hallucination(text), text


def test_prompt_lists_remembered_labels_first_then_household_objects_and_rooms() -> None:
    p = stt_prompt(["dining_table", "cup", "cup"], ["living_room", "kitchen"])
    assert p.startswith("Hey Loci, where is my dining table, cup, bottle")
    assert p.count("cup") == 1 and "What changed in the living room, kitchen?" in p
    assert stt_prompt().startswith("Hey Loci, where is my cup, bottle")
    many = stt_prompt([f"thing{i}" for i in range(100)])
    assert many.count("thing") == 40  # at most 40 labels


def test_transcribe_samples_uses_vad_prompt_and_guards(monkeypatch) -> None:
    stt, model = _stt([_seg(" Where is my laptop?")], monkeypatch)
    stt.vocabulary = lambda: (["laptop"], ["kitchen"])
    assert stt.transcribe_samples(np.zeros(16000, np.float32)) == "Where is my laptop?"
    opts = model.calls[0]
    assert opts["vad_filter"] is True and opts["vad_parameters"]["min_speech_duration_ms"] == 200
    assert opts["beam_size"] == 1 and opts["condition_on_previous_text"] is False
    assert opts["initial_prompt"].startswith("Hey Loci, where is my laptop")

    stt, _ = _stt([_seg(" Thank you.")], monkeypatch)
    assert stt.transcribe_samples(np.zeros(16000, np.float32)) == ""
    stt, _ = _stt([_seg(" cup", no_speech=0.9, logprob=-1.5)], monkeypatch)
    assert stt.transcribe_samples(np.zeros(16000, np.float32)) == ""
    stt, _ = _stt([], monkeypatch)  # VAD removed everything
    assert stt.transcribe_samples(np.zeros(16000, np.float32)) == ""


def test_decode_options_follow_the_environment(monkeypatch) -> None:
    stt, model = _stt([_seg("cup")], monkeypatch)
    monkeypatch.setenv("WHISPER_BEAM", "5")
    monkeypatch.setenv("WHISPER_VAD", "0")
    stt.vocabulary = lambda: (_ for _ in ()).throw(RuntimeError("memory busy"))
    opts = stt.decode_options()
    assert opts["beam_size"] == 5 and "vad_filter" not in opts
    assert opts["initial_prompt"].startswith("Hey Loci, where is my cup")  # vocabulary failure


def test_bridge_decodes_through_the_pipeline_stt_off_the_event_loop(monkeypatch) -> None:
    stt, model = _stt([_seg(" where is my cup")], monkeypatch)
    loop_threads: list[bool] = []
    real = stt.transcribe_samples

    def spy(samples, language="en"):
        try:
            asyncio.get_running_loop()
            loop_threads.append(True)
        except RuntimeError:
            loop_threads.append(False)
        return real(samples, language)

    stt.transcribe_samples = spy
    voice = SimpleNamespace(_stt=stt, tts_engine="none")
    bridge = Esp32VoiceBridge(voice, rate=16000)
    rng = np.random.default_rng(0)
    pcm = (rng.normal(0, 3000, 16000)).astype("<i2").tobytes()
    assert asyncio.run(bridge.transcribe_pcm(pcm)) == "where is my cup"
    assert loop_threads == [False]  # ran in an executor thread
    assert "vad_filter" in model.calls[0]


def test_bridge_warm_up_calls_the_stt_and_survives_failures() -> None:
    called = []
    voice = SimpleNamespace(_stt=SimpleNamespace(warm_up=lambda: called.append(1)))
    Esp32VoiceBridge(voice).warm_up()
    assert called == [1]

    def boom():
        raise RuntimeError("no model")

    bridge = Esp32VoiceBridge(SimpleNamespace(_stt=SimpleNamespace(warm_up=boom)))
    bridge.warm_up()
    assert "no model" in (bridge.last_error or "")


def test_edge_tts_that_stalls_gives_no_audio_instead_of_hanging(monkeypatch) -> None:
    class Communicate:
        def __init__(self, text, voice):
            pass

        async def stream(self):
            await asyncio.sleep(30)
            yield {"type": "audio", "data": b"late"}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Communicate))
    monkeypatch.setenv("TTS_TIMEOUT_S", "0.2")
    tts = TTS()
    t0 = time.perf_counter()
    assert asyncio.run(tts._synthesize_edge("hello")) == b""
    assert time.perf_counter() - t0 < 5


def test_queued_clips_expire() -> None:
    cfg = Esp32Config(max_fps=0, audio_ttl_s=0.1)
    hub = Esp32Hub(MagicIngestion(), cfg, detector=lambda _b: [])
    hub.queue_audio(b"\x01\x00" * 100, "d")
    time.sleep(0.2)
    hub.queue_audio(b"\x02\x00" * 100, "d")
    assert asyncio.run(hub.next_audio("d")) == b"\x02\x00" * 100
    dev = hub.device("d")
    assert dev.audio_dropped_total == 1 and dev.audio_sent_total == 1
    assert asyncio.run(hub.next_audio("d")) is None
    assert Esp32Config.from_env({"LOCI_ESP32_AUDIO_TTL_S": "15"}).audio_ttl_s == 15.0


class MagicIngestion:
    frame_count = 0
    last_detections: list = []

    def process_detections(self, detections, timestamp_ms=None):
        self.frame_count += 1
