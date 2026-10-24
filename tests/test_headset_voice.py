"""Tests for the headset voice loop (task T11) using fakes only: no sounddevice, no
microphone, no Whisper, no Piper. Every hardware-adjacent path is exercised through
injected fake modules, byte streams and callables.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app import headset_voice as hv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONSOLE = REPO / "deploy" / "handheld" / "scripts" / "voice_console.py"
RATE = hv.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _tone(
    seconds: float, freq: float = 440.0, amplitude: int = 8000, rate: int = RATE
) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.int16)


def _noise(seconds: float, amplitude: int = 40, rate: int = RATE, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-amplitude, amplitude + 1, size=int(seconds * rate)).astype(np.int16)


class FakeAudio:
    """Returns a fixed recording; remembers everything played."""

    name = "fake"

    def __init__(self, recording: bytes | None = None) -> None:
        self.recording = recording if recording is not None else hv.pcm16_to_wav(_tone(0.5))
        self.played: list[bytes] = []
        self.record_calls: list[tuple[float, bool]] = []

    def record(self, max_seconds: float, stop: threading.Event | None = None) -> bytes:
        self.record_calls.append((max_seconds, stop is not None))
        return self.recording

    def play(self, wav_bytes: bytes) -> None:
        self.played.append(wav_bytes)

    def list_devices(self) -> list[dict]:
        return [
            {
                "index": 0,
                "name": "MacBook Pro Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
            },
            {
                "index": 1,
                "name": "MacBook Pro Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {"index": 2, "name": "WH-1000XM4", "max_input_channels": 1, "max_output_channels": 2},
        ]


class FakeStreamAudio(FakeAudio):
    """Serves a long signal in successive ``record(chunk_s)`` calls (VAD fallback path)."""

    def __init__(self, signal: np.ndarray) -> None:
        super().__init__()
        self.signal = signal.astype(np.int16)
        self.pos = 0

    def record(self, max_seconds: float, stop: threading.Event | None = None) -> bytes:
        n = int(max_seconds * RATE)
        chunk = self.signal[self.pos : self.pos + n]
        self.pos += n
        if len(chunk) == 0 and stop is not None:
            stop.set()
        return hv.pcm16_to_wav(chunk)


class FakeChunkAudio(FakeAudio):
    """Offers ``chunks()`` like SoundDeviceAudio/CliAudio do (VAD persistent-stream path)."""

    def __init__(self, signal: np.ndarray) -> None:
        super().__init__()
        self.signal = signal.astype(np.int16)
        self.pos = 0
        self.opened = 0
        self.closed = 0

    def chunks(self, chunk_s: float, stop: threading.Event | None = None):
        self.opened += 1
        n = int(chunk_s * RATE)
        try:
            while self.pos < len(self.signal):
                if stop is not None and stop.is_set():
                    return
                chunk = self.signal[self.pos : self.pos + n]
                self.pos += n
                yield chunk
            if stop is not None:
                stop.set()  # the microphone "went away": a real backend would end the stream
        finally:
            self.closed += 1


class FakePTT:
    """Scripted presses; sets ``stop`` when the script runs out."""

    name = "fake"

    def __init__(self, presses: int = 1) -> None:
        self.remaining = presses
        self.releases = 0

    def wait_for_press(self, stop: threading.Event) -> bool:
        if self.remaining <= 0:
            stop.set()
            return False
        self.remaining -= 1
        return True

    def wait_for_release(self, timeout_s: float) -> None:
        self.releases += 1


class FakeSTT:
    def __init__(self, text: str = "where is my mug", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.inputs: list[bytes] = []

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        self.inputs.append(audio_bytes)
        if self.fail:
            raise RuntimeError("stt exploded")
        return self.text


class FakeTTS:
    def __init__(self, empty: bool = False) -> None:
        self.texts: list[str] = []
        self.empty = empty
        self.wav = hv.pcm16_to_wav(_tone(0.3, 220.0), 22050)

    async def synthesize(self, text: str) -> bytes:
        self.texts.append(text)
        return b"" if self.empty else self.wav


def _events(loop: hv.HeadsetVoiceLoop) -> list[str]:
    return [e["type"] for e in loop.events]


# ---------------------------------------------------------------------------
# WAV helpers and earcon
# ---------------------------------------------------------------------------


def test_wav_round_trip_int16_mono() -> None:
    rng = np.random.default_rng(1)
    samples = rng.integers(-32768, 32767, size=4000, dtype=np.int16)
    wav = hv.pcm16_to_wav(samples, 16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    decoded, rate = hv.wav_to_pcm16(wav)
    assert rate == 16000
    assert decoded.dtype == np.int16
    np.testing.assert_array_equal(decoded, samples)
    assert hv.wav_duration_s(wav) == pytest.approx(0.25)


def test_wav_decode_stereo_and_8bit() -> None:
    left = np.full(100, 1000, dtype=np.int16)
    right = np.full(100, 3000, dtype=np.int16)
    stereo = np.stack([left, right], axis=1)
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(stereo.astype("<i2").tobytes())
    mono, rate = hv.wav_to_pcm16(buf.getvalue())
    assert rate == 8000
    assert mono.shape == (100,)
    assert int(mono[0]) == 2000

    buf8 = io.BytesIO()
    with wave.open(buf8, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(bytes([128, 255, 0]))
    pcm8, _ = hv.wav_to_pcm16(buf8.getvalue())
    assert pcm8.tolist() == [0, 127 << 8, -128 << 8]


def test_resample_linear_doubles_length_and_keeps_frequency() -> None:
    src = _tone(1.0, 200.0, rate=8000)
    out = hv.resample_linear(src, 8000, 16000)
    assert out.dtype == np.int16
    assert len(out) == 16000
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    peak_hz = np.fft.rfftfreq(len(out), 1 / 16000)[int(np.argmax(spectrum))]
    assert peak_hz == pytest.approx(200.0, abs=2.0)
    assert hv.resample_linear(src, 8000, 8000) is src


def test_earcon_tone_shape() -> None:
    wav = hv.earcon_wav()
    samples, rate = hv.wav_to_pcm16(wav)
    assert rate == RATE
    assert len(samples) == int(round(hv.EARCON_S * RATE)) == 1280
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    peak_hz = np.fft.rfftfreq(len(samples), 1 / RATE)[int(np.argmax(spectrum))]
    assert peak_hz == pytest.approx(880.0, abs=15.0)
    peak = int(np.max(np.abs(samples)))
    assert 0.25 * 32767 < peak <= 0.3 * 32767 + 1
    # fades: the first and last samples are quiet, the middle is not
    assert abs(int(samples[0])) < 200 and abs(int(samples[-1])) < 200
    assert hv.rms(samples[200:1000]) > 5000
    assert hv.rms(np.zeros(10, dtype=np.int16)) == 0.0


# ---------------------------------------------------------------------------
# One utterance end to end
# ---------------------------------------------------------------------------


def test_one_utterance_end_to_end_events_in_order() -> None:
    audio = FakeAudio()
    ptt = FakePTT(presses=1)
    stt = FakeSTT("where is my mug")
    tts = FakeTTS()
    seen: list[dict] = []
    asked: list[str] = []

    def answer_fn(text: str) -> str:
        asked.append(text)
        return "Your mug is in the kitchen."

    loop = hv.HeadsetVoiceLoop(
        answer_fn, stt, tts, audio, ptt, max_seconds=4.0, on_event=seen.append
    )
    stop = threading.Event()
    loop.run(stop)

    types = [e["type"] for e in seen]
    assert types[0] == "started"
    assert types[1:9] == [
        "press",
        "earcon",
        "recorded",
        "transcript",
        "answer",
        "tts",
        "played",
        "done",
    ]
    assert types[-1] == "stopped"
    assert asked == ["where is my mug"]
    assert stt.inputs == [audio.recording]
    assert tts.texts == ["Your mug is in the kitchen."]
    # earcon first, then the spoken answer
    assert len(audio.played) == 2
    assert hv.wav_duration_s(audio.played[0]) == pytest.approx(hv.EARCON_S)
    assert audio.played[1] == tts.wav
    # the recorder got the release event and the max length
    assert audio.record_calls == [(4.0, True)]
    assert ptt.releases == 1
    for ev in seen:
        assert set(ev) >= {"type", "text", "latency_ms", "ts_ms"}
    by_type = {e["type"]: e for e in seen}
    assert by_type["transcript"]["text"] == "where is my mug"
    assert by_type["answer"]["text"] == "Your mug is in the kitchen."
    assert by_type["done"]["latency_ms"] >= 0
    assert loop.turns == 1 and loop.errors == 0 and not loop.running
    status = loop.status()
    assert status["ptt"] == "fake" and status["audio"] == "fake"
    assert status["last_events"][-1]["type"] == "stopped"
    assert loop.last_events(3)[-1]["type"] == "stopped"


def test_no_earcon_and_empty_transcript() -> None:
    audio = FakeAudio()
    stt = FakeSTT("")
    tts = FakeTTS()
    calls: list[str] = []
    loop = hv.HeadsetVoiceLoop(
        lambda t: calls.append(t) or "x", stt, tts, audio, FakePTT(1), earcon=False
    )
    loop.run(threading.Event())
    assert "earcon" not in _events(loop)
    assert calls == []
    assert tts.texts == [hv.NO_SPEECH_TEXT]
    assert audio.played == [tts.wav]


def test_handle_once_without_tts_audio() -> None:
    audio = FakeAudio()
    loop = hv.HeadsetVoiceLoop(
        lambda t: "answer", FakeSTT("hi"), FakeTTS(empty=True), audio, FakePTT(0)
    )
    result = loop.handle_once(audio.recording)
    assert result["text"] == "hi" and result["answer"] == "answer"
    assert result["audio_bytes"] == 0 and result["audio"] == b""
    assert audio.played == []
    assert "tts_unavailable" in _events(loop)
    assert "played" not in _events(loop)


def test_sync_stt_and_tts_are_accepted() -> None:
    class SyncSTT:
        def transcribe(self, b: bytes) -> str:
            return "sync text"

    class SyncTTS:
        def synthesize(self, t: str) -> bytes:
            return b"RIFFfake"

    audio = FakeAudio()
    loop = hv.HeadsetVoiceLoop(lambda t: t.upper(), SyncSTT(), SyncTTS(), audio, FakePTT(0))
    result = loop.handle_once(b"")
    assert result["answer"] == "SYNC TEXT"
    assert audio.played == [b"RIFFfake"]


def test_run_maybe_async_inside_running_loop() -> None:
    import asyncio

    async def inner() -> int:
        return 42

    async def outer() -> int:
        return hv.run_maybe_async(inner())

    assert asyncio.run(outer()) == 42
    assert hv.run_maybe_async(7) == 7


def test_loop_continues_after_stt_exception_and_answer_fn_exception() -> None:
    audio = FakeAudio()
    stt = FakeSTT("q", fail=True)
    tts = FakeTTS()
    loop = hv.HeadsetVoiceLoop(lambda t: "fine", stt, tts, audio, FakePTT(2), earcon=False)
    loop.run(threading.Event())
    types = _events(loop)
    assert types.count("press") == 2
    assert types.count("error") == 2
    assert loop.errors == 2 and loop.turns == 2
    assert types[-1] == "stopped"

    audio = FakeAudio()
    n = {"calls": 0}

    def flaky(text: str) -> str:
        n["calls"] += 1
        if n["calls"] == 1:
            raise ValueError("boom")
        return "second time lucky"

    tts = FakeTTS()
    loop = hv.HeadsetVoiceLoop(flaky, FakeSTT("q"), tts, audio, FakePTT(2), earcon=False)
    loop.run(threading.Event())
    assert tts.texts == [hv.ANSWER_FAILED_TEXT, "second time lucky"]
    assert loop.errors == 1
    assert len(audio.played) == 2


def test_run_backs_off_when_ptt_returns_without_press() -> None:
    class GivingUpPTT(FakePTT):
        def __init__(self) -> None:
            super().__init__(0)
            self.calls = 0

        def wait_for_press(self, stop: threading.Event) -> bool:
            self.calls += 1
            return False  # e.g. stdin closed, device unplugged: never sets stop

    ptt = GivingUpPTT()
    loop = hv.HeadsetVoiceLoop(lambda t: t, FakeSTT(), FakeTTS(), FakeAudio(), ptt)
    loop.idle_backoff_s = 0.02
    stop = threading.Event()
    t = hv.start_voice_thread(loop, stop)
    import time

    time.sleep(0.2)
    stop.set()
    t.join(2.0)
    assert not t.is_alive()
    assert 1 <= ptt.calls <= 15  # backed off instead of spinning thousands of times
    assert _events(loop).count("ptt_idle") == 1


def test_run_on_daemon_thread_stops() -> None:
    class BlockingPTT(FakePTT):
        def wait_for_press(self, stop: threading.Event) -> bool:
            stop.wait(5.0)
            return False

    loop = hv.HeadsetVoiceLoop(lambda t: t, FakeSTT(), FakeTTS(), FakeAudio(), BlockingPTT())
    stop = threading.Event()
    t = hv.start_voice_thread(loop, stop)
    assert t.daemon and t.name == "headset-voice"
    stop.set()
    t.join(2.0)
    assert not t.is_alive()
    assert _events(loop) == ["started", "stopped"]


# ---------------------------------------------------------------------------
# VadPTT segmentation
# ---------------------------------------------------------------------------


def _vad_signal() -> np.ndarray:
    return np.concatenate([_noise(1.0), _tone(1.5), _noise(2.0, seed=1)])


@pytest.mark.parametrize("audio_cls", [FakeStreamAudio, FakeChunkAudio])
def test_vad_segments_silence_tone_silence(audio_cls) -> None:
    audio = audio_cls(_vad_signal())
    ptt = hv.VadPTT(audio, threshold=500.0, silence_s=1.2, chunk_s=0.1, pre_roll_s=0.2)
    assert ptt.captures_audio is True
    stop = threading.Event()
    assert ptt.wait_for_press(stop) is True
    ptt.wait_for_release(10.0)
    rec = ptt.take_recording()
    samples, rate = hv.wav_to_pcm16(rec)
    assert rate == RATE
    # pre-roll (0.2 s) + tone (1.5 s) + trailing silence (1.2 s) = 2.9 s
    assert len(samples) / RATE == pytest.approx(2.9, abs=0.15)
    # the loud part sits inside the segment, preceded by the quiet pre-roll
    assert hv.rms(samples[: int(0.15 * RATE)]) < 500
    assert hv.rms(samples[int(0.3 * RATE) : int(1.5 * RATE)]) > 4000
    # second take: the signal is exhausted -> no press, stop honoured
    assert ptt.wait_for_press(stop) is False
    assert ptt.take_recording() == hv.silence_wav(0.0)


def test_vad_loop_integration_uses_captured_audio() -> None:
    audio = FakeChunkAudio(_vad_signal())
    ptt = hv.VadPTT(audio, threshold=500.0, silence_s=1.2, chunk_s=0.1)
    stt = FakeSTT("what changed")
    loop = hv.HeadsetVoiceLoop(lambda t: "nothing", stt, FakeTTS(), audio, ptt, earcon=False)
    loop.run(threading.Event())
    assert loop.turns == 1
    assert audio.record_calls == []  # the loop did not record a second time
    assert len(stt.inputs) == 1
    assert hv.wav_duration_s(stt.inputs[0]) == pytest.approx(3.0, abs=0.2)
    assert audio.closed == audio.opened >= 1


def test_vad_max_seconds_and_silence_only() -> None:
    audio = FakeStreamAudio(_tone(5.0))
    ptt = hv.VadPTT(audio, threshold=500.0, silence_s=1.2, chunk_s=0.1, pre_roll_s=0.1)
    assert ptt.wait_for_press(threading.Event())
    ptt.wait_for_release(1.0)
    assert hv.wav_duration_s(ptt.take_recording()) == pytest.approx(1.1, abs=0.1)

    quiet = FakeStreamAudio(_noise(1.0))
    ptt = hv.VadPTT(quiet, threshold=500.0, chunk_s=0.1)
    stop = threading.Event()
    assert ptt.wait_for_press(stop) is False
    assert stop.is_set()


# ---------------------------------------------------------------------------
# EvdevButtonPTT
# ---------------------------------------------------------------------------


class ByteStream:
    """Feeds pre-packed input_event bytes to EvdevButtonPTT in arbitrary slices."""

    def __init__(self, chunks: list[bytes], *, then: bytes | None = None) -> None:
        self.chunks = deque(chunks)
        self.then = then  # b"" = keep timing out, None = device gone
        self.calls = 0

    def __call__(self, timeout_s: float) -> bytes | None:
        self.calls += 1
        if self.chunks:
            return self.chunks.popleft()
        return self.then


def _key(code: int, value: int) -> bytes:
    return hv.pack_input_event(hv.EV_KEY, code, value, sec=1, usec=2) + hv.pack_input_event(
        hv.EV_SYN, 0, 0
    )


def test_input_event_struct_and_parse() -> None:
    assert struct.calcsize("llHHi") == hv.INPUT_EVENT_SIZE
    raw = _key(hv.KEY_PLAYPAUSE, 1)
    events, rest = hv.parse_input_events(raw + b"\x01\x02")
    assert rest == b"\x01\x02"
    assert events[0] == hv.InputEvent(1, 2, hv.EV_KEY, hv.KEY_PLAYPAUSE, 1)
    assert events[1].type == hv.EV_SYN


def test_evdev_press_and_release() -> None:
    stream = ByteStream(
        [
            _key(115, 1),  # KEY_VOLUMEUP: not the button
            _key(115, 0),
            _key(hv.KEY_PLAYPAUSE, 2),  # autorepeat: ignored
            _key(hv.KEY_PLAYPAUSE, 1),  # press
            _key(hv.KEY_PLAYPAUSE, 0),  # release
        ],
        then=b"",
    )
    ptt = hv.EvdevButtonPTT(reader=stream, poll_s=0.05)
    assert ptt.name == "evdev"
    assert ptt.wait_for_press(threading.Event()) is True
    assert stream.calls == 4
    ptt.wait_for_release(5.0)
    assert stream.calls == 5


def test_evdev_split_reads_and_other_codes() -> None:
    raw = _key(hv.KEY_NEXTSONG, 1)
    stream = ByteStream([raw[:7], raw[7:30], raw[30:]], then=None)
    ptt = hv.EvdevButtonPTT(reader=stream, codes=[hv.KEY_NEXTSONG], poll_s=0.05)
    assert ptt.wait_for_press(threading.Event()) is True
    # device gone (None) -> release returns, press returns False
    ptt.wait_for_release(5.0)
    assert ptt.wait_for_press(threading.Event()) is False


def test_evdev_toggle_mode() -> None:
    stream = ByteStream(
        [
            _key(hv.KEY_PLAYCD, 1),
            _key(hv.KEY_PLAYCD, 0),  # the click's own release must NOT end the recording
            _key(hv.KEY_PAUSECD, 1),  # second click ends it
            _key(hv.KEY_PAUSECD, 0),
        ],
        then=b"",
    )
    ptt = hv.EvdevButtonPTT(reader=stream, toggle=True, poll_s=0.05)
    assert ptt.wait_for_press(threading.Event()) is True
    assert stream.calls == 1
    ptt.wait_for_release(5.0)
    assert stream.calls == 3


def test_evdev_release_timeout_and_stop() -> None:
    stream = ByteStream([], then=b"")
    ptt = hv.EvdevButtonPTT(reader=stream, poll_s=0.02)
    import time

    t0 = time.monotonic()
    ptt.wait_for_release(0.15)
    assert 0.1 <= time.monotonic() - t0 < 1.0
    stop = threading.Event()
    stop.set()
    assert ptt.wait_for_press(stop) is False


PROC_TEXT = """I: Bus=0019 Vendor=0000 Product=0001 Version=0000
N: Name="Power Button"
P: Phys=LNXPWRBN/button/input0
H: Handlers=kbd event0
B: EV=3

I: Bus=0005 Vendor=054c Product=0ce0 Version=0100
N: Name="WH-1000XM4 (AVRCP)"
P: Phys=dc:a6:32:aa:bb:cc
S: Sysfs=/devices/virtual/input/input23
U: Uniq=94:db:56:11:22:33
H: Handlers=kbd event5
B: PROP=0
B: EV=100007
B: KEY=2fc800 145200000000 0 10300 49e800000c00 e16800000000f f810000010000ffc

I: Bus=0003 Vendor=046d Product=c52b Version=0111
N: Name="Logitech USB Receiver"
H: Handlers=sysrq kbd event3 leds
"""


def test_find_input_device_from_proc_text() -> None:
    assert hv.find_input_device("avrcp", PROC_TEXT) == "/dev/input/event5"
    assert hv.find_input_device("wh-1000", PROC_TEXT) == "/dev/input/event5"
    assert hv.find_input_device("logitech", PROC_TEXT) == "/dev/input/event3"
    assert hv.find_input_device("nothing like this", PROC_TEXT) is None
    devs = hv.list_input_devices(PROC_TEXT)
    assert [d["name"] for d in devs] == [
        "Power Button",
        "WH-1000XM4 (AVRCP)",
        "Logitech USB Receiver",
    ]
    assert devs[1]["event"] == "/dev/input/event5"
    ptt = hv.EvdevButtonPTT.from_spec("/dev/input/event9", toggle=True)
    assert ptt.device_path == "/dev/input/event9" and ptt.toggle
    assert hv.EvdevButtonPTT.from_spec("")._name == hv.DEFAULT_PTT_DEVICE


# ---------------------------------------------------------------------------
# KeyboardPTT
# ---------------------------------------------------------------------------


def test_keyboard_ptt_on_a_pipe_like_stdin() -> None:
    out = io.StringIO()
    ptt = hv.KeyboardPTT(stdin=io.StringIO("\n\n"), stdout=out, require_tty=False)
    assert ptt.wait_for_press(threading.Event()) is True
    ptt.wait_for_release(1.0)
    assert ptt.wait_for_press(threading.Event()) is False  # EOF
    assert "Press Enter" in out.getvalue()
    strict = hv.KeyboardPTT(stdin=io.StringIO(""), stdout=out)
    with pytest.raises(RuntimeError, match="TTY|terminal"):
        strict.wait_for_press(threading.Event())


# ---------------------------------------------------------------------------
# Device selection, SoundDeviceAudio with a fake sounddevice, CliAudio commands
# ---------------------------------------------------------------------------


def test_select_device_by_substring() -> None:
    devices = FakeAudio().list_devices()
    assert hv.select_device(devices, None, "input") is None
    assert hv.select_device(devices, "", "input") is None
    assert hv.select_device(devices, "default", "output") is None
    assert hv.select_device(devices, "wh-1000", "input") == 2
    assert hv.select_device(devices, "WH-1000XM4", "output") == 2
    assert hv.select_device(devices, "macbook", "input") == 0
    assert hv.select_device(devices, "macbook", "output") == 1
    assert hv.select_device(devices, "1", "output") == 1
    assert hv.select_device(devices, 2, "output") == 2
    with pytest.raises(LookupError, match="no input device matching 'airpods'"):
        hv.select_device(devices, "airpods", "input")
    with pytest.raises(LookupError):
        hv.select_device(devices, "macbook pro microphone", "output")


class FakeSD:
    """Just enough of the sounddevice module for SoundDeviceAudio."""

    class PortAudioError(Exception):
        pass

    def __init__(self, signal: np.ndarray, rate: int = 48000, allow_16k: bool = False) -> None:
        self.signal = signal.astype(np.int16)
        self.rate = rate
        self.allow_16k = allow_16k
        self.devices = [
            {
                "index": 0,
                "name": "Built-in Mic",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": rate,
            },
            {
                "index": 1,
                "name": "Built-in Output",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": rate,
            },
            {
                "index": 2,
                "name": "WH-1000XM4",
                "max_input_channels": 1,
                "max_output_channels": 2,
                "default_samplerate": rate,
            },
        ]
        self.played: list[tuple[int, int | None, int]] = []
        self.streams: list[dict] = []

    def query_devices(self, device=None, kind=None):
        if device is not None:
            return self.devices[device]
        if kind is not None:
            return self.devices[0] if kind == "input" else self.devices[1]
        return list(self.devices)

    def check_input_settings(self, device=None, samplerate=None, channels=None, dtype=None):
        if samplerate != self.rate and not self.allow_16k:
            raise FakeSD.PortAudioError(f"{samplerate} Hz unsupported")

    def InputStream(self, samplerate, channels, dtype, device, blocksize):  # noqa: N802
        sd = self

        class _Stream:
            def __init__(self) -> None:
                self.pos = 0
                self.info = {
                    "samplerate": samplerate,
                    "device": device,
                    "blocksize": blocksize,
                    "closed": False,
                }
                sd.streams.append(self.info)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.info["closed"] = True

            def read(self, n):
                chunk = sd.signal[self.pos : self.pos + n]
                self.pos += n
                if len(chunk) < n:
                    chunk = np.concatenate([chunk, np.zeros(n - len(chunk), dtype=np.int16)])
                return chunk.reshape(-1, 1), False

        return _Stream()

    def play(self, data, samplerate, device=None, blocking=True):
        if samplerate != self.rate:
            raise FakeSD.PortAudioError("rate")
        self.played.append((int(samplerate), device, len(data)))


def test_sounddevice_audio_records_resamples_and_selects_devices() -> None:
    sd = FakeSD(_tone(2.0, 300.0, rate=48000), rate=48000)
    audio = hv.SoundDeviceAudio("wh-1000", "wh-1000", sd_module=sd, chunk_s=0.1)
    assert audio.name == "sounddevice"
    names = [d["name"] for d in audio.list_devices()]
    assert names == ["Built-in Mic", "Built-in Output", "WH-1000XM4"]
    wav = audio.record(0.5)
    samples, rate = hv.wav_to_pcm16(wav)
    assert rate == RATE
    assert len(samples) == int(0.5 * RATE)
    assert sd.streams[-1]["samplerate"] == 48000 and sd.streams[-1]["device"] == 2
    assert sd.streams[-1]["closed"] is True
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    assert np.fft.rfftfreq(len(samples), 1 / RATE)[int(np.argmax(spectrum))] == pytest.approx(
        300, abs=5
    )

    # playback of a 22.05 kHz Piper-style WAV: the fake only accepts 48 kHz -> resampled
    audio.play(hv.pcm16_to_wav(_tone(0.2, rate=22050), 22050))
    assert sd.played == [(48000, 2, int(0.2 * 48000))]
    audio.play(b"")
    assert len(sd.played) == 1

    # a substring that matches nothing falls back to the default device, logged once
    other = hv.SoundDeviceAudio("airpods", None, sd_module=sd, chunk_s=0.1)
    other.record(0.1)
    assert sd.streams[-1]["device"] is None

    # stop event ends the recording early
    stop = threading.Event()
    stop.set()
    assert hv.wav_duration_s(audio.record(5.0, stop=stop)) == 0.0

    # 16 kHz native capture needs no resampling
    sd16 = FakeSD(_tone(1.0), rate=16000, allow_16k=True)
    native = hv.SoundDeviceAudio(sd_module=sd16, chunk_s=0.1)
    chunks = list(native.chunks(0.1, stop))
    assert chunks == []
    wav = native.record(0.3)
    assert hv.wav_duration_s(wav) == pytest.approx(0.3)
    assert sd16.streams[-1]["samplerate"] == 16000


def test_cli_audio_commands_linux_and_mac() -> None:
    have = {"pw-record", "pw-play", "arecord", "aplay", "afplay", "pactl"}

    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in have else None

    linux = hv.CliAudio("bluez_input.XX", "bluez_output.XX", system="linux", which=which)
    assert linux.record_command("/x/rec.wav", 3.2) == [
        "pw-record", "--rate", "16000", "--channels", "1", "--format", "s16",
        "--target", "bluez_input.XX", "/x/rec.wav",
    ]  # fmt: skip
    assert linux.play_command("/x/out.wav") == [
        "pw-play",
        "--target",
        "bluez_output.XX",
        "/x/out.wav",
    ]
    assert linux.stream_command()[:4] == ["arecord", "-q", "-t", "raw"]
    assert linux.stream_command()[-1] == "-"

    have.discard("pw-record")
    have.discard("pw-play")
    alsa = hv.CliAudio("hw:1,0", None, system="linux", which=which)
    assert alsa.record_command("/x/rec.wav", 3.2) == [
        "arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1",
        "-d", "4", "-D", "hw:1,0", "/x/rec.wav",
    ]  # fmt: skip
    assert alsa.play_command("/x/out.wav") == ["aplay", "-q", "/x/out.wav"]
    have.discard("arecord")
    have.add("pw-record")
    assert alsa.stream_command()[:2] == ["pw-record", "--raw"]

    mac = hv.CliAudio(system="darwin", which=which)
    assert mac.play_command("/x/out.wav") == ["afplay", "/x/out.wav"]
    assert mac.play_command("/x/out.mp3") == ["afplay", "/x/out.mp3"]
    assert mac.list_devices() == []
    with pytest.raises(RuntimeError, match="cannot record"):
        mac.record_command("/x/rec.wav", 1.0)
    with pytest.raises(RuntimeError, match="cannot record"):
        mac.record(1.0)


def test_cli_audio_play_and_list_with_fake_runner() -> None:
    calls: list[list[str]] = []

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.returncode = 0

    def runner(cmd, **kwargs):
        calls.append(list(cmd))
        assert "shell" not in kwargs
        if cmd[:3] == ["pactl", "list", "short"]:
            return Result("51\tbluez_output.94_DB_56.1\tPipeWire\ts16le 2ch 48000Hz\tRUNNING\n")
        if cmd == ["aplay", "-l"]:
            return Result("card 1: Headset [WH-1000XM4], device 0: USB Audio [USB Audio]\n")
        return Result("")

    def which(name: str) -> str | None:
        return "/usr/bin/" + name if name in {"aplay", "pactl"} else None

    audio = hv.CliAudio(None, "bluez_output.94_DB_56.1", system="linux", runner=runner, which=which)
    audio.play(hv.silence_wav(0.1))
    assert calls[-1][:4] == ["aplay", "-q", "-D", "bluez_output.94_DB_56.1"]
    assert calls[-1][-1].endswith(".wav")
    devices = audio.list_devices()
    names = {d["name"] for d in devices}
    assert "bluez_output.94_DB_56.1" in names
    assert any(n.startswith("hw:1,0") for n in names)
    assert all(int(d["max_output_channels"]) == 1 for d in devices if d["backend"] == "alsa")
    assert audio.is_linux and not audio.is_mac


def test_cli_audio_record_with_fake_popen(tmp_path: Path) -> None:
    class Proc:
        def __init__(self, cmd, **kwargs) -> None:
            Path(cmd[-1]).write_bytes(hv.pcm16_to_wav(_tone(0.2, rate=48000), 48000))
            self.signals: list[int] = []

        def poll(self):
            return None if not self.signals else 0

        def send_signal(self, sig):
            self.signals.append(sig)

        def wait(self, timeout=None):
            return 0

    def which(name: str) -> str | None:
        return "/usr/bin/arecord" if name == "arecord" else None

    audio = hv.CliAudio(system="linux", which=which, popen=Proc)
    wav = audio.record(0.05)
    samples, rate = hv.wav_to_pcm16(wav)
    assert rate == RATE and len(samples) == int(0.2 * RATE)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def test_make_voice_loop_from_env_none_and_headset() -> None:
    fake_audio = FakeAudio()
    assert (
        hv.make_voice_loop_from_env(str, FakeSTT(), FakeTTS(), env={"LOCI_VOICE": "none"}) is None
    )
    assert hv.make_voice_loop_from_env(str, FakeSTT(), FakeTTS(), env={}) is None
    assert (
        hv.make_voice_loop_from_env(str, FakeSTT(), FakeTTS(), env={"LOCI_VOICE": "bogus"}) is None
    )

    env = {
        "LOCI_VOICE": "headset",
        "LOCI_PTT": "keyboard",
        "LOCI_VOICE_MAX_S": "6.5",
        "LOCI_EARCON": "0",
    }
    loop = hv.make_voice_loop_from_env(str, FakeSTT(), FakeTTS(), env=env, audio=fake_audio)
    assert isinstance(loop, hv.HeadsetVoiceLoop)
    assert isinstance(loop._ptt, hv.KeyboardPTT)
    assert loop._audio is fake_audio
    assert loop.max_seconds == 6.5 and loop.earcon is False
    assert loop.status()["ptt"] == "keyboard"

    env = {
        "LOCI_VOICE": "headset",
        "LOCI_PTT": "vad",
        "LOCI_VAD_THRESHOLD": "750",
        "LOCI_VAD_SILENCE_S": "0.8",
        "LOCI_VOICE_MAX_S": "not-a-number",
    }
    loop = hv.make_voice_loop_from_env(str, FakeSTT(), FakeTTS(), env=env, audio=fake_audio)
    assert isinstance(loop._ptt, hv.VadPTT)
    assert loop._ptt.threshold == 750.0 and loop._ptt.silence_s == 0.8
    assert loop.max_seconds == 10.0 and loop.earcon is True

    env = {
        "LOCI_VOICE": "headset",
        "LOCI_PTT": "evdev",
        "LOCI_PTT_DEVICE": "wh-1000",
        "LOCI_PTT_TOGGLE": "1",
    }
    loop = hv.make_voice_loop_from_env(str, FakeSTT(), FakeTTS(), env=env, audio=fake_audio)
    assert isinstance(loop._ptt, hv.EvdevButtonPTT)
    assert loop._ptt.toggle is True and loop._ptt._name == "wh-1000"

    ptt = FakePTT(0)
    loop = hv.make_voice_loop_from_env(
        str, FakeSTT(), FakeTTS(), env={"LOCI_VOICE": "headset"}, audio=fake_audio, ptt=ptt
    )
    assert loop._ptt is ptt

    with pytest.raises(ValueError, match="LOCI_PTT"):
        hv.make_ptt_from_env(fake_audio, {"LOCI_PTT": "telepathy"})


def test_make_audio_from_env_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = hv.make_audio_from_env(
        {"LOCI_AUDIO_BACKEND": "cli", "LOCI_AUDIO_IN": "mic", "LOCI_AUDIO_OUT": ""}
    )
    assert isinstance(audio, hv.CliAudio)
    assert audio._in == "mic" and audio._out is None
    with pytest.raises(ValueError, match="LOCI_AUDIO_BACKEND"):
        hv.make_audio_from_env({"LOCI_AUDIO_BACKEND": "gramophone"})

    monkeypatch.setattr(hv, "sounddevice_available", lambda: False)
    assert isinstance(hv.make_audio_from_env({"LOCI_AUDIO_BACKEND": "auto"}), hv.CliAudio)
    with pytest.raises(RuntimeError, match="sounddevice"):
        hv.make_audio_from_env({"LOCI_AUDIO_BACKEND": "sounddevice"})

    monkeypatch.setattr(hv, "sounddevice_available", lambda: True)
    sd_audio = hv.make_audio_from_env({"LOCI_AUDIO_BACKEND": "auto", "LOCI_AUDIO_IN": "WH"})
    assert isinstance(sd_audio, hv.SoundDeviceAudio)
    assert sd_audio._input_spec == "WH"


def test_default_ptt_kind_depends_on_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hv.sys, "platform", "linux")
    assert isinstance(hv.make_ptt_from_env(FakeAudio(), {}), hv.EvdevButtonPTT)
    monkeypatch.setattr(hv.sys, "platform", "darwin")
    assert isinstance(hv.make_ptt_from_env(FakeAudio(), {}), hv.KeyboardPTT)


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(FakeAudio(), hv.AudioIO)
    assert isinstance(FakePTT(), hv.PushToTalk)
    assert isinstance(hv.CliAudio(system="linux"), hv.AudioIO)
    assert isinstance(hv.VadPTT(FakeAudio()), hv.PushToTalk)
    assert isinstance(hv.EvdevButtonPTT(reader=lambda t: None), hv.PushToTalk)


# ---------------------------------------------------------------------------
# voice_console.py
# ---------------------------------------------------------------------------


def _load_console():
    spec = importlib.util.spec_from_file_location("voice_console_under_test", CONSOLE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_console_loopback_text_canned_with_tts_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    mod = _load_console()
    fake_tts = FakeTTS()
    monkeypatch.setattr(mod, "build_tts", lambda args: fake_tts)
    out = tmp_path / "answer.wav"
    code = mod.main(["--loopback-text", "where is my mug", "--canned", "--tts-out", str(out)])
    assert code == 0
    assert out.exists() and out.read_bytes() == fake_tts.wav
    captured = capsys.readouterr().out
    assert "transcript" in captured and "where is my mug" in captured
    import json

    summary = json.loads(captured.strip().splitlines()[-1])
    assert summary["text"] == "where is my mug"
    assert "mug" in summary["answer"]
    assert summary["tts_out"] == str(out)
    assert fake_tts.texts == [summary["answer"]]


def test_console_loopback_wav_with_fake_stt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    mod = _load_console()
    wav_path = tmp_path / "q.wav"
    wav_path.write_bytes(hv.pcm16_to_wav(_tone(0.4)))
    monkeypatch.setattr(mod, "build_stt", lambda args: FakeSTT("what changed in the kitchen"))
    code = mod.main(["--loopback-wav", str(wav_path), "--canned", "--no-tts"])
    assert code == 0
    out = capsys.readouterr().out
    # intent wording depends on voice_pipeline.parse_intent (T10); only check the plumbing
    assert "what changed in the kitchen" in out and "Loopback console" in out


def test_console_remote_falls_back_when_server_down(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    mod = _load_console()
    code = mod.main(
        [
            "--loopback-text",
            "where are my keys",
            "--no-tts",
            "--host",
            "127.0.0.1",
            "--port",
            "1",
            "--timeout",
            "1",
        ]
    )
    assert code == 0
    assert "keys" in capsys.readouterr().out


def test_console_subprocess_list_devices_and_help() -> None:
    env = dict(os.environ, LOCI_AUDIO_BACKEND="cli")
    res = subprocess.run(
        [sys.executable, str(CONSOLE), "--list-devices"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "audio devices" in res.stdout and "input (button) devices" in res.stdout
    res = subprocess.run(
        [sys.executable, str(CONSOLE), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert res.returncode == 0
    assert "--loopback-text" in res.stdout and "--ptt" in res.stdout
    res = subprocess.run(
        [sys.executable, str(CONSOLE), "--loopback-text", "hi", "--loopback-wav", "x.wav"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert res.returncode == 2


def test_console_subprocess_loopback_text_canned() -> None:
    env = dict(os.environ, TTS_ENGINE="openai")
    env.pop("OPENAI_API_KEY", None)
    res = subprocess.run(
        [sys.executable, str(CONSOLE), "--loopback-text", "where is my mug", "--canned"],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "tts_unavailable" in res.stdout  # no TTS backend configured -> still an answer
    assert '"answer"' in res.stdout.strip().splitlines()[-1]


def test_console_in_process_answers_from_the_simulator(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """``--in-process`` runs the non-realtime simulator through handheld.build_handheld
    and answers from that memory (no server, no TTS, no audio hardware)."""
    pytest.importorskip("fastapi")  # demo_spatial.app.handheld pulls the app stack
    for key, value in {
        "LOCI_DETECTION_SOURCE": "sim",
        "LOCI_SIM_REALTIME": "0",
        "LOCI_SIM_ORACLE_POSE": "1",
        "LOCI_PERSIST": "0",
        "LOCI_LABEL_EMBEDDER": "hash",
        "LOCI_VOICE": "none",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("LOCI_RECORD_PATH", raising=False)
    mod = _load_console()
    code = mod.main(["--loopback-text", "where is my mug", "--in-process", "--no-tts"])
    out = capsys.readouterr().out
    assert code == 0, out
    last = json.loads(out.strip().splitlines()[-1])
    assert "kitchen" in last["answer"] and "cup" in last["answer"]
    assert last["text"] == "where is my mug"


# ---------------------------------------------------------------------------
# Review fixes: configuration errors stop the loop once; STT missing is said aloud
# ---------------------------------------------------------------------------


def test_ptt_config_error_stops_the_loop_once() -> None:
    class BadPTT:
        name = "bad"
        calls = 0

        def wait_for_press(self, stop: threading.Event) -> bool:
            BadPTT.calls += 1
            raise hv.PTTConfigError("needs an interactive terminal")

        def wait_for_release(self, timeout_s: float) -> None:
            return None

    loop = hv.HeadsetVoiceLoop(str, FakeSTT(), FakeTTS(), FakeAudio(), BadPTT(), earcon=False)
    stop = threading.Event()
    t0 = time.monotonic()
    loop.run(stop)  # returns by itself: no retry every 0.5 s
    assert time.monotonic() - t0 < 0.4 and BadPTT.calls == 1
    assert loop.running is False and loop.errors == 1
    assert loop.fatal is not None and "interactive terminal" in loop.fatal
    assert _events(loop) == ["started", "error", "stopped"]
    assert loop.events[1]["fatal"] is True and loop.status()["fatal"] == loop.fatal
    # KeyboardPTT without a TTY is exactly that kind of error
    ptt = hv.KeyboardPTT(stdin=io.StringIO(""), stdout=io.StringIO())
    with pytest.raises(hv.PTTConfigError):
        ptt.wait_for_press(threading.Event())
    assert issubclass(hv.PTTConfigError, RuntimeError)


def test_stt_unavailable_is_spoken_instead_of_try_again() -> None:
    class NoSTT(FakeSTT):
        is_available = False

    stt = NoSTT()
    tts = FakeTTS()
    audio = FakeAudio()
    loop = hv.HeadsetVoiceLoop(lambda t: "answer", stt, tts, audio, FakePTT(1), earcon=False)
    loop.run(threading.Event())
    kinds = _events(loop)
    assert "stt_unavailable" in kinds and "transcript" not in kinds
    assert stt.inputs == []  # never asked to transcribe
    assert tts.texts == [hv.STT_UNAVAILABLE_TEXT]
    assert hv.NO_SPEECH_TEXT not in tts.texts
    assert len(audio.played) == 1 and loop.status()["stt_available"] is False
    # an STT without the attribute (sync fakes, older pipelines) is treated as available
    assert hv.HeadsetVoiceLoop(str, FakeSTT(), tts, audio, FakePTT(0)).status()["stt_available"]
