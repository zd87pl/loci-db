"""WhisperSTT local path (review fix): PCM WAV is decoded with the stdlib ``wave`` module
and handed to faster-whisper as float32 samples, so PyAV is never involved for the
headset's 16 kHz recordings; other containers keep the BytesIO path behind a one-time
``av.open`` keyword shim for PyAV >= 15.  Fakes only: no model, no audio hardware.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import types
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app import voice_pipeline as vp  # noqa: E402

WEBM = b"\x1aE\xdf\xa3" + b"\x00" * 64  # EBML magic, what the phone page uploads


def _wav(rate: int = 16000, seconds: float = 0.25, channels: int = 1, width: int = 2) -> bytes:
    n = int(rate * seconds)
    t = np.arange(n) / rate
    mono = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sig = np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    if width == 2:
        raw = (sig * 32767).astype("<i2").tobytes()
    elif width == 1:
        raw = (sig * 127 + 128).astype(np.uint8).tobytes()
    else:
        raw = (sig * 2147483647).astype("<i4").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(raw)
    return buf.getvalue()


def test_wav_bytes_to_float32_16k_mono() -> None:
    arr = vp.wav_bytes_to_float32(_wav())
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32 and arr.ndim == 1
    assert arr.size == 4000 and arr.flags["C_CONTIGUOUS"]
    assert 0.45 < float(np.abs(arr).max()) <= 0.5


@pytest.mark.parametrize("width", [1, 2, 4])
def test_wav_bytes_to_float32_resamples_and_downmixes(width: int) -> None:
    arr = vp.wav_bytes_to_float32(_wav(rate=44100, seconds=0.5, channels=2, width=width))
    assert arr is not None and arr.dtype == np.float32
    assert arr.size == pytest.approx(8000, abs=2)  # 0.5 s at 16 kHz
    assert 0.4 < float(np.abs(arr).max()) <= 0.52


def test_wav_bytes_to_float32_rejects_non_wav_and_garbage() -> None:
    assert vp.wav_bytes_to_float32(WEBM) is None
    assert vp.wav_bytes_to_float32(b"RIFF\x00\x00\x00\x00WAVE" + b"\xff" * 40) is None
    assert vp.wav_bytes_to_float32(b"") is None
    assert vp.wav_bytes_to_float32(_wav()[:30]) is None


class FakeModel:
    def __init__(self) -> None:
        self.inputs: list[object] = []
        self.kwargs: list[dict] = []

    def transcribe(self, audio, language="en", beam_size=1):
        self.inputs.append(audio)
        self.kwargs.append({"language": language, "beam_size": beam_size})
        seg = types.SimpleNamespace(text=" where is my mug ")
        return iter([seg]), None


def test_local_transcription_passes_samples_for_wav_and_bytesio_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHISPER_LOCAL", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    stt = vp.WhisperSTT()
    model = FakeModel()
    stt._local_model = model  # the lazy loader is bypassed: no faster-whisper needed
    assert asyncio.run(stt.transcribe(_wav())) == "where is my mug"
    assert isinstance(model.inputs[0], np.ndarray) and model.inputs[0].dtype == np.float32
    assert model.kwargs[0] == {"language": "en", "beam_size": 1}
    monkeypatch.setattr(vp, "ensure_pyav_compat", lambda: False)
    assert asyncio.run(stt.transcribe(WEBM, language="de")) == "where is my mug"
    assert isinstance(model.inputs[1], io.BytesIO) and model.kwargs[1]["language"] == "de"


def test_pyav_compat_shim_drops_only_the_metadata_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple, dict]] = []

    def strict_open(*args, **kwargs):
        if "metadata_errors" in kwargs or "metadata_encoding" in kwargs:
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        if "boom" in kwargs:
            raise TypeError("boom")
        calls.append((args, kwargs))
        return "container"

    fake_av = types.ModuleType("av")
    fake_av.open = strict_open  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(vp, "_PYAV_PATCHED", False)
    assert vp.ensure_pyav_compat() is True
    src = io.BytesIO(b"x")
    assert fake_av.open(src, mode="r", metadata_errors="ignore", metadata_encoding="utf-8") == (
        "container"
    )
    assert calls == [((src,), {"mode": "r"})]
    with pytest.raises(TypeError, match="boom"):
        fake_av.open(src, boom=1)
    first = fake_av.open
    assert vp.ensure_pyav_compat() is True and fake_av.open is first  # idempotent
    monkeypatch.setattr(vp, "_PYAV_PATCHED", False)
    assert vp.ensure_pyav_compat() is True and fake_av.open is first  # no double wrap


def test_pyav_compat_without_pyav(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "av", None)
    monkeypatch.setattr(vp, "_PYAV_PATCHED", False)
    assert vp.ensure_pyav_compat() is False
