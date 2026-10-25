"""PCM conversions for the ESP32 thin client (board speaks 16 kHz mono PCM16)."""

from __future__ import annotations

import io
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app import pcm_audio  # noqa: E402


def _tone_pcm(rate: int, seconds: float, freq: float = 440.0, amp: float = 0.5) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype("<i2").tobytes()


def _wav(rate: int, seconds: float, channels: int = 1, width: int = 2) -> bytes:
    n = int(rate * seconds)
    t = np.arange(n) / rate
    mono = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sig = np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    if width == 1:
        raw = (sig * 127 + 128).astype(np.uint8).tobytes()
    elif width == 2:
        raw = (sig * 32767).astype("<i2").tobytes()
    else:
        raw = (sig * 2147483647).astype("<i4").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(raw)
    return buf.getvalue()


def test_sniff_mime() -> None:
    assert pcm_audio.sniff_mime(_wav(16000, 0.1)) == "audio/wav"
    assert pcm_audio.sniff_mime(b"ID3\x04" + b"\0" * 16) == "audio/mpeg"
    assert pcm_audio.sniff_mime(b"\xff\xfb\x90\x00" + b"\0" * 16) == "audio/mpeg"
    assert pcm_audio.sniff_mime(b"OggS" + b"\0" * 16) == "audio/ogg"
    assert pcm_audio.sniff_mime(b"\x1aE\xdf\xa3" + b"\0" * 16) == "audio/webm"
    assert pcm_audio.sniff_mime(b"\x00\x01\x02\x03") == "application/octet-stream"


def test_resample_changes_length_and_keeps_tone() -> None:
    src = _tone_pcm(48000, 0.5)
    out = pcm_audio.resample_pcm16(src, 48000, 16000)
    assert len(out) == len(src) // 3
    assert pcm_audio.resample_pcm16(src, 16000, 16000) == src
    # the 440 Hz tone keeps its amplitude after resampling
    assert 0.3 < pcm_audio.pcm16_rms(out) < 0.4


def test_wav_to_pcm16_downmixes_and_resamples() -> None:
    stereo_24k = _wav(24000, 0.5, channels=2)
    pcm = pcm_audio.wav_to_pcm16(stereo_24k, 16000)
    assert len(pcm) == 16000 // 2 * 2  # 0.5 s of mono PCM16 at 16 kHz
    assert 0.3 < pcm_audio.pcm16_rms(pcm) < 0.4
    assert len(pcm_audio.wav_to_pcm16(_wav(16000, 0.25, width=1))) == 4000 * 2
    assert len(pcm_audio.wav_to_pcm16(_wav(16000, 0.25, width=4))) == 4000 * 2


def test_decode_to_pcm16_wav_fast_path_and_errors() -> None:
    pcm = pcm_audio.decode_to_pcm16(_wav(16000, 0.25), 16000)
    assert len(pcm) == 4000 * 2
    with pytest.raises(ValueError):
        pcm_audio.decode_to_pcm16(b"", 16000)
    with pytest.raises(ValueError):
        pcm_audio.decode_to_pcm16(b"definitely not audio at all" * 4, 16000)


def test_decode_to_pcm16_mp3_via_pyav() -> None:
    av = pytest.importorskip("av")
    # Encode a 0.5 s tone as MP3 with PyAV, then decode it back to the board format.
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp3") as out:
        stream = out.add_stream("libmp3lame", rate=24000)
        stream.layout = "mono"
        samples = (np.sin(2 * np.pi * 440.0 * np.arange(12000) / 24000) * 0.5 * 32767).astype("<i2")
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 24000
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    mp3 = buf.getvalue()
    assert pcm_audio.sniff_mime(mp3) == "audio/mpeg"
    pcm = pcm_audio.decode_to_pcm16(mp3, 16000)
    seconds = pcm_audio.pcm16_duration_s(pcm, 16000)
    assert 0.45 < seconds < 0.65  # MP3 adds encoder padding
    assert 0.25 < pcm_audio.pcm16_rms(pcm) < 0.45


def test_pcm_wav_float_roundtrip() -> None:
    pcm = _tone_pcm(16000, 0.3)
    wav_bytes = pcm_audio.pcm16_to_wav(pcm, 16000)
    assert pcm_audio.sniff_mime(wav_bytes) == "audio/wav"
    assert pcm_audio.wav_to_pcm16(wav_bytes, 16000) == pcm
    f32 = pcm_audio.pcm16_to_float32(pcm)
    assert f32.dtype == np.float32 and f32.flags["C_CONTIGUOUS"]
    assert f32.size == 4800 and 0.49 < float(np.abs(f32).max()) <= 0.5
    assert pcm_audio.pcm16_to_float32(b"\x01").size == 0
    assert pcm_audio.pcm16_duration_s(pcm, 16000) == pytest.approx(0.3)


def test_gain_and_rms() -> None:
    pcm = _tone_pcm(16000, 0.2, amp=0.25)
    louder = pcm_audio.apply_gain_pcm16(pcm, 2.0)
    assert pcm_audio.pcm16_rms(louder) == pytest.approx(2 * pcm_audio.pcm16_rms(pcm), rel=0.01)
    clipped = pcm_audio.apply_gain_pcm16(pcm, 100.0)
    assert np.frombuffer(clipped, dtype="<i2").max() == 32767
    assert pcm_audio.pcm16_rms(b"") == 0.0
    assert pcm_audio.pcm16_rms(b"\x00" * 3200) == 0.0
