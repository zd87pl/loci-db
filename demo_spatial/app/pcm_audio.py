"""PCM helpers for thin clients that speak raw audio (ESP32 speaker and microphone).

The board plays and records **16 kHz, mono, 16-bit little-endian PCM** (``audio/L16``).
The host's TTS engines produce MP3 (edge-tts, OpenAI) or WAV (Piper), and the host's
STT wants float32 samples, so everything funnels through the two conversions here:

* :func:`decode_to_pcm16` — any container the host can decode → PCM16 at the board rate.
  WAV takes a stdlib-only fast path; everything else goes through PyAV (bundled with
  faster-whisper) and its resampler.
* :func:`pcm16_to_wav` / :func:`pcm16_to_float32` — board PCM → what STT wants.

No import of ``av`` at module import time, so the module is cheap to load in tests.
"""

from __future__ import annotations

import io
import wave

import numpy as np

DEFAULT_RATE = 16000
SAMPLE_WIDTH = 2  # bytes per PCM16 sample


def sniff_mime(audio: bytes) -> str:
    """Best-effort MIME type from the first bytes of an audio buffer."""
    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return "audio/wav"
    if audio.startswith(b"ID3") or (
        len(audio) > 1 and audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0
    ):
        return "audio/mpeg"
    if audio.startswith(b"OggS"):
        return "audio/ogg"
    if audio.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm"
    if audio.startswith(b"fLaC"):
        return "audio/flac"
    return "application/octet-stream"


def resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of mono PCM16; good enough for speech."""
    if src_rate == dst_rate or not pcm:
        return bytes(pcm)
    src = np.frombuffer(pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH], dtype="<i2").astype(np.float32)
    n_dst = int(round(src.size * dst_rate / src_rate))
    if n_dst <= 0:
        return b""
    x_src = np.arange(src.size, dtype=np.float64)
    x_dst = np.linspace(0.0, max(src.size - 1, 0), n_dst, dtype=np.float64)
    out = np.interp(x_dst, x_src, src)
    return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()


def wav_to_pcm16(wav_bytes: bytes, rate: int = DEFAULT_RATE) -> bytes:
    """Decode a PCM WAV (8/16/32-bit, any channel count) to mono PCM16 at ``rate``."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        src_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) * 256.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 65536.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width}")
    if channels > 1:
        usable = samples.size - samples.size % channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    mono = np.clip(np.round(samples), -32768, 32767).astype("<i2").tobytes()
    return resample_pcm16(mono, src_rate, rate)


def decode_to_pcm16(audio: bytes, rate: int = DEFAULT_RATE) -> bytes:
    """Decode MP3/WAV/OGG/WebM/... to mono PCM16 at ``rate``.

    Raises ``ValueError`` when the buffer is empty or cannot be decoded.
    """
    if not audio:
        raise ValueError("empty audio buffer")
    if sniff_mime(audio) == "audio/wav":
        try:
            return wav_to_pcm16(audio, rate)
        except (wave.Error, EOFError, ValueError):
            pass  # fall through to the generic decoder (e.g. float WAV)
    try:
        import av
    except ImportError as e:  # pragma: no cover - av ships with faster-whisper
        raise ValueError("PyAV is required to decode non-WAV audio") from e
    out = bytearray()
    try:
        with av.open(io.BytesIO(audio)) as container:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
            for frame in container.decode(stream):
                for res in resampler.resample(frame):
                    out += res.to_ndarray().tobytes()
            for res in resampler.resample(None):  # flush
                out += res.to_ndarray().tobytes()
    except Exception as e:
        raise ValueError(f"could not decode audio ({sniff_mime(audio)}): {e}") from e
    if not out:
        raise ValueError("decoder produced no samples")
    return bytes(out)


def pcm16_to_wav(pcm: bytes, rate: int = DEFAULT_RATE, channels: int = 1) -> bytes:
    """Wrap raw PCM16 in a WAV container (what STT backends accept)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm[: len(pcm) - len(pcm) % (SAMPLE_WIDTH * channels)])
    return buf.getvalue()


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Raw PCM16 → C-contiguous float32 in [-1, 1) (faster-whisper input)."""
    usable = len(pcm) - len(pcm) % SAMPLE_WIDTH
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float32) / 32768.0
    return np.ascontiguousarray(arr)


def pcm16_duration_s(pcm: bytes, rate: int = DEFAULT_RATE) -> float:
    return (len(pcm) // SAMPLE_WIDTH) / float(rate)


def pcm16_rms(pcm: bytes) -> float:
    """RMS level in [0, 1]; ~0.0 means silence (useful for push-to-talk sanity checks)."""
    arr = pcm16_to_float32(pcm)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def apply_gain_pcm16(pcm: bytes, gain: float) -> bytes:
    """Scale PCM16 by ``gain`` with clipping (small speakers often need a boost)."""
    if gain == 1.0 or not pcm:
        return bytes(pcm)
    arr = np.frombuffer(pcm[: len(pcm) - len(pcm) % SAMPLE_WIDTH], dtype="<i2").astype(np.float32)
    return np.clip(np.round(arr * gain), -32768, 32767).astype("<i2").tobytes()
