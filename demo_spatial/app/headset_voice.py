"""Headset voice loop for the handheld demo (task T11).

A blind user presses the headset button, asks a question, and hears the answer
through the same headset. No phone is involved: the loop runs on the device next
to the FastAPI server and calls the same answer function the typed ``/api/query``
path uses.

Pieces (all injectable, all testable with fakes):

* :class:`AudioIO` protocol with :class:`SoundDeviceAudio` (PortAudio through the
  ``sounddevice`` package, imported lazily) and :class:`CliAudio` (``pw-record`` /
  ``arecord`` / ``pw-play`` / ``aplay`` on Linux, ``afplay`` on macOS; never a shell).
* :class:`PushToTalk` protocol with :class:`KeyboardPTT` (Enter / Enter),
  :class:`VadPTT` (hands-free energy gate) and :class:`EvdevButtonPTT` (the headset's
  media button read straight from ``/dev/input/eventN`` without the ``evdev`` package).
* :class:`HeadsetVoiceLoop`: press -> earcon -> record -> STT -> answer -> TTS -> play,
  one event per stage, exceptions logged and the loop keeps going.
* :func:`make_voice_loop_from_env`: the ``LOCI_VOICE*`` / ``LOCI_PTT*`` /
  ``LOCI_AUDIO*`` environment contract used by ``main.py`` (T7).

Audio convention: every WAV handed to ``stt.transcribe`` is 16 kHz mono 16-bit PCM.
``tts.synthesize`` may return WAV at any rate (Piper: 22.05 kHz) or MP3 (OpenAI);
:meth:`AudioIO.play` copes with both.

Nothing optional is imported at module import time: ``sounddevice`` is imported inside
:class:`SoundDeviceAudio`, and the module works with numpy alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import logging
import os
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
"""Sample rate of every recording handed to STT (Hz)."""

EARCON_HZ = 880.0
EARCON_S = 0.08

NO_SPEECH_TEXT = "Sorry, I didn't catch that. Press the button and try again."
STT_UNAVAILABLE_TEXT = (
    "Speech recognition is not set up on this device, so I cannot hear your question yet."
)


class PTTConfigError(RuntimeError):
    """A push-to-talk source that can never work in this process (no TTY, bad device
    path): the loop reports it once and stops instead of retrying twice a second."""


ANSWER_FAILED_TEXT = "Sorry, something went wrong while answering. Please try again."


# ---------------------------------------------------------------------------
# WAV / PCM helpers (numpy only)
# ---------------------------------------------------------------------------


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average a ``(n, channels)`` array down to ``(n,)``; 1-D input is returned as is."""
    arr = np.asarray(samples)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    mean = arr.astype(np.float64).mean(axis=1)
    if arr.dtype.kind in "iu":
        return np.rint(mean).astype(np.int16)
    return mean


def pcm16_to_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode int16 mono samples (or int16-scaled floats) as a WAV byte string."""
    pcm = to_mono(np.asarray(samples))
    if pcm.dtype != np.int16:
        pcm = np.clip(np.rint(pcm), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.astype("<i2", copy=False).tobytes())
    return buf.getvalue()


def wav_to_pcm16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode a WAV byte string to ``(int16 mono samples, sample_rate)``.

    8/16/24/32-bit PCM and any channel count are accepted; channels are averaged.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    elif width == 1:
        data = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
    elif width == 4:
        data = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int16)
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        full = (b[:, 2] << 24) | (b[:, 1] << 16) | (b[:, 0] << 8)
        data = (full >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if channels > 1:
        usable = (len(data) // channels) * channels
        data = to_mono(data[:usable].reshape(-1, channels))
    return data.astype(np.int16, copy=False), int(rate)


def wav_duration_s(wav_bytes: bytes) -> float:
    """Length of a WAV byte string in seconds (0.0 when it does not parse)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or 1)
    except (wave.Error, EOFError):
        return 0.0


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler; adequate for speech and for VAD energy."""
    arr = np.asarray(samples)
    if src_rate == dst_rate or arr.size == 0:
        return arr
    n_out = int(round(arr.shape[0] * dst_rate / src_rate))
    x_old = np.arange(arr.shape[0], dtype=np.float64) * (dst_rate / src_rate)
    x_new = np.arange(n_out, dtype=np.float64)
    out = np.interp(x_new, x_old, arr.astype(np.float64))
    if arr.dtype.kind in "iu":
        return np.clip(np.rint(out), -32768, 32767).astype(np.int16)
    return out.astype(arr.dtype, copy=False)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level of a chunk (int16 scale when the input is int16)."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def earcon_wav(
    freq_hz: float = EARCON_HZ,
    duration_s: float = EARCON_S,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.3,
) -> bytes:
    """Short confirmation tone (default 880 Hz, 80 ms) with 5 ms fades, as WAV bytes."""
    n = max(1, int(round(duration_s * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    tone = np.sin(2.0 * np.pi * freq_hz * t)
    fade = min(n // 4, int(round(0.005 * sample_rate)))
    env = np.ones(n, dtype=np.float64)
    if fade > 0:
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
    pcm = np.rint(tone * env * amplitude * 32767.0).astype(np.int16)
    return pcm16_to_wav(pcm, sample_rate)


def silence_wav(seconds: float = 0.1, sample_rate: int = SAMPLE_RATE) -> bytes:
    """A WAV of digital silence (used by loopback tooling)."""
    return pcm16_to_wav(np.zeros(int(seconds * sample_rate), dtype=np.int16), sample_rate)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioIO(Protocol):
    """Microphone + speaker. ``record`` returns 16 kHz mono 16-bit PCM WAV bytes."""

    def record(self, max_seconds: float, stop: threading.Event | None = None) -> bytes: ...

    def play(self, wav_bytes: bytes) -> None: ...

    def list_devices(self) -> list[dict]: ...


@runtime_checkable
class PushToTalk(Protocol):
    """Press/release source. ``wait_for_press`` returns False when ``stop`` is set."""

    def wait_for_press(self, stop: threading.Event) -> bool: ...

    def wait_for_release(self, timeout_s: float) -> None: ...


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device(
    devices: Sequence[Mapping[str, Any]], spec: str | int | None, kind: str
) -> int | None:
    """Pick a device index from a ``list_devices()`` result.

    ``spec`` is ``None``/empty/``"default"`` for the system default (returns ``None``),
    an integer or digit string for an explicit index, or a case-insensitive substring
    of the device name. ``kind`` is ``"input"`` or ``"output"`` and restricts the
    search to devices with channels of that kind. Raises :class:`LookupError` when a
    substring matches nothing.
    """
    if spec is None:
        return None
    if isinstance(spec, bool):
        raise TypeError("device spec must be a string or an integer")
    if isinstance(spec, int):
        return spec
    text = spec.strip()
    if not text or text.lower() in {"default", "system"}:
        return None
    if text.isdigit():
        return int(text)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    needle = text.lower()
    for i, dev in enumerate(devices):
        if int(dev.get(key, 0) or 0) <= 0:
            continue
        if needle in str(dev.get("name", "")).lower():
            return int(dev.get("index", i))
    names = [str(d.get("name", "")) for d in devices if int(d.get(key, 0) or 0) > 0]
    raise LookupError(f"no {kind} device matching {spec!r}; available: {names}")


# ---------------------------------------------------------------------------
# AudioIO implementations
# ---------------------------------------------------------------------------


class SoundDeviceAudio:
    """PortAudio backend via ``sounddevice`` (imported lazily; ``libportaudio2`` needed).

    Devices are chosen by case-insensitive name substring (``LOCI_AUDIO_IN`` /
    ``LOCI_AUDIO_OUT``) and re-resolved on every call, so a headset that connects after
    start-up is picked up. When a substring matches nothing the system default is used
    and an error is logged once per direction. Recordings are captured at 16 kHz when
    the device allows it and resampled otherwise.
    """

    name = "sounddevice"

    def __init__(
        self,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        *,
        sample_rate: int = SAMPLE_RATE,
        chunk_s: float = 0.05,
        sd_module: Any | None = None,
    ) -> None:
        self._input_spec = input_device
        self._output_spec = output_device
        self._rate = int(sample_rate)
        self._chunk_s = float(chunk_s)
        self._sd = sd_module
        self._warned: set[str] = set()

    # -- plumbing -----------------------------------------------------------

    def _lib(self) -> Any:
        if self._sd is None:
            import sounddevice as sd

            self._sd = sd
        return self._sd

    def list_devices(self) -> list[dict]:
        sd = self._lib()
        out: list[dict] = []
        for i, dev in enumerate(sd.query_devices()):
            out.append(
                {
                    "index": int(dev.get("index", i)),
                    "name": str(dev.get("name", "")),
                    "max_input_channels": int(dev.get("max_input_channels", 0) or 0),
                    "max_output_channels": int(dev.get("max_output_channels", 0) or 0),
                    "default_samplerate": float(dev.get("default_samplerate", 0) or 0),
                    "hostapi": int(dev.get("hostapi") if dev.get("hostapi") is not None else -1),
                }
            )
        return out

    def _device(self, kind: str) -> int | None:
        spec = self._input_spec if kind == "input" else self._output_spec
        if spec is None:
            return None
        try:
            return select_device(self.list_devices(), spec, kind)
        except LookupError as e:
            if kind not in self._warned:
                logger.error("%s; using the system default %s device", e, kind)
                self._warned.add(kind)
            return None

    def _default_rate(self, sd: Any, device: int | None, kind: str) -> int:
        try:
            info = (
                sd.query_devices(device, kind)
                if device is not None
                else sd.query_devices(kind=kind)
            )
            rate = int(float(info["default_samplerate"]))
        except Exception:  # pragma: no cover - depends on the host's PortAudio
            rate = 48_000
        return rate or 48_000

    def _input_rate(self, sd: Any, device: int | None) -> int:
        try:
            sd.check_input_settings(device=device, samplerate=self._rate, channels=1, dtype="int16")
            return self._rate
        except Exception as e:
            rate = self._default_rate(sd, device, "input")
            logger.debug("16 kHz capture unsupported (%s); capturing at %d Hz", e, rate)
            return rate

    # -- capture --------------------------------------------------------------

    def chunks(self, chunk_s: float, stop: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Yield 16 kHz int16 mono chunks from one persistent input stream until ``stop``."""
        sd = self._lib()
        device = self._device("input")
        rate = self._input_rate(sd, device)
        block = max(1, int(round(rate * chunk_s)))
        with sd.InputStream(
            samplerate=rate, channels=1, dtype="int16", device=device, blocksize=block
        ) as stream:
            while not (stop is not None and stop.is_set()):
                data, _overflowed = stream.read(block)
                mono = to_mono(np.asarray(data)).astype(np.int16, copy=False)
                if rate != self._rate:
                    mono = resample_linear(mono, rate, self._rate)
                yield mono

    def record(self, max_seconds: float, stop: threading.Event | None = None) -> bytes:
        max_samples = max(0, int(max_seconds * self._rate))
        got: list[np.ndarray] = []
        total = 0
        gen = self.chunks(self._chunk_s, stop)
        try:
            for chunk in gen:
                got.append(chunk)
                total += len(chunk)
                if total >= max_samples:
                    break
        finally:
            gen.close()
        samples = np.concatenate(got)[:max_samples] if got else np.zeros(0, dtype=np.int16)
        return pcm16_to_wav(samples, self._rate)

    # -- playback -------------------------------------------------------------

    def play(self, wav_bytes: bytes) -> None:
        if not wav_bytes:
            return
        if wav_bytes[:4] != b"RIFF":
            # Compressed audio (e.g. MP3 from a cloud TTS): hand it to a system player.
            out_spec = self._output_spec if isinstance(self._output_spec, str) else None
            CliAudio(output_device=out_spec).play(wav_bytes)
            return
        sd = self._lib()
        samples, rate = wav_to_pcm16(wav_bytes)
        device = self._device("output")
        try:
            sd.play(samples, rate, device=device, blocking=True)
        except Exception as e:
            target = self._default_rate(sd, device, "output")
            if target == rate:
                raise
            logger.debug("playback at %d Hz failed (%s); resampling to %d Hz", rate, e, target)
            sd.play(resample_linear(samples, rate, target), target, device=device, blocking=True)


class CliAudio:
    """Fallback backend driving command-line tools through ``subprocess`` (no shell).

    Linux: ``pw-record``/``pw-play`` when PipeWire tools exist, else
    ``arecord -f S16_LE -r 16000 -c 1`` / ``aplay``. macOS: playback with ``afplay`` on a
    temporary file; recording is not supported there (install ``sounddevice``).
    Device strings are passed through (``--target`` for PipeWire, ``-D`` for ALSA).
    """

    name = "cli"

    def __init__(
        self,
        input_device: str | None = None,
        output_device: str | None = None,
        *,
        system: str | None = None,
        runner: Callable[..., Any] = subprocess.run,
        popen: Callable[..., Any] = subprocess.Popen,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._in = input_device or None
        self._out = output_device or None
        self._system = system or sys.platform
        self._run = runner
        self._popen = popen
        self._which = which

    @property
    def is_linux(self) -> bool:
        return self._system.startswith("linux")

    @property
    def is_mac(self) -> bool:
        return self._system == "darwin"

    # -- command construction (pure, unit-tested) -----------------------------

    def record_command(self, out_path: str, max_seconds: float) -> list[str]:
        if not self.is_linux:
            raise RuntimeError(
                "CliAudio cannot record on this platform; install sounddevice "
                "(pip install sounddevice) or use voice_console.py --loopback-wav"
            )
        if self._which("pw-record"):
            cmd = ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16"]
            if self._in:
                cmd += ["--target", self._in]
            return [*cmd, out_path]
        if self._which("arecord"):
            seconds = max(1, int(np.ceil(max_seconds)))
            cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(seconds)]
            if self._in:
                cmd += ["-D", self._in]
            return [*cmd, out_path]
        raise RuntimeError(
            "neither pw-record nor arecord found (apt install pipewire-bin alsa-utils)"
        )

    def stream_command(self) -> list[str]:
        """Command that writes raw s16le 16 kHz mono PCM to stdout until killed."""
        if not self.is_linux:
            raise RuntimeError("CliAudio streaming capture is Linux-only; install sounddevice")
        if self._which("arecord"):
            cmd = ["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", "16000", "-c", "1"]
            if self._in:
                cmd += ["-D", self._in]
            return [*cmd, "-"]
        if self._which("pw-record"):
            cmd = ["pw-record", "--raw", "--rate", "16000", "--channels", "1", "--format", "s16"]
            if self._in:
                cmd += ["--target", self._in]
            return [*cmd, "-"]
        raise RuntimeError(
            "neither arecord nor pw-record found (apt install alsa-utils pipewire-bin)"
        )

    def play_command(self, path: str) -> list[str]:
        compressed = not path.endswith(".wav")
        if self.is_mac:
            if self._which("afplay"):
                return ["afplay", path]
            raise RuntimeError("afplay not found")
        if self.is_linux:
            if compressed:
                if self._which("mpg123"):
                    return ["mpg123", "-q", path]
                if self._which("ffplay"):
                    return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path]
            if self._which("pw-play"):
                cmd = ["pw-play"]
                if self._out:
                    cmd += ["--target", self._out]
                return [*cmd, path]
            if self._which("aplay"):
                cmd = ["aplay", "-q"]
                if self._out:
                    cmd += ["-D", self._out]
                return [*cmd, path]
            raise RuntimeError(
                "neither pw-play nor aplay found (apt install pipewire-bin alsa-utils)"
            )
        raise RuntimeError(f"CliAudio has no player for platform {self._system!r}")

    # -- AudioIO --------------------------------------------------------------

    def list_devices(self) -> list[dict]:
        out: list[dict] = []
        if not self.is_linux:
            return out
        if self._which("pactl"):
            for kind, what in (("input", "sources"), ("output", "sinks")):
                try:
                    res = self._run(
                        ["pactl", "list", "short", what],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=5,
                    )
                except (OSError, subprocess.SubprocessError) as e:
                    logger.debug("pactl failed: %s", e)
                    continue
                for line in str(res.stdout or "").splitlines():
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    out.append(
                        {
                            "index": int(parts[0]) if parts[0].isdigit() else -1,
                            "name": parts[1],
                            "max_input_channels": 1 if kind == "input" else 0,
                            "max_output_channels": 1 if kind == "output" else 0,
                            "backend": "pipewire",
                        }
                    )
        for tool, kind in (("arecord", "input"), ("aplay", "output")):
            if not self._which(tool):
                continue
            try:
                res = self._run(
                    [tool, "-l"], capture_output=True, text=True, check=False, timeout=5
                )
            except (OSError, subprocess.SubprocessError) as e:
                logger.debug("%s -l failed: %s", tool, e)
                continue
            pattern = r"^card (\d+): (\S+) \[(.*?)\], device (\d+): (.*?) \["
            for m in re.finditer(pattern, str(res.stdout or ""), re.M):
                out.append(
                    {
                        "index": -1,
                        "name": f"hw:{m.group(1)},{m.group(4)} {m.group(3)} {m.group(5)}".strip(),
                        "max_input_channels": 1 if kind == "input" else 0,
                        "max_output_channels": 1 if kind == "output" else 0,
                        "backend": "alsa",
                    }
                )
        return out

    def record(self, max_seconds: float, stop: threading.Event | None = None) -> bytes:
        with tempfile.TemporaryDirectory(prefix="loci-voice-") as tmp:
            out_path = os.path.join(tmp, "rec.wav")
            cmd = self.record_command(out_path, max_seconds)
            proc = self._popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + max_seconds
            while proc.poll() is None and time.monotonic() < deadline:
                if stop is not None and stop.is_set():
                    break
                time.sleep(0.02)
            if proc.poll() is None:
                # SIGINT makes arecord/pw-record finish the WAV header cleanly.
                with contextlib.suppress(OSError):
                    proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            path = Path(out_path)
            if not path.exists():
                return silence_wav(0.0)
            data = path.read_bytes()
        try:
            samples, rate = wav_to_pcm16(data)
        except (wave.Error, EOFError, ValueError):
            return data
        if rate != SAMPLE_RATE:
            samples = resample_linear(samples, rate, SAMPLE_RATE)
        return pcm16_to_wav(samples, SAMPLE_RATE)

    def chunks(self, chunk_s: float, stop: threading.Event | None = None) -> Iterator[np.ndarray]:
        """Stream 16 kHz int16 chunks from a long-running ``arecord``/``pw-record`` process."""
        cmd = self.stream_command()
        block_bytes = max(2, int(round(SAMPLE_RATE * chunk_s)) * 2)
        proc = self._popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        try:
            pending = b""
            while not (stop is not None and stop.is_set()):
                data = proc.stdout.read(block_bytes - len(pending))
                if not data:
                    break
                pending += data
                if len(pending) < block_bytes:
                    continue
                yield np.frombuffer(pending, dtype="<i2").astype(np.int16)
                pending = b""
        finally:
            if proc.poll() is None:
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2.0)

    def play(self, wav_bytes: bytes) -> None:
        if not wav_bytes:
            return
        suffix = ".wav" if wav_bytes[:4] == b"RIFF" else ".mp3"
        with tempfile.TemporaryDirectory(prefix="loci-voice-") as tmp:
            path = os.path.join(tmp, f"out{suffix}")
            Path(path).write_bytes(wav_bytes)
            cmd = self.play_command(path)
            self._run(
                cmd,
                check=False,
                timeout=120,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


# ---------------------------------------------------------------------------
# PushToTalk implementations
# ---------------------------------------------------------------------------


class KeyboardPTT:
    """Press Enter to start talking, Enter again to stop. Needs an interactive TTY."""

    name = "keyboard"

    def __init__(
        self,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        require_tty: bool = True,
        poll_s: float = 0.2,
    ) -> None:
        self._in = stdin if stdin is not None else sys.stdin
        self._out = stdout if stdout is not None else sys.stdout
        self._require_tty = require_tty
        self._poll_s = poll_s

    def _fileno(self) -> int | None:
        try:
            return self._in.fileno()
        except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
            return None

    def _isatty(self) -> bool:
        fd = self._fileno()
        return fd is not None and os.isatty(fd)

    def _say(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self._out.write(text + "\n")
            self._out.flush()

    def _readline(self, timeout_s: float | None, stop: threading.Event | None) -> str | None:
        """Read one line; ``None`` on EOF, timeout or ``stop``."""
        fd = self._fileno()
        if fd is None:
            line = self._in.readline()
            return line if line else None
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if stop is not None and stop.is_set():
                return None
            if deadline is not None and time.monotonic() >= deadline:
                return None
            ready, _, _ = select.select([fd], [], [], self._poll_s)
            if ready:
                line = self._in.readline()
                return line if line else None

    def wait_for_press(self, stop: threading.Event) -> bool:
        if self._require_tty and not self._isatty():
            raise PTTConfigError(
                "KeyboardPTT needs an interactive terminal; set LOCI_PTT=vad or LOCI_PTT=evdev"
            )
        self._say("Press Enter to talk (Ctrl-C quits)...")
        return self._readline(None, stop) is not None

    def wait_for_release(self, timeout_s: float) -> None:
        self._say("Recording - press Enter to stop...")
        self._readline(timeout_s, None)


class VadPTT:
    """Hands-free trigger: speech starts when a chunk's RMS exceeds ``threshold`` and ends
    after ``silence_s`` of quiet. Pure numpy over short chunks pulled through the
    :class:`AudioIO` (``audio.chunks(chunk_s, stop)`` when the backend offers a
    persistent stream, else repeated ``audio.record(chunk_s)``).

    Because the detector already holds the microphone, it also keeps the audio: the
    loop sees ``captures_audio = True`` and calls :meth:`take_recording` instead of
    recording a second time. A ``pre_roll_s`` ring buffer keeps the word onset.
    """

    name = "vad"
    captures_audio = True

    def __init__(
        self,
        audio: AudioIO,
        *,
        threshold: float = 500.0,
        silence_s: float = 1.2,
        chunk_s: float = 0.1,
        pre_roll_s: float = 0.3,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._audio = audio
        self.threshold = float(threshold)
        self.silence_s = float(silence_s)
        self._chunk_s = float(chunk_s)
        self._pre_roll = max(0, int(round(pre_roll_s / chunk_s)))  # chunks kept before onset
        self._rate = int(sample_rate)
        self._stream: Iterator[np.ndarray] | None = None
        self._buffer: list[np.ndarray] = []

    def _chunk_iter(self, stop: threading.Event | None) -> Iterator[np.ndarray]:
        chunks = getattr(self._audio, "chunks", None)
        if callable(chunks):
            yield from chunks(self._chunk_s, stop)
            return
        while not (stop is not None and stop.is_set()):
            wav = self._audio.record(self._chunk_s, stop)
            if not wav:
                return
            pcm, rate = wav_to_pcm16(wav)
            if rate != self._rate:
                pcm = resample_linear(pcm, rate, self._rate)
            if len(pcm) == 0:
                return
            yield pcm

    def _next(self) -> np.ndarray | None:
        if self._stream is None:
            return None
        try:
            return next(self._stream)
        except StopIteration:
            return None

    def _close_stream(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()  # type: ignore[attr-defined]
            self._stream = None

    def wait_for_press(self, stop: threading.Event) -> bool:
        self._close_stream()
        self._buffer = []
        self._stream = self._chunk_iter(stop)
        ring: deque[np.ndarray] = deque(maxlen=self._pre_roll + 1)  # pre-roll + trigger chunk
        while not stop.is_set():
            chunk = self._next()
            if chunk is None:
                self._close_stream()
                return False
            ring.append(chunk)
            if rms(chunk) >= self.threshold:
                self._buffer = list(ring)
                return True
        self._close_stream()
        return False

    def wait_for_release(self, timeout_s: float) -> None:
        max_samples = int(timeout_s * self._rate)
        total = 0
        quiet_s = 0.0
        while total < max_samples:
            chunk = self._next()
            if chunk is None:
                break
            self._buffer.append(chunk)
            total += len(chunk)
            if rms(chunk) >= self.threshold:
                quiet_s = 0.0
            else:
                quiet_s += len(chunk) / self._rate
                if quiet_s >= self.silence_s:
                    break
        self._close_stream()

    def take_recording(self) -> bytes:
        samples = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.int16)
        self._buffer = []
        return pcm16_to_wav(samples, self._rate)


# -- Linux input events -------------------------------------------------------

EV_SYN = 0
EV_KEY = 1
KEY_NEXTSONG = 163
KEY_PLAYPAUSE = 164
KEY_PLAYCD = 200
KEY_PAUSECD = 201
DEFAULT_BUTTON_CODES: frozenset[int] = frozenset(
    {KEY_PLAYPAUSE, KEY_PLAYCD, KEY_PAUSECD, KEY_NEXTSONG}
)
INPUT_EVENT_FORMAT = "llHHi"
INPUT_EVENT_SIZE = struct.calcsize(INPUT_EVENT_FORMAT)  # 24 bytes on 64-bit Linux
PROC_INPUT_DEVICES = "/proc/bus/input/devices"
DEFAULT_PTT_DEVICE = "AVRCP"  # bluez names the headset's button device "<Headset> (AVRCP)"


@dataclass(frozen=True)
class InputEvent:
    sec: int
    usec: int
    type: int
    code: int
    value: int


def pack_input_event(type_: int, code: int, value: int, sec: int = 0, usec: int = 0) -> bytes:
    """Build one ``struct input_event`` (used by tests and loopback tooling)."""
    return struct.pack(INPUT_EVENT_FORMAT, sec, usec, type_, code, value)


def parse_input_events(data: bytes) -> tuple[list[InputEvent], bytes]:
    """Split a byte stream into complete input events; the unparsed tail is returned."""
    n = len(data) // INPUT_EVENT_SIZE
    events = [
        InputEvent(*struct.unpack_from(INPUT_EVENT_FORMAT, data, i * INPUT_EVENT_SIZE))
        for i in range(n)
    ]
    return events, data[n * INPUT_EVENT_SIZE :]


def list_input_devices(proc_text: str | None = None) -> list[dict]:
    """Parse ``/proc/bus/input/devices`` into ``[{"name", "event", "handlers"}]``."""
    if proc_text is None:
        try:
            proc_text = Path(PROC_INPUT_DEVICES).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
    out: list[dict] = []
    for block in re.split(r"\n\s*\n", proc_text):
        m_name = re.search(r'^N: Name="(.*)"\s*$', block, re.M)
        if not m_name:
            continue
        m_h = re.search(r"^H: Handlers=(.*)$", block, re.M)
        handlers = m_h.group(1).strip() if m_h else ""
        m_ev = re.search(r"\bevent(\d+)\b", handlers)
        out.append(
            {
                "name": m_name.group(1),
                "event": f"/dev/input/event{m_ev.group(1)}" if m_ev else None,
                "handlers": handlers,
            }
        )
    return out


def find_input_device(name_substring: str, proc_text: str | None = None) -> str | None:
    """Return ``/dev/input/eventN`` of the first device whose name contains the substring."""
    needle = name_substring.strip().lower()
    for dev in list_input_devices(proc_text):
        if needle in str(dev["name"]).lower() and dev["event"]:
            return str(dev["event"])
    return None


class EvdevButtonPTT:
    """The headset's media button as push-to-talk, read from ``/dev/input/eventN``.

    ``KEY_PLAYPAUSE``/``KEY_PLAYCD``/``KEY_PAUSECD``/``KEY_NEXTSONG`` count as the
    button; value 1 is press, 0 release (2 = autorepeat, ignored). In ``toggle`` mode
    (headsets that only emit click pairs) a click starts and the next click stops.
    The device is (re)located by name in ``/proc/bus/input/devices`` whenever it is
    missing, so pairing after start-up and reconnects work. The process must be able
    to read the event node (member of the ``input`` group).
    """

    name = "evdev"

    def __init__(
        self,
        device: str | None = None,
        *,
        name_substring: str | None = None,
        toggle: bool = False,
        codes: Iterable[int] = DEFAULT_BUTTON_CODES,
        reader: Callable[[float], bytes | None] | None = None,
        reconnect_s: float = 1.0,
        poll_s: float = 0.25,
    ) -> None:
        self._path = device
        self._name = name_substring if name_substring is not None else DEFAULT_PTT_DEVICE
        self.toggle = bool(toggle)
        self._codes = frozenset(int(c) for c in codes)
        self._reader = reader
        self._reconnect_s = float(reconnect_s)
        self._poll_s = float(poll_s)
        self._fd: int | None = None
        self._pending = b""
        self._warned: set[str] = set()

    @classmethod
    def from_spec(cls, spec: str | None, **kwargs: Any) -> EvdevButtonPTT:
        """``spec`` is ``/dev/input/eventN`` or a name substring (empty -> ``AVRCP``)."""
        text = (spec or "").strip()
        if text.startswith("/dev/"):
            return cls(device=text, **kwargs)
        return cls(name_substring=text or DEFAULT_PTT_DEVICE, **kwargs)

    @property
    def device_path(self) -> str | None:
        return self._path

    # -- low level ------------------------------------------------------------

    def _warn_once(self, key: str, msg: str, *args: Any) -> None:
        if key not in self._warned:
            logger.warning(msg, *args)
            self._warned.add(key)

    def _close(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None
        self._pending = b""

    def _open(self) -> bool:
        path = self._path or find_input_device(self._name)
        if not path:
            self._warn_once(
                "missing",
                "no input device matching %r in %s (is the headset connected?)",
                self._name,
                PROC_INPUT_DEVICES,
            )
            return False
        try:
            self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            self._warn_once("perm", "cannot read %s: add the user to the 'input' group", path)
            return False
        except OSError as e:
            self._warn_once("open", "cannot open %s: %s", path, e)
            return False
        logger.info("push-to-talk button: %s", path)
        self._warned.clear()
        return True

    def _read(self, timeout_s: float) -> bytes | None:
        """Raw bytes (possibly empty on timeout) or ``None`` when the device is gone."""
        if self._reader is not None:
            return self._reader(timeout_s)
        if self._fd is None:
            return None
        ready, _, _ = select.select([self._fd], [], [], max(0.0, timeout_s))
        if not ready:
            return b""
        try:
            data = os.read(self._fd, INPUT_EVENT_SIZE * 64)
        except BlockingIOError:
            return b""
        except OSError as e:
            logger.info("button device vanished: %s", e)
            self._close()
            return None
        if not data:
            self._close()
            return None
        return data

    def _poll(self, timeout_s: float) -> list[InputEvent] | None:
        data = self._read(timeout_s)
        if data is None:
            return None
        events, self._pending = parse_input_events(self._pending + data)
        return [e for e in events if e.type == EV_KEY and e.code in self._codes]

    def _ensure_open(self, stop: threading.Event) -> bool:
        if self._reader is not None or self._fd is not None:
            return True
        while not stop.is_set():
            if self._open():
                return True
            stop.wait(self._reconnect_s)
        return False

    # -- PushToTalk -----------------------------------------------------------

    def wait_for_press(self, stop: threading.Event) -> bool:
        while not stop.is_set():
            if not self._ensure_open(stop):
                return False
            events = self._poll(self._poll_s)
            if events is None:
                if self._reader is not None:
                    return False
                stop.wait(self._reconnect_s)
                continue
            if any(e.value == 1 for e in events):
                return True
        return False

    def wait_for_release(self, timeout_s: float) -> None:
        want = 1 if self.toggle else 0
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            events = self._poll(min(self._poll_s, remaining))
            if events is None:
                return
            if any(e.value == want for e in events):
                return

    def close(self) -> None:
        self._close()


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def _await(aw: Any) -> Any:
    return await aw


def run_maybe_async(value: Any) -> Any:
    """Return ``value``; when it is awaitable, run it to completion on this thread.

    Works both from a plain thread (``asyncio.run``) and from inside a running event
    loop (a helper thread runs it so the caller's loop is never re-entered).
    """
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(value))
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["result"] = asyncio.run(_await(value))
        except BaseException as e:  # re-raised on the caller's thread
            box["error"] = e

    t = threading.Thread(target=_runner, name="voice-await", daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


class HeadsetVoiceLoop:
    """press -> earcon -> record -> STT -> answer -> TTS -> play, forever.

    ``stt`` is a ``voice_pipeline.WhisperSTT`` (``async transcribe(wav) -> str``),
    ``tts`` a ``voice_pipeline.TTS`` (``async synthesize(text) -> bytes``); sync fakes
    work too. ``answer_fn(text) -> str`` is the server's text path
    (``lambda t: runtime.answer_text(t)["answer"]``).

    Every stage emits ``{"type", "text", "latency_ms", "ts_ms", ...}`` through
    ``on_event`` (called on the loop thread) and into :attr:`events`; ``status()`` is
    what ``/api/handheld/status`` shows under ``"voice"``. Exceptions are logged, an
    ``error`` event is emitted and the loop continues with the next press.
    """

    def __init__(
        self,
        answer_fn: Callable[[str], str],
        stt: Any,
        tts: Any,
        audio: AudioIO,
        ptt: PushToTalk,
        *,
        max_seconds: float = 10.0,
        earcon: bool = True,
        on_event: Callable[[dict], None] | None = None,
        history: int = 50,
        no_speech_text: str = NO_SPEECH_TEXT,
    ) -> None:
        self._answer_fn = answer_fn
        self._stt = stt
        self._tts = tts
        self._audio = audio
        self._ptt = ptt
        self.max_seconds = float(max_seconds)
        self.earcon = bool(earcon)
        self._on_event = on_event
        self._no_speech_text = no_speech_text
        self._earcon_wav = earcon_wav() if earcon else b""
        self.events: deque[dict] = deque(maxlen=history)
        self.turns = 0
        self.errors = 0
        self.running = False
        self.fatal: str | None = None
        self.idle_backoff_s = 0.5
        self._lock = threading.Lock()
        if getattr(stt, "is_available", True) is False:
            logger.warning(
                "headset voice: no STT backend (install faster-whisper with WHISPER_LOCAL=1 "
                "or set OPENAI_API_KEY); presses will be answered with a spoken notice"
            )

    # -- events ---------------------------------------------------------------

    def _emit(
        self, type_: str, text: str = "", latency_ms: float | None = None, **extra: Any
    ) -> dict:
        ev: dict[str, Any] = {
            "type": type_,
            "text": text,
            "latency_ms": None if latency_ms is None else round(float(latency_ms), 1),
            "ts_ms": int(time.time() * 1000),
        }
        ev.update(extra)
        with self._lock:
            self.events.append(ev)
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:
                logger.exception("voice on_event callback failed")
        return ev

    def last_events(self, n: int = 20) -> list[dict]:
        with self._lock:
            items = list(self.events)
        return items[-n:]

    def status(self) -> dict:
        return {
            "running": self.running,
            "fatal": self.fatal,
            "turns": self.turns,
            "errors": self.errors,
            "stt_available": getattr(self._stt, "is_available", True) is not False,
            "ptt": getattr(self._ptt, "name", type(self._ptt).__name__),
            "audio": getattr(self._audio, "name", type(self._audio).__name__),
            "max_seconds": self.max_seconds,
            "earcon": self.earcon,
            "last_events": self.last_events(10),
        }

    # -- running --------------------------------------------------------------

    def run(self, stop: threading.Event) -> None:
        """Serve presses until ``stop`` is set. Safe to run on a daemon thread."""
        self.running = True
        self._emit("started", text=f"ptt={self.status()['ptt']} audio={self.status()['audio']}")
        idle = 0
        try:
            while not stop.is_set():
                try:
                    pressed = self._turn(stop)
                except PTTConfigError as e:
                    self.errors += 1
                    self.fatal = f"{type(e).__name__}: {e}"
                    logger.error("headset voice loop stopped: %s", e)
                    self._emit("error", text=self.fatal, fatal=True)
                    return
                except Exception as e:
                    self.errors += 1
                    logger.exception("voice turn failed: %s", e)
                    self._emit("error", text=f"{type(e).__name__}: {e}")
                    stop.wait(0.5)
                    continue
                if pressed or stop.is_set():
                    idle = 0
                    continue
                # wait_for_press gave up without a press (device gone, stdin closed):
                # never spin, and say so once.
                idle += 1
                if idle == 1:
                    logger.warning("push-to-talk source returned without a press; retrying")
                    self._emit("ptt_idle")
                stop.wait(self.idle_backoff_s)
        finally:
            self.running = False
            self._emit("stopped")

    def _turn(self, stop: threading.Event) -> bool:
        """One press-to-answer cycle; False when ``wait_for_press`` returned without a press."""
        if not self._ptt.wait_for_press(stop):
            return False
        self.turns += 1
        t_press = time.perf_counter()
        self._emit("press", text=str(self.turns))
        if self._earcon_wav:
            self._audio.play(self._earcon_wav)
            self._emit("earcon", latency_ms=(time.perf_counter() - t_press) * 1000)
        t0 = time.perf_counter()
        wav = self._capture()
        seconds = wav_duration_s(wav)
        self._emit(
            "recorded",
            text=f"{seconds:.1f}s",
            latency_ms=(time.perf_counter() - t0) * 1000,
            seconds=round(seconds, 2),
            bytes=len(wav),
        )
        self.handle_once(wav, t_start=t_press)
        return True

    def _capture(self) -> bytes:
        if getattr(self._ptt, "captures_audio", False):
            self._ptt.wait_for_release(self.max_seconds)
            return self._ptt.take_recording()  # type: ignore[attr-defined]
        rec_stop = threading.Event()

        def _watch_release() -> None:
            try:
                self._ptt.wait_for_release(self.max_seconds)
            except Exception:
                logger.exception("wait_for_release failed")
            finally:
                rec_stop.set()

        threading.Thread(target=_watch_release, name="voice-ptt-release", daemon=True).start()
        try:
            return self._audio.record(self.max_seconds, stop=rec_stop)
        finally:
            rec_stop.set()

    def handle_once(self, wav_bytes: bytes, *, t_start: float | None = None) -> dict:
        """STT -> answer -> TTS -> play for one recording; returns a summary dict."""
        start = t_start if t_start is not None else time.perf_counter()
        t0 = time.perf_counter()
        stt_available = getattr(self._stt, "is_available", True) is not False
        if stt_available:
            text = str(run_maybe_async(self._stt.transcribe(wav_bytes)) or "").strip()
            self._emit("transcript", text=text, latency_ms=(time.perf_counter() - t0) * 1000)
        else:
            text = ""
            self._emit("stt_unavailable", latency_ms=(time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        if not stt_available:
            answer = STT_UNAVAILABLE_TEXT
        elif text:
            try:
                answer = str(self._answer_fn(text))
            except Exception as e:
                self.errors += 1
                logger.exception("answer_fn failed for %r: %s", text, e)
                self._emit("error", text=f"{type(e).__name__}: {e}")
                answer = ANSWER_FAILED_TEXT
        else:
            answer = self._no_speech_text
        self._emit("answer", text=answer, latency_ms=(time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        audio = bytes(run_maybe_async(self._tts.synthesize(answer)) or b"")
        self._emit(
            "tts",
            text=f"{len(audio)} bytes",
            latency_ms=(time.perf_counter() - t0) * 1000,
            bytes=len(audio),
        )
        if audio:
            t0 = time.perf_counter()
            self._audio.play(audio)
            self._emit("played", latency_ms=(time.perf_counter() - t0) * 1000)
        else:
            logger.warning("no TTS audio for the answer; check TTS_ENGINE / PIPER_MODEL_PATH")
            self._emit("tts_unavailable", text=answer)
        total_ms = (time.perf_counter() - start) * 1000
        self._emit("done", text=answer, latency_ms=total_ms)
        return {
            "text": text,
            "answer": answer,
            "audio": audio,
            "audio_bytes": len(audio),
            "latency_ms": round(total_ms, 1),
        }


def start_voice_thread(loop: HeadsetVoiceLoop, stop: threading.Event) -> threading.Thread:
    """Run ``loop.run(stop)`` on a daemon thread named ``headset-voice`` (for T7)."""
    t = threading.Thread(target=loop.run, args=(stop,), name="headset-voice", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------


def _flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "none"}


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", key, raw, default)
        return default


def sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def make_audio_from_env(
    env: Mapping[str, str] | None = None, *, backend: str | None = None
) -> AudioIO:
    """``LOCI_AUDIO_BACKEND`` auto|sounddevice|cli with ``LOCI_AUDIO_IN``/``LOCI_AUDIO_OUT``."""
    env = os.environ if env is None else env
    kind = (backend or env.get("LOCI_AUDIO_BACKEND") or "auto").strip().lower()
    inp = (env.get("LOCI_AUDIO_IN") or "").strip() or None
    out = (env.get("LOCI_AUDIO_OUT") or "").strip() or None
    if kind in {"auto", "sounddevice"}:
        if sounddevice_available():
            return SoundDeviceAudio(inp, out)
        if kind == "sounddevice":
            raise RuntimeError(
                "LOCI_AUDIO_BACKEND=sounddevice but the package is not importable: "
                "pip install sounddevice && sudo apt install libportaudio2"
            )
        logger.info("sounddevice unavailable; using the CLI audio backend")
        return CliAudio(inp, out)
    if kind == "cli":
        return CliAudio(inp, out)
    raise ValueError(f"unknown LOCI_AUDIO_BACKEND={kind!r} (auto|sounddevice|cli)")


def make_ptt_from_env(
    audio: AudioIO, env: Mapping[str, str] | None = None, *, kind: str | None = None
) -> PushToTalk:
    """``LOCI_PTT`` keyboard|evdev|vad (default evdev on Linux, keyboard elsewhere)."""
    env = os.environ if env is None else env
    default = "evdev" if sys.platform.startswith("linux") else "keyboard"
    kind = (kind or env.get("LOCI_PTT") or default).strip().lower()
    if kind == "keyboard":
        return KeyboardPTT()
    if kind == "vad":
        return VadPTT(
            audio,
            threshold=_float(env, "LOCI_VAD_THRESHOLD", 500.0),
            silence_s=_float(env, "LOCI_VAD_SILENCE_S", 1.2),
        )
    if kind == "evdev":
        return EvdevButtonPTT.from_spec(
            env.get("LOCI_PTT_DEVICE"), toggle=_flag(env, "LOCI_PTT_TOGGLE", False)
        )
    raise ValueError(f"unknown LOCI_PTT={kind!r} (keyboard|evdev|vad)")


def make_voice_loop_from_env(
    answer_fn: Callable[[str], str],
    stt: Any,
    tts: Any,
    *,
    env: Mapping[str, str] | None = None,
    audio: AudioIO | None = None,
    ptt: PushToTalk | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> HeadsetVoiceLoop | None:
    """Build the loop from ``LOCI_VOICE`` (none|headset) and friends; ``None`` when off.

    ``audio``/``ptt`` override the environment (tests inject fakes). Keys read:
    ``LOCI_VOICE``, ``LOCI_PTT``, ``LOCI_AUDIO_BACKEND``, ``LOCI_VOICE_MAX_S`` (10),
    ``LOCI_EARCON`` (1), ``LOCI_PTT_DEVICE``, ``LOCI_AUDIO_IN``, ``LOCI_AUDIO_OUT``,
    ``LOCI_VAD_THRESHOLD`` (500), ``LOCI_VAD_SILENCE_S`` (1.2), ``LOCI_PTT_TOGGLE`` (0).
    """
    env = os.environ if env is None else env
    mode = (env.get("LOCI_VOICE") or "none").strip().lower()
    if mode in {"", "none", "0", "off", "false"}:
        return None
    if mode != "headset":
        logger.warning("LOCI_VOICE=%r not understood (none|headset); voice loop disabled", mode)
        return None
    audio = audio if audio is not None else make_audio_from_env(env)
    ptt = ptt if ptt is not None else make_ptt_from_env(audio, env)
    return HeadsetVoiceLoop(
        answer_fn,
        stt,
        tts,
        audio,
        ptt,
        max_seconds=_float(env, "LOCI_VOICE_MAX_S", 10.0),
        earcon=_flag(env, "LOCI_EARCON", True),
        on_event=on_event,
    )


__all__ = [
    "ANSWER_FAILED_TEXT",
    "DEFAULT_BUTTON_CODES",
    "DEFAULT_PTT_DEVICE",
    "EARCON_HZ",
    "EARCON_S",
    "INPUT_EVENT_FORMAT",
    "INPUT_EVENT_SIZE",
    "NO_SPEECH_TEXT",
    "PTTConfigError",
    "STT_UNAVAILABLE_TEXT",
    "SAMPLE_RATE",
    "AudioIO",
    "CliAudio",
    "EvdevButtonPTT",
    "HeadsetVoiceLoop",
    "InputEvent",
    "KeyboardPTT",
    "PushToTalk",
    "SoundDeviceAudio",
    "VadPTT",
    "earcon_wav",
    "find_input_device",
    "list_input_devices",
    "make_audio_from_env",
    "make_ptt_from_env",
    "make_voice_loop_from_env",
    "pack_input_event",
    "parse_input_events",
    "pcm16_to_wav",
    "resample_linear",
    "rms",
    "run_maybe_async",
    "select_device",
    "silence_wav",
    "sounddevice_available",
    "start_voice_thread",
    "to_mono",
    "wav_duration_s",
    "wav_to_pcm16",
]
