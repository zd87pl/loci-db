"""Bluetooth audio I/O management for Raspberry Pi 5.

Handles microphone capture and audio playback over Bluetooth headphones
using PulseAudio/PipeWire. Provides push-to-talk and voice-activity
detection modes for hands-free operation.
"""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import wave

logger = logging.getLogger(__name__)

# Audio capture parameters (match Whisper's expected format)
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM

# Voice activity detection
VAD_SILENCE_THRESHOLD = 500  # RMS threshold for silence detection
VAD_SILENCE_DURATION_S = 1.5  # seconds of silence to end recording
VAD_MAX_DURATION_S = 10.0    # max recording duration


class AudioIO:
    """Manages Bluetooth audio input/output on Raspberry Pi 5.

    Usage::

        audio = AudioIO()
        audio_bytes = await audio.record_until_silence()
        await audio.play(tts_bytes)
    """

    def __init__(self) -> None:
        self._pyaudio = None
        self._stream = None

    def _get_pyaudio(self):
        if self._pyaudio is None:
            try:
                import pyaudio
                self._pyaudio = pyaudio.PyAudio()
            except ImportError:
                logger.error("pyaudio not installed — audio I/O unavailable")
        return self._pyaudio

    async def record_until_silence(
        self,
        silence_threshold: int = VAD_SILENCE_THRESHOLD,
        silence_duration: float = VAD_SILENCE_DURATION_S,
        max_duration: float = VAD_MAX_DURATION_S,
        chunk_size: int = 1024,
    ) -> bytes:
        """Record audio from the default input device until silence is detected.

        Returns WAV-formatted audio bytes suitable for Whisper transcription.
        """
        import struct

        pa = self._get_pyaudio()
        if pa is None:
            return b""

        import pyaudio
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=chunk_size,
        )

        frames: list[bytes] = []
        silent_chunks = 0
        chunks_for_silence = int(silence_duration * SAMPLE_RATE / chunk_size)
        max_chunks = int(max_duration * SAMPLE_RATE / chunk_size)

        try:
            loop = asyncio.get_event_loop()
            for _ in range(max_chunks):
                data = await loop.run_in_executor(
                    None, stream.read, chunk_size,
                )
                frames.append(data)

                # Compute RMS for voice activity detection
                samples = struct.unpack(f"<{chunk_size}h", data)
                rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

                if rms < silence_threshold:
                    silent_chunks += 1
                    if silent_chunks >= chunks_for_silence and len(frames) > chunks_for_silence:
                        break
                else:
                    silent_chunks = 0
        finally:
            stream.stop_stream()
            stream.close()

        # Convert to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

    async def play(self, audio_bytes: bytes) -> None:
        """Play audio bytes through the default output device.

        Accepts WAV or MP3 format. Uses aplay for WAV, ffplay for MP3.
        """
        if not audio_bytes:
            return

        # Detect format by magic bytes
        if audio_bytes[:4] == b"RIFF":
            await self._play_wav(audio_bytes)
        else:
            await self._play_with_ffplay(audio_bytes)

    async def _play_wav(self, wav_bytes: bytes) -> None:
        """Play WAV audio using aplay (ALSA)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "aplay", "-q", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate(input=wav_bytes)
        except FileNotFoundError:
            logger.warning("aplay not found, trying paplay")
            await self._play_with_paplay(wav_bytes)

    async def _play_with_paplay(self, wav_bytes: bytes) -> None:
        """Play WAV via PulseAudio paplay."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            proc = await asyncio.create_subprocess_exec(
                "paplay", tmp.name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()

    async def _play_with_ffplay(self, audio_bytes: bytes) -> None:
        """Play any audio format via ffplay (ffmpeg)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate(input=audio_bytes)
        except FileNotFoundError:
            logger.error("ffplay not found — cannot play audio")

    @staticmethod
    def list_bluetooth_devices() -> list[dict]:
        """List paired Bluetooth audio devices."""
        try:
            result = subprocess.run(
                ["bluetoothctl", "devices", "Paired"],
                capture_output=True, text=True, timeout=5,
            )
            devices = []
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Device "):
                    parts = line.split(" ", 2)
                    if len(parts) >= 3:
                        devices.append({"mac": parts[1], "name": parts[2]})
            return devices
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    @staticmethod
    def get_default_sink() -> str | None:
        """Get the current PulseAudio/PipeWire default audio sink."""
        try:
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() or None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def cleanup(self) -> None:
        """Release audio resources."""
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None
