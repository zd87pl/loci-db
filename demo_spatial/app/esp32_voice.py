"""Voice bridge for the ESP32 thin client: board PCM in, answer PCM out.

The board records 16 kHz mono PCM16 on push-to-talk and plays the same format. This
module sits between that raw audio and the demo's :class:`VoicePipeline`:

* **STT**: PCM → float32 samples → faster-whisper *directly* (no container, no PyAV), using
  the local model the pipeline already holds. Any ``(samples, language) -> text`` coroutine
  can be injected instead (tests do).
* **Understanding and answer text**: ``VoicePipeline.handle_text_query`` unchanged, so the
  ESP32 gets the same intents and phrasing as the phone page and the headset.
* **TTS**: whatever the pipeline's engine returns (MP3 from edge-tts, WAV from Piper) is
  decoded and resampled to the board's PCM format by :mod:`app.pcm_audio`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .pcm_audio import (
    apply_gain_pcm16,
    decode_to_pcm16,
    pcm16_duration_s,
    pcm16_rms,
    pcm16_to_float32,
    pcm16_to_wav,
)

logger = logging.getLogger(__name__)

Transcriber = Callable[[np.ndarray, str], Awaitable[str]]
Synthesizer = Callable[[str], Awaitable[bytes]]

COULD_NOT_HEAR = "Sorry, I couldn't hear that. Hold the button and ask again."
MIN_UTTERANCE_S = 0.25
MIN_RMS = 0.003  # below this the clip is treated as silence (open mic, bumped button)


@dataclass
class SpokenAnswer:
    transcription: str
    intent_kind: str
    object_name: str
    answer_text: str
    pcm: bytes
    latency_ms: float
    tts_error: str | None = None
    stt_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "transcription": self.transcription,
            "intent": {"kind": self.intent_kind, "object_name": self.object_name},
            "answer": self.answer_text,
            "has_audio": bool(self.pcm),
            "audio_seconds": round(pcm16_duration_s(self.pcm), 2) if self.pcm else 0.0,
            "tts_error": self.tts_error,
            "latency_ms": round(self.latency_ms, 1),
            "stt_ms": round(self.stt_ms, 1),
        }


class Esp32VoiceBridge:
    """Glue between raw board audio and the shared voice pipeline."""

    def __init__(
        self,
        voice: Any,
        *,
        rate: int = 16000,
        gain: float = 1.0,
        transcriber: Transcriber | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self._voice = voice
        self.rate = rate
        self.gain = gain
        self._transcriber = transcriber
        self._synthesizer = synthesizer
        self.last_error: str | None = None
        self.queries = 0

    # -- speech to text ----------------------------------------------------------

    async def transcribe_pcm(self, pcm: bytes, language: str = "en") -> str:
        """Board PCM16 at ``self.rate`` → text ('' for silence, too-short clips or failures)."""
        if pcm16_duration_s(pcm, self.rate) < MIN_UTTERANCE_S:
            return ""
        if pcm16_rms(pcm) < MIN_RMS:
            logger.info("ESP32 voice: clip is silent (rms below %.3f)", MIN_RMS)
            return ""
        samples = pcm16_to_float32(pcm)
        try:
            if self._transcriber is not None:
                return (await self._transcriber(samples, language)).strip()
            return (await self._default_transcribe(samples, pcm, language)).strip()
        except Exception as e:
            self.last_error = f"stt: {e}"
            logger.error("ESP32 voice: transcription failed: %s", e)
            return ""

    async def _default_transcribe(self, samples: np.ndarray, pcm: bytes, language: str) -> str:
        stt = getattr(self._voice, "_stt", None)
        model = self._local_whisper(stt)
        if model is not None:
            loop = asyncio.get_running_loop()

            def run() -> str:
                segments, _info = model.transcribe(samples, language=language, beam_size=1)
                return " ".join(seg.text.strip() for seg in segments)

            return await loop.run_in_executor(None, run)
        if stt is not None and hasattr(stt, "transcribe"):
            # Not a local model (cloud Whisper configured): hand over a WAV container.
            return await stt.transcribe(pcm16_to_wav(pcm, self.rate), language)
        logger.warning("ESP32 voice: no STT backend (set WHISPER_LOCAL=1)")
        return ""

    @staticmethod
    def _local_whisper(stt: Any) -> Any:
        if stt is None or not getattr(stt, "_use_local", False):
            return None
        loader = getattr(stt, "_load_local_model", None)
        if callable(loader):
            loader()
        return getattr(stt, "_local_model", None)

    # -- text to speech ----------------------------------------------------------

    async def _synthesize(self, text: str) -> bytes:
        if self._synthesizer is not None:
            return await self._synthesizer(text)
        tts = getattr(self._voice, "_tts", None)
        if tts is None:
            return b""
        return await tts.synthesize(text)

    def _to_pcm(self, audio: bytes) -> tuple[bytes, str | None]:
        if not audio:
            return b"", "tts produced no audio"
        try:
            pcm = decode_to_pcm16(audio, self.rate)
        except ValueError as e:
            self.last_error = f"tts decode: {e}"
            return b"", str(e)
        return apply_gain_pcm16(pcm, self.gain), None

    async def say(self, text: str) -> bytes:
        """Text → board PCM (``b""`` when the TTS engine is unavailable)."""
        try:
            audio = await self._synthesize(text)
        except Exception as e:
            self.last_error = f"tts: {e}"
            logger.error("ESP32 voice: TTS failed: %s", e)
            return b""
        pcm, err = self._to_pcm(audio)
        if err:
            logger.warning("ESP32 voice: no speech for %r: %s", text[:40], err)
        return pcm

    # -- full turns --------------------------------------------------------------

    async def answer_text(self, text: str, *, speak: bool = True) -> SpokenAnswer:
        """Typed question → answer text (+ PCM when ``speak``)."""
        t0 = time.perf_counter()
        self.queries += 1
        result = await self._voice.handle_text_query(text)
        pcm, err = b"", None
        if speak:
            audio = result.answer_audio or b""
            if not audio:
                try:
                    audio = await self._synthesize(result.answer_text)
                except Exception as e:  # TTS engine exploded; answer text still goes back
                    err = str(e)
            if audio:
                pcm, err = self._to_pcm(audio)
            elif err is None:
                err = "tts produced no audio"
        return SpokenAnswer(
            transcription=text,
            intent_kind=result.intent.kind,
            object_name=result.intent.object_name,
            answer_text=result.answer_text,
            pcm=pcm,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tts_error=err,
        )

    async def answer_pcm(
        self, pcm: bytes, *, language: str = "en", speak: bool = True
    ) -> SpokenAnswer:
        """Push-to-talk clip → transcription → answer (+ PCM)."""
        t0 = time.perf_counter()
        text = await self.transcribe_pcm(pcm, language)
        stt_ms = (time.perf_counter() - t0) * 1000.0
        if not text:
            out = await self.say(COULD_NOT_HEAR) if speak else b""
            return SpokenAnswer(
                transcription="",
                intent_kind="unknown",
                object_name="",
                answer_text=COULD_NOT_HEAR,
                pcm=out,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                tts_error=None if out or not speak else "tts produced no audio",
                stt_ms=stt_ms,
            )
        answer = await self.answer_text(text, speak=speak)
        answer.transcription = text
        answer.stt_ms = stt_ms
        answer.latency_ms = (time.perf_counter() - t0) * 1000.0
        return answer

    # -- status ------------------------------------------------------------------

    def status(self) -> dict:
        if self._transcriber is not None:
            stt = "custom"
        elif getattr(getattr(self._voice, "_stt", None), "_use_local", False):
            stt = "faster-whisper (local)"
        elif getattr(self._voice, "stt_available", False):
            stt = "pipeline"
        else:
            stt = "unavailable"
        return {
            "rate": self.rate,
            "gain": self.gain,
            "stt": stt,
            "tts": "custom" if self._synthesizer else getattr(self._voice, "tts_engine", None),
            "queries": self.queries,
            "last_error": self.last_error,
        }
