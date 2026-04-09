"""Voice pipeline for RPi5: local Whisper STT + Piper TTS + Bluetooth audio.

Configured for fully offline operation:
  - STT: faster-whisper (int8 on ARM CPU)
  - TTS: Piper TTS (ONNX, ~200ms per utterance)
  - Audio I/O: PulseAudio/PipeWire Bluetooth profile

Falls back to edge-tts (free cloud) if Piper model is unavailable.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spatial_memory import SpatialMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent model
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Parsed intent from a voice query."""
    kind: str           # "where_is" | "history" | "list_objects" | "changes" | "unknown"
    object_name: str
    raw_text: str


_WHERE_PATTERNS = [
    r"where(?:'s| is| are| did i (?:put|leave|place|set))?\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"(?:find|locate|show me)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"(?:have you seen|did you see)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"(?:what happened to|where did)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$|go)",
]

_HISTORY_PATTERNS = [
    r"(?:when|last time)\s+(?:was|did|have)\s+(?:i\s+)?(?:seen|used|placed|put|left)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"when\s+(?:did you|did i|have you)\s+(?:last\s+)?(?:see|seen|spot|notice)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"when\s+(?:was\s+)?(?:my\s+|the\s+)?(.+?)\s+(?:last\s+)?(?:seen|spotted|detected)",
    r"how\s+long\s+(?:ago|since)\s+(?:was|did|have)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$|seen|spotted)",
    r"history\s+(?:of\s+)?(?:my\s+)?(.+?)(?:\?|$)",
]

_LIST_PATTERNS = [
    r"(?:what(?:'s| is| are)?\s+(?:in\s+the\s+room|around|nearby|here|visible))",
    r"(?:list|show|tell me)\s+(?:all\s+)?(?:the\s+)?objects",
    r"what\s+(?:objects|things|items)\s+(?:can you|do you)\s+(?:see|know about)",
    r"what\s+do\s+(?:i|you)\s+(?:have|see|know)",
]

_CHANGES_PATTERNS = [
    r"what(?:'s| has| have)\s+(?:changed|moved|been moved)",
    r"(?:any|what)\s+(?:new|recent)\s+(?:changes|updates|movements)",
]


def parse_intent(text: str) -> QueryIntent:
    """Extract structured intent from transcribed voice query."""
    cleaned = text.strip().lower().rstrip("?!.")

    for pat in _CHANGES_PATTERNS:
        if re.search(pat, cleaned):
            return QueryIntent(kind="changes", object_name="", raw_text=text)

    for pat in _LIST_PATTERNS:
        if re.search(pat, cleaned):
            return QueryIntent(kind="list_objects", object_name="", raw_text=text)

    for pat in _HISTORY_PATTERNS:
        m = re.search(pat, cleaned)
        if m:
            return QueryIntent(kind="history", object_name=m.group(1).strip().rstrip("?"), raw_text=text)

    for pat in _WHERE_PATTERNS:
        m = re.search(pat, cleaned)
        if m:
            return QueryIntent(kind="where_is", object_name=m.group(1).strip().rstrip("?"), raw_text=text)

    simple_match = re.match(r"^(?:my\s+|the\s+)?(.+)$", cleaned)
    if simple_match and len(cleaned) < 40:
        return QueryIntent(kind="where_is", object_name=simple_match.group(1), raw_text=text)

    return QueryIntent(kind="unknown", object_name="", raw_text=text)


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def _format_age(age: float) -> str:
    if age < 10:
        return "just now"
    elif age < 60:
        return f"{int(age)} seconds ago"
    elif age < 3600:
        mins = int(age / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    else:
        hrs = int(age / 3600)
        return f"{hrs} hour{'s' if hrs != 1 else ''} ago"


def _format_clock_time(timestamp_ms: int) -> str:
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%I:%M %p").lstrip("0")


def _describe_position(cx: float, cy: float, depth_m: float | None = None) -> str:
    h_pos = "on the left" if cx < 0.33 else ("on the right" if cx > 0.67 else "in the center")
    v_pos = "toward the back" if cy < 0.35 else ("near the front" if cy > 0.70 else "in the middle area")
    base = f"{h_pos}, {v_pos}"
    if depth_m is not None:
        base += f", about {depth_m:.1f} meters away"
    return base


def _describe_position_relative(target, all_objects: list) -> str:
    if not all_objects or len(all_objects) < 2:
        return ""
    others = [o for o in all_objects if o.label != target.label]
    if not others:
        return ""
    closest = min(others, key=lambda o: ((o.cx - target.cx) ** 2 + (o.cy - target.cy) ** 2) ** 0.5)
    dx = target.cx - closest.cx
    dy = target.cy - closest.cy
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist > 0.5:
        return ""
    if abs(dx) > abs(dy):
        direction = "to the right of" if dx > 0 else "to the left of"
    else:
        direction = "in front of" if dy > 0 else "behind"
    return f"It's {direction} the {closest.label}."


def build_response_text(intent: QueryIntent, memory: "SpatialMemory") -> str:
    """Build a natural language response from intent + LOCI-DB results."""
    if intent.kind == "list_objects":
        objects = memory.current_objects()
        if not objects:
            return "I haven't seen any objects yet. Try pointing the camera around the room."
        parts = []
        for o in objects[:6]:
            pos = _describe_position(o.cx, o.cy, o.depth_m)
            age_str = _format_age(o.age_seconds)
            parts.append(f"{o.label} ({pos}, seen {age_str})")
        result = "I'm tracking: " + "; ".join(parts) + "."
        if len(objects) > 6:
            result += f" Plus {len(objects) - 6} more objects."
        return result

    if intent.kind == "changes":
        recent = memory.recent_changes(window_seconds=60)
        if not recent:
            return "No objects have moved in the last minute."
        parts = []
        for o in recent[:5]:
            pos = _describe_position(o.cx, o.cy, o.depth_m)
            time_str = _format_clock_time(o.timestamp_ms)
            parts.append(f"{o.label} at {time_str}, {pos}")
        return "Recently spotted: " + "; ".join(parts) + "."

    if intent.kind in ("where_is", "history") and intent.object_name:
        results = memory.where_is(intent.object_name, limit=5)
        if not results:
            return f"I haven't seen your {intent.object_name} yet. Try scanning the area."

        obj = results[0]
        pos = _describe_position(obj.cx, obj.cy, obj.depth_m)
        age_str = _format_age(obj.age_seconds)
        clock_time = _format_clock_time(obj.timestamp_ms)

        answer = f"Your {intent.object_name} was last seen {pos}. That was {age_str}, at {clock_time}."

        all_objects = memory.current_objects()
        rel = _describe_position_relative(obj, all_objects)
        if rel:
            answer += f" {rel}"

        if intent.kind == "history" and len(results) > 1:
            answer += " Previous sightings:"
            for prev in results[1:4]:
                prev_pos = _describe_position(prev.cx, prev.cy, prev.depth_m)
                prev_time = _format_clock_time(prev.timestamp_ms)
                prev_age = _format_age(prev.age_seconds)
                answer += f" At {prev_time} ({prev_age}), {prev_pos}."

        return answer

    return "I'm not sure what you're asking. Try saying 'where is my' followed by an object name."


# ---------------------------------------------------------------------------
# STT (local faster-whisper only)
# ---------------------------------------------------------------------------

class WhisperSTT:
    """Speech-to-text via local faster-whisper (int8 on ARM CPU)."""

    def __init__(self) -> None:
        self._model_size = os.environ.get("WHISPER_MODEL", "base")
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("Loaded local Whisper model: %s (int8)", self._model_size)
        except ImportError:
            logger.error("faster-whisper not installed — STT unavailable")
        except Exception as e:
            logger.error("Failed to load Whisper model: %s", e)

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """Transcribe audio bytes to text."""
        self._load_model()
        if self._model is None:
            return ""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: list(self._model.transcribe(
                    io.BytesIO(audio_bytes),
                    language=language,
                    beam_size=1,
                )[0])
            )
            return " ".join(seg.text.strip() for seg in result)
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return ""

    @property
    def is_available(self) -> bool:
        return True  # always available if faster-whisper is installed


# ---------------------------------------------------------------------------
# TTS (Piper local, edge-tts fallback)
# ---------------------------------------------------------------------------

class TTS:
    """Text-to-speech: Piper TTS (local) with edge-tts fallback.

    Configure via environment:
      PIPER_MODEL_PATH  — path to .onnx Piper voice model
      EDGE_TTS_VOICE    — fallback voice (default: en-US-AnaNeural)
    """

    def __init__(self) -> None:
        self._piper_model_path = os.environ.get("PIPER_MODEL_PATH", "")
        self._edge_voice = os.environ.get("EDGE_TTS_VOICE", "en-US-AnaNeural")
        self._piper_voice = None
        self._engine = "piper" if self._piper_model_path else "edge"

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (WAV or MP3)."""
        if not text.strip():
            return b""

        if self._piper_model_path:
            result = await self._synthesize_piper(text)
            if result:
                return result
            logger.warning("Piper TTS failed, falling back to edge-tts")

        return await self._synthesize_edge(text)

    async def _synthesize_piper(self, text: str) -> bytes:
        if not self._piper_model_path:
            return b""
        try:
            import piper
            if self._piper_voice is None:
                self._piper_voice = piper.PiperVoice.load(self._piper_model_path)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._run_piper_sync, text)
        except ImportError:
            logger.debug("piper-tts not installed")
            return b""
        except Exception as e:
            logger.error("Piper TTS error: %s", e)
            return b""

    def _run_piper_sync(self, text: str) -> bytes:
        buf = io.BytesIO()
        import wave
        with wave.open(buf, "wb") as wf:
            self._piper_voice.synthesize(text, wf)
        return buf.getvalue()

    async def _synthesize_edge(self, text: str) -> bytes:
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self._edge_voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()
        except ImportError:
            logger.error("edge-tts not installed — no TTS available")
            return b""
        except Exception as e:
            logger.error("Edge TTS error: %s", e)
            return b""

    @property
    def engine(self) -> str:
        return self._engine


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """End-to-end voice query pipeline: audio -> text -> answer -> audio."""

    def __init__(self, memory: "SpatialMemory") -> None:
        self._memory = memory
        self._stt = WhisperSTT()
        self._tts = TTS()

    @dataclass
    class QueryResult:
        transcription: str
        intent: QueryIntent
        answer_text: str
        answer_audio: bytes
        latency_ms: float

    async def handle_audio_query(
        self,
        audio_bytes: bytes,
        language: str = "en",
        synthesize_response: bool = True,
    ) -> "VoicePipeline.QueryResult":
        """Full pipeline: audio -> transcribe -> understand -> query -> respond."""
        t0 = time.perf_counter()

        transcript = await self._stt.transcribe(audio_bytes, language)
        if not transcript:
            answer = "Sorry, I couldn't understand the audio. Please try again."
            audio = await self._tts.synthesize(answer) if synthesize_response else b""
            return self.QueryResult(
                transcription="", intent=QueryIntent("unknown", "", ""),
                answer_text=answer, answer_audio=audio,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        intent = parse_intent(transcript)
        answer = build_response_text(intent, self._memory)
        audio = await self._tts.synthesize(answer) if synthesize_response else b""

        return self.QueryResult(
            transcription=transcript, intent=intent,
            answer_text=answer, answer_audio=audio,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    async def handle_text_query(self, text: str) -> "VoicePipeline.QueryResult":
        """Text-only pipeline (skip STT)."""
        t0 = time.perf_counter()
        intent = parse_intent(text)
        answer = build_response_text(intent, self._memory)
        audio = await self._tts.synthesize(answer)
        return self.QueryResult(
            transcription=text, intent=intent,
            answer_text=answer, answer_audio=audio,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @property
    def stt_available(self) -> bool:
        return self._stt.is_available

    @property
    def tts_engine(self) -> str:
        return self._tts.engine

    def status(self) -> dict:
        return {
            "stt_available": self.stt_available,
            "tts_engine": self.tts_engine,
        }
