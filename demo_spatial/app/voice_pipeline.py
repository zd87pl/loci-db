"""Voice pipeline: Whisper STT → intent extraction → spatial query → TTS output.

Supports three TTS engines (configured via TTS_ENGINE env var):
  - "openai"  (default): OpenAI TTS API — best quality, requires API key
  - "piper":  Local Piper TTS — zero cloud dependency, requires model download
  - "edge":   Microsoft Edge TTS — free cloud, no API key needed

STT uses OpenAI Whisper API by default. Falls back to faster-whisper
for fully offline operation if WHISPER_LOCAL=1 is set.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spatial_memory import SpatialMemory
    from .vlm_client import VLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent model
# ---------------------------------------------------------------------------

@dataclass
class QueryIntent:
    """Parsed intent from a voice query."""

    kind: str           # "where_is" | "history" | "list_objects" | "changes" | "unknown"
    object_name: str    # the target object, empty string if not applicable
    raw_text: str       # original transcribed text


_WHERE_PATTERNS = [
    r"where(?:'s| is| are| did i (?:put|leave|place|set))?\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"(?:find|locate|show me)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"(?:have you seen|did you see)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
    r"(?:what happened to|where did)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$|go)",
]

_HISTORY_PATTERNS = [
    r"(?:when|last time)\s+(?:was|did|have)\s+(?:i\s+)?(?:seen|used|placed)\s+(?:my\s+|the\s+)?(.+?)(?:\?|$)",
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

    # Check change/movement queries
    for pat in _CHANGES_PATTERNS:
        if re.search(pat, cleaned):
            return QueryIntent(kind="changes", object_name="", raw_text=text)

    # Check list-all queries
    for pat in _LIST_PATTERNS:
        if re.search(pat, cleaned):
            return QueryIntent(kind="list_objects", object_name="", raw_text=text)

    # Check history queries
    for pat in _HISTORY_PATTERNS:
        m = re.search(pat, cleaned)
        if m:
            obj = m.group(1).strip().rstrip("?")
            return QueryIntent(kind="history", object_name=obj, raw_text=text)

    # Check where-is queries
    for pat in _WHERE_PATTERNS:
        m = re.search(pat, cleaned)
        if m:
            obj = m.group(1).strip().rstrip("?")
            return QueryIntent(kind="where_is", object_name=obj, raw_text=text)

    # Generic fallback: treat as where_is with the full cleaned text as object
    # (handles short queries like "my keys?" or "glasses?")
    simple_match = re.match(r"^(?:my\s+|the\s+)?(.+)$", cleaned)
    if simple_match and len(cleaned) < 40:
        return QueryIntent(kind="where_is", object_name=simple_match.group(1), raw_text=text)

    return QueryIntent(kind="unknown", object_name="", raw_text=text)


def build_response_text(intent: QueryIntent, memory: "SpatialMemory", vlm_answer: str = "") -> str:
    """Build a natural language response from intent + LOCI-DB query results."""
    if vlm_answer:
        return vlm_answer

    if intent.kind == "list_objects":
        objects = memory.current_objects()
        if not objects:
            return "I haven't seen any objects yet. Try pointing the camera around the room."
        labels = [o.label for o in objects[:8]]
        if len(objects) > 8:
            labels.append(f"and {len(objects) - 8} more")
        return "I can see: " + ", ".join(labels) + "."

    if intent.kind == "changes":
        recent = memory.recent_changes(window_seconds=60)
        if not recent:
            return "No objects have moved in the last minute."
        labels = [o.label for o in recent[:5]]
        return "Recently spotted: " + ", ".join(labels) + "."

    if intent.kind in ("where_is", "history") and intent.object_name:
        results = memory.where_is(intent.object_name, limit=3)
        if not results:
            return f"I haven't seen your {intent.object_name} yet. Try scanning the area."

        obj = results[0]
        # Describe position in human terms
        cx, cy = obj.cx, obj.cy
        h_pos = "on the left" if cx < 0.33 else ("on the right" if cx > 0.67 else "in the center")
        v_pos = "toward the back" if cy < 0.35 else ("near the front" if cy > 0.70 else "")
        pos_parts = [h_pos]
        if v_pos:
            pos_parts.append(v_pos)
        pos = " and ".join(pos_parts)

        age = obj.age_seconds
        if age < 10:
            time_desc = "just now"
        elif age < 60:
            time_desc = f"{int(age)} seconds ago"
        elif age < 3600:
            time_desc = f"{int(age / 60)} minute{'s' if age >= 120 else ''} ago"
        else:
            time_desc = f"{int(age / 3600)} hour{'s' if age >= 7200 else ''} ago"

        return f"Your {intent.object_name} was last seen {pos}, {time_desc}."

    return "I'm not sure what you're asking. Try saying 'where is my [object]?'"


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

class WhisperSTT:
    """Speech-to-text via OpenAI Whisper API or local faster-whisper."""

    def __init__(self) -> None:
        self._use_local = os.environ.get("WHISPER_LOCAL", "0") == "1"
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._model_name = os.environ.get("WHISPER_MODEL", "whisper-1")
        self._local_model = None
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.AsyncOpenAI(api_key=self._openai_key)
        return self._openai_client

    def _load_local_model(self):
        if self._local_model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            model_size = os.environ.get("WHISPER_LOCAL_MODEL", "base")
            self._local_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Loaded local Whisper model: %s", model_size)
        except ImportError:
            logger.warning("faster-whisper not installed — falling back to OpenAI API")
            self._use_local = False

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """Transcribe audio bytes to text. Returns empty string on failure."""
        if self._use_local:
            return await self._transcribe_local(audio_bytes, language)
        if self._openai_key:
            return await self._transcribe_openai(audio_bytes, language)
        logger.warning("No STT backend available (set OPENAI_API_KEY or WHISPER_LOCAL=1)")
        return ""

    async def _transcribe_openai(self, audio_bytes: bytes, language: str) -> str:
        client = self._get_openai_client()
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.webm"
            response = await client.audio.transcriptions.create(
                model=self._model_name,
                file=audio_file,
                language=language,
                response_format="text",
            )
            return str(response).strip()
        except Exception as e:
            logger.error("Whisper API transcription failed: %s", e)
            return ""

    async def _transcribe_local(self, audio_bytes: bytes, language: str) -> str:
        self._load_local_model()
        if self._local_model is None:
            return await self._transcribe_openai(audio_bytes, language)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: list(self._local_model.transcribe(
                    io.BytesIO(audio_bytes),
                    language=language,
                    beam_size=1,
                )[0])
            )
            return " ".join(seg.text.strip() for seg in result)
        except Exception as e:
            logger.error("Local Whisper transcription failed: %s", e)
            return ""

    @property
    def is_available(self) -> bool:
        return bool(self._openai_key) or self._use_local


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

class TTS:
    """Text-to-speech with pluggable backends.

    Engine selection via TTS_ENGINE env var:
      - "openai"  (default): OpenAI TTS API (tts-1 / tts-1-hd)
      - "piper":  Local Piper TTS (model files must be present)
      - "edge":   Microsoft Edge TTS (free, cloud)
    """

    def __init__(self) -> None:
        self._engine = os.environ.get("TTS_ENGINE", "openai").lower()
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._openai_voice = os.environ.get("TTS_VOICE", "nova")     # nova, alloy, echo, fable, onyx, shimmer
        self._openai_model = os.environ.get("TTS_MODEL", "tts-1")
        self._piper_model = os.environ.get("PIPER_MODEL_PATH", "")
        self._edge_voice = os.environ.get("EDGE_TTS_VOICE", "en-US-AnaNeural")
        self._openai_client = None
        self._piper_voice = None

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.AsyncOpenAI(api_key=self._openai_key)
        return self._openai_client

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (MP3 or WAV depending on engine)."""
        if not text.strip():
            return b""

        if self._engine == "piper":
            result = await self._synthesize_piper(text)
            if result:
                return result
            # Fall through to next engine
            logger.warning("Piper TTS failed, falling back to edge-tts")

        if self._engine in ("edge", "piper"):
            result = await self._synthesize_edge(text)
            if result:
                return result
            logger.warning("Edge TTS failed, falling back to OpenAI")

        # Default: OpenAI TTS
        if self._openai_key:
            return await self._synthesize_openai(text)

        logger.error("No TTS backend available")
        return b""

    async def _synthesize_openai(self, text: str) -> bytes:
        client = self._get_openai_client()
        try:
            response = await client.audio.speech.create(
                model=self._openai_model,
                voice=self._openai_voice,
                input=text,
                response_format="mp3",
            )
            return response.content
        except Exception as e:
            logger.error("OpenAI TTS failed: %s", e)
            return b""

    async def _synthesize_piper(self, text: str) -> bytes:
        """Synthesize using local Piper TTS model."""
        if not self._piper_model:
            logger.debug("PIPER_MODEL_PATH not set")
            return b""
        try:
            import piper
            if self._piper_voice is None:
                self._piper_voice = piper.PiperVoice.load(self._piper_model)

            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                None, self._run_piper_sync, text
            )
            return audio_bytes
        except ImportError:
            logger.debug("piper-tts not installed")
            return b""
        except Exception as e:
            logger.error("Piper TTS error: %s", e)
            return b""

    def _run_piper_sync(self, text: str) -> bytes:
        """Run Piper synthesis synchronously (called from executor)."""
        buf = io.BytesIO()
        import wave
        with wave.open(buf, "wb") as wf:
            self._piper_voice.synthesize(text, wf)
        return buf.getvalue()

    async def _synthesize_edge(self, text: str) -> bytes:
        """Synthesize using Microsoft Edge TTS (edge-tts library)."""
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, self._edge_voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            return buf.getvalue()
        except ImportError:
            logger.debug("edge-tts not installed")
            return b""
        except Exception as e:
            logger.error("Edge TTS error: %s", e)
            return b""

    @property
    def engine(self) -> str:
        return self._engine


# ---------------------------------------------------------------------------
# High-level VoicePipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """End-to-end voice query pipeline: audio → text → answer → audio.

    Usage::

        pipeline = VoicePipeline(memory, vlm_client)
        result = await pipeline.handle_audio_query(audio_bytes)
        # result.answer_text  — text of the answer
        # result.answer_audio — audio bytes for playback
    """

    def __init__(
        self,
        memory: "SpatialMemory",
        vlm_client: "VLMClient | None" = None,
    ) -> None:
        self._memory = memory
        self._vlm = vlm_client
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
        """Full pipeline: audio → transcribe → understand → query → respond.

        Args:
            audio_bytes: Raw audio (WebM/WAV/MP3 from browser mic).
            language: ISO 639-1 language code for STT.
            synthesize_response: If False, skip TTS (text-only response).

        Returns:
            QueryResult with transcription, intent, answer text, and audio.
        """
        import time
        t0 = time.perf_counter()

        # Step 1: Transcribe
        transcript = await self._stt.transcribe(audio_bytes, language)
        if not transcript:
            answer = "Sorry, I couldn't understand the audio. Please try again."
            audio = await self._tts.synthesize(answer) if synthesize_response else b""
            return self.QueryResult(
                transcription="",
                intent=QueryIntent("unknown", "", ""),
                answer_text=answer,
                answer_audio=audio,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Step 2: Parse intent
        intent = parse_intent(transcript)

        # Step 3: Query spatial memory + generate answer
        answer = await self._generate_answer(intent)

        # Step 4: Synthesize speech
        audio = await self._tts.synthesize(answer) if synthesize_response else b""

        latency_ms = (time.perf_counter() - t0) * 1000
        return self.QueryResult(
            transcription=transcript,
            intent=intent,
            answer_text=answer,
            answer_audio=audio,
            latency_ms=round(latency_ms, 1),
        )

    async def handle_text_query(self, text: str) -> "VoicePipeline.QueryResult":
        """Text-only pipeline (skip STT, still generates TTS)."""
        import time
        t0 = time.perf_counter()
        intent = parse_intent(text)
        answer = await self._generate_answer(intent)
        audio = await self._tts.synthesize(answer)
        return self.QueryResult(
            transcription=text,
            intent=intent,
            answer_text=answer,
            answer_audio=audio,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    async def _generate_answer(self, intent: QueryIntent) -> str:
        """Query LOCI-DB and optionally use VLM to compose a natural answer."""
        # Get candidate objects from spatial memory
        if intent.kind in ("where_is", "history") and intent.object_name:
            objects = self._memory.where_is(intent.object_name, limit=3)
            objects_data = [o.to_dict() for o in objects]
        elif intent.kind == "list_objects":
            objects_data = [o.to_dict() for o in self._memory.current_objects()]
        elif intent.kind == "changes":
            objects_data = [o.to_dict() for o in self._memory.recent_changes(60)]
        else:
            objects_data = []

        # Use VLM for natural language composition if available and objects found
        if self._vlm and self._vlm.is_available and objects_data:
            try:
                vlm_answer = await self._vlm.answer_location_question(
                    intent.raw_text, objects_data
                )
                if vlm_answer:
                    return vlm_answer
            except Exception as e:
                logger.warning("VLM answer generation failed: %s", e)

        # Fallback: rule-based answer
        return build_response_text(intent, self._memory)

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
            "vlm_available": self._vlm.is_available if self._vlm else False,
        }
