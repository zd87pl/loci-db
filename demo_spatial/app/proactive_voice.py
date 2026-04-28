"""Proactive voice output layer — bridges hazard detection to TTS.

Consumes Alert objects from ProactiveEngine and speaks them with:
  - Priority queuing: IMMEDIATE alerts interrupt WARNING/INFORMATIONAL output
  - VAD-aware: never interrupts user's own speech
  - Mode switching: guide / quiet / reactive
  - Spatial memory enrichment for informational alerts

Typical integration::

    from demo_spatial.app.voice_pipeline import TTS
    from demo_spatial.app.proactive_voice import ProactiveVoice, VoiceMode

    pv = ProactiveVoice(tts, memory=spatial_memory)
    asyncio.create_task(pv.run())           # background speaker loop

    # Per camera frame — from hazard engine:
    for alert in engine.process(dets, depth_map):
        pv.enqueue_alert(alert)

    # Per STT result — check mode commands first:
    mode_response = await pv.handle_mode_command(transcript)
    if mode_response:
        pv.enqueue_text(mode_response)
    else:
        # route to normal VoicePipeline intent handling
        ...
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .spatial_memory import SpatialMemory
    from .voice_pipeline import TTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice mode
# ---------------------------------------------------------------------------

class VoiceMode(str, Enum):
    GUIDE    = "guide"     # proactive alerts fully enabled
    QUIET    = "quiet"     # all proactive alerts suppressed
    REACTIVE = "reactive"  # default — respond to explicit queries only


_GUIDE_PATTERNS = [
    r"\bguide\s+me\b",
    r"\bstart\s+(?:guiding|navigation|alerts?)\b",
    r"\bturn\s+on\s+(?:alerts?|guidance|navigation)\b",
    r"\benable\s+(?:alerts?|guidance)\b",
    r"\bproactive\s+(?:mode|alerts?)\b",
]

_QUIET_PATTERNS = [
    r"\bquiet\s+(?:mode|down|please)?\b",
    r"\bsilence\b",
    r"\bstop\s+(?:talking|alerts?|guiding)\b",
    r"\bturn\s+off\s+(?:alerts?|guidance)\b",
    r"\bdisable\s+(?:alerts?|guidance)\b",
    r"\bmute\b",
]

_SCENE_PATTERNS = [
    r"\bwhat(?:'s| is)\s+(?:around|nearby|here)\b",
    r"\bwhat\s+do\s+(?:you\s+)?see\b",
    r"\blook\s+around\b",
    r"\bdescribe\s+(?:the\s+)?(?:scene|surroundings?|area)\b",
    r"\bscan\s+(?:the\s+)?(?:area|room)\b",
    r"\bwhat(?:'s| is)\s+in\s+(?:the\s+)?(?:area|room)\b",
]

_MODE_RESPONSES: dict[VoiceMode, str] = {
    VoiceMode.GUIDE:    "Guide mode on. I'll alert you to obstacles ahead.",
    VoiceMode.QUIET:    "Quiet mode. Proactive alerts paused.",
    VoiceMode.REACTIVE: "Reactive mode. I'll answer questions on request.",
}


def parse_mode_intent(text: str) -> VoiceMode | str | None:
    """Check if *text* is a mode-switching command.

    Returns:
        VoiceMode  — if the user wants to change the alert mode
        "scene_summary" — if the user wants a one-shot scene description
        None       — if the text is not a mode command
    """
    cleaned = text.strip().lower()
    for pat in _GUIDE_PATTERNS:
        if re.search(pat, cleaned):
            return VoiceMode.GUIDE
    for pat in _QUIET_PATTERNS:
        if re.search(pat, cleaned):
            return VoiceMode.QUIET
    for pat in _SCENE_PATTERNS:
        if re.search(pat, cleaned):
            return "scene_summary"
    return None


# ---------------------------------------------------------------------------
# Internal priority levels
# ---------------------------------------------------------------------------

class _Priority(int, Enum):
    IMMEDIATE      = 0   # hazard: stop/caution alerts
    WARNING        = 1   # hazard: approaching-vehicle/animal alerts
    QUERY_RESPONSE = 2   # user-initiated query answers and mode confirmations
    INFORMATIONAL  = 3   # background/novel-object annotations


def _alert_priority(category_value: str) -> _Priority:
    return {
        "immediate":    _Priority.IMMEDIATE,
        "warning":      _Priority.WARNING,
        "informational": _Priority.INFORMATIONAL,
    }.get(category_value, _Priority.INFORMATIONAL)


@dataclass(order=True)
class _QueueItem:
    priority: int
    speech_text: str  = field(compare=False)
    timestamp_ms: int = field(compare=False)


# ---------------------------------------------------------------------------
# Audio playback helpers
# ---------------------------------------------------------------------------

def _play_sync(audio_bytes: bytes) -> None:
    """Blocking audio playback. Tries pydub+sounddevice, then CLI fallbacks."""
    if not audio_bytes:
        return

    try:
        import io

        import numpy as np
        import sounddevice as sd
        from pydub import AudioSegment

        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= 2 ** (seg.sample_width * 8 - 1)
        if seg.channels == 2:
            samples = samples.reshape((-1, 2))
        sd.play(samples, samplerate=seg.frame_rate, blocking=True)
        return
    except Exception:
        pass

    import os
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        for player in ("mpg123", "mpv", "aplay"):
            try:
                subprocess.run([player, "-q", tmp], timeout=30, check=False)
                return
            except FileNotFoundError:
                continue
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    logger.warning("proactive_voice: no audio playback backend available — %d bytes dropped", len(audio_bytes))


async def _play_async(audio_bytes: bytes) -> None:
    """Async wrapper around _play_sync, runs in default executor."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _play_sync, audio_bytes)


# ---------------------------------------------------------------------------
# Spatial helper
# ---------------------------------------------------------------------------

def _format_age(age_seconds: float) -> str:
    if age_seconds < 10:
        return "just now"
    if age_seconds < 60:
        return f"{int(age_seconds)} seconds ago"
    mins = int(age_seconds / 60)
    return f"{mins} minute{'s' if mins != 1 else ''} ago"


def _direction_to_x_range(direction_value: str) -> tuple[float, float]:
    """Map a Direction string to an (x_min, x_max) region for spatial queries."""
    return {
        "left":   (0.00, 0.35),
        "center": (0.30, 0.70),
        "right":  (0.65, 1.00),
        "behind": (0.30, 0.70),
    }.get(direction_value, (0.25, 0.75))


# ---------------------------------------------------------------------------
# ProactiveVoice
# ---------------------------------------------------------------------------

class ProactiveVoice:
    """Priority-queued, VAD-aware proactive voice output.

    Thread model: single asyncio event loop. All public methods are safe to
    call from the same thread/loop that runs ``run()``.
    """

    def __init__(
        self,
        tts: TTS,
        memory: SpatialMemory | None = None,
        mode: VoiceMode = VoiceMode.REACTIVE,
        max_queue_size: int = 16,
    ) -> None:
        self._tts = tts
        self._memory = memory
        self._mode = mode
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._user_speaking = False
        self._current_priority: int | None = None
        self._speak_task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    @property
    def mode(self) -> VoiceMode:
        return self._mode

    def set_mode(self, mode: VoiceMode) -> None:
        prev = self._mode
        self._mode = mode
        if mode == VoiceMode.QUIET and prev != VoiceMode.QUIET:
            self._drain_queue()
            # Interrupt any non-critical in-flight speech immediately
            if (
                self._speak_task is not None
                and not self._speak_task.done()
                and self._current_priority is not None
                and self._current_priority > _Priority.IMMEDIATE
            ):
                self._speak_task.cancel()

    # ------------------------------------------------------------------
    # VAD integration
    # ------------------------------------------------------------------

    def set_user_speaking(self, is_speaking: bool) -> None:
        """Call from VAD callback to prevent interrupting the user's speech."""
        self._user_speaking = is_speaking

    # ------------------------------------------------------------------
    # Alert / text enqueueing
    # ------------------------------------------------------------------

    def enqueue_alert(self, alert: Any) -> None:
        """Accept an Alert from ProactiveEngine.

        Silently dropped in QUIET and REACTIVE modes. The alert object must
        expose ``.category`` (str or enum), ``.speech_text`` (str), and
        ``.direction`` (str or enum with ``.value``).
        """
        if self._mode != VoiceMode.GUIDE:
            return

        cat_val = alert.category if isinstance(alert.category, str) else alert.category.value
        priority = _alert_priority(cat_val)
        speech = self._enrich_speech(alert, cat_val)

        item = _QueueItem(
            priority=priority.value,
            speech_text=speech,
            timestamp_ms=int(time.time() * 1000),
        )
        self._enqueue(item)
        self._maybe_interrupt(priority.value)

    def enqueue_text(
        self,
        text: str,
        priority: _Priority = _Priority.QUERY_RESPONSE,
    ) -> None:
        """Queue an arbitrary message (mode confirmations, query responses)."""
        item = _QueueItem(
            priority=priority.value,
            speech_text=text,
            timestamp_ms=int(time.time() * 1000),
        )
        self._enqueue(item)
        self._maybe_interrupt(priority.value)

    # ------------------------------------------------------------------
    # Background speaker loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Process the speech queue continuously. Run as an asyncio Task."""
        self._running = True
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # Defer non-critical speech if user is currently speaking (VAD)
            if self._user_speaking and item.priority > _Priority.IMMEDIATE.value:
                self._queue.task_done()
                continue

            self._speak_task = asyncio.create_task(self._do_speak(item))
            try:
                await self._speak_task
            except asyncio.CancelledError:
                pass  # interrupted by a higher-priority alert
            finally:
                self._speak_task = None
                self._current_priority = None
                self._queue.task_done()

    def stop(self) -> None:
        """Signal the run() loop to exit and cancel any in-flight speech."""
        self._running = False
        if self._speak_task is not None and not self._speak_task.done():
            self._speak_task.cancel()

    # ------------------------------------------------------------------
    # Mode-command integration point
    # ------------------------------------------------------------------

    async def handle_mode_command(self, text: str) -> str | None:
        """Check if *text* is a mode-switching command.

        Returns the spoken confirmation text, or None if not a mode command.
        Call this before routing to the normal VoicePipeline intent handler.

        Example::

            response = await pv.handle_mode_command(transcript)
            if response:
                pv.enqueue_text(response)
            else:
                result = await pipeline.handle_text_query(transcript)
                pv.enqueue_text(result.answer_text)
        """
        result = parse_mode_intent(text)
        if result is None:
            return None
        if result == "scene_summary":
            return self._build_scene_summary()
        assert isinstance(result, VoiceMode)
        self.set_mode(result)
        return _MODE_RESPONSES[result]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _do_speak(self, item: _QueueItem) -> None:
        self._current_priority = item.priority
        try:
            audio = await self._tts.synthesize(item.speech_text)
            if audio:
                await _play_async(audio)
        except asyncio.CancelledError:
            raise  # must propagate so run() can resume the queue

    def _maybe_interrupt(self, new_priority: int) -> None:
        """Cancel current speech when an outranking item arrives and user isn't speaking."""
        if (
            self._speak_task is not None
            and not self._speak_task.done()
            and self._current_priority is not None
            and new_priority < self._current_priority
            and not self._user_speaking
        ):
            self._speak_task.cancel()

    def _enqueue(self, item: _QueueItem) -> None:
        """Put item in queue; silently drop if full (prefer newest alerts)."""
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.debug("proactive_voice: queue full, dropping alert '%s'", item.speech_text[:40])

    def _enrich_speech(self, alert: Any, cat_val: str) -> str:
        """For INFORMATIONAL alerts, append spatial memory context if available."""
        speech: str = alert.speech_text
        if cat_val != "informational" or self._memory is None:
            return speech

        dir_val = alert.direction if isinstance(alert.direction, str) else alert.direction.value
        x_min, x_max = _direction_to_x_range(dir_val)

        try:
            nearby = self._memory.objects_in_region(
                x_min=x_min, x_max=x_max, y_min=0.0, y_max=1.0, limit=3
            )
        except Exception:
            return speech

        if not nearby:
            return speech

        recent = nearby[0]
        age = _format_age(recent.age_seconds)
        return f"{speech} Your {recent.label} was last seen here {age}."

    def _build_scene_summary(self) -> str:
        """One-shot description of all currently tracked objects."""
        if self._memory is None:
            return "No spatial memory available."
        objects = self._memory.current_objects()
        if not objects:
            return "I haven't tracked any objects yet. Try scanning the area."
        parts: list[str] = []
        for o in objects[:5]:
            age = _format_age(o.age_seconds)
            parts.append(f"your {o.label} {o.position_description}, seen {age}")
        result = "Around you: " + "; ".join(parts) + "."
        if len(objects) > 5:
            result += f" And {len(objects) - 5} more."
        return result

    def _drain_queue(self) -> None:
        """Discard all queued speech items (used when entering QUIET mode)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break
