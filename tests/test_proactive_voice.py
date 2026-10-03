"""Tests for the proactive voice output module."""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Allow importing demo_spatial.app modules without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_spatial.app.proactive_voice import (
    ProactiveVoice,
    VoiceMode,
    _Priority,
    _alert_priority,
    _direction_to_x_range,
    _format_age,
    parse_mode_intent,
)
from demo_spatial.app.voice_pipeline import QueryIntent, parse_intent


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _FakeAlert:
    category: str
    speech_text: str
    label: str
    distance_m: float
    direction: str
    priority: float = 1.0
    timestamp_ms: int = 0

    def __post_init__(self) -> None:
        if not self.timestamp_ms:
            self.timestamp_ms = int(time.time() * 1000)


class _FakeTTS:
    def __init__(self, return_bytes: bytes = b"audio") -> None:
        self.calls: list[str] = []
        self._bytes = return_bytes

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self._bytes


class _FakeMemory:
    def __init__(self, objects=None) -> None:
        self._objects = objects or []

    def current_objects(self):
        return self._objects

    def objects_in_region(self, **_kwargs):
        return self._objects


@dataclass
class _FakeObject:
    label: str
    cx: float = 0.5
    cy: float = 0.5
    age_seconds: float = 30.0
    position_description: str = "in the center, in the middle area"


# ---------------------------------------------------------------------------
# parse_mode_intent
# ---------------------------------------------------------------------------

class TestParseModeIntent:
    def test_guide_me(self):
        assert parse_mode_intent("guide me") == VoiceMode.GUIDE

    def test_guide_me_mixed_case(self):
        assert parse_mode_intent("Guide Me please") == VoiceMode.GUIDE

    def test_quiet_mode(self):
        assert parse_mode_intent("quiet mode") == VoiceMode.QUIET

    def test_silence(self):
        assert parse_mode_intent("silence") == VoiceMode.QUIET

    def test_stop_talking(self):
        assert parse_mode_intent("stop talking") == VoiceMode.QUIET

    def test_mute(self):
        assert parse_mode_intent("mute") == VoiceMode.QUIET

    def test_scene_summary_whats_around(self):
        assert parse_mode_intent("what's around me?") == "scene_summary"

    def test_scene_summary_look_around(self):
        assert parse_mode_intent("look around") == "scene_summary"

    def test_scene_summary_what_do_you_see(self):
        assert parse_mode_intent("what do you see") == "scene_summary"

    def test_none_for_spatial_query(self):
        assert parse_mode_intent("where are my keys?") is None

    def test_none_for_history_query(self):
        assert parse_mode_intent("when did you last see my wallet?") is None

    def test_none_for_empty(self):
        assert parse_mode_intent("") is None


# ---------------------------------------------------------------------------
# _alert_priority
# ---------------------------------------------------------------------------

class TestAlertPriority:
    def test_immediate(self):
        assert _alert_priority("immediate") == _Priority.IMMEDIATE

    def test_warning(self):
        assert _alert_priority("warning") == _Priority.WARNING

    def test_informational(self):
        assert _alert_priority("informational") == _Priority.INFORMATIONAL

    def test_unknown_defaults_to_informational(self):
        assert _alert_priority("unknown") == _Priority.INFORMATIONAL


# ---------------------------------------------------------------------------
# _format_age
# ---------------------------------------------------------------------------

class TestFormatAge:
    def test_just_now(self):
        assert _format_age(5) == "just now"

    def test_seconds(self):
        assert _format_age(45) == "45 seconds ago"

    def test_one_minute(self):
        assert _format_age(60) == "1 minute ago"

    def test_plural_minutes(self):
        assert _format_age(120) == "2 minutes ago"


# ---------------------------------------------------------------------------
# _direction_to_x_range
# ---------------------------------------------------------------------------

class TestDirectionToXRange:
    def test_left(self):
        x_min, x_max = _direction_to_x_range("left")
        assert x_min == 0.0
        assert x_max < 0.5

    def test_right(self):
        x_min, x_max = _direction_to_x_range("right")
        assert x_min > 0.5
        assert x_max == 1.0

    def test_center(self):
        x_min, x_max = _direction_to_x_range("center")
        assert x_min < 0.5
        assert x_max > 0.5

    def test_behind_same_as_center(self):
        assert _direction_to_x_range("behind") == _direction_to_x_range("center")


# ---------------------------------------------------------------------------
# ProactiveVoice.set_mode / mode property
# ---------------------------------------------------------------------------

class TestProactiveVoiceMode:
    def _make_pv(self, mode=VoiceMode.GUIDE):
        tts = _FakeTTS()
        return ProactiveVoice(tts, mode=mode), tts

    def test_default_mode(self):
        pv, _ = self._make_pv(VoiceMode.REACTIVE)
        assert pv.mode == VoiceMode.REACTIVE

    def test_set_mode_guide(self):
        pv, _ = self._make_pv()
        pv.set_mode(VoiceMode.QUIET)
        assert pv.mode == VoiceMode.QUIET
        pv.set_mode(VoiceMode.GUIDE)
        assert pv.mode == VoiceMode.GUIDE

    def test_alerts_dropped_in_quiet_mode(self):
        pv, tts = self._make_pv(VoiceMode.QUIET)
        alert = _FakeAlert("immediate", "Stop! door ahead.", "door", 0.8, "center")
        pv.enqueue_alert(alert)
        assert pv._queue.empty()

    def test_alerts_dropped_in_reactive_mode(self):
        pv, _ = self._make_pv(VoiceMode.REACTIVE)
        alert = _FakeAlert("warning", "Car to your left.", "car", 2.0, "left")
        pv.enqueue_alert(alert)
        assert pv._queue.empty()

    def test_alerts_enqueued_in_guide_mode(self):
        pv, _ = self._make_pv(VoiceMode.GUIDE)
        alert = _FakeAlert("immediate", "Stop! door ahead.", "door", 0.8, "center")
        pv.enqueue_alert(alert)
        assert pv._queue.qsize() == 1

    def test_drain_on_quiet(self):
        pv, _ = self._make_pv(VoiceMode.GUIDE)
        alert = _FakeAlert("warning", "Car ahead.", "car", 2.0, "center")
        pv.enqueue_alert(alert)
        assert not pv._queue.empty()
        pv.set_mode(VoiceMode.QUIET)
        assert pv._queue.empty()


# ---------------------------------------------------------------------------
# ProactiveVoice: queue priority ordering
# ---------------------------------------------------------------------------

class TestQueuePriority:
    def test_immediate_has_lower_priority_number(self):
        assert _Priority.IMMEDIATE < _Priority.WARNING
        assert _Priority.WARNING < _Priority.QUERY_RESPONSE
        assert _Priority.QUERY_RESPONSE < _Priority.INFORMATIONAL

    def test_immediate_alert_dequeues_before_warning(self):
        pv = ProactiveVoice(_FakeTTS(), mode=VoiceMode.GUIDE, max_queue_size=32)
        warning = _FakeAlert("warning", "Car ahead.", "car", 2.0, "center")
        immediate = _FakeAlert("immediate", "Stop! door.", "door", 0.5, "center")

        pv.enqueue_alert(warning)
        pv.enqueue_alert(immediate)

        first = pv._queue.get_nowait()
        assert first.priority == _Priority.IMMEDIATE.value
        second = pv._queue.get_nowait()
        assert second.priority == _Priority.WARNING.value


# ---------------------------------------------------------------------------
# ProactiveVoice: VAD / user-speaking suppression
# ---------------------------------------------------------------------------

class TestVADSuppression:
    def test_set_user_speaking_flag(self):
        pv = ProactiveVoice(_FakeTTS())
        pv.set_user_speaking(True)
        assert pv._user_speaking is True
        pv.set_user_speaking(False)
        assert pv._user_speaking is False

    @pytest.mark.asyncio
    async def test_non_critical_deferred_while_user_speaking(self):
        """WARNING alerts are skipped (task_done'd) when user is speaking."""
        tts = _FakeTTS()
        pv = ProactiveVoice(tts, mode=VoiceMode.GUIDE)
        pv.set_user_speaking(True)

        alert = _FakeAlert("warning", "Car ahead.", "car", 2.0, "center")
        pv.enqueue_alert(alert)

        # Run one iteration of the loop
        async def _run_once():
            pv._running = True
            try:
                item = await asyncio.wait_for(pv._queue.get(), timeout=0.2)
                if pv._user_speaking and item.priority > _Priority.IMMEDIATE.value:
                    pv._queue.task_done()
                    return "deferred"
                pv._queue.task_done()
                return "spoke"
            except asyncio.TimeoutError:
                return "timeout"

        result = await _run_once()
        assert result == "deferred"
        assert not tts.calls  # TTS was never called


# ---------------------------------------------------------------------------
# ProactiveVoice: speech enrichment with spatial memory
# ---------------------------------------------------------------------------

class TestSpeechEnrichment:
    def test_informational_enriched_with_nearby_object(self):
        tts = _FakeTTS()
        obj = _FakeObject(label="keys", age_seconds=120.0)
        memory = _FakeMemory(objects=[obj])
        pv = ProactiveVoice(tts, memory=memory, mode=VoiceMode.GUIDE)

        alert = _FakeAlert("informational", "Note: bag ahead.", "bag", 4.0, "center")
        pv.enqueue_alert(alert)

        item = pv._queue.get_nowait()
        assert "keys" in item.speech_text
        assert "minutes ago" in item.speech_text

    def test_non_informational_not_enriched(self):
        tts = _FakeTTS()
        obj = _FakeObject(label="keys", age_seconds=30.0)
        memory = _FakeMemory(objects=[obj])
        pv = ProactiveVoice(tts, memory=memory, mode=VoiceMode.GUIDE)

        alert = _FakeAlert("immediate", "Stop! door ahead.", "door", 0.8, "center")
        pv.enqueue_alert(alert)

        item = pv._queue.get_nowait()
        assert item.speech_text == "Stop! door ahead."

    def test_no_enrichment_without_memory(self):
        pv = ProactiveVoice(_FakeTTS(), memory=None, mode=VoiceMode.GUIDE)
        alert = _FakeAlert("informational", "Note: bag ahead.", "bag", 4.0, "center")
        pv.enqueue_alert(alert)

        item = pv._queue.get_nowait()
        assert item.speech_text == "Note: bag ahead."

    def test_no_enrichment_when_no_nearby_objects(self):
        pv = ProactiveVoice(_FakeTTS(), memory=_FakeMemory(objects=[]), mode=VoiceMode.GUIDE)
        alert = _FakeAlert("informational", "Note: bag ahead.", "bag", 4.0, "center")
        pv.enqueue_alert(alert)

        item = pv._queue.get_nowait()
        assert item.speech_text == "Note: bag ahead."


# ---------------------------------------------------------------------------
# ProactiveVoice: scene summary
# ---------------------------------------------------------------------------

class TestSceneSummary:
    def test_scene_summary_no_memory(self):
        pv = ProactiveVoice(_FakeTTS(), memory=None)
        summary = pv._build_scene_summary()
        assert "No spatial memory" in summary

    def test_scene_summary_empty_memory(self):
        pv = ProactiveVoice(_FakeTTS(), memory=_FakeMemory(objects=[]))
        summary = pv._build_scene_summary()
        assert "haven't tracked" in summary

    def test_scene_summary_with_objects(self):
        objects = [
            _FakeObject(label="phone", age_seconds=30.0),
            _FakeObject(label="keys", age_seconds=120.0),
        ]
        pv = ProactiveVoice(_FakeTTS(), memory=_FakeMemory(objects=objects))
        summary = pv._build_scene_summary()
        assert "phone" in summary
        assert "keys" in summary
        assert "Around you" in summary

    def test_scene_summary_caps_at_five_and_reports_more(self):
        objects = [_FakeObject(label=f"obj{i}", age_seconds=float(i)) for i in range(8)]
        pv = ProactiveVoice(_FakeTTS(), memory=_FakeMemory(objects=objects))
        summary = pv._build_scene_summary()
        assert "3 more" in summary


# ---------------------------------------------------------------------------
# ProactiveVoice: handle_mode_command
# ---------------------------------------------------------------------------

class TestHandleModeCommand:
    @pytest.mark.asyncio
    async def test_guide_command_sets_mode(self):
        pv = ProactiveVoice(_FakeTTS(), mode=VoiceMode.REACTIVE)
        response = await pv.handle_mode_command("guide me")
        assert response is not None
        assert pv.mode == VoiceMode.GUIDE
        assert "Guide mode" in response

    @pytest.mark.asyncio
    async def test_quiet_command_sets_mode(self):
        pv = ProactiveVoice(_FakeTTS(), mode=VoiceMode.GUIDE)
        response = await pv.handle_mode_command("quiet mode")
        assert response is not None
        assert pv.mode == VoiceMode.QUIET

    @pytest.mark.asyncio
    async def test_scene_summary_command(self):
        obj = _FakeObject(label="cup", age_seconds=15.0)
        pv = ProactiveVoice(_FakeTTS(), memory=_FakeMemory(objects=[obj]))
        response = await pv.handle_mode_command("what's around me?")
        assert response is not None
        assert "cup" in response
        assert pv.mode != VoiceMode.QUIET  # mode should not change

    @pytest.mark.asyncio
    async def test_non_mode_command_returns_none(self):
        pv = ProactiveVoice(_FakeTTS())
        response = await pv.handle_mode_command("where are my keys?")
        assert response is None


# ---------------------------------------------------------------------------
# voice_pipeline.py: parse_intent mode intents
# ---------------------------------------------------------------------------

class TestParseModeIntentInPipeline:
    def test_guide_me_parsed(self):
        intent = parse_intent("guide me")
        assert intent.kind == "mode_guide"

    def test_quiet_mode_parsed(self):
        intent = parse_intent("quiet mode")
        assert intent.kind == "mode_quiet"

    def test_whats_around_me_parsed(self):
        intent = parse_intent("what's around me?")
        assert intent.kind == "scene_summary"

    def test_where_is_still_works(self):
        intent = parse_intent("where are my keys?")
        assert intent.kind == "where_is"
        assert intent.object_name == "keys"

    def test_mode_takes_priority_over_spatial(self):
        # "guide me to the kitchen" should still be mode_guide not where_is
        intent = parse_intent("guide me")
        assert intent.kind == "mode_guide"


# ---------------------------------------------------------------------------
# ProactiveVoice: run() loop integration (with mocked audio)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_loop_speaks_enqueued_text():
    tts = _FakeTTS(return_bytes=b"audio")
    pv = ProactiveVoice(tts, mode=VoiceMode.REACTIVE)

    played: list[bytes] = []

    async def _fake_play(audio_bytes: bytes) -> None:
        played.append(audio_bytes)

    with patch("demo_spatial.app.proactive_voice._play_async", side_effect=_fake_play):
        run_task = asyncio.create_task(pv.run())
        pv.enqueue_text("Hello, I am your guide.", priority=_Priority.QUERY_RESPONSE)

        # Give the run loop time to process
        await asyncio.sleep(0.2)
        pv.stop()
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass

    assert "Hello, I am your guide." in tts.calls


@pytest.mark.asyncio
async def test_run_loop_immediate_interrupts_warning():
    """Enqueuing an IMMEDIATE alert while a WARNING is playing should cancel it."""
    tts = _FakeTTS(return_bytes=b"audio")
    pv = ProactiveVoice(tts, mode=VoiceMode.GUIDE)

    cancelled: list[str] = []

    async def slow_play(audio_bytes: bytes) -> None:
        await asyncio.sleep(10)  # simulate long-playing audio

    async def fast_play(audio_bytes: bytes) -> None:
        pass

    play_calls = 0

    async def _fake_play(audio_bytes: bytes) -> None:
        nonlocal play_calls
        play_calls += 1
        if play_calls == 1:
            await slow_play(audio_bytes)
        else:
            await fast_play(audio_bytes)

    with patch("demo_spatial.app.proactive_voice._play_async", side_effect=_fake_play):
        run_task = asyncio.create_task(pv.run())

        warning = _FakeAlert("warning", "Car ahead.", "car", 2.0, "center")
        pv.enqueue_alert(warning)
        await asyncio.sleep(0.05)  # let run() start the warning

        immediate = _FakeAlert("immediate", "Stop! door.", "door", 0.4, "center")
        pv.enqueue_alert(immediate)
        await asyncio.sleep(0.3)

        pv.stop()
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass

    # Immediate alert should have been spoken
    assert any("Stop!" in c or "door" in c for c in tts.calls)
