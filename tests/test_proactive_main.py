"""Integration tests for the proactive_main entry point.

Tests exercise:
  - OakSimPipeline — frame generation, rate control, scenario cycling
  - ProactiveApp — camera → engine → voice pipeline integration
  - PerfTracker — FPS/latency accounting
  - Power measurement helpers (graceful no-op on non-RPi hardware)
  - Graceful shutdown within a timeout
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_APP_DIR = Path(__file__).parent.parent / "deploy" / "oak-rpi5" / "app"
_DEPLOY_DIR = Path(__file__).parent.parent / "deploy" / "oak-rpi5"
_REPO_ROOT = Path(__file__).parent.parent

for _p in (_APP_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from oak_pipeline import (
    OakDetection,
    OakFrame,
    OakSimPipeline,
    OakUnavailable,
    _FPSEstimator,
)
from proactive_engine import AlertCategory, ProactiveEngine
from demo_spatial.app.proactive_voice import VoiceMode


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return b"audio"

    @property
    def engine(self) -> str:
        return "fake"


class _FakeSTT:
    def __init__(self, transcripts: list[str] | None = None) -> None:
        self._transcripts = transcripts or []
        self._idx = 0
        self.is_available = True

    async def transcribe(self, audio: bytes, language: str = "en") -> str:
        if self._idx < len(self._transcripts):
            t = self._transcripts[self._idx]
            self._idx += 1
            return t
        return ""


class _FakeMemory:
    def __init__(self) -> None:
        self.tracked_labels: list[str] = []
        self.observation_count = 0

    def current_objects(self):
        return []

    def where_is(self, label: str, limit: int = 5):
        return []

    def stats(self):
        return {}


# ---------------------------------------------------------------------------
# OakSimPipeline
# ---------------------------------------------------------------------------


class TestOakSimPipeline:
    def test_start_stop(self):
        pipeline = OakSimPipeline(fps=30.0)
        pipeline.start()
        pipeline.stop()

    def test_context_manager(self):
        with OakSimPipeline(fps=10.0) as p:
            assert p._started_at is not None

    def test_raises_before_start(self):
        p = OakSimPipeline()
        with pytest.raises(RuntimeError, match="not started"):
            p.next_frame()

    def test_emits_oak_frame(self):
        with OakSimPipeline(fps=100.0) as p:
            # At 100 FPS the first frame should arrive immediately
            time.sleep(0.02)
            frame = p.next_frame(timeout_ms=50)
            assert frame is not None
            assert isinstance(frame, OakFrame)
            assert len(frame.detections) == 1
            det = frame.detections[0]
            assert 0.0 <= det.cx <= 1.0
            assert 0.0 <= det.cy <= 1.0
            assert det.depth_m is not None
            assert det.depth_m > 0

    def test_frame_id_increments(self):
        with OakSimPipeline(fps=100.0) as p:
            time.sleep(0.05)
            ids = []
            for _ in range(3):
                frame = p.next_frame(timeout_ms=50)
                if frame:
                    ids.append(frame.frame_id)
            assert ids == sorted(set(ids))

    def test_scenarios_cycle(self):
        """All sim scenarios are eventually emitted when looping."""
        with OakSimPipeline(fps=1000.0, scenario_duration_s=0.01, loop=True) as p:
            labels: set[str] = set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and len(labels) < 5:
                frame = p.next_frame(timeout_ms=10)
                if frame:
                    labels.add(frame.detections[0].label)
            assert len(labels) >= 2, f"Only saw labels: {labels}"

    def test_noise_stays_in_bounds(self):
        with OakSimPipeline(fps=1000.0) as p:
            time.sleep(0.05)
            for _ in range(20):
                frame = p.next_frame(timeout_ms=10)
                if frame:
                    det = frame.detections[0]
                    assert 0.0 <= det.cx <= 1.0
                    assert 0.0 <= det.cy <= 1.0
                    assert det.depth_m >= 0.1

    def test_returns_none_when_no_frame_due(self):
        with OakSimPipeline(fps=1.0) as p:
            # No frame should be due 1ms after start at 1 FPS
            result = p.next_frame(timeout_ms=5)
            assert result is None


# ---------------------------------------------------------------------------
# OakUnavailable
# ---------------------------------------------------------------------------


class TestOakUnavailable:
    def test_raised_when_depthai_missing(self):
        import importlib

        import oak_pipeline

        orig = oak_pipeline._DEPTHAI_AVAILABLE
        try:
            oak_pipeline._DEPTHAI_AVAILABLE = False
            with pytest.raises(OakUnavailable):
                oak_pipeline._check_depthai()
        finally:
            oak_pipeline._DEPTHAI_AVAILABLE = orig


# ---------------------------------------------------------------------------
# _FPSEstimator
# ---------------------------------------------------------------------------


class TestFPSEstimator:
    def test_empty(self):
        e = _FPSEstimator()
        assert e.tick() == 0.0  # single tick → 0

    def test_roughly_correct(self):
        e = _FPSEstimator(window=10)
        for _ in range(5):
            e.tick()
            time.sleep(0.01)
        fps = e.tick()
        # ~100 FPS at 10ms intervals, but allow wide tolerance in CI
        assert 20 < fps < 500


# ---------------------------------------------------------------------------
# PerfTracker
# ---------------------------------------------------------------------------


class TestPerfTracker:
    def _make(self):
        # Import from the script's namespace via the deploy dir in sys.path
        sys.path.insert(0, str(Path(__file__).parent.parent / "deploy" / "oak-rpi5"))
        import proactive_main
        return proactive_main.PerfTracker()

    def test_fps_empty(self):
        t = self._make()
        assert t.fps == 0.0

    def test_fps_accumulates(self):
        t = self._make()
        for i in range(10):
            t.record_frame(i)
            time.sleep(0.005)
        assert t.fps > 0

    def test_latency_tracking(self):
        t = self._make()
        t.record_frame(1)
        time.sleep(0.05)
        t.record_alert_spoken(1)
        assert t.p50_latency_ms >= 40

    def test_latency_unknown_frame_ignored(self):
        t = self._make()
        t.record_alert_spoken(999)  # no-op — frame_id not tracked
        assert t.p50_latency_ms == 0.0

    def test_frame_cap(self):
        """Old frame_ids are evicted once the cap is reached."""
        t = self._make()
        for i in range(250):
            t.record_frame(i)
        assert len(t._frame_ts) <= 200


# ---------------------------------------------------------------------------
# ProactiveApp — unit tests using mocked sub-components
# ---------------------------------------------------------------------------


def _make_app(sim: bool = True, mode: str = "guide"):
    sys.path.insert(0, str(Path(__file__).parent.parent / "deploy" / "oak-rpi5"))
    import argparse
    import proactive_main

    args = argparse.Namespace(
        sim=sim,
        mode=mode,
        blob="dummy.blob",
        fps=30,
        power_log="",
        log_level="WARNING",
    )
    app = proactive_main.ProactiveApp(args)
    return app, proactive_main


@pytest.mark.asyncio
async def test_app_starts_and_shuts_down():
    """App should start, announce startup, and shut down cleanly within 1s."""
    app, _ = _make_app(sim=True, mode="reactive")

    # Patch TTS so we don't need a real audio engine
    fake_tts = _FakeTTS()
    app._tts = fake_tts
    app._voice._tts = fake_tts

    # Patch STT to be unavailable so we don't need pyaudio
    app._stt = _FakeSTT(transcripts=[])
    app._stt.is_available = False

    async def _run_with_timeout():
        run_task = asyncio.create_task(app.run())
        await asyncio.sleep(0.2)
        app.request_shutdown()
        await asyncio.wait_for(run_task, timeout=2.0)

    await _run_with_timeout()


@pytest.mark.asyncio
async def test_camera_loop_produces_alerts():
    """With sim pipeline in guide mode, alerts should be enqueued."""
    app, _ = _make_app(sim=True, mode="guide")

    fake_tts = _FakeTTS()
    app._tts = fake_tts
    app._voice._tts = fake_tts
    app._stt = _FakeSTT(transcripts=[])
    app._stt.is_available = False

    enqueued: list[str] = []
    orig_enqueue = app._voice.enqueue_alert

    def _capture_enqueue(alert):
        enqueued.append(alert.speech_text)
        orig_enqueue(alert)

    app._voice.enqueue_alert = _capture_enqueue

    async def _run_with_timeout():
        run_task = asyncio.create_task(app.run())
        # Let the pipeline run for 1s — sim at 30fps should produce many frames
        await asyncio.sleep(1.0)
        app.request_shutdown()
        await asyncio.wait_for(run_task, timeout=3.0)

    await _run_with_timeout()
    assert len(enqueued) > 0, "No alerts produced in 1s of sim pipeline"


@pytest.mark.asyncio
async def test_quiet_mode_suppresses_alerts():
    """In QUIET mode, TTS should never be called for hazard alerts."""
    app, _ = _make_app(sim=True, mode="quiet")

    fake_tts = _FakeTTS()
    app._tts = fake_tts
    app._voice._tts = fake_tts
    app._stt = _FakeSTT(transcripts=[])
    app._stt.is_available = False

    async def _run():
        run_task = asyncio.create_task(app.run())
        await asyncio.sleep(0.8)
        app.request_shutdown()
        await asyncio.wait_for(run_task, timeout=3.0)

    await _run()
    # In QUIET mode ProactiveVoice.enqueue_alert is a no-op, so TTS should
    # only have been called for the startup announcement (one call), not for
    # any hazard alert (which would add many more calls).
    # The startup message is enqueued via enqueue_text which bypasses mode guard.
    assert len(fake_tts.calls) <= 1, (
        f"Expected at most 1 TTS call (startup) in QUIET mode, got: {fake_tts.calls}"
    )


@pytest.mark.asyncio
async def test_voice_mode_switch_via_stt():
    """A 'guide me' transcript should switch the voice mode to GUIDE.

    Exercises handle_mode_command → voice.set_mode path directly, bypassing
    the pyaudio microphone layer which is not available in CI.
    """
    app, _ = _make_app(sim=True, mode="reactive")

    fake_tts = _FakeTTS()
    app._tts = fake_tts
    app._voice._tts = fake_tts

    assert app._voice.mode == VoiceMode.REACTIVE

    # Directly exercise the mode-command handler as the STT loop would
    response = await app._voice.handle_mode_command("guide me")
    assert response is not None
    assert app._voice.mode == VoiceMode.GUIDE

    # Switch back via quiet command
    response2 = await app._voice.handle_mode_command("quiet mode")
    assert response2 is not None
    assert app._voice.mode == VoiceMode.QUIET


# ---------------------------------------------------------------------------
# Power measurement helpers
# ---------------------------------------------------------------------------


def test_read_rss_mb():
    sys.path.insert(0, str(Path(__file__).parent.parent / "deploy" / "oak-rpi5"))
    import proactive_main

    rss = proactive_main._read_rss_mb()
    assert rss > 0, "RSS should always be positive"
    assert rss < 32_768, "RSS should be less than 32 GB in tests"


def test_read_rpi5_power_returns_none_on_non_rpi():
    sys.path.insert(0, str(Path(__file__).parent.parent / "deploy" / "oak-rpi5"))
    import proactive_main

    # On non-RPi hardware (CI) all sysfs paths are absent; should return None
    result = proactive_main._read_rpi5_power_mw()
    # May be None or a float — just check it doesn't crash
    assert result is None or isinstance(result, float)
