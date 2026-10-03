"""Tests for the proactive hazard detection engine."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

# Allow importing proactive_engine without the full depthai stack
sys.path.insert(0, str(Path(__file__).parent.parent / "deploy" / "oak-rpi5" / "app"))

from proactive_engine import (
    DIST_CRITICAL,
    DIST_IMMEDIATE,
    DIST_WARNING,
    AlertCategory,
    Direction,
    ProactiveEngine,
    _CooldownTracker,
    _build_speech,
    _closer_band,
    _compute_priority,
)


# ---------------------------------------------------------------------------
# Minimal detection stub — mirrors OakDetection without the depthai dep
# ---------------------------------------------------------------------------


@dataclass
class Det:
    label: str
    cx: float
    cy: float
    depth_m: float | None
    confidence: float = 0.9
    label_id: int = 0
    width: float = 0.1
    height: float = 0.1


# ---------------------------------------------------------------------------
# _closer_band
# ---------------------------------------------------------------------------


class TestCloserBand:
    def test_same_band_returns_false(self):
        assert not _closer_band(1.0, 1.2)

    def test_crossed_into_immediate(self):
        # was at 2m (WARNING band), now at 1.0m (IMMEDIATE band)
        assert _closer_band(1.0, 2.0)

    def test_crossed_into_critical(self):
        assert _closer_band(0.4, 1.0)

    def test_moved_further_returns_false(self):
        assert not _closer_band(2.0, 1.0)


# ---------------------------------------------------------------------------
# _compute_priority
# ---------------------------------------------------------------------------


class TestComputePriority:
    def test_immediate_higher_than_warning_same_distance(self):
        p_imm = _compute_priority(1.0, AlertCategory.IMMEDIATE)
        p_warn = _compute_priority(1.0, AlertCategory.WARNING)
        assert p_imm > p_warn

    def test_closer_object_higher_priority(self):
        p_close = _compute_priority(0.5, AlertCategory.WARNING)
        p_far = _compute_priority(2.5, AlertCategory.WARNING)
        assert p_close > p_far

    def test_zero_distance_clamped(self):
        # Should not raise ZeroDivisionError
        p = _compute_priority(0.0, AlertCategory.IMMEDIATE)
        assert p == 3.0 / 0.1


# ---------------------------------------------------------------------------
# _build_speech
# ---------------------------------------------------------------------------


class TestBuildSpeech:
    def test_immediate_critical(self):
        s = _build_speech("door", 0.3, AlertCategory.IMMEDIATE, Direction.CENTER)
        assert s.startswith("Stop!")

    def test_immediate_non_critical(self):
        s = _build_speech("pole", 1.0, AlertCategory.IMMEDIATE, Direction.LEFT)
        assert "Caution" in s
        assert "left" in s

    def test_warning(self):
        s = _build_speech("car", 2.0, AlertCategory.WARNING, Direction.RIGHT)
        assert "right" in s
        assert "Car" in s

    def test_distance_under_one_metre_in_cm(self):
        s = _build_speech("curb", 0.8, AlertCategory.IMMEDIATE, Direction.CENTER)
        assert "centimetres" in s

    def test_distance_over_one_metre(self):
        s = _build_speech("person", 1.5, AlertCategory.WARNING, Direction.CENTER)
        assert "metres" in s


# ---------------------------------------------------------------------------
# _CooldownTracker
# ---------------------------------------------------------------------------


class TestCooldownTracker:
    def test_first_alert_always_passes(self):
        tracker = _CooldownTracker()
        assert tracker.should_alert("door", 1.0, AlertCategory.IMMEDIATE, 0)

    def test_within_cooldown_suppressed(self):
        tracker = _CooldownTracker()
        t0 = 0
        tracker.record("door", 1.0, t0)
        # 1 second later — door cooldown is 10s
        assert not tracker.should_alert("door", 1.0, AlertCategory.IMMEDIATE, t0 + 1_000)

    def test_after_cooldown_passes(self):
        tracker = _CooldownTracker()
        t0 = 0
        tracker.record("door", 1.0, t0)
        assert tracker.should_alert("door", 1.0, AlertCategory.IMMEDIATE, t0 + 11_000)

    def test_distance_escalation_overrides_cooldown(self):
        tracker = _CooldownTracker()
        t0 = 0
        tracker.record("car", 2.5, t0)  # Warning band
        # 1 second later but object now in Immediate band — should escalate
        assert tracker.should_alert("car", 1.0, AlertCategory.WARNING, t0 + 1_000)

    def test_no_escalation_moving_further(self):
        tracker = _CooldownTracker()
        t0 = 0
        tracker.record("car", 1.0, t0)  # Immediate band
        # Object now further — no escalation, still in cooldown
        assert not tracker.should_alert("car", 2.5, AlertCategory.WARNING, t0 + 1_000)


# ---------------------------------------------------------------------------
# ProactiveEngine
# ---------------------------------------------------------------------------


class TestProactiveEngine:
    def _engine(self) -> ProactiveEngine:
        return ProactiveEngine(max_alerts_per_frame=5)

    def test_no_detections_returns_empty(self):
        eng = self._engine()
        assert eng.process([]) == []

    def test_detection_without_depth_skipped(self):
        eng = self._engine()
        det = Det(label="door", cx=0.5, cy=0.5, depth_m=None)
        assert eng.process([det]) == []

    def test_immediate_in_path(self):
        eng = self._engine()
        det = Det(label="door", cx=0.5, cy=0.5, depth_m=1.0)
        alerts = eng.process([det])
        assert len(alerts) == 1
        assert alerts[0].category == AlertCategory.IMMEDIATE

    def test_door_off_path_demoted_to_warning(self):
        eng = self._engine()
        det = Det(label="door", cx=0.1, cy=0.5, depth_m=1.0)  # far left
        alerts = eng.process([det])
        assert len(alerts) == 1
        assert alerts[0].category == AlertCategory.WARNING

    def test_warning_class_car(self):
        eng = self._engine()
        det = Det(label="car", cx=0.5, cy=0.5, depth_m=2.5)
        alerts = eng.process([det])
        assert len(alerts) == 1
        assert alerts[0].category == AlertCategory.WARNING

    def test_car_beyond_warning_range_ignored(self):
        eng = self._engine()
        det = Det(label="car", cx=0.5, cy=0.5, depth_m=3.5)
        assert eng.process([det]) == []

    def test_overhead_object_suppressed(self):
        eng = self._engine()
        det = Det(label="person", cx=0.5, cy=0.05, depth_m=1.0)  # cy < 0.15
        assert eng.process([det]) == []

    def test_passed_object_suppressed(self):
        eng = self._engine()
        det = Det(label="pole", cx=0.5, cy=0.95, depth_m=0.5)  # cy > 0.90
        assert eng.process([det]) == []

    def test_priority_order_respected(self):
        eng = self._engine()
        close_person = Det(label="person", cx=0.5, cy=0.5, depth_m=0.8)
        far_car = Det(label="car", cx=0.5, cy=0.5, depth_m=2.8)
        alerts = eng.process([far_car, close_person])
        assert len(alerts) == 2
        assert alerts[0].label == "person"

    def test_cooldown_suppresses_repeat(self):
        eng = self._engine()
        t0 = int(time.time() * 1000)
        det = Det(label="door", cx=0.5, cy=0.5, depth_m=1.0)
        alerts1 = eng.process([det], now_ms=t0)
        assert len(alerts1) == 1
        # 2 seconds later — still in cooldown (door = 10s)
        alerts2 = eng.process([det], now_ms=t0 + 2_000)
        assert len(alerts2) == 0

    def test_escalation_fires_within_cooldown(self):
        eng = self._engine()
        t0 = int(time.time() * 1000)
        det1 = Det(label="car", cx=0.5, cy=0.5, depth_m=2.5)  # Warning band
        eng.process([det1], now_ms=t0)
        det2 = Det(label="car", cx=0.5, cy=0.5, depth_m=1.0)  # Immediate band
        alerts = eng.process([det2], now_ms=t0 + 1_000)
        assert len(alerts) == 1

    def test_max_alerts_capped(self):
        eng = ProactiveEngine(max_alerts_per_frame=2)
        dets = [
            Det(label="door",   cx=0.5, cy=0.5, depth_m=0.8),
            Det(label="pole",   cx=0.5, cy=0.5, depth_m=1.0),
            Det(label="person", cx=0.5, cy=0.5, depth_m=1.2),
        ]
        alerts = eng.process(dets)
        assert len(alerts) == 2

    def test_direction_left_right_center(self):
        eng = self._engine()
        for cx, expected_dir in [(0.1, Direction.LEFT), (0.5, Direction.CENTER), (0.9, Direction.RIGHT)]:
            det = Det(label="pole", cx=cx, cy=0.5, depth_m=1.0)
            alerts = eng.process([det])
            if alerts:
                assert alerts[0].direction == expected_dir

    def test_speech_text_present(self):
        eng = self._engine()
        det = Det(label="stairs", cx=0.5, cy=0.5, depth_m=1.0)
        alerts = eng.process([det])
        assert len(alerts) == 1
        assert len(alerts[0].speech_text) > 0

    def test_critical_distance_says_stop(self):
        eng = self._engine()
        det = Det(label="stairs", cx=0.5, cy=0.5, depth_m=0.4)
        alerts = eng.process([det])
        assert "Stop" in alerts[0].speech_text
