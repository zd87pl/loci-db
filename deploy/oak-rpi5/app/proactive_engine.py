"""Proactive hazard detection engine for OAK-D Lite spatial awareness.

Consumes a stream of (detections, depth_map) frames from oak_pipeline.py and
emits prioritised Alert objects with spoken text for visually impaired users.

Hazard tiers:
  - IMMEDIATE (<1.5 m): stairs, curbs, doors, poles, people in path
  - WARNING (1.5–3 m): approaching vehicles, cyclists, animals
  - INFORMATIONAL: novel objects above a confidence / novelty threshold

Alert suppression:
  - Per-class cooldown (avoid re-alerting "door" every frame)
  - Distance-band escalation (always re-alert when object moves closer)
  - Objects in upper frame edge (overhead / already passed) are suppressed
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Hazard taxonomy
# ---------------------------------------------------------------------------

IMMEDIATE_CLASSES: frozenset[str] = frozenset(
    {"stairs", "curb", "door", "pole", "person", "traffic light", "fire hydrant", "bollard"}
)
WARNING_CLASSES: frozenset[str] = frozenset(
    {"car", "truck", "bus", "bicycle", "motorcycle", "dog", "cat"}
)

# Distance thresholds (metres)
DIST_CRITICAL = 0.5
DIST_IMMEDIATE = 1.5
DIST_WARNING = 3.0

# Novelty score threshold for informational alerts
NOVELTY_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Cooldown configuration
# ---------------------------------------------------------------------------

DEFAULT_COOLDOWN_S: dict[str, float] = {
    "immediate": 5.0,
    "warning": 10.0,
    "informational": 30.0,
}

# Per-class overrides (seconds). Classes not listed fall back to DEFAULT_COOLDOWN_S.
CLASS_COOLDOWN_S: dict[str, float] = {
    "stairs":        8.0,
    "curb":          8.0,
    "door":         10.0,
    "pole":          8.0,
    "person":        5.0,
    "car":           5.0,
    "truck":         5.0,
    "bus":           5.0,
    "bicycle":       6.0,
    "motorcycle":    5.0,
    "dog":           8.0,
    "cat":          10.0,
}

# ---------------------------------------------------------------------------
# Path geometry
# ---------------------------------------------------------------------------

# "In the path" = center 40% of frame width (0.30–0.70 normalised)
PATH_X_MIN = 0.30
PATH_X_MAX = 0.70

# Objects in the upper 15% of frame are overhead/already passed
OVERHEAD_CY_THRESHOLD = 0.15

# Objects in the bottom 10% of frame are very close to feet / effectively passed
PASSED_CY_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class AlertCategory(str, Enum):
    IMMEDIATE = "immediate"
    WARNING = "warning"
    INFORMATIONAL = "informational"


class Direction(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    BEHIND = "behind"


@dataclass
class Alert:
    """A single hazard alert ready to be spoken or logged."""

    category: AlertCategory
    label: str
    distance_m: float
    direction: Direction
    priority: float      # higher = more urgent; sort descending
    speech_text: str
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


# ---------------------------------------------------------------------------
# Internal cooldown tracker
# ---------------------------------------------------------------------------


@dataclass
class _CooldownEntry:
    last_alerted_ms: int
    last_distance_m: float


class _CooldownTracker:
    """Per-class cooldown with distance-band escalation."""

    def __init__(self) -> None:
        self._entries: dict[str, _CooldownEntry] = {}

    def should_alert(
        self,
        label: str,
        distance_m: float,
        category: AlertCategory,
        now_ms: int,
    ) -> bool:
        entry = self._entries.get(label)
        if entry is None:
            return True

        cooldown_s = CLASS_COOLDOWN_S.get(
            label, DEFAULT_COOLDOWN_S.get(category.value, 10.0)
        )
        elapsed_s = (now_ms - entry.last_alerted_ms) / 1000.0

        if elapsed_s >= cooldown_s:
            return True

        # Escalate immediately when object enters a closer distance band
        if _closer_band(distance_m, entry.last_distance_m):
            return True

        return False

    def record(self, label: str, distance_m: float, now_ms: int) -> None:
        self._entries[label] = _CooldownEntry(
            last_alerted_ms=now_ms,
            last_distance_m=distance_m,
        )


def _closer_band(current_m: float, previous_m: float) -> bool:
    """Return True when the object has crossed into a nearer distance band."""
    thresholds = (DIST_CRITICAL, DIST_IMMEDIATE, DIST_WARNING)

    def band_idx(d: float) -> int:
        for i, t in enumerate(thresholds):
            if d <= t:
                return i
        return len(thresholds)

    return band_idx(current_m) < band_idx(previous_m)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ProactiveEngine:
    """Evaluate OAK-D Lite frames and emit spoken hazard alerts.

    Typical usage::

        engine = ProactiveEngine()
        for frame in pipeline.frames():
            alerts = engine.process(frame.detections, frame.depth_map)
            for alert in alerts:
                tts.speak(alert.speech_text)

    The engine is stateful (tracks per-class cooldowns across frames) and is
    not thread-safe. Create one instance per pipeline thread.
    """

    def __init__(
        self,
        path_x_min: float = PATH_X_MIN,
        path_x_max: float = PATH_X_MAX,
        max_alerts_per_frame: int = 3,
    ) -> None:
        self._path_x_min = path_x_min
        self._path_x_max = path_x_max
        self._max_alerts = max_alerts_per_frame
        self._cooldown = _CooldownTracker()

    def process(
        self,
        detections: list,
        depth_map: np.ndarray | None = None,
        now_ms: int | None = None,
    ) -> list[Alert]:
        """Process one frame of detections and return prioritised alerts.

        Parameters
        ----------
        detections:
            List of OakDetection objects (from oak_pipeline.py). Each must
            have ``label``, ``cx``, ``cy``, and ``depth_m`` attributes.
        depth_map:
            Optional HxW uint16 depth array (millimetres). Passed through for
            future use; depth values are already in detections from the
            pipeline.
        now_ms:
            Current epoch time in milliseconds. Defaults to ``time.time()``.

        Returns
        -------
        list[Alert]
            Up to ``max_alerts_per_frame`` alerts, sorted by priority
            (highest first).
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        candidates: list[Alert] = []

        for det in detections:
            if det.depth_m is None:
                continue

            if self._is_suppressed(det):
                continue

            category = self._classify(det)
            if category is None:
                continue

            direction = self._direction(det)

            # Off-path immediate hazards are demoted to warning priority
            effective_category = category
            if category == AlertCategory.IMMEDIATE and not self._is_in_path(det):
                effective_category = AlertCategory.WARNING

            priority = _compute_priority(det.depth_m, effective_category)

            if not self._cooldown.should_alert(det.label, det.depth_m, effective_category, now_ms):
                continue

            speech = _build_speech(det.label, det.depth_m, effective_category, direction)
            candidates.append(Alert(
                category=effective_category,
                label=det.label,
                distance_m=det.depth_m,
                direction=direction,
                priority=priority,
                speech_text=speech,
                timestamp_ms=now_ms,
            ))

        # Rank by priority, cap at max_alerts_per_frame
        candidates.sort(key=lambda a: a.priority, reverse=True)
        alerts = candidates[: self._max_alerts]

        for alert in alerts:
            self._cooldown.record(alert.label, alert.distance_m, now_ms)

        return alerts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(self, det) -> AlertCategory | None:
        """Return the alert tier for a detection, or None to skip."""
        d = det.depth_m
        label = det.label.lower()

        if label in IMMEDIATE_CLASSES:
            if d <= DIST_IMMEDIATE:
                return AlertCategory.IMMEDIATE
            if d <= DIST_WARNING:
                return AlertCategory.WARNING
        elif label in WARNING_CLASSES:
            if d <= DIST_WARNING:
                return AlertCategory.WARNING

        return None

    def _is_in_path(self, det) -> bool:
        """True if detection center is within the center 40% of frame width."""
        return self._path_x_min <= det.cx <= self._path_x_max

    @staticmethod
    def _is_suppressed(det) -> bool:
        """True for objects that are overhead or already passed the user."""
        return det.cy < OVERHEAD_CY_THRESHOLD or det.cy > PASSED_CY_THRESHOLD

    def _direction(self, det) -> Direction:
        if det.cy > PASSED_CY_THRESHOLD:
            return Direction.BEHIND
        if det.cx < self._path_x_min:
            return Direction.LEFT
        if det.cx > self._path_x_max:
            return Direction.RIGHT
        return Direction.CENTER


# ---------------------------------------------------------------------------
# Pure functions (easier to test independently)
# ---------------------------------------------------------------------------


def _compute_priority(distance_m: float, category: AlertCategory) -> float:
    """Priority = severity × 1/distance. Higher value = more urgent."""
    severity = {
        AlertCategory.IMMEDIATE: 3.0,
        AlertCategory.WARNING: 2.0,
        AlertCategory.INFORMATIONAL: 1.0,
    }[category]
    return severity / max(distance_m, 0.1)


def _build_speech(
    label: str,
    distance_m: float,
    category: AlertCategory,
    direction: Direction,
) -> str:
    """Generate a concise spoken alert string."""
    if distance_m >= 1.0:
        dist_str = f"{distance_m:.1f} metres"
    else:
        dist_str = f"{int(distance_m * 100)} centimetres"

    dir_str = {
        Direction.LEFT:   "to your left",
        Direction.RIGHT:  "to your right",
        Direction.CENTER: "ahead",
        Direction.BEHIND: "behind you",
    }[direction]

    if category == AlertCategory.IMMEDIATE:
        if distance_m <= DIST_CRITICAL:
            return f"Stop! {label} {dir_str}, {dist_str}."
        return f"Caution: {label} {dir_str}, {dist_str}."

    if category == AlertCategory.WARNING:
        return f"{label.capitalize()} {dir_str}, {dist_str}."

    return f"Note: {label} {dir_str}."
