"""Unit tests for ADR-2: Cross-Frame Temporal Consensus Buffer.

Covers:
- Single-frame detections are NOT stored (buffered, not passed to observe())
- Consensus reached on N-th confirmation triggers observe()
- Expired detections are evicted from the buffer
- Averaged position and max confidence used when consensus reached
- High-IoU detections count as same object
- Low-IoU detections treated as different objects
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, call, patch

# Ensure demo_spatial is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.scene_ingestion import SceneIngestion


def _make_ingestion(**kwargs) -> SceneIngestion:
    """Build a SceneIngestion with a mock SpatialMemory and no YOLO/VLM."""
    memory = MagicMock()
    memory.observe = MagicMock(return_value="fake-state-id")
    ingestion = SceneIngestion(
        memory=memory,
        vlm_client=None,
        use_vlm_fallback=False,
        consensus_window_ms=kwargs.pop("consensus_window_ms", 1000),
        consensus_min_count=kwargs.pop("consensus_min_count", 2),
        consensus_iou_threshold=kwargs.pop("consensus_iou_threshold", 0.4),
    )
    return ingestion


# ---------------------------------------------------------------------------
# Single-frame gate
# ---------------------------------------------------------------------------


def test_single_frame_not_stored() -> None:
    """A detection seen only once must NOT be stored (buffered only)."""
    ing = _make_ingestion()
    ing._update_consensus("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=1000, depth_m=None)
    ing._memory.observe.assert_not_called()


# ---------------------------------------------------------------------------
# Consensus threshold
# ---------------------------------------------------------------------------


def test_consensus_reached_triggers_storage() -> None:
    """observe() must be called exactly once when consensus_min_count is reached."""
    ing = _make_ingestion(consensus_min_count=2)
    ts = 1000
    ing._update_consensus("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts, depth_m=None)
    ing._memory.observe.assert_not_called()  # first detection buffered

    ing._update_consensus(
        "cup", cx=0.51, cy=0.51, confidence=0.9, timestamp_ms=ts + 100, depth_m=None
    )
    ing._memory.observe.assert_called_once()


def test_consensus_with_min_count_three() -> None:
    """With consensus_min_count=3, observe() fires only on the 3rd detection."""
    ing = _make_ingestion(consensus_min_count=3)
    ts = 1000
    ing._update_consensus("chair", cx=0.5, cy=0.5, confidence=0.7, timestamp_ms=ts, depth_m=None)
    ing._update_consensus(
        "chair", cx=0.5, cy=0.5, confidence=0.75, timestamp_ms=ts + 100, depth_m=None
    )
    ing._memory.observe.assert_not_called()  # 2 detections — not yet

    ing._update_consensus(
        "chair", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts + 200, depth_m=None
    )
    ing._memory.observe.assert_called_once()


# ---------------------------------------------------------------------------
# Expired detections eviction
# ---------------------------------------------------------------------------


def test_expired_detections_evicted_from_buffer() -> None:
    """Detections outside consensus_window_ms must be evicted and not count."""
    ing = _make_ingestion(consensus_window_ms=500, consensus_min_count=2)

    # First detection at t=0
    ing._update_consensus("keys", cx=0.5, cy=0.5, confidence=0.7, timestamp_ms=0, depth_m=None)
    # Second detection at t=600 — first has expired (window=500ms)
    ing._update_consensus("keys", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=600, depth_m=None)
    # With the expired entry evicted, we only have 1 fresh entry — no consensus
    ing._memory.observe.assert_not_called()


# ---------------------------------------------------------------------------
# Averaged position and max confidence
# ---------------------------------------------------------------------------


def test_consensus_uses_averaged_position_and_max_confidence() -> None:
    """observe() must be called with averaged cx/cy and max confidence."""
    ing = _make_ingestion(consensus_min_count=2)
    ts = 1000
    # Use nearby positions so IoU > threshold (box_size=0.1 default)
    ing._update_consensus("bottle", cx=0.50, cy=0.50, confidence=0.7, timestamp_ms=ts, depth_m=None)
    ing._update_consensus(
        "bottle", cx=0.52, cy=0.52, confidence=0.9, timestamp_ms=ts + 100, depth_m=None
    )

    ing._memory.observe.assert_called_once()
    call_kwargs = ing._memory.observe.call_args
    # Averaged: cx=(0.50+0.52)/2=0.51, cy=(0.50+0.52)/2=0.51; max conf=0.9
    assert abs(call_kwargs.kwargs["cx"] - 0.51) < 1e-6
    assert abs(call_kwargs.kwargs["cy"] - 0.51) < 1e-6
    assert abs(call_kwargs.kwargs["confidence"] - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# IoU matching
# ---------------------------------------------------------------------------


def test_high_iou_overlap_counts_as_same_object() -> None:
    """Two detections with IoU > threshold must count as the same object."""
    ing = _make_ingestion(consensus_iou_threshold=0.4, consensus_min_count=2)
    ts = 1000
    # Nearly identical positions → high IoU
    ing._update_consensus("phone", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts, depth_m=None)
    ing._update_consensus(
        "phone", cx=0.52, cy=0.52, confidence=0.85, timestamp_ms=ts + 50, depth_m=None
    )
    ing._memory.observe.assert_called_once()


def test_low_iou_overlap_treated_as_different_object() -> None:
    """Two detections with IoU <= threshold must be treated as different objects."""
    ing = _make_ingestion(consensus_iou_threshold=0.4, consensus_min_count=2)
    ts = 1000
    # Far apart positions → near-zero IoU
    ing._update_consensus("phone", cx=0.1, cy=0.1, confidence=0.8, timestamp_ms=ts, depth_m=None)
    ing._update_consensus(
        "phone", cx=0.9, cy=0.9, confidence=0.85, timestamp_ms=ts + 50, depth_m=None
    )
    # Both are buffered as different spatial positions, consensus not reached for either
    ing._memory.observe.assert_not_called()


# ---------------------------------------------------------------------------
# IoU helper unit test
# ---------------------------------------------------------------------------


def test_iou_identical_boxes() -> None:
    """IoU of a box with itself is 1.0."""
    assert abs(SceneIngestion._iou(0.5, 0.5, 0.5, 0.5) - 1.0) < 1e-6


def test_iou_non_overlapping_boxes() -> None:
    """IoU of non-overlapping boxes is 0.0."""
    assert SceneIngestion._iou(0.0, 0.0, 1.0, 1.0) == 0.0


def test_iou_partial_overlap() -> None:
    """IoU of boxes with partial overlap is between 0 and 1."""
    val = SceneIngestion._iou(0.5, 0.5, 0.55, 0.55)
    assert 0.0 < val < 1.0


# ---------------------------------------------------------------------------
# Label isolation
# ---------------------------------------------------------------------------


def test_consensus_buffer_isolated_by_label() -> None:
    """Detections for different labels must not count towards each other's consensus."""
    ing = _make_ingestion(consensus_min_count=2)
    ts = 1000
    ing._update_consensus("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts, depth_m=None)
    ing._update_consensus(
        "bottle", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts + 50, depth_m=None
    )
    # Both are different labels — neither has 2 confirmations yet
    ing._memory.observe.assert_not_called()
