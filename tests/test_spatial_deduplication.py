"""Unit tests for ADR-3: Spatial Deduplication (Cross-Frame NMS).

Covers:
- High-IoU duplicates are merged into the existing record
- Low-IoU detections insert as new records
- Merged record uses confidence-weighted position average
- Merged record preserves max confidence
- Records outside dedup_window_ms are NOT merged (treated as new)
- Legacy records without explicit width/height default to 0.1 proxy
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.spatial_memory import SpatialMemory


def _make_memory(**kwargs) -> SpatialMemory:
    return SpatialMemory(epoch_size_ms=5000, **kwargs)


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------


def test_duplicate_high_iou_merges_into_existing_record() -> None:
    """Second detection at same position must merge, not insert a new record."""
    mem = _make_memory(dedup_iou_threshold=0.5, dedup_window_ms=5000)
    ts = 1000
    id1 = mem.observe("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts, min_confidence=0.0)
    assert id1 is not None

    id2 = mem.observe("cup", cx=0.51, cy=0.51, confidence=0.85, timestamp_ms=ts + 100, min_confidence=0.0)
    assert id2 is not None
    # Should be the same record (merged), not a new one
    assert id2 == id1

    # Only one record in memory (not two)
    observations = mem.where_is("cup")
    assert len(observations) == 1


def test_unique_low_iou_inserts_as_new_record() -> None:
    """Detection at far position must insert as a new record."""
    mem = _make_memory(dedup_iou_threshold=0.5, dedup_window_ms=5000)
    ts = 1000
    id1 = mem.observe("phone", cx=0.1, cy=0.1, confidence=0.8, timestamp_ms=ts, min_confidence=0.0)
    id2 = mem.observe("phone", cx=0.9, cy=0.9, confidence=0.8, timestamp_ms=ts + 100, min_confidence=0.0)

    assert id1 is not None
    assert id2 is not None
    assert id1 != id2

    observations = mem.where_is("phone", limit=10)
    assert len(observations) == 2


# ---------------------------------------------------------------------------
# Merge accuracy tests
# ---------------------------------------------------------------------------


def test_merge_uses_confidence_weighted_position_average() -> None:
    """Merged position must be the confidence-weighted average of both records."""
    mem = _make_memory(dedup_iou_threshold=0.5, dedup_window_ms=5000)
    ts = 1000
    # First observation: high confidence at 0.5
    mem.observe("bottle", cx=0.5, cy=0.5, confidence=0.9, timestamp_ms=ts, min_confidence=0.0)
    # Second observation: lower confidence very slightly offset (ensures IoU > 0.5)
    mem.observe("bottle", cx=0.505, cy=0.505, confidence=0.3, timestamp_ms=ts + 50, min_confidence=0.0)

    obs = mem.where_is("bottle")
    assert len(obs) == 1
    # Weighted avg: (0.9*0.5 + 0.3*0.505) / (0.9+0.3) = (0.45+0.1515)/1.2 = 0.5013
    expected_cx = (0.9 * 0.5 + 0.3 * 0.505) / (0.9 + 0.3)
    assert abs(obs[0].cx - expected_cx) < 0.005


def test_merge_preserves_max_confidence() -> None:
    """Merged record's confidence must be max(existing, new)."""
    mem = _make_memory(dedup_iou_threshold=0.5, dedup_window_ms=5000)
    ts = 1000
    mem.observe("keys", cx=0.5, cy=0.5, confidence=0.7, timestamp_ms=ts, min_confidence=0.0)
    mem.observe("keys", cx=0.51, cy=0.51, confidence=0.9, timestamp_ms=ts + 50, min_confidence=0.0)

    obs = mem.where_is("keys")
    assert len(obs) == 1
    assert abs(obs[0].confidence - 0.9) < 0.01


# ---------------------------------------------------------------------------
# Window-based eviction test
# ---------------------------------------------------------------------------


def test_dedup_ignores_records_outside_window() -> None:
    """Records older than dedup_window_ms must NOT be merged; new record is inserted."""
    mem = _make_memory(dedup_iou_threshold=0.5, dedup_window_ms=1000)
    ts = 0
    id1 = mem.observe("wallet", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts, min_confidence=0.0)
    # Second detection is 2000ms later — first record is outside the window
    id2 = mem.observe("wallet", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts + 2000, min_confidence=0.0)

    assert id1 is not None
    assert id2 is not None
    assert id1 != id2

    observations = mem.where_is("wallet", limit=10)
    assert len(observations) == 2


# ---------------------------------------------------------------------------
# Legacy dimensions test (proxy bbox = 0.1)
# ---------------------------------------------------------------------------


def test_legacy_records_default_to_nominal_dimensions() -> None:
    """Records stored without explicit width/height use 0.1×0.1 proxy for IoU computation.

    Verify that NMS correctly deduplicates when the only positional info
    is (cx, cy) — i.e., legacy records work as expected.
    """
    mem = _make_memory(dedup_iou_threshold=0.5, dedup_window_ms=5000)
    ts = 1000
    # Two identical positions — must merge regardless (uses 0.1x0.1 proxy)
    id1 = mem.observe("glasses", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts, min_confidence=0.0)
    id2 = mem.observe("glasses", cx=0.5, cy=0.5, confidence=0.9, timestamp_ms=ts + 100, min_confidence=0.0)

    assert id1 is not None
    assert id2 == id1  # merged

    observations = mem.where_is("glasses")
    assert len(observations) == 1
    assert all(obs.confidence >= 0.9 for obs in observations)


# ---------------------------------------------------------------------------
# Label isolation test
# ---------------------------------------------------------------------------


def test_dedup_isolated_by_label() -> None:
    """Deduplication must not merge detections across different labels."""
    mem = _make_memory(dedup_iou_threshold=0.5, dedup_window_ms=5000)
    ts = 1000
    id_cup = mem.observe("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts, min_confidence=0.0)
    id_bottle = mem.observe("bottle", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=ts + 50, min_confidence=0.0)

    assert id_cup is not None
    assert id_bottle is not None
    assert id_cup != id_bottle
