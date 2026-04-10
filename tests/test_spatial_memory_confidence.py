"""Unit tests for confidence filtering in SpatialMemory.

Covers:
- Storage gate: observe() rejects detections below min_confidence
- Storage gate: observe() stores detections at or above min_confidence
- per-label confidence override via label_confidence_overrides
- Query-time confidence filter via where_is(min_confidence=...)
"""

from __future__ import annotations

import os
import sys

# Ensure demo_spatial is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.spatial_memory import SpatialMemory


def _make_memory(**kwargs) -> SpatialMemory:
    return SpatialMemory(epoch_size_ms=5000, **kwargs)


# ---------------------------------------------------------------------------
# Storage gate tests
# ---------------------------------------------------------------------------


def test_observe_rejects_below_min_confidence() -> None:
    """observe() must return None and not store when confidence < min_confidence."""
    mem = _make_memory()
    result = mem.observe("cup", cx=0.5, cy=0.5, confidence=0.3, min_confidence=0.55)
    assert result is None

    # Confirm nothing was stored by querying
    hits = mem.where_is("cup")
    assert hits == []


def test_observe_stores_at_min_confidence() -> None:
    """observe() must store and return a state_id when confidence >= min_confidence."""
    mem = _make_memory()

    # Exactly at the threshold
    result_at = mem.observe("cup", cx=0.5, cy=0.5, confidence=0.55, min_confidence=0.55)
    assert result_at is not None

    # Above the threshold
    result_above = mem.observe("cup", cx=0.6, cy=0.6, confidence=0.9, min_confidence=0.55)
    assert result_above is not None

    hits = mem.where_is("cup")
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# Per-label override tests
# ---------------------------------------------------------------------------


def test_label_confidence_overrides_respected() -> None:
    """label_confidence_overrides should override the default min_confidence per label."""
    mem = _make_memory(label_confidence_overrides={"phone": 0.3, "keys": 0.8})

    # "phone" has a lower threshold — 0.4 should pass (0.4 >= 0.3)
    phone_result = mem.observe("phone", cx=0.2, cy=0.2, confidence=0.4, min_confidence=0.55)
    assert phone_result is not None, "phone should be stored (override threshold 0.3)"

    # "keys" has a higher threshold — 0.7 should be rejected (0.7 < 0.8)
    keys_result = mem.observe("keys", cx=0.8, cy=0.8, confidence=0.7, min_confidence=0.55)
    assert keys_result is None, "keys should be rejected (override threshold 0.8)"

    # "cup" uses the default min_confidence=0.55; 0.4 should be rejected
    cup_result = mem.observe("cup", cx=0.5, cy=0.5, confidence=0.4, min_confidence=0.55)
    assert cup_result is None, "cup should be rejected (default threshold 0.55)"


# ---------------------------------------------------------------------------
# Query-time confidence filter tests
# ---------------------------------------------------------------------------


def test_query_min_confidence_filter() -> None:
    """where_is(min_confidence=...) must exclude results below the threshold."""
    mem = _make_memory()

    # Store one low-confidence and one high-confidence observation.
    # Use a very low storage gate so both get stored.
    low_id = mem.observe("book", cx=0.1, cy=0.1, confidence=0.4, min_confidence=0.0)
    high_id = mem.observe("book", cx=0.9, cy=0.9, confidence=0.9, min_confidence=0.0)
    assert low_id is not None
    assert high_id is not None

    # Query without filter — both should be returned
    all_hits = mem.where_is("book")
    assert len(all_hits) == 2

    # Query with min_confidence=0.7 — only the high-confidence one should return
    filtered_hits = mem.where_is("book", min_confidence=0.7)
    assert len(filtered_hits) == 1
    assert filtered_hits[0].confidence >= 0.7
