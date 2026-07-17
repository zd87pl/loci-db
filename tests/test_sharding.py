"""Tests for temporal sharding helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from loci.temporal import sharding
from loci.temporal.sharding import collection_name, epoch_id, epochs_in_range


def test_epoch_id_basic() -> None:
    assert epoch_id(0, 5000) == 0
    assert epoch_id(4999, 5000) == 0
    assert epoch_id(5000, 5000) == 1
    assert epoch_id(12345, 5000) == 2


def test_collection_name() -> None:
    assert collection_name(0) == "loci_0"
    assert collection_name(42) == "loci_42"


def test_epochs_in_range_single() -> None:
    assert epochs_in_range(0, 4999, 5000) == [0]


def test_epochs_in_range_multiple() -> None:
    assert epochs_in_range(0, 10000, 5000) == [0, 1, 2]


def test_epochs_in_range_boundary() -> None:
    assert epochs_in_range(5000, 5000, 5000) == [1]


# ---------------------------------------------------------------------------
# Negative timestamps are clamped to 0 (same behaviour on Rust and Python)
# ---------------------------------------------------------------------------


def test_epoch_id_negative_clamped() -> None:
    assert epoch_id(-1, 5000) == 0
    assert epoch_id(-1_000_000, 5000) == 0


def test_epoch_id_negative_clamped_python_fallback() -> None:
    with patch.object(sharding, "_RUST_AVAILABLE", False):
        assert epoch_id(-1, 5000) == 0
        assert epoch_id(-99_999, 5000) == 0


def test_epochs_in_range_negative_clamped() -> None:
    assert epochs_in_range(-10_000, 2_000, 5000) == [0]
    assert epochs_in_range(-10_000, -1, 5000) == [0]


def test_epochs_in_range_negative_clamped_python_fallback() -> None:
    with patch.object(sharding, "_RUST_AVAILABLE", False):
        assert epochs_in_range(-10_000, 2_000, 5000) == [0]
        assert epochs_in_range(-10_000, -1, 5000) == [0]


# ---------------------------------------------------------------------------
# Materialisation cap
# ---------------------------------------------------------------------------


def test_epochs_in_range_raises_beyond_cap() -> None:
    """An all-time window with a small epoch size must not OOM."""
    with pytest.raises(ValueError, match="epoch_size_ms"):
        epochs_in_range(0, 2_000_000_000_000, 5000)  # ~400M epochs


def test_epochs_in_range_cap_python_fallback() -> None:
    with patch.object(sharding, "_RUST_AVAILABLE", False), pytest.raises(ValueError):
        epochs_in_range(0, 2_000_000_000_000, 5000)


def test_epochs_in_range_below_cap_ok() -> None:
    result = epochs_in_range(0, 4_999_999, 5000)
    assert len(result) == 1000
    assert result[0] == 0
    assert result[-1] == 999
