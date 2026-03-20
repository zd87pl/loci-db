"""Tests for temporal sharding helpers."""

from __future__ import annotations

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
