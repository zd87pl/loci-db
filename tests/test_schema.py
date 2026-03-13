"""Tests for WorldState dataclass validation."""

from __future__ import annotations

import pytest

from engram.schema import WorldState


def _make(**overrides) -> WorldState:
    defaults = dict(x=0.5, y=0.5, z=0.5, timestamp_ms=1000, vector=[1.0])
    defaults.update(overrides)
    return WorldState(**defaults)


def test_valid_state() -> None:
    s = _make()
    assert s.x == 0.5


@pytest.mark.parametrize("field", ["x", "y", "z"])
def test_spatial_below_zero(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _make(**{field: -0.1})


@pytest.mark.parametrize("field", ["x", "y", "z"])
def test_spatial_above_one(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _make(**{field: 1.1})


def test_confidence_below_zero() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _make(confidence=-0.1)


def test_confidence_above_one() -> None:
    with pytest.raises(ValueError, match="confidence"):
        _make(confidence=1.5)


def test_confidence_boundary_values() -> None:
    _make(confidence=0.0)
    _make(confidence=1.0)


def test_negative_timestamp_rejected() -> None:
    with pytest.raises(ValueError, match="timestamp_ms"):
        _make(timestamp_ms=-1)


def test_zero_timestamp_ok() -> None:
    s = _make(timestamp_ms=0)
    assert s.timestamp_ms == 0


def test_invalid_scale_level() -> None:
    with pytest.raises(ValueError, match="scale_level"):
        _make(scale_level="invalid")


def test_valid_scale_levels() -> None:
    for sl in ("patch", "frame", "sequence"):
        s = _make(scale_level=sl)
        assert s.scale_level == sl
