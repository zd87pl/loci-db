"""Tests for WorldState dataclass validation."""

from __future__ import annotations

import pytest

from loci.schema import WorldState


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


def test_metadata_defaults_to_empty_dict() -> None:
    s = _make()
    assert s.metadata == {}


def test_metadata_default_not_shared_between_instances() -> None:
    a = _make()
    b = _make()
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_metadata_round_trips_through_every_client_serializer() -> None:
    """Anti-drift guard: all three client serializer copies must carry metadata."""
    import loci.async_client as async_mod
    import loci.local_client as local_mod
    from loci.client import LociClient

    pairs = [
        (LociClient._state_to_payload, LociClient._payload_to_state),
        (async_mod._state_to_payload, async_mod._payload_to_state),
        (local_mod._state_to_payload, local_mod._payload_to_state),
    ]
    state = _make(metadata={"label": "doorway", "n": 3})
    for to_payload, from_payload in pairs:
        payload = to_payload(state, {})
        restored = from_payload(payload, "pid-1", [1.0])
        assert restored.metadata == {"label": "doorway", "n": 3}
        # Old payloads written before the metadata field existed restore to {}.
        legacy = {k: v for k, v in payload.items() if k != "metadata"}
        assert from_payload(legacy, "pid-2", [1.0]).metadata == {}
