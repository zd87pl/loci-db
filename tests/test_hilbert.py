"""Tests for Hilbert curve encoding — roundtrip, locality, and edge cases."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from engram.spatial.hilbert import decode, encode

# ------------------------------------------------------------------
# Roundtrip property test
# ------------------------------------------------------------------


@given(
    x=st.floats(min_value=0.0, max_value=1.0),
    y=st.floats(min_value=0.0, max_value=1.0),
    z=st.floats(min_value=0.0, max_value=1.0),
    t=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_encode_decode_roundtrip(x: float, y: float, z: float, t: float) -> None:
    """encode → decode should recover coordinates within quantisation tolerance."""
    resolution = 4
    hid = encode(x, y, z, t, resolution_order=resolution)
    rx, ry, rz, rt = decode(hid, resolution_order=resolution)
    tol = 1.0 / ((1 << resolution) - 1) + 1e-9
    assert abs(rx - x) <= tol, f"x mismatch: {rx} vs {x}"
    assert abs(ry - y) <= tol, f"y mismatch: {ry} vs {y}"
    assert abs(rz - z) <= tol, f"z mismatch: {rz} vs {z}"
    assert abs(rt - t) <= tol, f"t mismatch: {rt} vs {t}"


# ------------------------------------------------------------------
# Locality preservation
# ------------------------------------------------------------------


def test_nearby_points_have_similar_hilbert_ids() -> None:
    """Points close in 4D space should have Hilbert IDs closer together
    on average than distant points."""
    near_ids = [encode(0.5 + d, 0.5, 0.5, 0.5) for d in [0.0, 0.01, 0.02, 0.03]]
    far_ids = [encode(0.0, 0.0, 0.0, 0.0), encode(1.0, 1.0, 1.0, 1.0)]

    near_spread = max(near_ids) - min(near_ids)
    far_spread = max(far_ids) - min(far_ids)
    assert near_spread <= far_spread


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_origin() -> None:
    hid = encode(0.0, 0.0, 0.0, 0.0)
    assert hid == 0
    assert decode(hid) == (0.0, 0.0, 0.0, 0.0)


def test_max_corner() -> None:
    hid = encode(1.0, 1.0, 1.0, 1.0)
    x, y, z, t = decode(hid)
    assert (x, y, z, t) == (1.0, 1.0, 1.0, 1.0)


def test_deterministic() -> None:
    a = encode(0.3, 0.7, 0.1, 0.9)
    b = encode(0.3, 0.7, 0.1, 0.9)
    assert a == b


def test_different_resolutions() -> None:
    for order in (2, 4, 6):
        hid = encode(0.5, 0.5, 0.5, 0.5, resolution_order=order)
        x, y, z, t = decode(hid, resolution_order=order)
        tol = 1.0 / ((1 << order) - 1) + 1e-9
        assert abs(x - 0.5) <= tol
        assert abs(y - 0.5) <= tol
        assert abs(z - 0.5) <= tol
        assert abs(t - 0.5) <= tol
