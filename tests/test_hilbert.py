"""Tests for Hilbert curve encoding — roundtrip, locality, and edge cases."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loci.spatial.hilbert import decode, encode

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


# ------------------------------------------------------------------
# LUT-based query_buckets tests
# ------------------------------------------------------------------


def test_lut_matches_direct_computation() -> None:
    """LUT-based query_buckets produces identical results to itertools path."""
    import itertools
    import math

    from hilbertcurve.hilbertcurve import HilbertCurve

    from loci.spatial.hilbert import HilbertIndex, SpatialBounds, _clamp

    index = HilbertIndex(resolutions=[4])
    bounds = SpatialBounds(
        x_min=0.2, x_max=0.6, y_min=0.3, y_max=0.7,
        z_min=0.0, z_max=1.0, t_min=0.0, t_max=1.0,
    )
    lut_result = index.query_buckets(bounds, resolution=4, overlap_factor=1.2)

    # Compute expected result directly
    curve = HilbertCurve(p=4, n=4)
    side = (1 << 4) - 1
    min_pad = 1.0 / side

    def _expand(lo: float, hi: float) -> tuple[float, float]:
        span = hi - lo
        pad = max(span * 0.1, min_pad)
        return max(0.0, lo - pad), min(1.0, hi + pad)

    x_lo, x_hi = _expand(0.2, 0.6)
    y_lo, y_hi = _expand(0.3, 0.7)
    z_lo, z_hi = _expand(0.0, 1.0)
    t_lo, t_hi = _expand(0.0, 1.0)

    ids: set[int] = set()
    for ix, iy, iz, it in itertools.product(
        range(_clamp(math.floor(x_lo * side), 0, side), _clamp(math.ceil(x_hi * side), 0, side) + 1),
        range(_clamp(math.floor(y_lo * side), 0, side), _clamp(math.ceil(y_hi * side), 0, side) + 1),
        range(_clamp(math.floor(z_lo * side), 0, side), _clamp(math.ceil(z_hi * side), 0, side) + 1),
        range(_clamp(math.floor(t_lo * side), 0, side), _clamp(math.ceil(t_hi * side), 0, side) + 1),
    ):
        ids.add(curve.distance_from_point([ix, iy, iz, it]))

    assert lut_result == sorted(ids)


def test_query_buckets_fallback_without_lut() -> None:
    """query_buckets works for resolutions without a precomputed LUT."""
    from loci.spatial.hilbert import HilbertIndex, SpatialBounds

    index = HilbertIndex(resolutions=[4, 8])
    bounds = SpatialBounds(
        x_min=0.4, x_max=0.6, y_min=0.4, y_max=0.6,
        z_min=0.4, z_max=0.6, t_min=0.4, t_max=0.6,
    )
    result = index.query_buckets(bounds, resolution=8, overlap_factor=1.0)
    assert isinstance(result, list)
    assert len(result) > 0
    assert result == sorted(set(result))


def test_lut_performance() -> None:
    """LUT-based query_buckets completes in under 5ms (was ~47ms)."""
    import time

    from loci.spatial.hilbert import HilbertIndex, SpatialBounds

    index = HilbertIndex(resolutions=[4])
    bounds = SpatialBounds(
        x_min=0.1, x_max=0.9, y_min=0.1, y_max=0.9,
        z_min=0.0, z_max=1.0, t_min=0.0, t_max=1.0,
    )

    # Warm up
    index.query_buckets(bounds, resolution=4)

    start = time.perf_counter()
    for _ in range(100):
        index.query_buckets(bounds, resolution=4)
    elapsed = (time.perf_counter() - start) / 100

    assert elapsed < 0.005, f"query_buckets took {elapsed * 1000:.1f}ms, expected < 5ms"


def test_lut_precomputed_for_small_resolutions() -> None:
    """LUT is precomputed for resolutions with grid side <= 16."""
    from loci.spatial.hilbert import HilbertIndex

    index = HilbertIndex(resolutions=[2, 4, 8])
    assert 2 in index._luts  # 2^2 = 4 <= 16
    assert 4 in index._luts  # 2^4 = 16 <= 16
    assert 8 not in index._luts  # 2^8 = 256 > 16
