"""Tests for Hilbert curve encoding — roundtrip, locality, and edge cases."""

from __future__ import annotations

import os
import random

import pytest
from hilbertcurve.hilbertcurve import HilbertCurve
from hypothesis import given, settings
from hypothesis import strategies as st

import loci.spatial.hilbert as hilbert_module
from loci.spatial.hilbert import HilbertIndex, SpatialBounds, _clamp, decode, encode

try:
    import loci_core as _rust
except ImportError:
    _rust = None

requires_rust = pytest.mark.skipif(_rust is None, reason="loci_core extension not importable")

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
        x_min=0.2,
        x_max=0.6,
        y_min=0.3,
        y_max=0.7,
        z_min=0.0,
        z_max=1.0,
        t_min=0.0,
        t_max=1.0,
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
        range(
            _clamp(math.floor(x_lo * side), 0, side), _clamp(math.ceil(x_hi * side), 0, side) + 1
        ),
        range(
            _clamp(math.floor(y_lo * side), 0, side), _clamp(math.ceil(y_hi * side), 0, side) + 1
        ),
        range(
            _clamp(math.floor(z_lo * side), 0, side), _clamp(math.ceil(z_hi * side), 0, side) + 1
        ),
        range(
            _clamp(math.floor(t_lo * side), 0, side), _clamp(math.ceil(t_hi * side), 0, side) + 1
        ),
    ):
        ids.add(curve.distance_from_point([ix, iy, iz, it]))

    assert lut_result == sorted(ids)


def test_query_buckets_fallback_without_lut() -> None:
    """query_buckets works for resolutions without a precomputed LUT."""
    from loci.spatial.hilbert import HilbertIndex, SpatialBounds

    index = HilbertIndex(resolutions=[4, 8])
    bounds = SpatialBounds(
        x_min=0.4,
        x_max=0.6,
        y_min=0.4,
        y_max=0.6,
        z_min=0.4,
        z_max=0.6,
        t_min=0.4,
        t_max=0.6,
    )
    result = index.query_buckets(bounds, resolution=8, overlap_factor=1.0)
    assert isinstance(result, list)
    assert len(result) > 0
    assert result == sorted(set(result))


@pytest.mark.skipif(
    os.environ.get("LOCI_PERF_TESTS") != "1",
    reason="Wall-clock perf test; opt-in via LOCI_PERF_TESTS=1",
)
def test_lut_performance() -> None:
    """LUT-based query_buckets stays well below the pre-LUT baseline (~47ms).

    This is a regression guard, not a hard SLO. Wall-clock assertions are
    intrinsically flaky on shared CI runners, so the test is opt-in. Set
    ``LOCI_PERF_TESTS=1`` to enable it locally or in a dedicated perf job.
    """
    import time

    from loci.spatial.hilbert import HilbertIndex, SpatialBounds

    index = HilbertIndex(resolutions=[4])
    bounds = SpatialBounds(
        x_min=0.1,
        x_max=0.9,
        y_min=0.1,
        y_max=0.9,
        z_min=0.0,
        z_max=1.0,
        t_min=0.0,
        t_max=1.0,
    )

    # Warm up
    index.query_buckets(bounds, resolution=4)

    start = time.perf_counter()
    for _ in range(100):
        index.query_buckets(bounds, resolution=4)
    elapsed = (time.perf_counter() - start) / 100

    # Generous threshold — 10x the typical measurement is still a clear
    # regression signal versus the 47ms pre-LUT baseline.
    assert elapsed < 0.020, f"query_buckets took {elapsed * 1000:.1f}ms, expected < 20ms"


def test_lut_built_lazily_for_small_resolutions() -> None:
    """LUT is built lazily on first query, only for grid side <= 16.

    Construction must not pay the ~350ms LUT build cost — clients create
    HilbertIndex instances eagerly and only some of them ever run spatial
    queries.
    """
    index = HilbertIndex(resolutions=[2, 4, 8])
    assert index._luts == {}  # nothing built at construction time

    narrow = SpatialBounds(
        x_min=0.5, x_max=0.5, y_min=0.5, y_max=0.5, z_min=0.5, z_max=0.5, t_min=0.5, t_max=0.5
    )
    index.query_buckets(narrow, resolution=2)
    assert 2 in index._luts  # 2^2 = 4 <= 16
    index.query_buckets(narrow, resolution=4)
    assert 4 in index._luts  # 2^4 = 16 <= 16
    index.query_buckets(narrow, resolution=8)
    assert 8 not in index._luts  # 2^8 = 256 > 16: itertools fallback, no LUT


# ------------------------------------------------------------------
# Input validation — both backends must fail identically
# ------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_encode_rejects_non_finite(bad: float) -> None:
    """encode() raises ValueError for NaN/inf on whichever backend is active."""
    with pytest.raises(ValueError, match="finite"):
        encode(bad, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="finite"):
        encode(0.5, 0.5, 0.5, bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_encode_rejects_non_finite_python_fallback(bad: float, monkeypatch) -> None:
    """The pure-Python path raises the same ValueError as the Rust path."""
    monkeypatch.setattr(hilbert_module, "_RUST_AVAILABLE", False)
    with pytest.raises(ValueError, match="finite"):
        encode(0.5, bad, 0.5, 0.5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_hilbert_index_encode_rejects_non_finite(bad: float) -> None:
    index = HilbertIndex(resolutions=[4])
    with pytest.raises(ValueError, match="finite"):
        index.encode(0.5, 0.5, bad, 0.5)


@requires_rust
def test_rust_rejects_invalid_order() -> None:
    """Orders that would truncate the u64 Hilbert distance raise ValueError."""
    with pytest.raises(ValueError, match="order"):
        _rust.encode_hilbert_4d(0.5, 0.5, 0.5, 0.5, 0)
    with pytest.raises(ValueError, match="order"):
        _rust.encode_hilbert_4d(0.5, 0.5, 0.5, 0.5, 17)  # 4 * 17 > 64 bits
    with pytest.raises(ValueError, match="order"):
        _rust.encode_hilbert_3d(0.5, 0.5, 0.5, 22)  # 3 * 22 > 64 bits
    with pytest.raises(ValueError, match="order"):
        _rust.decode_hilbert_3d(0, 0)
    with pytest.raises(ValueError, match="order"):
        _rust.spatial_bounds_to_hilbert_buckets_4d(0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.0, 1.0, 17, 1.2)
    # Maximum valid orders are accepted.
    assert _rust.encode_hilbert_4d(1.0, 1.0, 1.0, 1.0, 16) > 0
    assert _rust.encode_hilbert_3d(1.0, 1.0, 1.0, 21) > 0


@requires_rust
def test_rust_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValueError, match="finite"):
        _rust.encode_hilbert_4d(float("nan"), 0.5, 0.5, 0.5, 4)
    with pytest.raises(ValueError, match="finite"):
        _rust.encode_hilbert_3d(0.5, float("inf"), 0.5, 4)


def test_encode_high_order_falls_back_to_python() -> None:
    """Orders beyond the Rust u64 limit still work via the big-int Python path.

    order=17 needs 68 bits for a 4D Hilbert distance; the Rust extension
    rejects it, so encode() must fall back to pure Python instead of
    silently truncating.
    """
    expected = HilbertCurve(p=17, n=4).distance_from_point([(1 << 17) - 1] * 4)
    assert encode(1.0, 1.0, 1.0, 1.0, resolution_order=17) == int(expected)


# ------------------------------------------------------------------
# Rust <-> Python parity (bit-for-bit)
# ------------------------------------------------------------------


def _python_reference_encode(
    curve: HilbertCurve, side: int, x: float, y: float, z: float, t: float
) -> int:
    """The pure-Python quantise-and-encode reference used by the parity tests."""
    coords = [
        _clamp(int(round(x * side)), 0, side),
        _clamp(int(round(y * side)), 0, side),
        _clamp(int(round(z * side)), 0, side),
        _clamp(int(round(t * side)), 0, side),
    ]
    return int(curve.distance_from_point(coords))


@requires_rust
def test_rust_python_encode_parity_randomized() -> None:
    """Rust and pure-Python encoders agree bit-for-bit over a randomized grid,
    including exact half-cell boundary values where the rounding mode matters."""
    rng = random.Random(20260717)
    for order in (1, 2, 4, 8, 12, 16):
        curve = HilbertCurve(p=order, n=4)
        side = (1 << order) - 1

        points: list[tuple[float, float, float, float]] = [
            tuple(rng.random() for _ in range(4))  # type: ignore[misc]
            for _ in range(64)
        ]
        points += [(0.0,) * 4, (1.0,) * 4, (0.5,) * 4]

        # Half-cell boundary values: x * side lands exactly on k + 0.5 where
        # banker's rounding (Python round) and round-half-away (old Rust
        # f64::round) disagree for even k.
        ties = 0
        for k in range(0, min(side, 64)):
            x = (k + 0.5) / side
            if x * side == k + 0.5:  # exact tie in f64
                ties += 1
                points.append((x, x, x, x))
        assert side == 1 or ties > 0, f"no exact ties generated at order {order}"

        for x, y, z, t in points:
            expected = _python_reference_encode(curve, side, x, y, z, t)
            assert _rust.encode_hilbert_4d(x, y, z, t, order) == expected, (
                f"parity mismatch at order {order} for {(x, y, z, t)}"
            )


@requires_rust
def test_rust_matches_python_bankers_rounding_at_half_boundary() -> None:
    """0.5/side * side == 0.5 exactly; Python round(0.5) == 0 (banker's).

    A half-away-from-zero Rust quantiser would round to cell 1 and produce a
    nonzero Hilbert distance — this is the regression guard for that bug.
    """
    for order in (1, 2, 4, 8, 12, 16):
        side = (1 << order) - 1
        x = 0.5 / side
        assert x * side == 0.5  # exact tie
        assert _rust.encode_hilbert_4d(x, x, x, x, order) == 0
        assert encode(x, x, x, x, resolution_order=order) == 0


@requires_rust
def test_hilbert_index_encode_uses_rust_and_matches_python(monkeypatch) -> None:
    """HilbertIndex.encode routes through Rust and matches the Python path."""
    rng = random.Random(42)
    index = HilbertIndex(resolutions=[4, 8, 12])
    points = [tuple(rng.random() for _ in range(4)) for _ in range(32)]
    points += [(0.0,) * 4, (1.0,) * 4, (0.5,) * 4]

    rust_results = [index.encode(*p) for p in points]
    monkeypatch.setattr(hilbert_module, "_RUST_AVAILABLE", False)
    python_results = [index.encode(*p) for p in points]

    assert rust_results == python_results


@requires_rust
def test_rust_bucket_cover_4d_matches_python_query_buckets() -> None:
    """spatial_bounds_to_hilbert_buckets_4d matches HilbertIndex.query_buckets."""
    index = HilbertIndex(resolutions=[4, 6])
    cases = [
        (SpatialBounds(0.2, 0.6, 0.3, 0.7, 0.0, 1.0, 0.0, 1.0), 4, 1.2),
        (SpatialBounds(0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6), 4, 1.0),
        (SpatialBounds(0.45, 0.55, 0.45, 0.55, 0.45, 0.55, 0.1, 0.2), 6, 1.2),
    ]
    for bounds, order, overlap in cases:
        expected = index.query_buckets(bounds, resolution=order, overlap_factor=overlap)
        got = _rust.spatial_bounds_to_hilbert_buckets_4d(
            bounds.x_min,
            bounds.x_max,
            bounds.y_min,
            bounds.y_max,
            bounds.z_min,
            bounds.z_max,
            bounds.t_min,
            bounds.t_max,
            order,
            overlap,
        )
        assert [int(b) for b in got] == expected
