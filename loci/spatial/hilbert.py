"""Multi-resolution Hilbert curve encoding for (x, y, z, t) coordinates.

Encodes a normalised 4D coordinate at multiple Hilbert resolutions for
efficient spatial pre-filtering in Qdrant.  Points near bucket boundaries
are captured by the overlap_factor in query_buckets().
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve


def _make_curve(resolution_order: int) -> HilbertCurve:
    """Create a 4-dimensional Hilbert curve with the given resolution."""
    return HilbertCurve(p=resolution_order, n=4)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class SpatialBounds:
    """Axis-aligned bounding box in normalised [0, 1] coordinates."""

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> SpatialBounds:
        return cls(
            x_min=d.get("x_min", 0.0),
            x_max=d.get("x_max", 1.0),
            y_min=d.get("y_min", 0.0),
            y_max=d.get("y_max", 1.0),
            z_min=d.get("z_min", 0.0),
            z_max=d.get("z_max", 1.0),
            t_min=d.get("t_min", 0.0),
            t_max=d.get("t_max", 1.0),
        )


class HilbertIndex:
    """Multi-resolution Hilbert encoding.

    Each point is encoded at multiple resolutions:
    - p=4:  coarse grid (16^4 cells)    — fast, low precision
    - p=8:  medium grid (256^4 cells)   — balanced
    - p=12: fine grid (4096^4 cells)    — slow, high precision

    Stored as payload: hilbert_r4, hilbert_r8, hilbert_r12

    Query filtering uses the coarsest resolution (p=4) by default,
    which produces manageable MatchAny filter sets (<100 IDs for
    typical spatial queries).  Higher resolutions are stored for
    payload-level post-filtering and narrow point lookups.
    """

    _LUT_MAX_SIDE = 16  # Precompute LUT for resolutions where 2^p <= this

    def __init__(self, resolutions: list[int] | None = None) -> None:
        self.resolutions = resolutions or [4, 8, 12]
        self._curves: dict[int, HilbertCurve] = {r: _make_curve(r) for r in self.resolutions}
        self._luts: dict[int, np.ndarray] = {}
        for r in self.resolutions:
            side = 1 << r
            if side <= self._LUT_MAX_SIDE:
                self._luts[r] = self._build_lut(r)

    def _build_lut(self, resolution: int) -> np.ndarray:
        """Precompute the full (x, y, z, t) → hilbert_distance LUT for *resolution*."""
        curve = self._curves[resolution]
        side = 1 << resolution
        lut = np.empty((side, side, side, side), dtype=np.uint32)
        for ix in range(side):
            for iy in range(side):
                for iz in range(side):
                    for it in range(side):
                        lut[ix, iy, iz, it] = curve.distance_from_point([ix, iy, iz, it])
        return lut

    def encode(
        self,
        x: float,
        y: float,
        z: float,
        t_normalized: float,
    ) -> dict[str, int]:
        """Encode a point at all configured resolutions.

        Returns:
            Dict with keys like ``hilbert_r4``, ``hilbert_r8``, etc.
        """
        result: dict[str, int] = {}
        for r in self.resolutions:
            curve = self._curves[r]
            side = (1 << r) - 1
            coords = [
                _clamp(int(round(x * side)), 0, side),
                _clamp(int(round(y * side)), 0, side),
                _clamp(int(round(z * side)), 0, side),
                _clamp(int(round(t_normalized * side)), 0, side),
            ]
            result[f"hilbert_r{r}"] = int(curve.distance_from_point(coords))
        return result

    def query_buckets(
        self,
        bounds: SpatialBounds | dict,
        resolution: int | None = None,
        overlap_factor: float = 1.2,
    ) -> list[int]:
        """Return Hilbert IDs covering the spatial bounds at the given resolution.

        The bounds are expanded by ``overlap_factor`` to catch points near
        bucket boundaries.  overlap_factor=1.2 means a 20% larger search
        region, followed by exact post-filter on returned results.

        Args:
            bounds: Spatial bounding box.
            resolution: Hilbert resolution to use (default: lowest configured).
            overlap_factor: Multiplier for expanding query region (>=1.0).

        Returns:
            Sorted list of unique Hilbert IDs.
        """
        res, curve, ranges = self._expanded_index_ranges(bounds, resolution, overlap_factor)
        (ix_lo, ix_hi), (iy_lo, iy_hi), (iz_lo, iz_hi), (it_lo, it_hi) = ranges

        if res in self._luts:
            sub = self._luts[res][
                ix_lo : ix_hi + 1,
                iy_lo : iy_hi + 1,
                iz_lo : iz_hi + 1,
                it_lo : it_hi + 1,
            ]
            return [int(x) for x in np.unique(sub)]

        # Fallback for resolutions without a precomputed LUT
        ids: set[int] = set()
        for ix, iy, iz, it in itertools.product(
            range(ix_lo, ix_hi + 1),
            range(iy_lo, iy_hi + 1),
            range(iz_lo, iz_hi + 1),
            range(it_lo, it_hi + 1),
        ):
            ids.add(curve.distance_from_point([ix, iy, iz, it]))

        return sorted(ids)

    def estimated_bucket_count(
        self,
        bounds: SpatialBounds | dict,
        resolution: int | None = None,
        overlap_factor: float = 1.2,
    ) -> int:
        """Estimate how many Hilbert buckets query_buckets() would enumerate."""
        _, _, ranges = self._expanded_index_ranges(bounds, resolution, overlap_factor)
        total = 1
        for lo, hi in ranges:
            total *= hi - lo + 1
        return total

    def payload_field(self, resolution: int | None = None) -> str:
        """Return the payload field name for the given resolution."""
        res = resolution if resolution is not None else self.resolutions[0]
        return f"hilbert_r{res}"

    def _expanded_index_ranges(
        self,
        bounds: SpatialBounds | dict,
        resolution: int | None = None,
        overlap_factor: float = 1.2,
    ) -> tuple[int, HilbertCurve, tuple[tuple[int, int], ...]]:
        if isinstance(bounds, dict):
            bounds = SpatialBounds.from_dict(bounds)

        res = resolution if resolution is not None else self.resolutions[0]
        if res not in self._curves:
            self._curves[res] = _make_curve(res)
        curve = self._curves[res]
        side = (1 << res) - 1

        min_pad = 1.0 / max(side, 1)

        def _expand(lo: float, hi: float) -> tuple[float, float]:
            span = hi - lo
            pad = max(span * (overlap_factor - 1.0) / 2.0, min_pad)
            return max(0.0, lo - pad), min(1.0, hi + pad)

        x_lo, x_hi = _expand(bounds.x_min, bounds.x_max)
        y_lo, y_hi = _expand(bounds.y_min, bounds.y_max)
        z_lo, z_hi = _expand(bounds.z_min, bounds.z_max)
        t_lo, t_hi = _expand(bounds.t_min, bounds.t_max)

        ranges = (
            (_clamp(math.floor(x_lo * side), 0, side), _clamp(math.ceil(x_hi * side), 0, side)),
            (_clamp(math.floor(y_lo * side), 0, side), _clamp(math.ceil(y_hi * side), 0, side)),
            (_clamp(math.floor(z_lo * side), 0, side), _clamp(math.ceil(z_hi * side), 0, side)),
            (_clamp(math.floor(t_lo * side), 0, side), _clamp(math.ceil(t_hi * side), 0, side)),
        )
        return res, curve, ranges


# ---------------------------------------------------------------------------
# Module-level backward-compatible functions (delegate to default p=4)
# ---------------------------------------------------------------------------

_DEFAULT_ORDER = 4
_DEFAULT_CURVE = _make_curve(_DEFAULT_ORDER)
_DEFAULT_INDEX = HilbertIndex(resolutions=[_DEFAULT_ORDER])


def encode(
    x: float,
    y: float,
    z: float,
    t_norm: float,
    *,
    resolution_order: int | None = None,
) -> int:
    """Encode a normalised (x, y, z, t) point to a Hilbert index.

    All coordinates must be in [0, 1].  They are quantised to integer
    grid coordinates in ``[0, 2**resolution_order - 1]`` before encoding.
    """
    order = resolution_order if resolution_order is not None else _DEFAULT_ORDER
    curve = _DEFAULT_CURVE if order == _DEFAULT_ORDER else _make_curve(order)
    side = (1 << order) - 1

    coords = [
        _clamp(int(round(x * side)), 0, side),
        _clamp(int(round(y * side)), 0, side),
        _clamp(int(round(z * side)), 0, side),
        _clamp(int(round(t_norm * side)), 0, side),
    ]
    return int(curve.distance_from_point(coords))


def decode(
    hilbert_id: int,
    *,
    resolution_order: int | None = None,
) -> tuple[float, float, float, float]:
    """Decode a Hilbert index back to normalised (x, y, z, t)."""
    order = resolution_order if resolution_order is not None else _DEFAULT_ORDER
    curve = _DEFAULT_CURVE if order == _DEFAULT_ORDER else _make_curve(order)
    side = (1 << order) - 1

    coords = curve.point_from_distance(hilbert_id)
    return (
        coords[0] / side,
        coords[1] / side,
        coords[2] / side,
        coords[3] / side,
    )
