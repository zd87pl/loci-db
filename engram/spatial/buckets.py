"""Bucket ID computation and range expansion for spatial queries.

Given a spatial bounding box, compute the set of Hilbert bucket IDs
that intersect it so we can use a ``MatchAny`` filter in Qdrant.
"""

from __future__ import annotations

import itertools
import math

from engram.spatial.hilbert import _DEFAULT_ORDER, _clamp, _make_curve


def compute_bucket_id(
    x: float,
    y: float,
    z: float,
    t_norm: float,
    *,
    resolution_order: int | None = None,
) -> int:
    """Return the Hilbert bucket ID for a single point.

    This is a convenience wrapper that mirrors :func:`engram.spatial.hilbert.encode`.
    """
    from engram.spatial.hilbert import encode

    return encode(x, y, z, t_norm, resolution_order=resolution_order)


def expand_bounding_box(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    *,
    resolution_order: int | None = None,
) -> list[int]:
    """Return the sorted list of Hilbert IDs that cover a bounding box.

    We enumerate every grid cell that overlaps the box and return the
    unique Hilbert indices.  Because resolution is deliberately kept low
    (default 16 divisions per axis) the enumeration is cheap.

    Args:
        x_min, x_max: Normalised x bounds.
        y_min, y_max: Normalised y bounds.
        z_min, z_max: Normalised z bounds.
        t_min, t_max: Normalised temporal bounds.
        resolution_order: Bits per dimension (must match encoding).

    Returns:
        Sorted list of unique Hilbert indices.
    """
    order = resolution_order if resolution_order is not None else _DEFAULT_ORDER
    curve = _make_curve(order)
    side = (1 << order) - 1

    # Use floor for lower bounds and ceil for upper bounds so that
    # every grid cell that could contain a point quantised with round()
    # is included.  This fixes a mismatch where encode() uses round()
    # but the old code used int() (truncation), causing boundary misses.
    ix_lo = _clamp(math.floor(x_min * side), 0, side)
    ix_hi = _clamp(math.ceil(x_max * side), 0, side)
    iy_lo = _clamp(math.floor(y_min * side), 0, side)
    iy_hi = _clamp(math.ceil(y_max * side), 0, side)
    iz_lo = _clamp(math.floor(z_min * side), 0, side)
    iz_hi = _clamp(math.ceil(z_max * side), 0, side)
    it_lo = _clamp(math.floor(t_min * side), 0, side)
    it_hi = _clamp(math.ceil(t_max * side), 0, side)

    ids: set[int] = set()
    for ix, iy, iz, it in itertools.product(
        range(ix_lo, ix_hi + 1),
        range(iy_lo, iy_hi + 1),
        range(iz_lo, iz_hi + 1),
        range(it_lo, it_hi + 1),
    ):
        ids.add(curve.distance_from_point([ix, iy, iz, it]))

    return sorted(ids)
