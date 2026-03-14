"""Hilbert curve / Morton code encoding for (x, y, z, t) coordinates.

Encodes a normalised 4D coordinate into a single int64 Hilbert index
for efficient spatial pre-filtering in Qdrant.
"""

from __future__ import annotations

from hilbertcurve.hilbertcurve import HilbertCurve


def _make_curve(resolution_order: int) -> HilbertCurve:
    """Create a 4-dimensional Hilbert curve with the given resolution.

    Args:
        resolution_order: Number of bits per dimension (p). The grid side
            length is ``2**p``, so with *p* = 8 we get a 256^4 grid.

    Returns:
        A :class:`HilbertCurve` instance for 4 dimensions.
    """
    return HilbertCurve(p=resolution_order, n=4)


# Default shared instance — 4D, 16 divisions per axis (p=4 → 2^4 = 16).
_DEFAULT_ORDER = 4
_DEFAULT_CURVE = _make_curve(_DEFAULT_ORDER)


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

    Args:
        x: Normalised x coordinate.
        y: Normalised y coordinate.
        z: Normalised z coordinate.
        t_norm: Normalised temporal coordinate.
        resolution_order: Bits per dimension (default 4 → 16 divisions).

    Returns:
        Hilbert index as a non-negative integer.
    """
    order = resolution_order if resolution_order is not None else _DEFAULT_ORDER
    curve = _DEFAULT_CURVE if order == _DEFAULT_ORDER else _make_curve(order)
    side = (1 << order) - 1  # 2^p - 1

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
    """Decode a Hilbert index back to normalised (x, y, z, t).

    Args:
        hilbert_id: Hilbert index produced by :func:`encode`.
        resolution_order: Bits per dimension (must match the value used for encoding).

    Returns:
        Tuple of ``(x, y, z, t_norm)`` each in [0, 1].
    """
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


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))
