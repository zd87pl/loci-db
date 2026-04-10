"""Bucket ID computation and range expansion for spatial queries.

Given a spatial bounding box, compute the set of Hilbert bucket IDs
that intersect it so we can use a ``MatchAny`` filter in Qdrant.
"""

from __future__ import annotations

from loci.spatial.hilbert import _DEFAULT_ORDER, HilbertIndex, SpatialBounds


def compute_bucket_id(
    x: float,
    y: float,
    z: float,
    t_norm: float,
    *,
    resolution_order: int | None = None,
) -> int:
    """Return the Hilbert bucket ID for a single point."""
    from loci.spatial.hilbert import encode

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
    overlap_factor: float = 1.0,
) -> list[int]:
    """Return the sorted list of Hilbert IDs that cover a bounding box.

    Delegates to :class:`HilbertIndex.query_buckets` which uses a
    precomputed numpy LUT for small resolutions.

    Args:
        x_min, x_max: Normalised x bounds.
        y_min, y_max: Normalised y bounds.
        z_min, z_max: Normalised z bounds.
        t_min, t_max: Normalised temporal bounds.
        resolution_order: Bits per dimension (must match encoding).
        overlap_factor: Expand each dimension by this factor (>=1.0).
            1.2 means 20% larger search region to catch boundary points.

    Returns:
        Sorted list of unique Hilbert indices.
    """
    order = resolution_order if resolution_order is not None else _DEFAULT_ORDER
    bounds = SpatialBounds(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        t_min=t_min,
        t_max=t_max,
    )
    index = HilbertIndex(resolutions=[order])
    return index.query_buckets(bounds, resolution=order, overlap_factor=overlap_factor)
