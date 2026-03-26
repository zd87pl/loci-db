"""Helpers for planning adaptive spatiotemporal Hilbert queries."""

from __future__ import annotations

from loci.spatial.adaptive import AdaptiveResolution
from loci.spatial.hilbert import HilbertIndex, SpatialBounds

_MAX_HILBERT_BUCKETS = 10_000


def bounds_for_epoch(
    spatial_bounds: SpatialBounds | dict,
    time_window_ms: tuple[int, int] | None,
    ep: int,
    epoch_size_ms: int,
) -> SpatialBounds:
    """Return 4D bounds including epoch-local normalized time."""
    bounds = (
        spatial_bounds
        if isinstance(spatial_bounds, SpatialBounds)
        else SpatialBounds.from_dict(spatial_bounds)
    )

    if time_window_ms is None:
        t_min, t_max = 0.0, 1.0
    else:
        epoch_start = ep * epoch_size_ms
        overlap_start = max(time_window_ms[0], epoch_start)
        overlap_end = min(time_window_ms[1], epoch_start + epoch_size_ms)
        t_min = min(1.0, max(0.0, (overlap_start - epoch_start) / epoch_size_ms))
        t_max = min(1.0, max(0.0, (overlap_end - epoch_start) / epoch_size_ms))

    return SpatialBounds(
        x_min=bounds.x_min,
        x_max=bounds.x_max,
        y_min=bounds.y_min,
        y_max=bounds.y_max,
        z_min=bounds.z_min,
        z_max=bounds.z_max,
        t_min=t_min,
        t_max=t_max,
    )


def choose_query_resolution(
    hilbert: HilbertIndex,
    adaptive: AdaptiveResolution | None,
    spatial_bounds: SpatialBounds | dict,
    time_window_ms: tuple[int, int] | None,
    ep: int,
    epoch_size_ms: int,
    overlap_factor: float,
) -> int:
    """Choose the Hilbert resolution for a query in a specific epoch."""
    available = sorted(hilbert.resolutions)
    base_resolution = available[0]
    if adaptive is None:
        return base_resolution

    bounds = bounds_for_epoch(spatial_bounds, time_window_ms, ep, epoch_size_ms)
    center_x = (bounds.x_min + bounds.x_max) / 2
    center_y = (bounds.y_min + bounds.y_max) / 2
    center_z = (bounds.z_min + bounds.z_max) / 2
    center_t = (bounds.t_min + bounds.t_max) / 2

    requested = adaptive.resolution_for(center_x, center_y, center_z, center_t)
    candidate = _map_to_available_resolution(requested, available)

    while candidate > base_resolution:
        bucket_count = hilbert.estimated_bucket_count(
            bounds,
            resolution=candidate,
            overlap_factor=overlap_factor,
        )
        if bucket_count <= _MAX_HILBERT_BUCKETS:
            break
        candidate = _next_lower_resolution(candidate, available)

    return candidate


def _map_to_available_resolution(requested: int, available: list[int]) -> int:
    for resolution in available:
        if resolution >= requested:
            return resolution
    return available[-1]


def _next_lower_resolution(current: int, available: list[int]) -> int:
    lower = [resolution for resolution in available if resolution < current]
    return lower[-1] if lower else available[0]
