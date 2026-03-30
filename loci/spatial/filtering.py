"""Exact payload filtering for spatiotemporal query results."""

from __future__ import annotations

from typing import Any

from loci.spatial.hilbert import SpatialBounds


def exact_payload_match(
    payload: dict[str, Any],
    spatial_bounds: SpatialBounds | dict | None = None,
    time_window_ms: tuple[int, int] | None = None,
) -> bool:
    """Return True when a payload satisfies the exact requested bounds."""
    if spatial_bounds is not None:
        bounds = (
            spatial_bounds
            if isinstance(spatial_bounds, SpatialBounds)
            else SpatialBounds.from_dict(spatial_bounds)
        )
        x = payload.get("x")
        y = payload.get("y")
        z = payload.get("z")
        if x is None or y is None or z is None:
            return False
        if not (bounds.x_min <= x <= bounds.x_max):
            return False
        if not (bounds.y_min <= y <= bounds.y_max):
            return False
        if not (bounds.z_min <= z <= bounds.z_max):
            return False

    if time_window_ms is not None:
        ts = payload.get("timestamp_ms")
        if ts is None:
            return False
        start_ms, end_ms = time_window_ms
        if not (start_ms <= ts <= end_ms):
            return False

    return True
