"""Exact payload filtering for spatiotemporal query results."""

from __future__ import annotations

from typing import Any

from loci.spatial.hilbert import SpatialBounds


def exact_payload_ranges(
    spatial_bounds: SpatialBounds | dict | None = None,
    time_window_ms: tuple[int, int] | None = None,
    min_confidence: float | None = None,
) -> dict[str, dict[str, float]]:
    """Build exact ranges to apply before backend top-k selection.

    Hilbert buckets can include points outside the requested box. Filtering
    only a bounded candidate list afterward can discard every result even
    when qualifying points exist deeper in the index.

    Inputs are not validated here: an inverted window or an out-of-range
    threshold yields a range that nothing (or everything) satisfies, exactly
    like :func:`exact_payload_match`, so callers see an empty result rather
    than an exception (the REST servers forward caller-supplied windows
    unchecked).
    """
    ranges: dict[str, dict[str, float]] = {}
    if spatial_bounds is not None:
        bounds = (
            spatial_bounds
            if isinstance(spatial_bounds, SpatialBounds)
            else SpatialBounds.from_dict(spatial_bounds)
        )
        for axis in ("x", "y", "z"):
            ranges[axis] = {
                "gte": getattr(bounds, f"{axis}_min"),
                "lte": getattr(bounds, f"{axis}_max"),
            }
    if time_window_ms is not None:
        start_ms, end_ms = time_window_ms
        ranges["timestamp_ms"] = {"gte": start_ms, "lte": end_ms}
    if min_confidence is not None:
        ranges["confidence"] = {"gte": min_confidence}
    return ranges


def exact_payload_match(
    payload: dict[str, Any],
    spatial_bounds: SpatialBounds | dict | None = None,
    time_window_ms: tuple[int, int] | None = None,
    min_confidence: float | None = None,
) -> bool:
    """Return True when a payload satisfies the exact requested bounds."""
    if min_confidence is not None:
        conf = payload.get("confidence")
        if conf is None or conf < min_confidence:
            return False

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
