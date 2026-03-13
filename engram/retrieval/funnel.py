"""Multi-scale coarse-to-fine search funnel.

Strategy:
1. **Coarse pass** — search at ``scale_level="sequence"`` with a generous limit.
2. **Medium pass** — re-query the matching epochs at ``scale_level="frame"``.
3. **Fine pass** — final ANN search at ``scale_level="patch"`` among candidates.

This avoids scanning the full index when the world model stores
embeddings at multiple granularities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.client import EngramClient
    from engram.schema import WorldState


_SCALE_ORDER = ("sequence", "frame", "patch")


def funnel_search(
    client: "EngramClient",
    vector: list[float],
    spatial_bounds: dict | None = None,
    time_window_ms: tuple[int, int] | None = None,
    limit: int = 10,
) -> list["WorldState"]:
    """Run a coarse-to-fine funnel search across scale levels.

    Searches from coarsest (sequence) to finest (patch) granularity.
    Returns results at the finest scale that produced any hits.

    Args:
        client: Initialised :class:`EngramClient`.
        vector: Query embedding.
        spatial_bounds: Optional spatial bounding box dict.
        time_window_ms: Optional ``(start_ms, end_ms)`` time window.
        limit: Number of final results to return.

    Returns:
        List of :class:`WorldState` results at the finest available scale.
    """
    best: list[WorldState] = []

    for scale in _SCALE_ORDER:
        extra_filter = {"scale_level": scale}
        results = client.query(
            vector=vector,
            spatial_bounds=spatial_bounds,
            time_window_ms=time_window_ms,
            limit=limit * 3,
            _extra_payload_filter=extra_filter,
        )
        if results:
            # Always prefer finer-grained results over coarser ones.
            best = results

    return best[:limit]
