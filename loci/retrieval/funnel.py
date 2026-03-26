"""Multi-scale coarse-to-fine search funnel.

Strategy:
1. **Coarse pass** — search at ``scale_level="sequence"`` with a generous limit.
2. **Medium pass** — re-query the matching epochs at ``scale_level="frame"``.
3. **Fine pass** — final ANN search at ``scale_level="patch"`` among candidates.

This avoids scanning the full index when the world model stores
embeddings at multiple granularities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from loci.temporal.sharding import epoch_id

if TYPE_CHECKING:
    from loci.schema import WorldState


class SyncFunnelClient(Protocol):
    _epoch_size_ms: int

    def query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        *,
        _extra_payload_filter: dict[str, object] | None = None,
        _epoch_ids: set[int] | None = None,
    ) -> list[WorldState]: ...


class AsyncFunnelClient(Protocol):
    _epoch_size_ms: int

    async def query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        *,
        _extra_payload_filter: dict[str, object] | None = None,
        _epoch_ids: set[int] | None = None,
    ) -> list[WorldState]: ...


_SCALE_ORDER = ("sequence", "frame", "patch")


def funnel_search(
    client: SyncFunnelClient,
    vector: list[float],
    spatial_bounds: dict | None = None,
    time_window_ms: tuple[int, int] | None = None,
    limit: int = 10,
) -> list[WorldState]:
    """Run a coarse-to-fine funnel search across scale levels.

    Searches from coarsest (sequence) to finest (patch) granularity.
    Returns results at the finest scale that produced any hits.

    Args:
        client: Initialised :class:`LociClient`.
        vector: Query embedding.
        spatial_bounds: Optional spatial bounding box dict.
        time_window_ms: Optional ``(start_ms, end_ms)`` time window.
        limit: Number of final results to return.

    Returns:
        List of :class:`WorldState` results at the finest available scale.
    """
    best: list[WorldState] = []
    candidate_epochs: set[int] | None = None
    candidate_scene_ids: set[str] | None = None

    for scale in _SCALE_ORDER:
        extra_filter: dict[str, object] = {"scale_level": scale}
        if candidate_scene_ids:
            extra_filter["scene_id"] = sorted(candidate_scene_ids)
        results = client.query(
            vector=vector,
            spatial_bounds=spatial_bounds,
            time_window_ms=time_window_ms,
            limit=limit * 3,
            _extra_payload_filter=extra_filter,
            _epoch_ids=candidate_epochs,
        )
        if results:
            # Always prefer finer-grained results over coarser ones.
            best = results
            candidate_epochs = matched_epoch_ids(results, client._epoch_size_ms)
            candidate_scene_ids = matched_scene_ids(results) or None

    return best[:limit]


async def async_funnel_search(
    client: AsyncFunnelClient,
    vector: list[float],
    spatial_bounds: dict | None = None,
    time_window_ms: tuple[int, int] | None = None,
    limit: int = 10,
) -> list[WorldState]:
    """Async variant of funnel_search with epoch carry-over."""
    best: list[WorldState] = []
    candidate_epochs: set[int] | None = None
    candidate_scene_ids: set[str] | None = None

    for scale in _SCALE_ORDER:
        extra_filter: dict[str, object] = {"scale_level": scale}
        if candidate_scene_ids:
            extra_filter["scene_id"] = sorted(candidate_scene_ids)
        results = await client.query(
            vector=vector,
            spatial_bounds=spatial_bounds,
            time_window_ms=time_window_ms,
            limit=limit * 3,
            _extra_payload_filter=extra_filter,
            _epoch_ids=candidate_epochs,
        )
        if results:
            best = results
            candidate_epochs = matched_epoch_ids(results, client._epoch_size_ms)
            candidate_scene_ids = matched_scene_ids(results) or None

    return best[:limit]


def matched_epoch_ids(results: list[WorldState], epoch_size_ms: int) -> set[int]:
    """Return the set of epoch IDs covered by a result set."""
    return {epoch_id(state.timestamp_ms, epoch_size_ms) for state in results}


def matched_scene_ids(results: list[WorldState]) -> set[str]:
    """Return the non-empty scene IDs covered by a result set."""
    return {state.scene_id for state in results if state.scene_id}
