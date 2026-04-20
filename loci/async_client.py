"""Async LociClient — parallel shard fan-out for high-throughput workloads."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from typing import cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from loci.cloud_transport import AsyncCloudTransport, CloudModeUnsupportedError
from loci.payload_filters import extra_filter_to_conditions
from loci.retrieval.predict import PredictRetrieveResult, rerank_prediction_candidates
from loci.schema import ScoredWorldState, WorldState
from loci.spatial.adaptive import AdaptiveResolution
from loci.spatial.filtering import exact_payload_match
from loci.spatial.hilbert import HilbertIndex
from loci.spatial.query_plan import bounds_for_epoch, choose_query_resolution
from loci.temporal.decay import apply_decay
from loci.temporal.sharding import collection_name, epoch_id, epochs_in_range

logger = logging.getLogger(__name__)

_EXACT_FILTER_OVERFETCH = 3
_SCROLL_PAGE_SIZE = 256

# Map public distance names to Qdrant enum values
_DISTANCE_MAP: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclidean": Distance.EUCLID,
}


class AsyncLociClient:
    """Async high-level client with parallel shard fan-out.

    All query operations fan out across temporal shards concurrently
    using ``asyncio.gather``, giving significant speedups when data
    spans many epochs.

    Args:
        qdrant_url: URL of the Qdrant instance.
        epoch_size_ms: Width of each temporal shard in milliseconds.
        spatial_resolution: Hilbert curve resolution order (bits per dimension).
        vector_size: Dimensionality of the embedding vectors.
        decay_lambda: Temporal decay rate for recency weighting.
        distance: Distance metric — ``"cosine"``, ``"dot"``, or ``"euclidean"``.
        max_retries: Maximum number of retry attempts for transient Qdrant failures.
        retry_backoff: Base delay in seconds for exponential backoff between retries.
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        epoch_size_ms: int = 5000,
        spatial_resolution: int = 4,
        vector_size: int = 512,
        decay_lambda: float = 1e-4,
        distance: str = "cosine",
        adaptive: bool = False,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        resolutions: list[int] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Cloud mode: base_url + api_key → route via LOCI Cloud HTTP API.
        if base_url is not None:
            if api_key is None:
                raise ValueError("cloud mode requires api_key")
            self._cloud: AsyncCloudTransport | None = AsyncCloudTransport(base_url, api_key)
            # _qdrant is unused in cloud mode; keep type as AsyncQdrantClient so
            # local-mode code paths type-check without pervasive None guards.
            self._qdrant: AsyncQdrantClient = cast(AsyncQdrantClient, None)
        else:
            if qdrant_url is None:
                raise ValueError("qdrant_url is required unless base_url is provided")
            self._cloud = None
            self._qdrant = AsyncQdrantClient(url=qdrant_url, api_key=api_key)
        self._epoch_size_ms = epoch_size_ms
        self._spatial_resolution = spatial_resolution
        self._vector_size = vector_size
        self._decay_lambda = decay_lambda
        if distance not in _DISTANCE_MAP:
            raise ValueError(f"distance must be one of {list(_DISTANCE_MAP)}, got {distance!r}")
        self._distance = _DISTANCE_MAP[distance]
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._known_collections: set[str] = set()
        self._collection_locks: dict[str, asyncio.Lock] = {}
        self._hilbert = HilbertIndex(resolutions=resolutions or [4, 8, 12])
        self._adaptive = (
            AdaptiveResolution(
                base_order=self._hilbert.resolutions[0],
                max_order=max(self._hilbert.resolutions),
                density_threshold=50,
            )
            if adaptive
            else None
        )

    async def _retry(self, fn, *args, **kwargs):
        """Execute an async fn with retry logic."""
        from loci.retry import async_with_retry

        return await async_with_retry(
            fn,
            *args,
            max_retries=self._max_retries,
            backoff_base=self._retry_backoff,
            **kwargs,
        )

    @property
    def density_stats(self):
        """Return adaptive resolution density stats, or None if not enabled."""
        return self._adaptive.stats() if self._adaptive is not None else None

    async def close(self) -> None:
        """Close the underlying Qdrant connection or cloud transport."""
        if self._cloud is not None:
            await self._cloud.close()
        else:
            await self._qdrant.close()

    async def __aenter__(self) -> AsyncLociClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def _discover_collections(self) -> None:
        """Populate _known_collections from Qdrant (for read-only clients)."""
        if self._known_collections:
            return
        try:
            response = await self._qdrant.get_collections()
            for col in response.collections:
                if col.name.startswith("loci_"):
                    self._known_collections.add(col.name)
        except Exception:
            logger.debug("Failed to discover collections", exc_info=True)

    async def _ensure_collection(self, name: str) -> None:
        """Create a Qdrant collection if it does not already exist (idempotent, async-safe)."""
        if name in self._known_collections:
            return

        # Per-collection lock prevents concurrent creation races
        if name not in self._collection_locks:
            self._collection_locks[name] = asyncio.Lock()
        async with self._collection_locks[name]:
            if name in self._known_collections:
                return

            exists = False
            try:
                await self._qdrant.get_collection(name)
                exists = True
            except UnexpectedResponse as exc:
                if exc.status_code != 404:
                    raise

            if not exists:
                await self._qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=self._distance,
                    ),
                )
                index_tasks = [
                    self._qdrant.create_payload_index(
                        collection_name=name,
                        field_name=f"hilbert_r{r}",
                        field_schema=PayloadSchemaType.INTEGER,
                    )
                    for r in self._hilbert.resolutions
                ]
                index_tasks.extend(
                    [
                        self._qdrant.create_payload_index(
                            collection_name=name,
                            field_name="timestamp_ms",
                            field_schema=PayloadSchemaType.INTEGER,
                        ),
                        self._qdrant.create_payload_index(
                            collection_name=name,
                            field_name="scale_level",
                            field_schema=PayloadSchemaType.KEYWORD,
                        ),
                        self._qdrant.create_payload_index(
                            collection_name=name,
                            field_name="scene_id",
                            field_schema=PayloadSchemaType.KEYWORD,
                        ),
                    ]
                )
                await asyncio.gather(*index_tasks)

            self._known_collections.add(name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def insert(self, state: WorldState) -> str:
        """Insert a single WorldState into the store.

        Args:
            state: The world state to persist.

        Returns:
            The unique ID assigned to this state.
        """
        if self._cloud is not None:
            return await self._cloud.insert(state)

        point_id = uuid.uuid4().hex

        ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
        col = collection_name(ep)
        await self._ensure_collection(col)

        t_norm = _normalise_time(state.timestamp_ms, ep, self._epoch_size_ms)
        hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)

        if self._adaptive is not None:
            self._adaptive.record(state.x, state.y, state.z, t_norm)

        payload = _state_to_payload(state, hilbert_ids)

        # Causal linking
        predecessor = await self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
        if predecessor is not None:
            prev_id, prev_col = predecessor
            payload["prev_state_id"] = prev_id
            await self._patch_next_link(prev_id, point_id, collection_hint=prev_col)

        await self._retry(
            self._qdrant.upsert,
            collection_name=col,
            points=[PointStruct(id=point_id, vector=state.vector, payload=payload)],
        )
        return point_id

    async def insert_batch(self, states: list[WorldState]) -> list[str]:
        """Insert a batch of WorldStates — truly batched, one upsert per epoch.

        Within a batch, states in the same scene are causally linked
        in timestamp order.

        Args:
            states: List of world states.

        Returns:
            List of assigned IDs (same order as *states*).
        """
        groups: dict[str, list[PointStruct]] = {}
        ids: list[str] = []

        # Track per-scene causal chains within the batch
        scene_chains: dict[str, tuple[str, str]] = {}  # scene_id → (latest point_id, collection)
        prev_collection_by_point: dict[str, str] = {}

        # Sort indices by (scene_id, timestamp_ms) for correct linking
        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))
        id_by_index: dict[int, str] = {}

        for orig_idx, state in indexed:
            point_id = uuid.uuid4().hex
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            col = collection_name(ep)
            await self._ensure_collection(col)

            t_norm = _normalise_time(state.timestamp_ms, ep, self._epoch_size_ms)
            hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)
            if self._adaptive is not None:
                self._adaptive.record(state.x, state.y, state.z, t_norm)

            payload = _state_to_payload(state, hilbert_ids)

            # Link within the batch
            if state.scene_id and state.scene_id in scene_chains:
                prev_id, prev_col = scene_chains[state.scene_id]
                payload["prev_state_id"] = prev_id
                prev_collection_by_point[point_id] = prev_col
            if state.scene_id:
                scene_chains[state.scene_id] = (point_id, col)

            groups.setdefault(col, []).append(
                PointStruct(id=point_id, vector=state.vector, payload=payload)
            )

        # Fan out upserts concurrently
        await asyncio.gather(
            *(
                self._retry(self._qdrant.upsert, collection_name=col, points=points)
                for col, points in groups.items()
            )
        )

        # Patch next_state_id links within the batch
        for col, points in groups.items():
            for point in points:
                prev_link_id = (point.payload or {}).get("prev_state_id")
                if prev_link_id:
                    prev_id_str = str(prev_link_id)
                    try:
                        await self._retry(
                            self._qdrant.set_payload,
                            collection_name=prev_collection_by_point.get(str(point.id), col),
                            payload={"next_state_id": point.id},
                            points=[prev_id_str],
                        )
                    except Exception:
                        logger.debug(
                            "Failed to patch next link %s→%s",
                            prev_link_id,
                            point.id,
                            exc_info=True,
                        )

        # Return IDs in original order
        ids = [id_by_index[i] for i in range(len(states))]
        return ids

    # ------------------------------------------------------------------
    # Read — parallel fan-out
    # ------------------------------------------------------------------

    async def query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        *,
        _extra_payload_filter: dict | None = None,
        _epoch_ids: set[int] | None = None,
    ) -> list[WorldState]:
        """Search for nearest neighbours with parallel shard fan-out.

        All matching epoch collections are searched concurrently.

        Args:
            vector: Query embedding vector.
            spatial_bounds: Optional spatial bounding box.
            time_window_ms: Optional ``(start_ms, end_ms)`` window.
            limit: Maximum number of results.

        Returns:
            List of :class:`WorldState` results sorted by decay-weighted similarity.
        """
        if self._cloud is not None:
            if _extra_payload_filter is not None or _epoch_ids is not None:
                raise CloudModeUnsupportedError("advanced filtering is not supported in cloud mode")
            return await self._cloud.query(
                vector=vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window_ms,
                limit=limit,
            )

        return [
            candidate.state
            for candidate in await self.query_scored(
                vector,
                spatial_bounds,
                time_window_ms,
                limit,
                _extra_payload_filter=_extra_payload_filter,
                _epoch_ids=_epoch_ids,
            )
        ]

    async def query_scored(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        *,
        _extra_payload_filter: dict | None = None,
        _epoch_ids: set[int] | None = None,
    ) -> list[ScoredWorldState]:
        """Search and return scored results for downstream reranking."""
        await self._discover_collections()

        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            epochs = epochs_in_range(start_ms, end_ms, self._epoch_size_ms)
        else:
            epochs = self._list_active_epochs()
        if _epoch_ids is not None:
            epochs = [ep for ep in epochs if ep in _epoch_ids]

        epoch_collections = [
            (e, collection_name(e)) for e in epochs if collection_name(e) in self._known_collections
        ]
        if not epoch_collections:
            return []

        shard_limit = limit * _EXACT_FILTER_OVERFETCH if spatial_bounds is not None else limit

        # Parallel fan-out across shards
        async def _search_shard(ep: int, col: str) -> list[dict]:
            must_conditions: list = []
            if spatial_bounds is not None:
                query_resolution = choose_query_resolution(
                    self._hilbert,
                    self._adaptive,
                    spatial_bounds,
                    time_window_ms,
                    ep,
                    self._epoch_size_ms,
                    1.2,
                )
                hids = self._hilbert.query_buckets(
                    bounds_for_epoch(spatial_bounds, time_window_ms, ep, self._epoch_size_ms),
                    resolution=query_resolution,
                    overlap_factor=1.2,
                )
                if not hids:
                    return []
                field = self._hilbert.payload_field(query_resolution)
                must_conditions.append(FieldCondition(key=field, match=MatchAny(any=hids)))

            if time_window_ms is not None:
                must_conditions.append(
                    FieldCondition(
                        key="timestamp_ms",
                        range=Range(gte=start_ms, lte=end_ms),
                    )
                )

            must_conditions.extend(extra_filter_to_conditions(_extra_payload_filter))

            query_filter = Filter(must=must_conditions) if must_conditions else None
            try:
                resp = await self._retry(
                    self._qdrant.query_points,
                    collection_name=col,
                    query=vector,
                    query_filter=query_filter,
                    limit=shard_limit,
                    with_vectors=True,
                )
                hits = resp.points
                return [
                    {
                        "score": hit.score,
                        "timestamp_ms": hit.payload.get("timestamp_ms", 0),
                        "payload": hit.payload,
                        "vector": hit.vector,
                        "id": hit.id,
                    }
                    for hit in hits
                ]
            except Exception:
                logger.debug("Search failed on %s", col, exc_info=True)
                return []

        shard_results = await asyncio.gather(
            *(_search_shard(ep, col) for ep, col in epoch_collections)
        )
        all_results: list[dict] = []
        for batch in shard_results:
            all_results.extend(batch)

        if spatial_bounds is not None or time_window_ms is not None:
            all_results = [
                r
                for r in all_results
                if exact_payload_match(
                    r["payload"],
                    spatial_bounds=spatial_bounds,
                    time_window_ms=time_window_ms,
                )
            ]

        # Apply temporal decay and re-rank
        now_ms = int(time.time() * 1000)
        apply_decay(all_results, now_ms, self._decay_lambda)
        all_results = all_results[:limit]

        return [
            ScoredWorldState(
                state=_payload_to_state(r["payload"], r["id"], r["vector"]),
                score=float(r["score"]),
                decayed_score=float(r.get("decayed_score", r["score"])),
            )
            for r in all_results
        ]

    async def predict_and_retrieve(
        self,
        context_vector: list[float],
        predictor_fn: Callable[[list[float]], list[float]],
        future_horizon_ms: int = 1000,
        limit: int = 5,
        current_position: tuple[float, float, float] | None = None,
        spatial_search_radius: float = 0.3,
        alpha: float = 0.7,
        return_prediction: bool = False,
    ) -> list[WorldState] | PredictRetrieveResult:
        """Predict a future state then retrieve nearest neighbours.

        When ``current_position`` is provided or ``return_prediction`` is
        enabled, returns a full :class:`PredictRetrieveResult` with novelty
        scoring and timing. Otherwise falls back to the legacy API returning
        a plain list.

        Args:
            context_vector: Current-state embedding.
            predictor_fn: User-supplied world model.
            future_horizon_ms: How far ahead to search (milliseconds).
            limit: Maximum number of results.
            current_position: Optional (x, y, z) for spatial + novelty scoring.
            spatial_search_radius: Search radius around current_position.
            alpha: Weight for vector_sim vs temporal_proximity (default 0.7).
            return_prediction: Include predicted vector in result.

        Returns:
            :class:`PredictRetrieveResult` when ``current_position`` is set
            or ``return_prediction`` is ``True``, otherwise a plain list of
            :class:`WorldState`.
        """
        t_predictor = time.perf_counter()
        predicted_vector = predictor_fn(context_vector)
        predictor_call_ms = (time.perf_counter() - t_predictor) * 1000
        now_ms = int(time.time() * 1000)

        if current_position is not None or return_prediction:
            t0 = time.perf_counter()
            spatial_bounds = None
            if current_position is not None:
                x, y, z = current_position
                spatial_bounds = {
                    "x_min": max(0.0, x - spatial_search_radius),
                    "x_max": min(1.0, x + spatial_search_radius),
                    "y_min": max(0.0, y - spatial_search_radius),
                    "y_max": min(1.0, y + spatial_search_radius),
                    "z_min": max(0.0, z - spatial_search_radius),
                    "z_max": min(1.0, z + spatial_search_radius),
                }

            raw_results = await self.query_scored(
                vector=predicted_vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=(now_ms, now_ms + future_horizon_ms),
                limit=limit * 2,
            )
            retrieval_latency_ms = (time.perf_counter() - t0) * 1000

            if raw_results:
                results, prediction_novelty = rerank_prediction_candidates(
                    raw_results,
                    now_ms=now_ms,
                    future_horizon_ms=future_horizon_ms,
                    alpha=alpha,
                    limit=limit,
                )
            else:
                results = []
                prediction_novelty = 1.0

            return PredictRetrieveResult(
                results=results,
                prediction_novelty=prediction_novelty,
                predicted_vector=predicted_vector if return_prediction else None,
                retrieval_latency_ms=retrieval_latency_ms,
                predictor_call_ms=predictor_call_ms,
            )

        return await self.query(
            vector=predicted_vector,
            time_window_ms=(now_ms, now_ms + future_horizon_ms),
            limit=limit,
        )

    async def funnel_query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
    ) -> list[WorldState]:
        """Multi-scale coarse-to-fine search across scale levels.

        Cascades from sequence → frame → patch, returning results at the
        finest scale that produced hits.

        Args:
            vector: Query embedding vector.
            spatial_bounds: Optional spatial bounding box.
            time_window_ms: Optional ``(start_ms, end_ms)`` time window.
            limit: Maximum number of results.

        Returns:
            List of :class:`WorldState` at the finest available scale.
        """
        from loci.retrieval.funnel import async_funnel_search

        return await async_funnel_search(self, vector, spatial_bounds, time_window_ms, limit)

    async def get_trajectory(
        self,
        state_id: str,
        steps_back: int = 10,
        steps_forward: int = 10,
    ) -> list[WorldState]:
        """Reconstruct a trajectory using scroll API with scene_id filter."""
        await self._discover_collections()
        anchor = await self._get_state_by_id(state_id)
        if anchor is None:
            return []
        if not anchor.scene_id:
            return [anchor]

        async def _scroll_shard(col: str) -> list[WorldState]:
            try:
                points = await self._scroll_all(
                    collection=col,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="scene_id",
                                match=MatchValue(value=anchor.scene_id),
                            ),
                        ]
                    ),
                    order_by="timestamp_ms",
                    with_vectors=True,
                )
                results = []
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    results.append(_payload_to_state(pt.payload, pt.id, vec))
                return results
            except Exception:
                return []

        shard_results = await asyncio.gather(
            *(_scroll_shard(col) for col in list(self._known_collections))
        )
        all_states: list[WorldState] = []
        for batch in shard_results:
            all_states.extend(batch)

        all_states.sort(key=lambda s: s.timestamp_ms)
        anchor_idx = None
        for i, s in enumerate(all_states):
            if s.id == state_id:
                anchor_idx = i
                break

        if anchor_idx is None:
            return [anchor]

        start = max(0, anchor_idx - steps_back)
        end = min(len(all_states), anchor_idx + steps_forward + 1)
        return all_states[start:end]

    async def get_causal_context(
        self,
        state_id: str,
        window_ms: int = 5000,
    ) -> list[WorldState]:
        """Return all states within ±window_ms in the same scene_id."""
        await self._discover_collections()
        anchor = await self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms

        async def _scroll_shard(col: str) -> list[WorldState]:
            try:
                points = await self._scroll_all(
                    collection=col,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="scene_id",
                                match=MatchValue(value=anchor.scene_id),
                            ),
                            FieldCondition(
                                key="timestamp_ms",
                                range=Range(gte=t_min, lte=t_max),
                            ),
                        ]
                    ),
                    order_by="timestamp_ms",
                    with_vectors=True,
                )
                results = []
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    results.append(_payload_to_state(pt.payload, pt.id, vec))
                return results
            except Exception:
                return []

        shard_results = await asyncio.gather(
            *(_scroll_shard(col) for col in list(self._known_collections))
        )
        context: list[WorldState] = []
        for batch in shard_results:
            context.extend(batch)

        context.sort(key=lambda s: s.timestamp_ms)
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_state_by_id(self, state_id: str) -> WorldState | None:
        for col in list(self._known_collections):
            try:
                results = await self._retry(
                    self._qdrant.retrieve,
                    collection_name=col,
                    ids=[state_id],
                    with_payload=True,
                    with_vectors=True,
                )
                if results:
                    vec = results[0].vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    return _payload_to_state(results[0].payload, results[0].id, vec)
            except Exception:  # noqa: S112  # retry loop across epochs
                continue
        return None

    async def _find_latest_predecessor(
        self, scene_id: str, before_ms: int
    ) -> tuple[str, str] | None:
        if not scene_id:
            return None
        await self._discover_collections()
        for collection in self._predecessor_search_collections(before_ms):
            try:
                points = await self._scroll_all(
                    collection=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="scene_id", match=MatchValue(value=scene_id)),
                            FieldCondition(key="timestamp_ms", range=Range(lt=before_ms)),
                        ]
                    ),
                    order_by="timestamp_ms",
                )
                if points:
                    # Scroll returns ascending order; last item is the latest predecessor
                    return str(points[-1].id), collection
            except Exception:
                logger.debug("Failed to find predecessor in %s", collection, exc_info=True)
        return None

    async def _patch_next_link(
        self, prev_id: str, next_id: str, collection_hint: str | None = None
    ) -> None:
        collections = list(self._known_collections)
        if collection_hint is not None:
            collections = [collection_hint] + [col for col in collections if col != collection_hint]
        for col in collections:
            try:
                await self._retry(
                    self._qdrant.set_payload,
                    collection_name=col,
                    payload={"next_state_id": next_id},
                    points=[prev_id],
                )
                return
            except Exception:  # noqa: S112  # retry loop across epochs
                continue

    async def _scroll_all(
        self,
        *,
        collection: str,
        scroll_filter: Filter | None = None,
        order_by: str | None = None,
        with_vectors: bool = False,
    ) -> list:
        """Return the full ordered scroll result for a collection."""
        offset: object | None = None
        all_points: list = []
        while True:
            hits = await self._retry(
                self._qdrant.scroll,
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=_SCROLL_PAGE_SIZE,
                order_by=order_by,
                with_vectors=with_vectors,
                offset=offset,
            )
            points, next_offset = hits if isinstance(hits, tuple) else (hits, None)
            all_points.extend(points)
            if not points or next_offset is None:
                break
            offset = next_offset
        return all_points

    def _predecessor_search_collections(self, before_ms: int) -> list[str]:
        target_epoch = epoch_id(before_ms, self._epoch_size_ms)
        epochs = [ep for ep in self._list_active_epochs() if ep <= target_epoch]
        return [collection_name(ep) for ep in sorted(epochs, reverse=True)]

    def _list_active_epochs(self) -> list[int]:
        epochs: list[int] = []
        for col in self._known_collections:
            if col.startswith("loci_"):
                with contextlib.suppress(ValueError):
                    epochs.append(int(col.split("_", 1)[1]))
        return sorted(epochs) if epochs else []


# ---------------------------------------------------------------------------
# Shared helpers (used by both sync and async clients)
# ---------------------------------------------------------------------------


def _normalise_time(timestamp_ms: int, ep: int, epoch_size_ms: int) -> float:
    epoch_start = ep * epoch_size_ms
    offset = timestamp_ms - epoch_start
    return min(1.0, max(0.0, offset / epoch_size_ms))


def _state_to_payload(state: WorldState, hilbert_ids: dict[str, int]) -> dict:
    payload = {
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "timestamp_ms": state.timestamp_ms,
        "scene_id": state.scene_id,
        "scale_level": state.scale_level,
        "confidence": state.confidence,
        "prev_state_id": state.prev_state_id,
        "next_state_id": state.next_state_id,
    }
    payload.update(hilbert_ids)
    return payload


def _payload_to_state(
    payload: dict, point_id: str, vector: list[float] | None = None
) -> WorldState:
    return WorldState(
        x=payload["x"],
        y=payload["y"],
        z=payload["z"],
        timestamp_ms=payload["timestamp_ms"],
        vector=vector if vector is not None else [],
        scene_id=payload.get("scene_id", ""),
        scale_level=payload.get("scale_level", "patch"),
        confidence=payload.get("confidence", 1.0),
        prev_state_id=payload.get("prev_state_id"),
        next_state_id=payload.get("next_state_id"),
        id=str(point_id),
    )
