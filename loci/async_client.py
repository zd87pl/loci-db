"""Async LociClient — parallel shard fan-out for high-throughput workloads."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, cast

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
from loci.retrieval.predict import (
    PredictRetrieveResult,
    _validate_predicted_vector,
    rerank_prediction_candidates,
)
from loci.schema import ScoredWorldState, WorldState
from loci.spatial.adaptive import AdaptiveResolution
from loci.spatial.filtering import exact_payload_match
from loci.spatial.hilbert import HilbertIndex
from loci.spatial.query_plan import bounds_for_epoch, choose_query_resolution
from loci.temporal.decay import DEFAULT_DECAY_LAMBDA, apply_decay
from loci.temporal.retention import RetentionManager, RetentionPolicy
from loci.temporal.sharding import collection_name, epoch_id

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
        spatial_resolution: Hilbert curve resolution order (bits per dimension)
            used as the default (coarsest) query resolution. Ignored when an
            explicit ``resolutions`` list is provided — ``resolutions`` wins.
        vector_size: Dimensionality of the embedding vectors.
        decay_lambda: Temporal decay rate for recency weighting (per ms;
            defaults to a one-hour half-life).
        distance: Distance metric — ``"cosine"``, ``"dot"``, or ``"euclidean"``.
        max_retries: Maximum number of retry attempts for transient Qdrant failures.
        retry_backoff: Base delay in seconds for exponential backoff between retries.
        collection_prefix: Optional tenant namespace prefix for Qdrant collections.
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        epoch_size_ms: int = 5000,
        spatial_resolution: int = 4,
        vector_size: int = 512,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        distance: str = "cosine",
        adaptive: bool = False,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        resolutions: list[int] | None = None,
        api_key: str | None = None,
        collection_prefix: str = "",
        base_url: str | None = None,
        retention_policy: RetentionPolicy | None = None,
    ) -> None:
        if epoch_size_ms <= 0:
            raise ValueError(f"epoch_size_ms must be positive, got {epoch_size_ms}")
        if distance not in _DISTANCE_MAP:
            raise ValueError(f"distance must be one of {list(_DISTANCE_MAP)}, got {distance!r}")

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
        self._distance = _DISTANCE_MAP[distance]
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._collection_prefix = collection_prefix
        self._known_collections: set[str] = set()
        self._discovered = False
        self._collection_locks: dict[str, asyncio.Lock] = {}
        self._locks_mutex = asyncio.Lock()
        # Explicit resolutions win; otherwise spatial_resolution becomes the
        # coarsest (default) query resolution alongside the finer defaults.
        self._hilbert = HilbertIndex(
            resolutions=resolutions
            if resolutions is not None
            else sorted({spatial_resolution, 8, 12})
        )
        self._adaptive = (
            AdaptiveResolution(
                base_order=self._hilbert.resolutions[0],
                max_order=max(self._hilbert.resolutions),
                density_threshold=50,
            )
            if adaptive
            else None
        )
        self._retention = (
            RetentionManager(
                policy=retention_policy,
                epoch_size_ms=self._epoch_size_ms,
                collection_prefix=collection_prefix,
            )
            if retention_policy is not None
            else None
        )

    def _col_name(self, ep: int) -> str:
        """Return the Qdrant collection name for an epoch, applying the tenant namespace prefix."""
        base = collection_name(ep)
        return f"{self._collection_prefix}{base}" if self._collection_prefix else base

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

    async def _discover_collections(self, force: bool = False) -> None:
        """Merge Qdrant's collection listing into _known_collections.

        Runs once per client by default; pass ``force=True`` to refresh (used
        when a query targets an epoch this client has not seen yet, e.g. one
        created by another writer).
        """
        if self._discovered and not force:
            return
        prefix = f"{self._collection_prefix}loci_" if self._collection_prefix else "loci_"
        try:
            response = await self._qdrant.get_collections()
            for col in response.collections:
                if col.name.startswith(prefix):
                    self._known_collections.add(col.name)
            self._discovered = True
        except Exception:
            logger.debug("Failed to discover collections", exc_info=True)

    async def _refresh_for_window(self, first_ep: int, last_ep: int) -> None:
        """Re-run discovery when part of a requested epoch window is unknown."""
        known = sum(1 for ep in self._list_active_epochs() if first_ep <= ep <= last_ep)
        if known < (last_ep - first_ep + 1):
            await self._discover_collections(force=True)

    def _epochs_intersecting(self, first_ep: int, last_ep: int) -> list[int]:
        """Known epochs within [first_ep, last_ep] without materialising the range."""
        return [ep for ep in self._list_active_epochs() if first_ep <= ep <= last_ep]

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._vector_size:
            raise ValueError(f"vector has dimension {len(vector)}, expected {self._vector_size}")

    async def _ensure_collection(self, name: str) -> None:
        """Create a Qdrant collection if it does not already exist (idempotent, async-safe)."""
        if name in self._known_collections:
            return

        # Per-collection lock prevents concurrent creation races.
        async with self._locks_mutex:
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
                try:
                    await self._qdrant.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=self._vector_size,
                            distance=self._distance,
                        ),
                    )
                except Exception as exc:
                    # A concurrent external writer won the create race; treat
                    # as success (the winner also creates the indexes).
                    if not _is_already_exists_error(exc):
                        raise
                    exists = True

            if not exists:
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

        self._validate_vector(state.vector)
        point_id = str(uuid.uuid4())

        ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
        col = self._col_name(ep)
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
        await self._maybe_purge()
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
        if self._cloud is not None:
            raise CloudModeUnsupportedError("insert_batch is not supported in cloud mode")
        for state in states:
            self._validate_vector(state.vector)

        groups: dict[str, list[PointStruct]] = {}
        ids: list[str] = []

        # Track per-scene causal chains within the batch
        scene_chains: dict[str, tuple[str, str]] = {}  # scene_id → (latest point_id, collection)
        prev_collection_by_point: dict[str, str] = {}

        # Sort indices by (scene_id, timestamp_ms) for correct linking
        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))
        id_by_index: dict[int, str] = {}

        for orig_idx, state in indexed:
            point_id = str(uuid.uuid4())
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            col = self._col_name(ep)
            await self._ensure_collection(col)

            t_norm = _normalise_time(state.timestamp_ms, ep, self._epoch_size_ms)
            hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)
            if self._adaptive is not None:
                self._adaptive.record(state.x, state.y, state.z, t_norm)

            payload = _state_to_payload(state, hilbert_ids)

            # Link within the batch; the first state per scene links to the
            # latest predecessor already in the store (matching the
            # sequential-insert behaviour).
            if state.scene_id:
                if state.scene_id in scene_chains:
                    prev_link: tuple[str, str] | None = scene_chains[state.scene_id]
                else:
                    prev_link = await self._find_latest_predecessor(
                        state.scene_id, state.timestamp_ms
                    )
                if prev_link is not None:
                    prev_id, prev_col = prev_link
                    payload["prev_state_id"] = prev_id
                    prev_collection_by_point[point_id] = prev_col
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
        await self._maybe_purge()
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
        overlap_factor: float = 1.2,
        min_confidence: float | None = None,
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
            _advanced = (
                _extra_payload_filter is not None
                or _epoch_ids is not None
                or min_confidence is not None
            )
            if _advanced:
                raise CloudModeUnsupportedError(
                    "advanced filtering (payload filters, epoch ids, min_confidence) "
                    "is not supported in cloud mode"
                )
            return await self._cloud.query(
                vector=vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window_ms,
                limit=limit,
                overlap_factor=overlap_factor,
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
                overlap_factor=overlap_factor,
                min_confidence=min_confidence,
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
        overlap_factor: float = 1.2,
        min_confidence: float | None = None,
    ) -> list[ScoredWorldState]:
        """Search and return scored results for downstream reranking.

        Scores follow the higher-is-better convention for every distance
        metric: raw Qdrant euclidean distances (smaller-is-better) are
        negated at the boundary so decay re-ranking, cross-shard merging,
        and truncation behave identically across metrics and backends.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("query_scored is not supported in cloud mode")
        await self._discover_collections()

        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            first_ep = epoch_id(start_ms, self._epoch_size_ms)
            last_ep = epoch_id(end_ms, self._epoch_size_ms)
            # Refresh discovery when part of the window is unknown — another
            # writer may have created those collections since discovery ran.
            await self._refresh_for_window(first_ep, last_ep)
            epochs = self._epochs_intersecting(first_ep, last_ep)
        else:
            epochs = self._list_active_epochs()
        if _epoch_ids is not None:
            epochs = [ep for ep in epochs if ep in _epoch_ids]

        epoch_collections = [
            (e, self._col_name(e)) for e in epochs if self._col_name(e) in self._known_collections
        ]
        if not epoch_collections:
            return []

        # Over-fetch per shard whenever post-search filtering (spatial exact
        # match, min_confidence) or decay re-ranking can reorder/drop hits;
        # otherwise per-shard truncation could evict the true top-k.
        needs_overfetch = (
            spatial_bounds is not None or min_confidence is not None or self._decay_lambda > 0
        )
        shard_limit = limit * _EXACT_FILTER_OVERFETCH if needs_overfetch else limit

        # Parallel fan-out across shards; None signals a failed shard.
        async def _search_shard(ep: int, col: str) -> list[dict] | None:
            must_conditions: list = []
            if spatial_bounds is not None:
                query_resolution = choose_query_resolution(
                    self._hilbert,
                    self._adaptive,
                    spatial_bounds,
                    time_window_ms,
                    ep,
                    self._epoch_size_ms,
                    overlap_factor,
                )
                hids = self._hilbert.query_buckets(
                    bounds_for_epoch(spatial_bounds, time_window_ms, ep, self._epoch_size_ms),
                    resolution=query_resolution,
                    overlap_factor=overlap_factor,
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

            if min_confidence is not None:
                must_conditions.append(
                    FieldCondition(key="confidence", range=Range(gte=min_confidence))
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
                results = []
                for hit in hits:
                    score = float(hit.score)
                    if self._distance == Distance.EUCLID:
                        # Qdrant returns raw euclidean distances (smaller is
                        # better); negate so all downstream code sees
                        # higher-is-better scores.
                        score = -score
                    results.append(
                        {
                            "score": score,
                            "timestamp_ms": hit.payload.get("timestamp_ms", 0),
                            "payload": hit.payload,
                            "vector": hit.vector,
                            "id": hit.id,
                        }
                    )
                return results
            except Exception as exc:
                logger.warning("Search failed on shard %s: %s", col, exc)
                return None

        shard_results = await asyncio.gather(
            *(_search_shard(ep, col) for ep, col in epoch_collections)
        )
        failed = sum(1 for batch in shard_results if batch is None)
        if failed == len(shard_results) and shard_results:
            logger.warning("All %d shard searches failed; returning no results", failed)
        all_results: list[dict] = []
        for batch in shard_results:
            all_results.extend(batch or [])

        if spatial_bounds is not None or time_window_ms is not None or min_confidence is not None:
            all_results = [
                r
                for r in all_results
                if exact_payload_match(
                    r["payload"],
                    spatial_bounds=spatial_bounds,
                    time_window_ms=time_window_ms,
                    min_confidence=min_confidence,
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
        current_timestamp_ms: int | None = None,
        spatial_search_radius: float = 0.3,
        alpha: float = 0.7,
        return_prediction: bool = False,
        *,
        calibrator: Any = None,
        search_time_window_ms: tuple[int, int] | None = None,
    ) -> list[WorldState] | PredictRetrieveResult:
        """Predict a future state then retrieve nearest neighbours.

        When ``current_position`` is provided or ``return_prediction`` is
        enabled, returns a full :class:`PredictRetrieveResult` with novelty
        scoring and timing. Otherwise falls back to the legacy API returning
        a plain list. By default, retrieval searches stored history for
        analogs; pass ``search_time_window_ms`` to restrict it to an absolute
        timestamp range.

        Args:
            context_vector: Current-state embedding.
            predictor_fn: User-supplied world model.
            future_horizon_ms: How far ahead to search (milliseconds).
            limit: Maximum number of results.
            current_position: Optional (x, y, z) for spatial + novelty scoring.
            current_timestamp_ms: Current time in ms for novelty scoring
                (defaults to wall-clock now).
            spatial_search_radius: Search radius around current_position.
            alpha: Weight for vector_sim vs temporal_proximity (default 0.7).
            return_prediction: Include predicted vector in result.
            search_time_window_ms: Optional explicit timestamp range to search.

        Returns:
            :class:`PredictRetrieveResult` when ``current_position`` is set
            or ``return_prediction`` is ``True``, otherwise a plain list of
            :class:`WorldState`.

        In cloud mode only the legacy path (plain query on the predicted
        vector) is supported; ``current_position``, ``return_prediction``,
        and ``calibrator`` require scored retrieval, which the cloud API
        does not expose yet.
        """
        if self._cloud is not None and (
            current_position is not None or return_prediction or calibrator is not None
        ):
            raise CloudModeUnsupportedError(
                "predict_and_retrieve with current_position, return_prediction, or "
                "calibrator is not supported in cloud mode"
            )
        t_predictor = time.perf_counter()
        predicted_vector = predictor_fn(context_vector)
        predictor_call_ms = (time.perf_counter() - t_predictor) * 1000
        _validate_predicted_vector(predicted_vector, len(context_vector))
        if current_timestamp_ms is not None:
            now_ms = current_timestamp_ms
        else:
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
                time_window_ms=search_time_window_ms,
                limit=limit * 2,
            )
            retrieval_latency_ms = (time.perf_counter() - t0) * 1000

            if raw_results:
                results, best_score = rerank_prediction_candidates(
                    raw_results,
                    now_ms=now_ms,
                    future_horizon_ms=future_horizon_ms,
                    alpha=alpha,
                    limit=limit,
                    use_temporal_proximity=search_time_window_ms is not None,
                    predicted_vector=predicted_vector,
                    time_window_ms=search_time_window_ms,
                )
            else:
                results = []
                best_score = 0.0

            if calibrator is not None:
                # Score against the window *before* observing so the current
                # sample does not contaminate its own baseline.
                prediction_novelty = calibrator.calibrated_novelty(best_score)
                novelty_samples = len(calibrator)
                calibrator.observe(best_score)
            else:
                prediction_novelty = max(0.0, min(1.0, 1.0 - best_score))
                novelty_samples = 0

            return PredictRetrieveResult(
                results=results,
                prediction_novelty=prediction_novelty,
                predicted_vector=predicted_vector if return_prediction else None,
                retrieval_latency_ms=retrieval_latency_ms,
                predictor_call_ms=predictor_call_ms,
                novelty_samples=novelty_samples,
            )

        return await self.query(
            vector=predicted_vector,
            time_window_ms=search_time_window_ms,
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
        if self._cloud is not None:
            raise CloudModeUnsupportedError("funnel_query is not supported in cloud mode")
        from loci.retrieval.funnel import async_funnel_search

        return await async_funnel_search(self, vector, spatial_bounds, time_window_ms, limit)

    async def get_trajectory(
        self,
        state_id: str,
        steps_back: int = 10,
        steps_forward: int = 10,
    ) -> list[WorldState]:
        """Reconstruct a trajectory using scroll API with scene_id filter."""
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_trajectory is not supported in cloud mode")
        await self._discover_collections()
        anchor = await self._get_state_by_id(state_id)
        if anchor is None:
            # The anchor may live in a collection created by another writer
            # after our last discovery; refresh once and retry.
            await self._discover_collections(force=True)
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
                    with_vectors=True,
                )
                results = []
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    results.append(_payload_to_state(pt.payload, pt.id, vec))
                return results
            except Exception as exc:
                logger.warning("Trajectory scroll failed on shard %s: %s", col, exc)
                return []

        shard_results = await asyncio.gather(
            *(_scroll_shard(col) for col in list(self._known_collections))
        )
        all_states: list[WorldState] = []
        for batch in shard_results:
            all_states.extend(batch)

        all_states.sort(key=lambda s: s.timestamp_ms)
        anchor_idx = None
        target = _normalize_id(state_id)
        for i, s in enumerate(all_states):
            if _normalize_id(s.id) == target:
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
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_causal_context is not supported in cloud mode")
        await self._discover_collections()
        anchor = await self._get_state_by_id(state_id)
        if anchor is None:
            await self._discover_collections(force=True)
            anchor = await self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms
        await self._refresh_for_window(
            epoch_id(t_min, self._epoch_size_ms), epoch_id(t_max, self._epoch_size_ms)
        )

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
                    with_vectors=True,
                )
                results = []
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    results.append(_payload_to_state(pt.payload, pt.id, vec))
                return results
            except Exception as exc:
                logger.warning("Causal-context scroll failed on shard %s: %s", col, exc)
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
        """Find the most recent state in the same scene before a timestamp.

        Scrolls unordered (server-side ordering breaks pagination) and picks
        the max-timestamp point client-side. The scan is bounded per epoch
        collection by the scene and timestamp filters.
        """
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
                )
                if points:
                    latest = max(points, key=_point_timestamp)
                    return str(latest.id), collection
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

    async def _maybe_purge(self) -> None:
        if self._retention is None:
            return
        try:
            dropped = await self._retention.maybe_purge_async(
                active_epochs=self._list_active_epochs(),
                now_ms=int(time.time() * 1000),
                delete_fn=self._qdrant.delete_collection,
            )
            self._forget_collections(dropped)
        except Exception as exc:
            logger.warning("Retention purge failed: %s", exc)

    def _forget_collections(self, names: list[str]) -> None:
        """Drop purged collections from local caches so a late insert into a
        purged epoch simply recreates the collection."""
        for name in names:
            self._known_collections.discard(name)
            self._collection_locks.pop(name, None)

    async def _scroll_all(
        self,
        *,
        collection: str,
        scroll_filter: Filter | None = None,
        with_vectors: bool = False,
    ) -> list:
        """Return the full scroll result for a collection.

        Scrolls unordered: Qdrant returns ``next_page_offset=None`` whenever
        ``order_by`` is set (and rejects ``offset`` + ``order_by``), which
        silently truncated ordered scrolls to a single page. Callers sort
        client-side by ``timestamp_ms`` instead.
        """
        offset: object | None = None
        all_points: list = []
        while True:
            hits = await self._retry(
                self._qdrant.scroll,
                collection_name=collection,
                scroll_filter=scroll_filter,
                limit=_SCROLL_PAGE_SIZE,
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
        return [self._col_name(ep) for ep in sorted(epochs, reverse=True)]

    def _list_active_epochs(self) -> list[int]:
        """Return epoch IDs for all known collections, respecting the tenant prefix."""
        prefix = f"{self._collection_prefix}loci_" if self._collection_prefix else "loci_"
        epochs: list[int] = []
        for col in self._known_collections:
            if col.startswith(prefix):
                with contextlib.suppress(ValueError):
                    epochs.append(int(col[len(prefix) :]))
        return sorted(epochs) if epochs else []


# ---------------------------------------------------------------------------
# Shared helpers (used by both sync and async clients)
# ---------------------------------------------------------------------------


def _normalize_id(point_id: object) -> str:
    """Normalise a point ID for comparison.

    Real Qdrant servers canonicalise UUID point IDs into lowercase hyphenated
    form, while historic Loci clients generated hyphen-less ``uuid4().hex``
    IDs. Comparing normalised forms keeps both representations matching.
    """
    return str(point_id).lower().replace("-", "")


def _is_already_exists_error(exc: Exception) -> bool:
    """Return True for Qdrant 'collection already exists' create conflicts."""
    if getattr(exc, "status_code", None) == 409:
        return True
    return "already exists" in str(exc).lower()


def _point_timestamp(point: Any) -> int:
    """Timestamp of a scrolled Qdrant point (0 when payload is missing)."""
    payload = getattr(point, "payload", None) or {}
    return int(payload.get("timestamp_ms", 0))


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
        "metadata": state.metadata,
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
        metadata=payload.get("metadata") or {},
        id=str(point_id),
    )
