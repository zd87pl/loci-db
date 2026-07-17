"""Async LociClient — concurrent data + summary searches for high-throughput workloads.

Storage layout (bounded collection set): raw points live in one
``{prefix}loci_data`` collection and consolidated summaries in one
``{prefix}loci_summary`` collection.  Epochs are a purely logical
concept — the unit of consolidation granularity and of Hilbert
t-normalisation — never a collection.
"""

from __future__ import annotations

import asyncio
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
    FilterSelector,
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
from loci.temporal.consolidation import (
    ConsolidationPolicy,
    coarse_id,
    coarse_time_range,
    consolidate_states,
    data_collection_name,
    fold_cutoff_ms,
    summary_collection_name,
)
from loci.temporal.decay import DEFAULT_DECAY_LAMBDA, apply_decay
from loci.temporal.retention import RetentionManager, RetentionPolicy
from loci.temporal.sharding import epoch_id

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
    """Async high-level client with concurrent collection fan-out.

    Query operations search the raw data collection and the summary
    collection concurrently using ``asyncio.gather``.

    Args:
        qdrant_url: URL of the Qdrant instance.
        epoch_size_ms: Width of each logical temporal epoch in milliseconds.
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
        retention_policy: Optional policy that purges expired raw points
            from the data collection (cutoff-based; never touches the
            summary collection).
        consolidation_policy: Optional policy that summarises raw epochs
            leaving the raw window into the ``loci_summary`` collection,
            then deletes their raw points (episodic-to-semantic memory
            aging).  Works standalone or combined with ``retention_policy``;
            when both are set, consolidation runs first on each maintenance
            pass and retention's purge applies to whatever raw points
            remain.  Summaries are never purged or re-consolidated by
            retention.  Note: a retention policy tighter than the raw
            window (fewer than ``raw_window_epochs + 1`` epochs retained)
            purges raw points before consolidation can fold them.
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
        consolidation_policy: ConsolidationPolicy | None = None,
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
        self._data_collection = data_collection_name(collection_prefix)
        self._summary_collection = summary_collection_name(collection_prefix)
        # Per-collection readiness cache and create locks — exactly two real
        # collections, so exactly two locks (serialising the check-then-create).
        self._collection_ready: dict[str, bool] = {}
        self._collection_locks: dict[str, asyncio.Lock] = {
            self._data_collection: asyncio.Lock(),
            self._summary_collection: asyncio.Lock(),
        }
        # Frontier of time this client has inserted; maintenance clocks off
        # max(wall clock, frontier) so future-dated streams still age out.
        self._newest_timestamp_ms = 0
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
            RetentionManager(policy=retention_policy, epoch_size_ms=self._epoch_size_ms)
            if retention_policy is not None
            else None
        )
        self._consolidation_policy = consolidation_policy

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

    async def _ensure_data_collection(self) -> None:
        await self._ensure_collection(self._data_collection, hilbert_indices=True)

    async def _ensure_summary_collection(self) -> None:
        # Summaries carry no Hilbert payload; spatial queries reach them
        # via the exact post-filter on x/y/z.
        await self._ensure_collection(self._summary_collection, hilbert_indices=False)

    async def _ensure_collection(self, name: str, *, hilbert_indices: bool) -> None:
        """Create a Qdrant collection if it does not already exist.

        Idempotent and async-safe: the collection's lock serialises the
        check-then-create within this process, and a create conflict from a
        concurrent external writer is treated as success (the race winner
        also creates the payload indexes).
        """
        if self._collection_ready.get(name):
            return
        async with self._collection_locks[name]:
            if self._collection_ready.get(name):
                return
            if not await self._collection_exists(name):
                await self._create_collection(name, hilbert_indices=hilbert_indices)
            self._collection_ready[name] = True

    async def _create_collection(self, name: str, *, hilbert_indices: bool) -> None:
        try:
            await self._qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )
        except Exception as exc:
            # A concurrent writer won the create race; treat as success
            # (the winner also creates the payload indexes).
            if not _is_already_exists_error(exc):
                raise
            return
        index_tasks: list[Any] = []
        if hilbert_indices:
            index_tasks.extend(
                self._qdrant.create_payload_index(
                    collection_name=name,
                    field_name=f"hilbert_r{r}",
                    field_schema=PayloadSchemaType.INTEGER,
                )
                for r in self._hilbert.resolutions
            )
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

    async def _collection_exists(self, name: str) -> bool:
        """Probe Qdrant for a collection using the HTTP server's 404 contract."""
        try:
            await self._qdrant.get_collection(name)
            return True
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            return False
        except ValueError as exc:
            # qdrant-client's local (":memory:"/path) engine signals a
            # missing collection with ValueError instead of an HTTP 404.
            if "not found" not in str(exc).lower():
                raise
            return False

    async def _searchable(self, name: str) -> bool:
        """True when a query/scan should touch *name*.

        Cached once a collection is known to exist (created by us or found
        by a probe — another writer may have created it); a failed probe is
        treated as "not there yet" so reads degrade gracefully.
        """
        if self._collection_ready.get(name):
            return True
        try:
            exists = await self._collection_exists(name)
        except Exception:
            logger.debug("Existence probe failed for %s", name, exc_info=True)
            return False
        if exists:
            self._collection_ready[name] = True
        return exists

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._vector_size:
            raise ValueError(f"vector has dimension {len(vector)}, expected {self._vector_size}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def insert(self, state: WorldState) -> str:
        """Insert a single WorldState into the store.

        The input *state* is not mutated.

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
        await self._ensure_data_collection()

        t_norm = _normalise_time(state.timestamp_ms, ep, self._epoch_size_ms)
        hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)

        if self._adaptive is not None:
            self._adaptive.record(state.x, state.y, state.z, t_norm)

        payload = _state_to_payload(state, hilbert_ids)

        # Causal linking
        predecessor = await self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
        if predecessor is not None:
            payload["prev_state_id"] = predecessor
            await self._patch_next_link(predecessor, point_id)

        await self._retry(
            self._qdrant.upsert,
            collection_name=self._data_collection,
            points=[PointStruct(id=point_id, vector=state.vector, payload=payload)],
        )
        self._newest_timestamp_ms = max(self._newest_timestamp_ms, state.timestamp_ms)
        await self._maybe_purge()
        return point_id

    async def insert_batch(self, states: list[WorldState]) -> list[str]:
        """Insert a batch of WorldStates — one upsert into the data collection.

        Within a batch, states in the same scene are causally linked in
        timestamp order.  Input states are not mutated.

        Args:
            states: List of world states.

        Returns:
            List of assigned IDs (same order as *states*).
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("insert_batch is not supported in cloud mode")
        for state in states:
            self._validate_vector(state.vector)
        await self._ensure_data_collection()

        id_by_index: dict[int, str] = {}
        scene_chains: dict[str, str] = {}  # scene_id → latest point_id in the batch
        points: list[PointStruct] = []

        # Sort by (scene_id, timestamp) to build correct causal chains
        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))

        for orig_idx, state in indexed:
            point_id = str(uuid.uuid4())
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            t_norm = _normalise_time(state.timestamp_ms, ep, self._epoch_size_ms)
            hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)
            if self._adaptive is not None:
                self._adaptive.record(state.x, state.y, state.z, t_norm)

            payload = _state_to_payload(state, hilbert_ids)

            # Causal link within the batch; the first state per scene links to
            # the latest predecessor already in the store (matching the
            # sequential-insert behaviour).
            if state.scene_id:
                if state.scene_id in scene_chains:
                    prev_id: str | None = scene_chains[state.scene_id]
                else:
                    prev_id = await self._find_latest_predecessor(
                        state.scene_id, state.timestamp_ms
                    )
                if prev_id is not None:
                    payload["prev_state_id"] = prev_id
                scene_chains[state.scene_id] = point_id

            points.append(PointStruct(id=point_id, vector=state.vector, payload=payload))

        if points:
            await self._retry(
                self._qdrant.upsert, collection_name=self._data_collection, points=points
            )

        # Patch next_state_id for intra-batch links
        for point in points:
            prev_link_id = (point.payload or {}).get("prev_state_id")
            if prev_link_id:
                await self._patch_next_link(str(prev_link_id), str(point.id))

        if states:
            self._newest_timestamp_ms = max(
                self._newest_timestamp_ms, max(s.timestamp_ms for s in states)
            )
        await self._maybe_purge()
        return [id_by_index[i] for i in range(len(states))]

    # ------------------------------------------------------------------
    # Read — concurrent fan-out
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
        """Search for nearest neighbours with spatial and temporal filtering.

        The data and summary collections are searched concurrently.

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

        Runs ONE search over the ``loci_data`` collection with
        must-conditions [timestamp Range (when a window or epoch
        restriction is given), Hilbert MatchAny (when spatial bounds are
        given)], PLUS one search over ``loci_summary`` (vector + timestamp
        Range only — summaries carry no Hilbert payload), concurrently via
        ``asyncio.gather``, then merges both result sets before the exact
        post-filter and decay re-rank.

        Hilbert bucket selection: when the time window maps to a SINGLE
        epoch, the cover uses that epoch's normalised t-bounds (full 4D
        selectivity).  When the window spans multiple epochs, or there is
        no window, the cover is computed with the full t range [0, 1] —
        spatial-only selectivity — and the timestamp Range condition
        carries the t-dimension instead.

        Scores follow the higher-is-better convention for every distance
        metric: raw Qdrant euclidean distances (smaller-is-better) are
        negated at the boundary so decay re-ranking, merging, and
        truncation behave identically across metrics and backends.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("query_scored is not supported in cloud mode")

        if _epoch_ids is not None and not _epoch_ids:
            # Explicit empty epoch restriction (funnel dead-end): no results.
            return []

        # Timestamp Range: intersection of the query window and the epoch
        # restriction's envelope (funnel narrowing).
        ts_lo, ts_hi = self._timestamp_range(time_window_ms, _epoch_ids)

        # Over-fetch whenever post-search filtering (spatial exact match,
        # min_confidence, epoch membership) or decay re-ranking can
        # reorder/drop hits, else truncation could evict the top-k.
        needs_overfetch = (
            spatial_bounds is not None
            or min_confidence is not None
            or _epoch_ids is not None
            or self._decay_lambda > 0
        )
        fetch_limit = limit * _EXACT_FILTER_OVERFETCH if needs_overfetch else limit

        base_conditions: list[FieldCondition] = []
        if ts_lo is not None:
            base_conditions.append(
                FieldCondition(key="timestamp_ms", range=Range(gte=ts_lo, lte=ts_hi))
            )
        if min_confidence is not None:
            base_conditions.append(
                FieldCondition(key="confidence", range=Range(gte=min_confidence))
            )
        base_conditions.extend(extra_filter_to_conditions(_extra_payload_filter))

        # --- Raw data search (Hilbert pre-filter + timestamp Range) ---
        data_conditions = list(base_conditions)
        skip_data = False
        if spatial_bounds is not None:
            hids, field = self._hilbert_cover(spatial_bounds, time_window_ms, overlap_factor)
            if hids:
                data_conditions.append(FieldCondition(key=field, match=MatchAny(any=hids)))
            else:
                skip_data = True  # no bucket overlaps the requested region

        search_jobs: list[tuple[str, list[FieldCondition], bool]] = []
        if not skip_data and await self._searchable(self._data_collection):
            search_jobs.append((self._data_collection, data_conditions, False))
        # --- Summary search (no Hilbert condition) ---
        if await self._searchable(self._summary_collection):
            search_jobs.append((self._summary_collection, base_conditions, True))

        batches = await asyncio.gather(
            *(
                self._search_collection(name, vector, conditions, fetch_limit, summary=summary)
                for name, conditions, summary in search_jobs
            )
        )
        failed = sum(1 for batch in batches if batch is None)
        if search_jobs and failed == len(search_jobs):
            logger.warning("All %d collection searches failed; returning no results", failed)
        all_results: list[dict] = []
        for batch in batches:
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
        if _epoch_ids is not None:
            all_results = [r for r in all_results if self._epoch_restriction_ok(r, _epoch_ids)]

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

    # ------------------------------------------------------------------
    # Temporal navigation
    # ------------------------------------------------------------------

    async def get_trajectory(
        self,
        state_id: str,
        steps_back: int = 10,
        steps_forward: int = 10,
    ) -> list[WorldState]:
        """Reconstruct a trajectory using the scroll API with a scene_id filter.

        Scans the raw data collection only — summaries are naturally
        excluded by collection.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_trajectory is not supported in cloud mode")
        anchor = await self._get_state_by_id(state_id)
        if anchor is None:
            return []
        if not anchor.scene_id:
            return [anchor]

        try:
            points = await self._scroll_all(
                collection=self._data_collection,
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
        except Exception as exc:
            logger.warning("Trajectory scroll failed on %s: %s", self._data_collection, exc)
            return [anchor]
        all_states = [_payload_to_state(pt.payload, pt.id, _point_vector(pt)) for pt in points]

        # Sort by timestamp and find anchor position
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
        """Return all states within ±window_ms in the same scene_id.

        A single Qdrant scroll over the raw data collection with a
        scene_id + timestamp range filter; summaries are naturally
        excluded by collection.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_causal_context is not supported in cloud mode")
        anchor = await self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms

        try:
            points = await self._scroll_all(
                collection=self._data_collection,
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
        except Exception as exc:
            logger.warning("Causal-context scroll failed on %s: %s", self._data_collection, exc)
            return []
        context = [_payload_to_state(pt.payload, pt.id, _point_vector(pt)) for pt in points]

        context.sort(key=lambda s: s.timestamp_ms)
        return context

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _timestamp_range(
        self,
        time_window_ms: tuple[int, int] | None,
        epoch_ids: set[int] | None,
    ) -> tuple[int | None, int | None]:
        """Inclusive timestamp Range for the search must-conditions.

        Intersects the explicit query window with the envelope of an
        internal epoch restriction (funnel narrowing).  The epoch envelope
        is widened to whole coarse groups so summaries representing a
        requested epoch stay reachable; exact membership is enforced by
        :meth:`_epoch_restriction_ok` afterwards.
        """
        lo: int | None = None
        hi: int | None = None
        if time_window_ms is not None:
            lo, hi = time_window_ms
        if epoch_ids:
            e_lo = min(epoch_ids) * self._epoch_size_ms
            e_hi = (max(epoch_ids) + 1) * self._epoch_size_ms - 1
            policy = self._consolidation_policy
            if policy is not None:
                first = coarse_id(min(epoch_ids), policy)
                last = coarse_id(max(epoch_ids), policy)
                e_lo = min(e_lo, coarse_time_range(first, policy, self._epoch_size_ms)[0])
                e_hi = max(e_hi, coarse_time_range(last, policy, self._epoch_size_ms)[1])
            lo = e_lo if lo is None else max(lo, e_lo)
            hi = e_hi if hi is None else min(hi, e_hi)
        return lo, hi

    def _hilbert_cover(
        self,
        spatial_bounds: dict,
        time_window_ms: tuple[int, int] | None,
        overlap_factor: float,
    ) -> tuple[list[int], str]:
        """Hilbert bucket cover and payload field for a spatial query.

        Single-epoch windows use that epoch's normalised t-bounds; anything
        else covers the full t range [0, 1] (see :meth:`query_scored`).
        """
        single_epoch = time_window_ms is not None and epoch_id(
            time_window_ms[0], self._epoch_size_ms
        ) == epoch_id(time_window_ms[1], self._epoch_size_ms)
        if single_epoch:
            assert time_window_ms is not None
            ep = epoch_id(time_window_ms[0], self._epoch_size_ms)
            window: tuple[int, int] | None = time_window_ms
        else:
            ep = 0
            window = None  # bounds_for_epoch then yields t in [0, 1]
        resolution = choose_query_resolution(
            self._hilbert,
            self._adaptive,
            spatial_bounds,
            window,
            ep,
            self._epoch_size_ms,
            overlap_factor,
        )
        hids = self._hilbert.query_buckets(
            bounds_for_epoch(spatial_bounds, window, ep, self._epoch_size_ms),
            resolution=resolution,
            overlap_factor=overlap_factor,
        )
        return hids, self._hilbert.payload_field(resolution)

    def _epoch_restriction_ok(self, result: dict, epoch_ids: set[int]) -> bool:
        """Exact epoch-membership post-filter for an internal restriction.

        Raw points must lie in a requested epoch.  Summaries represent a
        whole coarse group, so they pass when their coarse group covers any
        requested epoch (matching the old per-collection pruning).
        """
        ts = result["payload"].get("timestamp_ms", 0)
        ep = epoch_id(ts, self._epoch_size_ms)
        if not result["summary"]:
            return ep in epoch_ids
        policy = self._consolidation_policy
        if policy is None:
            return False
        return coarse_id(ep, policy) in {coarse_id(e, policy) for e in epoch_ids}

    async def _search_collection(
        self,
        name: str,
        vector: list[float],
        conditions: list[FieldCondition],
        limit: int,
        *,
        summary: bool,
    ) -> list[dict] | None:
        """Run one vector search; returns hit dicts, or None when it failed."""
        query_filter = Filter(must=list(conditions)) if conditions else None
        try:
            resp = await self._retry(
                self._qdrant.query_points,
                collection_name=name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_vectors=True,
            )
        except Exception as exc:
            logger.warning("Search failed on %s: %s", name, exc)
            return None
        return [self._hit_to_result(hit, summary=summary) for hit in resp.points]

    def _hit_to_result(self, hit: Any, *, summary: bool) -> dict:
        score = float(hit.score)
        if self._distance == Distance.EUCLID:
            # Qdrant returns raw euclidean distances (smaller is better);
            # negate so all downstream code sees higher-is-better scores.
            score = -score
        payload = hit.payload or {}
        return {
            "score": score,
            "timestamp_ms": payload.get("timestamp_ms", 0),
            "payload": payload,
            "vector": hit.vector,
            "id": hit.id,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Maintenance (consolidation first, then retention)
    # ------------------------------------------------------------------

    async def _maybe_purge(self) -> None:
        """Run maintenance: consolidation first, then retention purge."""
        now_ms = self._maintenance_now_ms()
        await self._maybe_consolidate(now_ms)
        if self._retention is None:
            return
        try:
            await self._retention.maybe_purge_async(now_ms, self._delete_raw_before)
        except Exception as exc:
            logger.warning("Retention purge failed: %s", exc)

    def _maintenance_now_ms(self) -> int:
        """Maintenance clock: max(wall clock, newest inserted timestamp).

        Future-dated streams (simulations, replays) advance the clock so
        their old points still leave the raw/retention windows.
        """
        return max(int(time.time() * 1000), self._newest_timestamp_ms)

    async def _delete_raw_before(self, cutoff_ms: int) -> None:
        """Retention deleter: purge raw points below the cutoff.

        Never touches the summary collection.  Qdrant deletes do not report
        a count, so this returns None (the manager tolerates that).
        """
        await self._delete_time_range(self._data_collection, 0, cutoff_ms)

    async def _delete_time_range(self, collection: str, t_min_ms: int, t_max_ms: int) -> None:
        """Delete points with ``t_min_ms <= timestamp_ms < t_max_ms``."""
        await self._retry(
            self._qdrant.delete,
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp_ms",
                            range=Range(gte=t_min_ms, lt=t_max_ms),
                        )
                    ]
                )
            ),
        )

    async def _maybe_consolidate(self, now_ms: int) -> None:
        """Fold raw epochs that left the raw window into the summary collection."""
        policy = self._consolidation_policy
        if policy is None:
            return
        cutoff_ms = fold_cutoff_ms(now_ms, self._epoch_size_ms, policy)
        if cutoff_ms <= 0:
            return
        stale_filter = Filter(must=[FieldCondition(key="timestamp_ms", range=Range(lt=cutoff_ms))])
        try:
            # Cheap staleness probe (limit-1) before the full scan.
            hits = await self._qdrant.scroll(
                collection_name=self._data_collection,
                scroll_filter=stale_filter,
                limit=1,
            )
            probe = hits[0] if isinstance(hits, tuple) else hits
            if not probe:
                return
            stale = await self._scroll_all(
                collection=self._data_collection,
                scroll_filter=stale_filter,
                with_vectors=True,
            )
        except Exception:
            logger.warning("Consolidation scan failed on %s", self._data_collection, exc_info=True)
            return
        by_epoch: dict[int, list[Any]] = {}
        for pt in stale:
            ep = epoch_id(_point_timestamp(pt), self._epoch_size_ms)
            by_epoch.setdefault(ep, []).append(pt)
        for ep in sorted(by_epoch):
            try:
                await self._consolidate_epoch(ep, by_epoch[ep], policy)
            except Exception:
                logger.warning("Consolidation failed for epoch %d", ep, exc_info=True)

    async def _consolidate_epoch(
        self, ep: int, raw_points: list[Any], policy: ConsolidationPolicy
    ) -> None:
        """Fold one raw epoch into its coarse summary group.

        The coarse group's existing summaries (selected by timestamp Range
        over its coarse time range) are combined with the epoch's raw
        states and re-consolidated, keeping the group at
        ``max_states_per_scene`` states per scene; the old summaries are
        replaced (delete Range + upsert) and the epoch's raw points deleted.
        """
        coarse = coarse_id(ep, policy)
        t_min, t_max = coarse_time_range(coarse, policy, self._epoch_size_ms)
        summary_exists = await self._searchable(self._summary_collection)
        existing: list[Any] = []
        if summary_exists:
            existing = await self._scroll_all(
                collection=self._summary_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="timestamp_ms", range=Range(gte=t_min, lte=t_max))]
                ),
                with_vectors=True,
            )
        combined = [
            _payload_to_state(pt.payload, pt.id, _point_vector(pt))
            for pt in [*existing, *raw_points]
        ]
        summaries = consolidate_states(combined, policy, seed=ep)
        if summary_exists:
            await self._delete_time_range(self._summary_collection, t_min, t_max + 1)
        if summaries:
            await self._ensure_summary_collection()
            # Summaries carry no Hilbert payload; spatial queries reach them
            # via the exact post-filter on x/y/z.
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=s.vector,
                    payload=_state_to_payload(s, {}),
                )
                for s in summaries
            ]
            await self._retry(
                self._qdrant.upsert, collection_name=self._summary_collection, points=points
            )
        await self._delete_time_range(
            self._data_collection, ep * self._epoch_size_ms, (ep + 1) * self._epoch_size_ms
        )
        logger.info("Consolidated epoch %d into %s", ep, self._summary_collection)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_state_by_id(self, state_id: str) -> WorldState | None:
        """Retrieve a single state by ID (data collection first, then summaries)."""
        for col in (self._data_collection, self._summary_collection):
            try:
                results = await self._retry(
                    self._qdrant.retrieve,
                    collection_name=col,
                    ids=[state_id],
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception:  # noqa: S112  # missing collection or transient failure
                continue
            if results:
                return _payload_to_state(
                    results[0].payload, results[0].id, _point_vector(results[0])
                )
        return None

    async def _find_latest_predecessor(self, scene_id: str, before_ms: int) -> str | None:
        """Latest raw point in *scene_id* strictly before *before_ms*.

        A single filtered scroll over the data collection — summaries are
        naturally excluded by collection and never become predecessors.
        Scrolls unordered (server-side ordering breaks pagination) and
        picks the max-timestamp point client-side.
        """
        if not scene_id:
            return None
        try:
            points = await self._scroll_all(
                collection=self._data_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="scene_id", match=MatchValue(value=scene_id)),
                        FieldCondition(key="timestamp_ms", range=Range(lt=before_ms)),
                    ]
                ),
            )
        except Exception:
            logger.debug("Failed to find predecessor in %s", self._data_collection, exc_info=True)
            return None
        if points:
            return str(max(points, key=_point_timestamp).id)
        return None

    async def _patch_next_link(self, prev_id: str, next_id: str) -> None:
        """Update the predecessor's next_state_id payload field (best effort)."""
        try:
            await self._retry(
                self._qdrant.set_payload,
                collection_name=self._data_collection,
                payload={"next_state_id": next_id},
                points=[prev_id],
            )
        except Exception:
            logger.debug("Failed to patch next link %s→%s", prev_id, next_id, exc_info=True)

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


def _point_vector(point: Any) -> list[float]:
    """Plain vector of a Qdrant point (unwraps single-entry named-vector dicts)."""
    vec = point.vector
    if isinstance(vec, dict):
        vec = list(vec.values())[0] if vec else []
    return vec if vec is not None else []


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
