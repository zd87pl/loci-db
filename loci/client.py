"""Main LociClient class — primary API surface for the Loci database.

Storage layout (bounded collection set): raw points live in one
``{prefix}loci_data`` collection and consolidated summaries in one
``{prefix}loci_summary`` collection.  Epochs are a purely logical
concept — the unit of consolidation granularity and of Hilbert
t-normalisation — never a collection.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, cast

from qdrant_client import QdrantClient
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

from loci.cloud_transport import CloudModeUnsupportedError, CloudTransport
from loci.payload_filters import extra_filter_to_conditions
from loci.retrieval.predict import PredictRetrieveResult, PredictThenRetrieve
from loci.retrieval.predict import predict_and_retrieve as _predict_and_retrieve
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


_EXACT_FILTER_OVERFETCH = 3
_SCROLL_PAGE_SIZE = 256

# Map public distance names to Qdrant enum values
_DISTANCE_MAP: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclidean": Distance.EUCLID,
}


class LociClient:
    """High-level client for inserting, querying, and navigating WorldStates.

    Wraps a Qdrant instance and adds Hilbert-curve spatial bucketing,
    logical temporal epochs, and predict-then-retrieve on top.  Raw points
    live in the single ``{prefix}loci_data`` collection; consolidated
    summaries live in ``{prefix}loci_summary``.

    Args:
        qdrant_url: URL of the Qdrant instance (e.g. ``"http://localhost:6333"``).
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

        # Cloud mode: both base_url and api_key provided → talk to LOCI Cloud API.
        # Local mode (default): qdrant_url is required and points at a Qdrant cluster.
        if base_url is not None:
            if api_key is None:
                raise ValueError("cloud mode requires api_key")
            self._cloud: CloudTransport | None = CloudTransport(base_url, api_key)
            # _qdrant is unused in cloud mode; keep type as QdrantClient so the
            # local-mode code paths type-check without pervasive None checks.
            self._qdrant: QdrantClient = cast(QdrantClient, None)
        else:
            if qdrant_url is None:
                raise ValueError("qdrant_url is required unless base_url is provided")
            self._cloud = None
            self._qdrant = QdrantClient(url=qdrant_url, api_key=api_key)
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
        self._collection_locks: dict[str, threading.Lock] = {
            self._data_collection: threading.Lock(),
            self._summary_collection: threading.Lock(),
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

    def _retry(self, fn, *args, **kwargs):
        """Execute fn with retry logic."""
        from loci.retry import with_retry

        wrapped = with_retry(self._max_retries, self._retry_backoff)(fn)
        return wrapped(*args, **kwargs)

    @property
    def density_stats(self):
        """Return adaptive resolution density stats, or None if not enabled."""
        return self._adaptive.stats() if self._adaptive is not None else None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_data_collection(self) -> None:
        self._ensure_collection(self._data_collection, hilbert_indices=True)

    def _ensure_summary_collection(self) -> None:
        # Summaries carry no Hilbert payload; spatial queries reach them
        # via the exact post-filter on x/y/z.
        self._ensure_collection(self._summary_collection, hilbert_indices=False)

    def _ensure_collection(self, name: str, *, hilbert_indices: bool) -> None:
        """Create a Qdrant collection if it does not already exist.

        Idempotent and thread-safe: the collection's lock serialises the
        check-then-create within this process, and a create conflict from a
        concurrent external writer is treated as success (the race winner
        also creates the payload indexes).
        """
        if self._collection_ready.get(name):
            return
        with self._collection_locks[name]:
            if self._collection_ready.get(name):
                return
            if not self._collection_exists(name):
                self._create_collection(name, hilbert_indices=hilbert_indices)
            self._collection_ready[name] = True

    def _create_collection(self, name: str, *, hilbert_indices: bool) -> None:
        try:
            self._qdrant.create_collection(
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
        if hilbert_indices:
            for r in self._hilbert.resolutions:
                self._qdrant.create_payload_index(
                    collection_name=name,
                    field_name=f"hilbert_r{r}",
                    field_schema=PayloadSchemaType.INTEGER,
                )
        self._qdrant.create_payload_index(
            collection_name=name,
            field_name="timestamp_ms",
            field_schema=PayloadSchemaType.INTEGER,
        )
        self._qdrant.create_payload_index(
            collection_name=name,
            field_name="scale_level",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._qdrant.create_payload_index(
            collection_name=name,
            field_name="scene_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def _collection_exists(self, name: str) -> bool:
        """Probe Qdrant for a collection using the HTTP server's 404 contract."""
        try:
            self._qdrant.get_collection(name)
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

    def _searchable(self, name: str) -> bool:
        """True when a query/scan should touch *name*.

        Cached once a collection is known to exist (created by us or found
        by a probe — another writer may have created it); a failed probe is
        treated as "not there yet" so reads degrade gracefully.
        """
        if self._collection_ready.get(name):
            return True
        try:
            exists = self._collection_exists(name)
        except Exception:
            logger.debug("Existence probe failed for %s", name, exc_info=True)
            return False
        if exists:
            self._collection_ready[name] = True
        return exists

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._vector_size:
            raise ValueError(f"vector has dimension {len(vector)}, expected {self._vector_size}")

    def close(self) -> None:
        """Close the underlying Qdrant connection (no-op in cloud mode)."""
        if self._cloud is None:
            self._qdrant.close()

    def __enter__(self) -> LociClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, state: WorldState) -> str:
        """Insert a single WorldState into the store.

        The input *state* is not mutated.

        Args:
            state: The world state to persist.

        Returns:
            The unique ID assigned to this state.
        """
        if self._cloud is not None:
            return self._cloud.insert(state)

        self._validate_vector(state.vector)
        point_id = str(uuid.uuid4())

        ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
        self._ensure_data_collection()

        t_norm = self._normalise_time(state.timestamp_ms, ep)
        hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)

        if self._adaptive is not None:
            self._adaptive.record(state.x, state.y, state.z, t_norm)

        payload = self._state_to_payload(state, hilbert_ids)

        # Causal linking
        predecessor = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
        if predecessor is not None:
            payload["prev_state_id"] = predecessor
            self._patch_next_link(predecessor, point_id)

        self._retry(
            self._qdrant.upsert,
            collection_name=self._data_collection,
            points=[PointStruct(id=point_id, vector=state.vector, payload=payload)],
        )
        self._newest_timestamp_ms = max(self._newest_timestamp_ms, state.timestamp_ms)
        self._maybe_purge()
        return point_id

    def insert_batch(self, states: list[WorldState]) -> list[str]:
        """Insert a batch of WorldStates efficiently.

        All points are upserted into the data collection in a single Qdrant
        call.  Within a batch, states in the same scene are causally linked
        in timestamp order.  Input states are not mutated.

        Args:
            states: List of world states.

        Returns:
            List of assigned IDs (same order as *states*).
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("insert_batch is not supported in cloud mode")
        for state in states:
            self._validate_vector(state.vector)
        self._ensure_data_collection()

        id_by_index: dict[int, str] = {}
        scene_chains: dict[str, str] = {}  # scene_id → latest point_id in the batch
        points: list[PointStruct] = []

        # Sort by (scene_id, timestamp) to build correct causal chains
        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))

        for orig_idx, state in indexed:
            point_id = str(uuid.uuid4())
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            t_norm = self._normalise_time(state.timestamp_ms, ep)
            hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)
            if self._adaptive is not None:
                self._adaptive.record(state.x, state.y, state.z, t_norm)

            payload = self._state_to_payload(state, hilbert_ids)

            # Causal link within the batch; the first state per scene links to
            # the latest predecessor already in the store (matching the
            # sequential-insert behaviour).
            if state.scene_id:
                if state.scene_id in scene_chains:
                    prev_id: str | None = scene_chains[state.scene_id]
                else:
                    prev_id = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
                if prev_id is not None:
                    payload["prev_state_id"] = prev_id
                scene_chains[state.scene_id] = point_id

            points.append(PointStruct(id=point_id, vector=state.vector, payload=payload))

        if points:
            self._retry(self._qdrant.upsert, collection_name=self._data_collection, points=points)

        # Patch next_state_id for intra-batch links
        for point in points:
            prev_link_id = (point.payload or {}).get("prev_state_id")
            if prev_link_id:
                self._patch_next_link(str(prev_link_id), str(point.id))

        if states:
            self._newest_timestamp_ms = max(
                self._newest_timestamp_ms, max(s.timestamp_ms for s in states)
            )
        self._maybe_purge()
        return [id_by_index[i] for i in range(len(states))]

    # ------------------------------------------------------------------
    # Read — standard
    # ------------------------------------------------------------------

    def query(
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

        Args:
            vector: Query embedding vector.
            spatial_bounds: Optional dict with keys ``x_min``, ``x_max``,
                ``y_min``, ``y_max``, ``z_min``, ``z_max``.
            time_window_ms: Optional ``(start_ms, end_ms)`` window.
            limit: Maximum number of results.
            overlap_factor: Expand spatial query by this factor to catch
                boundary points (default 1.2 = 20% expansion).

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
            return self._cloud.query(
                vector=vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window_ms,
                limit=limit,
                overlap_factor=overlap_factor,
            )

        return [
            candidate.state
            for candidate in self.query_scored(
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

    def query_scored(
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
        Range only — summaries carry no Hilbert payload), then merges both
        result sets before the exact post-filter and decay re-rank.

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
        if not skip_data and self._searchable(self._data_collection):
            search_jobs.append((self._data_collection, data_conditions, False))
        # --- Summary search (no Hilbert condition) ---
        if self._searchable(self._summary_collection):
            search_jobs.append((self._summary_collection, base_conditions, True))

        batches = [
            self._search_collection(name, vector, conditions, fetch_limit, summary=summary)
            for name, conditions, summary in search_jobs
        ]
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

        now_ms = int(time.time() * 1000)
        apply_decay(all_results, now_ms, self._decay_lambda)
        all_results = all_results[:limit]

        return [
            ScoredWorldState(
                state=self._payload_to_state(r["payload"], r["id"], r["vector"]),
                score=float(r["score"]),
                decayed_score=float(r.get("decayed_score", r["score"])),
            )
            for r in all_results
        ]

    # ------------------------------------------------------------------
    # Read — novel primitive
    # ------------------------------------------------------------------

    def predict_and_retrieve(
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
        """Predict a future state then retrieve nearest neighbours to it.

        When ``current_position`` is provided, returns a full
        :class:`PredictRetrieveResult` with novelty scoring and timing.
        Otherwise falls back to the legacy API returning a plain list.
        By default the retrieval step searches stored history for analogs;
        pass ``search_time_window_ms`` to restrict it to an absolute time range.

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
            :class:`PredictRetrieveResult` when current_position is set,
            otherwise a plain list of :class:`WorldState`.

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
        if current_position is not None or return_prediction:
            ptr = PredictThenRetrieve(self, calibrator=calibrator)
            return ptr.retrieve(
                context_vector=context_vector,
                predictor_fn=predictor_fn,
                future_horizon_ms=future_horizon_ms,
                current_position=current_position,
                current_timestamp_ms=current_timestamp_ms,
                spatial_search_radius=spatial_search_radius,
                limit=limit,
                alpha=alpha,
                return_prediction=return_prediction,
                search_time_window_ms=search_time_window_ms,
            )
        return _predict_and_retrieve(
            self,
            context_vector,
            predictor_fn,
            future_horizon_ms=future_horizon_ms,
            limit=limit,
            search_time_window_ms=search_time_window_ms,
        )

    def funnel_query(
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
        from loci.retrieval.funnel import funnel_search

        return funnel_search(self, vector, spatial_bounds, time_window_ms, limit)

    # ------------------------------------------------------------------
    # Temporal navigation
    # ------------------------------------------------------------------

    def get_trajectory(
        self,
        state_id: str,
        steps_back: int = 10,
        steps_forward: int = 10,
    ) -> list[WorldState]:
        """Reconstruct a trajectory using the scroll API with a scene_id filter.

        Scans the raw data collection only — summaries are naturally
        excluded by collection.

        Args:
            state_id: ID of the anchor state.
            steps_back: Number of predecessors to include.
            steps_forward: Number of successors to include.

        Returns:
            Ordered list of states from oldest to newest.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_trajectory is not supported in cloud mode")
        anchor = self._get_state_by_id(state_id)
        if anchor is None:
            return []
        if not anchor.scene_id:
            return [anchor]

        try:
            points = self._scroll_all(
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
        all_states = [self._payload_to_state(pt.payload, pt.id, _point_vector(pt)) for pt in points]

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

    def get_causal_context(
        self,
        state_id: str,
        window_ms: int = 5000,
    ) -> list[WorldState]:
        """Return all states within ±window_ms of the given state's timestamp
        in the same scene_id — the 'episodic context window'.

        A single Qdrant scroll over the raw data collection with a
        scene_id + timestamp range filter; summaries are naturally
        excluded by collection.

        Args:
            state_id: ID of the anchor state.
            window_ms: Time window radius in milliseconds.

        Returns:
            List of :class:`WorldState` sorted by timestamp.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_causal_context is not supported in cloud mode")
        anchor = self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms

        try:
            points = self._scroll_all(
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
        context = [self._payload_to_state(pt.payload, pt.id, _point_vector(pt)) for pt in points]

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

    def _search_collection(
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
            resp = self._retry(
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

    def _maybe_purge(self) -> None:
        """Run maintenance: consolidation first, then retention purge."""
        now_ms = self._maintenance_now_ms()
        self._maybe_consolidate(now_ms)
        if self._retention is None:
            return
        try:
            self._retention.maybe_purge(now_ms, self._delete_raw_before)
        except Exception as exc:
            logger.warning("Retention purge failed: %s", exc)

    def _maintenance_now_ms(self) -> int:
        """Maintenance clock: max(wall clock, newest inserted timestamp).

        Future-dated streams (simulations, replays) advance the clock so
        their old points still leave the raw/retention windows.
        """
        return max(int(time.time() * 1000), self._newest_timestamp_ms)

    def _delete_raw_before(self, cutoff_ms: int) -> None:
        """Retention deleter: purge raw points below the cutoff.

        Never touches the summary collection.  Qdrant deletes do not report
        a count, so this returns None (the manager tolerates that).
        """
        self._delete_time_range(self._data_collection, 0, cutoff_ms)

    def _delete_time_range(self, collection: str, t_min_ms: int, t_max_ms: int) -> None:
        """Delete points with ``t_min_ms <= timestamp_ms < t_max_ms``."""
        self._retry(
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

    def _maybe_consolidate(self, now_ms: int) -> None:
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
            hits = self._qdrant.scroll(
                collection_name=self._data_collection,
                scroll_filter=stale_filter,
                limit=1,
            )
            probe = hits[0] if isinstance(hits, tuple) else hits
            if not probe:
                return
            stale = self._scroll_all(
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
                self._consolidate_epoch(ep, by_epoch[ep], policy)
            except Exception:
                logger.warning("Consolidation failed for epoch %d", ep, exc_info=True)

    def _consolidate_epoch(
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
        summary_exists = self._searchable(self._summary_collection)
        existing: list[Any] = []
        if summary_exists:
            existing = self._scroll_all(
                collection=self._summary_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="timestamp_ms", range=Range(gte=t_min, lte=t_max))]
                ),
                with_vectors=True,
            )
        combined = [
            self._payload_to_state(pt.payload, pt.id, _point_vector(pt))
            for pt in [*existing, *raw_points]
        ]
        summaries = consolidate_states(combined, policy, seed=ep)
        if summary_exists:
            self._delete_time_range(self._summary_collection, t_min, t_max + 1)
        if summaries:
            self._ensure_summary_collection()
            # Summaries carry no Hilbert payload; spatial queries reach them
            # via the exact post-filter on x/y/z.
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=s.vector,
                    payload=self._state_to_payload(s, {}),
                )
                for s in summaries
            ]
            self._retry(
                self._qdrant.upsert, collection_name=self._summary_collection, points=points
            )
        self._delete_time_range(
            self._data_collection, ep * self._epoch_size_ms, (ep + 1) * self._epoch_size_ms
        )
        logger.info("Consolidated epoch %d into %s", ep, self._summary_collection)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_time(self, timestamp_ms: int, ep: int) -> float:
        """Map a timestamp to [0, 1] within its epoch."""
        epoch_start = ep * self._epoch_size_ms
        offset = timestamp_ms - epoch_start
        return min(1.0, max(0.0, offset / self._epoch_size_ms))

    @staticmethod
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

    @staticmethod
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

    def _get_state_by_id(self, state_id: str) -> WorldState | None:
        """Retrieve a single state by ID (data collection first, then summaries)."""
        for col in (self._data_collection, self._summary_collection):
            try:
                results = self._retry(
                    self._qdrant.retrieve,
                    collection_name=col,
                    ids=[state_id],
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception:  # noqa: S112  # missing collection or transient failure
                continue
            if results:
                return self._payload_to_state(
                    results[0].payload, results[0].id, _point_vector(results[0])
                )
        return None

    def _find_latest_predecessor(self, scene_id: str, before_ms: int) -> str | None:
        """Latest raw point in *scene_id* strictly before *before_ms*.

        A single filtered scroll over the data collection — summaries are
        naturally excluded by collection and never become predecessors.
        Scrolls unordered (server-side ordering breaks pagination) and
        picks the max-timestamp point client-side.
        """
        if not scene_id:
            return None
        try:
            points = self._scroll_all(
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

    def _patch_next_link(self, prev_id: str, next_id: str) -> None:
        """Update the predecessor's next_state_id payload field (best effort)."""
        try:
            self._retry(
                self._qdrant.set_payload,
                collection_name=self._data_collection,
                payload={"next_state_id": next_id},
                points=[prev_id],
            )
        except Exception:
            logger.debug("Failed to patch next link %s→%s", prev_id, next_id, exc_info=True)

    def _scroll_all(
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
            hits = self._retry(
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
