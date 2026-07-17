"""Main LociClient class — primary API surface for the Loci database."""

from __future__ import annotations

import contextlib
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
from loci.temporal.decay import DEFAULT_DECAY_LAMBDA, apply_decay
from loci.temporal.retention import RetentionManager, RetentionPolicy
from loci.temporal.sharding import collection_name, epoch_id

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
    return payload.get("timestamp_ms", 0)


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
    temporal sharding, and predict-then-retrieve on top.

    Args:
        qdrant_url: URL of the Qdrant instance (e.g. ``"http://localhost:6333"``).
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
        self._known_collections: set[str] = set()
        self._discovered = False
        self._collection_locks: dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()
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

    def _discover_collections(self, force: bool = False) -> None:
        """Merge Qdrant's collection listing into _known_collections.

        Runs once per client by default; pass ``force=True`` to refresh (used
        when a query targets an epoch this client has not seen yet, e.g. one
        created by another writer).
        """
        if self._discovered and not force:
            return
        prefix = f"{self._collection_prefix}loci_" if self._collection_prefix else "loci_"
        try:
            response = self._qdrant.get_collections()
            for col in response.collections:
                if col.name.startswith(prefix):
                    self._known_collections.add(col.name)
            self._discovered = True
        except Exception:
            logger.debug("Failed to discover collections", exc_info=True)

    def _refresh_for_window(self, first_ep: int, last_ep: int) -> None:
        """Re-run discovery when part of a requested epoch window is unknown."""
        known = sum(1 for ep in self._list_active_epochs() if first_ep <= ep <= last_ep)
        if known < (last_ep - first_ep + 1):
            self._discover_collections(force=True)

    def _epochs_intersecting(self, first_ep: int, last_ep: int) -> list[int]:
        """Known epochs within [first_ep, last_ep] without materialising the range."""
        return [ep for ep in self._list_active_epochs() if first_ep <= ep <= last_ep]

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._vector_size:
            raise ValueError(f"vector has dimension {len(vector)}, expected {self._vector_size}")

    def _ensure_collection(self, name: str) -> None:
        """Create a Qdrant collection if it does not already exist.

        Idempotent and thread-safe: a per-collection lock serialises the
        check-then-create within this process, and a create conflict from a
        concurrent external writer is treated as success.
        """
        if name in self._known_collections:
            return

        with self._locks_mutex:
            lock = self._collection_locks.setdefault(name, threading.Lock())
        with lock:
            if name in self._known_collections:
                return

            exists = False
            try:
                self._qdrant.get_collection(name)
                exists = True
            except UnexpectedResponse as exc:
                if exc.status_code != 404:
                    raise

            if not exists:
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
                    exists = True

            if not exists:
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

            self._known_collections.add(name)

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
        col = self._col_name(ep)
        self._ensure_collection(col)

        t_norm = self._normalise_time(state.timestamp_ms, ep)
        hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)

        if self._adaptive is not None:
            self._adaptive.record(state.x, state.y, state.z, t_norm)

        payload = self._state_to_payload(state, hilbert_ids)

        # Causal linking
        predecessor = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
        if predecessor is not None:
            prev_id, prev_col = predecessor
            payload["prev_state_id"] = prev_id
            self._patch_next_link(prev_id, point_id, collection_hint=prev_col)

        self._retry(
            self._qdrant.upsert,
            collection_name=col,
            points=[PointStruct(id=point_id, vector=state.vector, payload=payload)],
        )
        self._maybe_purge()
        return point_id

    def insert_batch(self, states: list[WorldState]) -> list[str]:
        """Insert a batch of WorldStates efficiently.

        Vectors are grouped by epoch and upserted in a single Qdrant
        call per collection.  Within a batch, states in the same scene
        are causally linked in timestamp order.  Input states are not
        mutated.

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
        id_by_index: dict[int, str] = {}
        scene_chains: dict[str, tuple[str, str]] = {}  # scene_id → (latest point_id, collection)
        prev_collection_by_point: dict[str, str] = {}

        # Sort by (scene_id, timestamp) to build correct causal chains
        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))

        for orig_idx, state in indexed:
            point_id = str(uuid.uuid4())
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            col = self._col_name(ep)
            self._ensure_collection(col)

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
                    prev_link: tuple[str, str] | None = scene_chains[state.scene_id]
                else:
                    prev_link = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
                if prev_link is not None:
                    prev_id, prev_col = prev_link
                    payload["prev_state_id"] = prev_id
                    prev_collection_by_point[point_id] = prev_col
                scene_chains[state.scene_id] = (point_id, col)

            groups.setdefault(col, []).append(
                PointStruct(id=point_id, vector=state.vector, payload=payload)
            )

        for col, points in groups.items():
            self._retry(self._qdrant.upsert, collection_name=col, points=points)

        # Patch next_state_id for intra-batch links
        for col, points in groups.items():
            for point in points:
                prev_link_id = (point.payload or {}).get("prev_state_id")
                if prev_link_id:
                    prev_id_str = str(prev_link_id)
                    try:
                        self._retry(
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
        """Search for nearest neighbours and return scores alongside states.

        Scores follow the higher-is-better convention for every distance
        metric: raw Qdrant euclidean distances (smaller-is-better) are
        negated at the boundary so decay re-ranking, cross-shard merging,
        and truncation behave identically across metrics and backends.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("query_scored is not supported in cloud mode")
        self._discover_collections()

        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            first_ep = epoch_id(start_ms, self._epoch_size_ms)
            last_ep = epoch_id(end_ms, self._epoch_size_ms)
            # Refresh discovery when part of the window is unknown — another
            # writer may have created those collections since discovery ran.
            self._refresh_for_window(first_ep, last_ep)
            epochs = self._epochs_intersecting(first_ep, last_ep)
        else:
            epochs = self._list_active_epochs()
        if _epoch_ids is not None:
            epochs = [ep for ep in epochs if ep in _epoch_ids]

        # Over-fetch per shard whenever post-search filtering (spatial exact
        # match, min_confidence) or decay re-ranking can reorder/drop hits;
        # otherwise per-shard truncation could evict the true top-k.
        needs_overfetch = (
            spatial_bounds is not None or min_confidence is not None or self._decay_lambda > 0
        )
        shard_limit = limit * _EXACT_FILTER_OVERFETCH if needs_overfetch else limit
        all_results: list[dict] = []
        shards_tried = 0
        shards_failed = 0
        for ep in epochs:
            col = self._col_name(ep)
            if col not in self._known_collections:
                continue

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
                    continue
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
            shards_tried += 1
            try:
                resp = self._retry(
                    self._qdrant.query_points,
                    collection_name=col,
                    query=vector,
                    query_filter=query_filter,
                    limit=shard_limit,
                    with_vectors=True,
                )
                hits = resp.points
            except Exception as exc:
                shards_failed += 1
                logger.warning("Search failed on shard %s: %s", col, exc)
                continue
            for hit in hits:
                score = float(hit.score)
                if self._distance == Distance.EUCLID:
                    # Qdrant returns raw euclidean distances (smaller is
                    # better); negate so all downstream code sees
                    # higher-is-better scores.
                    score = -score
                all_results.append(
                    {
                        "score": score,
                        "timestamp_ms": hit.payload.get("timestamp_ms", 0),
                        "payload": hit.payload,
                        "vector": hit.vector,
                        "id": hit.id,
                    }
                )

        if shards_tried > 0 and shards_failed == shards_tried:
            logger.warning("All %d shard searches failed; returning no results", shards_tried)

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
        """Reconstruct a trajectory using scroll API with scene_id filter.

        Uses a single Qdrant scroll call per shard (filtered by scene_id
        and ordered by timestamp) instead of N individual point lookups.

        Args:
            state_id: ID of the anchor state.
            steps_back: Number of predecessors to include.
            steps_forward: Number of successors to include.

        Returns:
            Ordered list of states from oldest to newest.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_trajectory is not supported in cloud mode")
        self._discover_collections()
        anchor = self._get_state_by_id(state_id)
        if anchor is None:
            # The anchor may live in a collection created by another writer
            # after our last discovery; refresh once and retry.
            self._discover_collections(force=True)
            anchor = self._get_state_by_id(state_id)
        if anchor is None:
            return []
        if not anchor.scene_id:
            return [anchor]

        all_states: list[WorldState] = []
        for col in list(self._known_collections):
            try:
                points = self._scroll_all(
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
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    all_states.append(self._payload_to_state(pt.payload, pt.id, vec))
            except Exception as exc:
                logger.warning("Trajectory scroll failed on shard %s: %s", col, exc)
                continue

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

        Uses a single Qdrant scroll query per shard with scene_id +
        timestamp range filter.

        Args:
            state_id: ID of the anchor state.
            window_ms: Time window radius in milliseconds.

        Returns:
            List of :class:`WorldState` sorted by timestamp.
        """
        if self._cloud is not None:
            raise CloudModeUnsupportedError("get_causal_context is not supported in cloud mode")
        self._discover_collections()
        anchor = self._get_state_by_id(state_id)
        if anchor is None:
            self._discover_collections(force=True)
            anchor = self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms
        self._refresh_for_window(
            epoch_id(t_min, self._epoch_size_ms), epoch_id(t_max, self._epoch_size_ms)
        )

        context: list[WorldState] = []
        for col in list(self._known_collections):
            try:
                points = self._scroll_all(
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
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    context.append(self._payload_to_state(pt.payload, pt.id, vec))
            except Exception as exc:
                logger.warning("Causal-context scroll failed on shard %s: %s", col, exc)
                continue

        context.sort(key=lambda s: s.timestamp_ms)
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_purge(self) -> None:
        if self._retention is None:
            return
        try:
            dropped = self._retention.maybe_purge(
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
            with self._locks_mutex:
                self._collection_locks.pop(name, None)

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
        """Retrieve a single state by its ID (scans known collections)."""
        for col in list(self._known_collections):
            try:
                results = self._retry(
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
                    return self._payload_to_state(results[0].payload, results[0].id, vec)
            except Exception:  # noqa: S112  # retry loop across epochs
                continue
        return None

    def _find_latest_predecessor(self, scene_id: str, before_ms: int) -> tuple[str, str] | None:
        """Find the most recent state in the same scene before a timestamp.

        Scrolls unordered (server-side ordering breaks pagination) and picks
        the max-timestamp point client-side. The scan is bounded per epoch
        collection by the scene and timestamp filters.
        """
        if not scene_id:
            return None
        self._discover_collections()
        for collection in self._predecessor_search_collections(before_ms):
            try:
                points = self._scroll_all(
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

    def _patch_next_link(
        self, prev_id: str, next_id: str, collection_hint: str | None = None
    ) -> None:
        """Update the predecessor's next_state_id payload field."""
        collections = list(self._known_collections)
        if collection_hint is not None:
            collections = [collection_hint] + [col for col in collections if col != collection_hint]
        for col in collections:
            try:
                self._retry(
                    self._qdrant.set_payload,
                    collection_name=col,
                    payload={"next_state_id": next_id},
                    points=[prev_id],
                )
                return
            except Exception:  # noqa: S112  # retry loop across epochs
                continue

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

    def _predecessor_search_collections(self, before_ms: int) -> list[str]:
        target_epoch = epoch_id(before_ms, self._epoch_size_ms)
        epochs = [ep for ep in self._list_active_epochs() if ep <= target_epoch]
        return [self._col_name(ep) for ep in sorted(epochs, reverse=True)]

    def _list_active_epochs(self) -> list[int]:
        """Return epoch IDs for all known collections."""
        prefix = f"{self._collection_prefix}loci_" if self._collection_prefix else "loci_"
        epochs: list[int] = []
        for col in self._known_collections:
            if col.startswith(prefix):
                with contextlib.suppress(ValueError):
                    epochs.append(int(col[len(prefix) :]))
        return sorted(epochs) if epochs else []
