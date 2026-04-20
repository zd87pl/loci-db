"""Main LociClient class — primary API surface for the Loci database."""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from typing import cast

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


class LociClient:
    """High-level client for inserting, querying, and navigating WorldStates.

    Wraps a Qdrant instance and adds Hilbert-curve spatial bucketing,
    temporal sharding, and predict-then-retrieve on top.

    Args:
        qdrant_url: URL of the Qdrant instance (e.g. ``"http://localhost:6333"``).
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
        collection_prefix: str = "",
        base_url: str | None = None,
    ) -> None:
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
        if distance not in _DISTANCE_MAP:
            raise ValueError(f"distance must be one of {list(_DISTANCE_MAP)}, got {distance!r}")
        self._distance = _DISTANCE_MAP[distance]
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._collection_prefix = collection_prefix
        self._known_collections: set[str] = set()
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

    def _discover_collections(self) -> None:
        """Populate _known_collections from Qdrant (for read-only clients)."""
        if self._known_collections:
            return
        try:
            response = self._qdrant.get_collections()
            for col in response.collections:
                if col.name.startswith("loci_"):
                    self._known_collections.add(col.name)
        except Exception:
            logger.debug("Failed to discover collections", exc_info=True)

    def _ensure_collection(self, name: str) -> None:
        """Create a Qdrant collection if it does not already exist (idempotent)."""
        if name in self._known_collections:
            return

        exists = False
        try:
            self._qdrant.get_collection(name)
            exists = True
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
        except Exception:
            raise

        if not exists:
            self._qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )
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

        point_id = uuid.uuid4().hex

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
        groups: dict[str, list[PointStruct]] = {}
        id_by_index: dict[int, str] = {}
        scene_chains: dict[str, tuple[str, str]] = {}  # scene_id → (latest point_id, collection)
        prev_collection_by_point: dict[str, str] = {}

        # Sort by (scene_id, timestamp) to build correct causal chains
        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))

        for orig_idx, state in indexed:
            point_id = uuid.uuid4().hex
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            col = self._col_name(ep)
            self._ensure_collection(col)

            t_norm = self._normalise_time(state.timestamp_ms, ep)
            hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)
            if self._adaptive is not None:
                self._adaptive.record(state.x, state.y, state.z, t_norm)

            payload = self._state_to_payload(state, hilbert_ids)

            # Causal link within the batch
            if state.scene_id and state.scene_id in scene_chains:
                prev_id, prev_col = scene_chains[state.scene_id]
                payload["prev_state_id"] = prev_id
                prev_collection_by_point[point_id] = prev_col
            if state.scene_id:
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
        """Search for nearest neighbours and return scores alongside states."""
        self._discover_collections()

        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            epochs = epochs_in_range(start_ms, end_ms, self._epoch_size_ms)
        else:
            epochs = self._list_active_epochs()
        if _epoch_ids is not None:
            epochs = [ep for ep in epochs if ep in _epoch_ids]

        shard_limit = limit * _EXACT_FILTER_OVERFETCH if spatial_bounds is not None else limit
        all_results: list[dict] = []
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

            must_conditions.extend(extra_filter_to_conditions(_extra_payload_filter))

            query_filter = Filter(must=must_conditions) if must_conditions else None
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
            except Exception:  # noqa: S112  # retry loop across shards
                continue
            for hit in hits:
                all_results.append(
                    {
                        "score": hit.score,
                        "timestamp_ms": hit.payload.get("timestamp_ms", 0),
                        "payload": hit.payload,
                        "vector": hit.vector,
                        "id": hit.id,
                    }
                )

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
        spatial_search_radius: float = 0.3,
        alpha: float = 0.7,
        return_prediction: bool = False,
    ) -> list[WorldState] | PredictRetrieveResult:
        """Predict a future state then retrieve nearest neighbours to it.

        When ``current_position`` is provided, returns a full
        :class:`PredictRetrieveResult` with novelty scoring and timing.
        Otherwise falls back to the legacy API returning a plain list.

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
            :class:`PredictRetrieveResult` when current_position is set,
            otherwise a plain list of :class:`WorldState`.
        """
        if current_position is not None or return_prediction:
            ptr = PredictThenRetrieve(self)
            return ptr.retrieve(
                context_vector=context_vector,
                predictor_fn=predictor_fn,
                future_horizon_ms=future_horizon_ms,
                current_position=current_position,
                spatial_search_radius=spatial_search_radius,
                limit=limit,
                alpha=alpha,
                return_prediction=return_prediction,
            )
        return _predict_and_retrieve(
            self,
            context_vector,
            predictor_fn,
            future_horizon_ms=future_horizon_ms,
            limit=limit,
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
        self._discover_collections()
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
                    order_by="timestamp_ms",
                    with_vectors=True,
                )
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    all_states.append(self._payload_to_state(pt.payload, pt.id, vec))
            except Exception:  # noqa: S112  # retry loop across epochs
                continue

        # Sort by timestamp and find anchor position
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
        self._discover_collections()
        anchor = self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms

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
                    order_by="timestamp_ms",
                    with_vectors=True,
                )
                for pt in points:
                    vec = pt.vector
                    if isinstance(vec, dict):
                        vec = list(vec.values())[0] if vec else []
                    context.append(self._payload_to_state(pt.payload, pt.id, vec))
            except Exception:  # noqa: S112  # retry loop across epochs
                continue

        context.sort(key=lambda s: s.timestamp_ms)
        return context

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
        """Find the most recent state in the same scene before a timestamp."""
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
                    order_by="timestamp_ms",
                )
                if points:
                    # Scroll returns ascending order; last item is the latest predecessor
                    return str(points[-1].id), collection
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
        order_by: str | None = None,
        with_vectors: bool = False,
    ) -> list:
        """Return the full ordered scroll result for a collection."""
        offset: object | None = None
        all_points: list = []
        while True:
            hits = self._retry(
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
