"""Main LociClient class — primary API surface for the Loci database."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

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

from loci.retrieval.predict import predict_and_retrieve as _predict_and_retrieve
from loci.schema import WorldState
from loci.spatial.adaptive import AdaptiveResolution
from loci.spatial.buckets import expand_bounding_box
from loci.spatial.hilbert import encode as hilbert_encode
from loci.temporal.decay import apply_decay
from loci.temporal.sharding import collection_name, epoch_id, epochs_in_range

logger = logging.getLogger(__name__)

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
        qdrant_url: str,
        epoch_size_ms: int = 5000,
        spatial_resolution: int = 4,
        vector_size: int = 512,
        decay_lambda: float = 1e-4,
        distance: str = "cosine",
        adaptive: bool = False,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        self._qdrant = QdrantClient(url=qdrant_url)
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
        self._adaptive = (
            AdaptiveResolution(
                base_order=spatial_resolution,
                max_order=spatial_resolution + 2,
                density_threshold=50,
            )
            if adaptive
            else None
        )

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
            self._qdrant.create_payload_index(
                collection_name=name,
                field_name="hilbert_id",
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
        point_id = uuid.uuid4().hex

        ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
        col = collection_name(ep)
        self._ensure_collection(col)

        t_norm = self._normalise_time(state.timestamp_ms, ep)
        hid = hilbert_encode(
            state.x,
            state.y,
            state.z,
            t_norm,
            resolution_order=self._spatial_resolution,
        )

        if self._adaptive is not None:
            self._adaptive.record(state.x, state.y, state.z, t_norm)

        payload = self._state_to_payload(state, hid)

        # Causal linking
        prev_id = self._find_latest_predecessor(col, state.scene_id, state.timestamp_ms)
        if prev_id is not None:
            payload["prev_state_id"] = prev_id
            self._patch_next_link(prev_id, point_id)

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
        scene_chains: dict[str, str] = {}  # scene_id → latest point_id

        # Sort by (scene_id, timestamp) to build correct causal chains
        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))

        for orig_idx, state in indexed:
            point_id = uuid.uuid4().hex
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            col = collection_name(ep)
            self._ensure_collection(col)

            t_norm = self._normalise_time(state.timestamp_ms, ep)
            hid = hilbert_encode(
                state.x,
                state.y,
                state.z,
                t_norm,
                resolution_order=self._spatial_resolution,
            )
            if self._adaptive is not None:
                self._adaptive.record(state.x, state.y, state.z, t_norm)

            payload = self._state_to_payload(state, hid)

            # Causal link within the batch
            if state.scene_id and state.scene_id in scene_chains:
                payload["prev_state_id"] = scene_chains[state.scene_id]
            if state.scene_id:
                scene_chains[state.scene_id] = point_id

            groups.setdefault(col, []).append(
                PointStruct(id=point_id, vector=state.vector, payload=payload)
            )

        for col, points in groups.items():
            self._retry(self._qdrant.upsert, collection_name=col, points=points)

        # Patch next_state_id for intra-batch links
        for col, points in groups.items():
            for point in points:
                prev_id = (point.payload or {}).get("prev_state_id")
                if prev_id:
                    try:
                        self._retry(
                            self._qdrant.set_payload,
                            collection_name=col,
                            payload={"next_state_id": point.id},
                            points=[prev_id],
                        )
                    except Exception:
                        logger.debug(
                            "Failed to patch next link %s→%s",
                            prev_id,
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
    ) -> list[WorldState]:
        """Search for nearest neighbours with spatial and temporal filtering.

        Args:
            vector: Query embedding vector.
            spatial_bounds: Optional dict with keys ``x_min``, ``x_max``,
                ``y_min``, ``y_max``, ``z_min``, ``z_max``.
            time_window_ms: Optional ``(start_ms, end_ms)`` window.
            limit: Maximum number of results.

        Returns:
            List of :class:`WorldState` results sorted by decay-weighted similarity.
        """
        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            epochs = epochs_in_range(start_ms, end_ms, self._epoch_size_ms)
        else:
            epochs = self._list_active_epochs()

        collections = [collection_name(e) for e in epochs]

        must_conditions: list = []

        if spatial_bounds is not None:
            hids = expand_bounding_box(
                spatial_bounds.get("x_min", 0.0),
                spatial_bounds.get("x_max", 1.0),
                spatial_bounds.get("y_min", 0.0),
                spatial_bounds.get("y_max", 1.0),
                spatial_bounds.get("z_min", 0.0),
                spatial_bounds.get("z_max", 1.0),
                0.0,
                1.0,
                resolution_order=self._spatial_resolution,
            )
            if hids:
                must_conditions.append(FieldCondition(key="hilbert_id", match=MatchAny(any=hids)))
            else:
                return []

        if time_window_ms is not None:
            must_conditions.append(
                FieldCondition(
                    key="timestamp_ms",
                    range=Range(gte=start_ms, lte=end_ms),
                )
            )

        if _extra_payload_filter:
            for key, value in _extra_payload_filter.items():
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        all_results: list[dict] = []
        for col in collections:
            if col not in self._known_collections:
                continue
            try:
                resp = self._retry(
                    self._qdrant.query_points,
                    collection_name=col,
                    query=vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_vectors=True,
                )
                hits = resp.points
            except Exception:
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

        now_ms = int(time.time() * 1000)
        apply_decay(all_results, now_ms, self._decay_lambda)
        all_results = all_results[:limit]

        return [self._payload_to_state(r["payload"], r["id"], r["vector"]) for r in all_results]

    # ------------------------------------------------------------------
    # Read — novel primitive
    # ------------------------------------------------------------------

    def predict_and_retrieve(
        self,
        context_vector: list[float],
        predictor_fn: Callable[[list[float]], list[float]],
        future_horizon_ms: int = 1000,
        limit: int = 5,
    ) -> list[WorldState]:
        """Predict a future state then retrieve nearest neighbours to it.

        Args:
            context_vector: Current-state embedding.
            predictor_fn: User-supplied world model.
            future_horizon_ms: How far ahead to search (milliseconds).
            limit: Maximum number of results.

        Returns:
            List of :class:`WorldState` neighbours of the predicted vector.
        """
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
        """Follow causal links to reconstruct a trajectory.

        Args:
            state_id: ID of the anchor state.
            steps_back: Number of predecessors to follow.
            steps_forward: Number of successors to follow.

        Returns:
            Ordered list of states from oldest to newest.
        """
        anchor = self._get_state_by_id(state_id)
        if anchor is None:
            return []

        backward: list[WorldState] = []
        current = anchor
        for _ in range(steps_back):
            if current.prev_state_id is None:
                break
            prev = self._get_state_by_id(current.prev_state_id)
            if prev is None:
                break
            backward.append(prev)
            current = prev
        backward.reverse()

        forward: list[WorldState] = []
        current = anchor
        for _ in range(steps_forward):
            if current.next_state_id is None:
                break
            nxt = self._get_state_by_id(current.next_state_id)
            if nxt is None:
                break
            forward.append(nxt)
            current = nxt

        return backward + [anchor] + forward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_time(self, timestamp_ms: int, ep: int) -> float:
        """Map a timestamp to [0, 1] within its epoch."""
        epoch_start = ep * self._epoch_size_ms
        offset = timestamp_ms - epoch_start
        return min(1.0, max(0.0, offset / self._epoch_size_ms))

    @staticmethod
    def _state_to_payload(state: WorldState, hilbert_id: int) -> dict:
        return {
            "x": state.x,
            "y": state.y,
            "z": state.z,
            "timestamp_ms": state.timestamp_ms,
            "hilbert_id": hilbert_id,
            "scene_id": state.scene_id,
            "scale_level": state.scale_level,
            "confidence": state.confidence,
            "prev_state_id": state.prev_state_id,
            "next_state_id": state.next_state_id,
        }

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
            except Exception:
                continue
        return None

    def _find_latest_predecessor(
        self, collection: str, scene_id: str, before_ms: int
    ) -> str | None:
        """Find the most recent state in the same scene before a timestamp."""
        if not scene_id:
            return None
        try:
            hits = self._retry(
                self._qdrant.scroll,
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="scene_id", match=MatchValue(value=scene_id)),
                        FieldCondition(key="timestamp_ms", range=Range(lt=before_ms)),
                    ]
                ),
                limit=1,
                order_by="timestamp_ms",
            )
            points = hits[0] if isinstance(hits, tuple) else hits
            if points:
                return str(points[0].id)
        except Exception:
            logger.debug("Failed to find predecessor in %s", collection, exc_info=True)
        return None

    def _patch_next_link(self, prev_id: str, next_id: str) -> None:
        """Update the predecessor's next_state_id payload field."""
        for col in list(self._known_collections):
            try:
                self._retry(
                    self._qdrant.set_payload,
                    collection_name=col,
                    payload={"next_state_id": next_id},
                    points=[prev_id],
                )
                return
            except Exception:
                continue

    def _list_active_epochs(self) -> list[int]:
        """Return epoch IDs for all known collections."""
        epochs: list[int] = []
        for col in self._known_collections:
            if col.startswith("loci_"):
                try:
                    epochs.append(int(col.split("_", 1)[1]))
                except ValueError:
                    pass
        return sorted(epochs) if epochs else [0]
