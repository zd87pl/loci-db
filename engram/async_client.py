"""Async EngramClient — parallel shard fan-out for high-throughput workloads."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Callable

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

from engram.schema import WorldState
from engram.spatial.buckets import expand_bounding_box
from engram.spatial.hilbert import encode as hilbert_encode
from engram.temporal.decay import apply_decay
from engram.temporal.sharding import collection_name, epoch_id, epochs_in_range

logger = logging.getLogger(__name__)

# Map public distance names to Qdrant enum values
_DISTANCE_MAP: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclidean": Distance.EUCLID,
}


class AsyncEngramClient:
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
    """

    def __init__(
        self,
        qdrant_url: str,
        epoch_size_ms: int = 5000,
        spatial_resolution: int = 4,
        vector_size: int = 512,
        decay_lambda: float = 1e-4,
        distance: str = "cosine",
    ) -> None:
        self._qdrant = AsyncQdrantClient(url=qdrant_url)
        self._epoch_size_ms = epoch_size_ms
        self._spatial_resolution = spatial_resolution
        self._vector_size = vector_size
        self._decay_lambda = decay_lambda
        if distance not in _DISTANCE_MAP:
            raise ValueError(
                f"distance must be one of {list(_DISTANCE_MAP)}, got {distance!r}"
            )
        self._distance = _DISTANCE_MAP[distance]
        self._known_collections: set[str] = set()
        self._collection_locks: dict[str, asyncio.Lock] = {}

    async def close(self) -> None:
        """Close the underlying Qdrant connection."""
        await self._qdrant.close()

    async def __aenter__(self) -> AsyncEngramClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

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
                await asyncio.gather(
                    self._qdrant.create_payload_index(
                        collection_name=name,
                        field_name="hilbert_id",
                        field_schema=PayloadSchemaType.INTEGER,
                    ),
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
                )

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
        point_id = uuid.uuid4().hex

        ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
        col = collection_name(ep)
        await self._ensure_collection(col)

        t_norm = _normalise_time(state.timestamp_ms, ep, self._epoch_size_ms)
        hid = hilbert_encode(
            state.x, state.y, state.z, t_norm,
            resolution_order=self._spatial_resolution,
        )

        payload = _state_to_payload(state, hid)

        # Causal linking
        prev_id = await self._find_latest_predecessor(
            col, state.scene_id, state.timestamp_ms
        )
        if prev_id is not None:
            payload["prev_state_id"] = prev_id
            await self._patch_next_link(prev_id, point_id)

        await self._qdrant.upsert(
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
        scene_chains: dict[str, str] = {}  # scene_id → latest point_id

        # Sort indices by (scene_id, timestamp_ms) for correct linking
        indexed = sorted(
            enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms)
        )
        id_by_index: dict[int, str] = {}

        for orig_idx, state in indexed:
            point_id = uuid.uuid4().hex
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            col = collection_name(ep)
            await self._ensure_collection(col)

            t_norm = _normalise_time(state.timestamp_ms, ep, self._epoch_size_ms)
            hid = hilbert_encode(
                state.x, state.y, state.z, t_norm,
                resolution_order=self._spatial_resolution,
            )
            payload = _state_to_payload(state, hid)

            # Link within the batch
            if state.scene_id and state.scene_id in scene_chains:
                payload["prev_state_id"] = scene_chains[state.scene_id]
            scene_chains[state.scene_id] = point_id

            groups.setdefault(col, []).append(
                PointStruct(id=point_id, vector=state.vector, payload=payload)
            )

        # Fan out upserts concurrently
        await asyncio.gather(
            *(
                self._qdrant.upsert(collection_name=col, points=points)
                for col, points in groups.items()
            )
        )

        # Patch next_state_id links within the batch
        for col, points in groups.items():
            for i, point in enumerate(points):
                prev_id = point.payload.get("prev_state_id")
                if prev_id:
                    # Find the predecessor in this batch and update it
                    for p in points:
                        if p.id == prev_id:
                            try:
                                await self._qdrant.set_payload(
                                    collection_name=col,
                                    payload={"next_state_id": point.id},
                                    points=[prev_id],
                                )
                            except Exception:
                                logger.debug(
                                    "Failed to patch next link %s→%s",
                                    prev_id, point.id, exc_info=True,
                                )
                            break

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
        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            epochs = epochs_in_range(start_ms, end_ms, self._epoch_size_ms)
        else:
            epochs = self._list_active_epochs()

        collections = [
            collection_name(e)
            for e in epochs
            if collection_name(e) in self._known_collections
        ]
        if not collections:
            return []

        # Build filter
        must_conditions: list = []

        if spatial_bounds is not None:
            hids = expand_bounding_box(
                spatial_bounds.get("x_min", 0.0),
                spatial_bounds.get("x_max", 1.0),
                spatial_bounds.get("y_min", 0.0),
                spatial_bounds.get("y_max", 1.0),
                spatial_bounds.get("z_min", 0.0),
                spatial_bounds.get("z_max", 1.0),
                0.0, 1.0,
                resolution_order=self._spatial_resolution,
            )
            if hids:
                must_conditions.append(
                    FieldCondition(key="hilbert_id", match=MatchAny(any=hids))
                )
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
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Parallel fan-out across shards
        async def _search_shard(col: str) -> list[dict]:
            try:
                hits = await self._qdrant.search(
                    collection_name=col,
                    query_vector=vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_vectors=True,
                )
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
            *(_search_shard(col) for col in collections)
        )
        all_results: list[dict] = []
        for batch in shard_results:
            all_results.extend(batch)

        # Apply temporal decay and re-rank
        now_ms = int(time.time() * 1000)
        apply_decay(all_results, now_ms, self._decay_lambda)
        all_results = all_results[:limit]

        return [
            _payload_to_state(r["payload"], r["id"], r["vector"])
            for r in all_results
        ]

    async def predict_and_retrieve(
        self,
        context_vector: list[float],
        predictor_fn: Callable[[list[float]], list[float]],
        future_horizon_ms: int = 1000,
        limit: int = 5,
    ) -> list[WorldState]:
        """Predict a future state then retrieve nearest neighbours.

        Args:
            context_vector: Current-state embedding.
            predictor_fn: User-supplied world model.
            future_horizon_ms: How far ahead to search (milliseconds).
            limit: Maximum number of results.

        Returns:
            List of :class:`WorldState` neighbours of the predicted vector.
        """
        predicted_vector = predictor_fn(context_vector)
        now_ms = int(time.time() * 1000)
        return await self.query(
            vector=predicted_vector,
            time_window_ms=(now_ms, now_ms + future_horizon_ms),
            limit=limit,
        )

    async def get_trajectory(
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
        anchor = await self._get_state_by_id(state_id)
        if anchor is None:
            return []

        backward: list[WorldState] = []
        current = anchor
        for _ in range(steps_back):
            if current.prev_state_id is None:
                break
            prev = await self._get_state_by_id(current.prev_state_id)
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
            nxt = await self._get_state_by_id(current.next_state_id)
            if nxt is None:
                break
            forward.append(nxt)
            current = nxt

        return backward + [anchor] + forward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_state_by_id(self, state_id: str) -> WorldState | None:
        for col in list(self._known_collections):
            try:
                results = await self._qdrant.retrieve(
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
            except Exception:
                continue
        return None

    async def _find_latest_predecessor(
        self, collection: str, scene_id: str, before_ms: int
    ) -> str | None:
        if not scene_id:
            return None
        try:
            hits = await self._qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="scene_id", match=MatchValue(value=scene_id)
                        ),
                        FieldCondition(
                            key="timestamp_ms", range=Range(lt=before_ms)
                        ),
                    ]
                ),
                limit=1,
                order_by="timestamp_ms",
            )
            points = hits[0] if isinstance(hits, tuple) else hits
            if points:
                return str(points[0].id)
        except Exception:
            logger.debug(
                "Failed to find predecessor in %s", collection, exc_info=True
            )
        return None

    async def _patch_next_link(self, prev_id: str, next_id: str) -> None:
        for col in list(self._known_collections):
            try:
                await self._qdrant.set_payload(
                    collection_name=col,
                    payload={"next_state_id": next_id},
                    points=[prev_id],
                )
                return
            except Exception:
                continue

    def _list_active_epochs(self) -> list[int]:
        epochs: list[int] = []
        for col in self._known_collections:
            if col.startswith("engram_"):
                try:
                    epochs.append(int(col.split("_", 1)[1]))
                except ValueError:
                    pass
        return sorted(epochs) if epochs else [0]


# ---------------------------------------------------------------------------
# Shared helpers (used by both sync and async clients)
# ---------------------------------------------------------------------------

def _normalise_time(timestamp_ms: int, ep: int, epoch_size_ms: int) -> float:
    epoch_start = ep * epoch_size_ms
    offset = timestamp_ms - epoch_start
    return min(1.0, max(0.0, offset / epoch_size_ms))


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
