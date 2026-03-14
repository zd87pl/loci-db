"""LocalEngramClient — full Engram with zero external dependencies.

Uses the in-memory backend instead of Qdrant.  Identical API surface
to :class:`EngramClient`, so code that works with LocalEngramClient
works with the real client by swapping the constructor.

Use cases:
- Tests without Docker
- Benchmarks that isolate Engram's indexing overhead
- Demos and prototyping
- CI environments
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from engram.backends.memory import MemoryStore
from engram.schema import WorldState
from engram.spatial.adaptive import AdaptiveResolution
from engram.spatial.buckets import expand_bounding_box
from engram.spatial.hilbert import encode as hilbert_encode
from engram.temporal.decay import apply_decay
from engram.temporal.sharding import collection_name, epoch_id, epochs_in_range

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Statistics from a single query execution.

    Attributes:
        shards_searched: Number of temporal shards (collections) searched.
        total_candidates: Points that passed payload filters before ANN.
        hilbert_ids_in_filter: Size of the Hilbert MatchAny set (0 if no spatial filter).
        decay_applied: Whether temporal decay re-ranking was applied.
        elapsed_ms: Wall-clock time for the query in milliseconds.
    """

    shards_searched: int = 0
    total_candidates: int = 0
    hilbert_ids_in_filter: int = 0
    decay_applied: bool = False
    elapsed_ms: float = 0.0


class LocalEngramClient:
    """Full Engram client backed by an in-memory store.

    API-compatible with :class:`EngramClient`.  No Qdrant required.

    Args:
        epoch_size_ms: Width of each temporal shard in milliseconds.
        spatial_resolution: Hilbert curve resolution order.
        vector_size: Dimensionality of embedding vectors.
        decay_lambda: Temporal decay rate.
        distance: Distance metric — ``"cosine"``, ``"dot"``, or ``"euclidean"``.
    """

    def __init__(
        self,
        epoch_size_ms: int = 5000,
        spatial_resolution: int = 4,
        vector_size: int = 512,
        decay_lambda: float = 1e-4,
        distance: str = "cosine",
        adaptive: bool = False,
    ) -> None:
        self._store = MemoryStore()
        self._epoch_size_ms = epoch_size_ms
        self._spatial_resolution = spatial_resolution
        self._vector_size = vector_size
        self._decay_lambda = decay_lambda
        self._distance = distance
        self._known_collections: set[str] = set()
        self._last_query_stats: QueryStats | None = None
        self._adaptive = (
            AdaptiveResolution(
                base_order=spatial_resolution,
                max_order=spatial_resolution + 2,
                density_threshold=50,
            )
            if adaptive
            else None
        )

    @property
    def density_stats(self):
        """Return adaptive resolution density stats, or None if not enabled."""
        return self._adaptive.stats() if self._adaptive is not None else None

    @property
    def last_query_stats(self) -> QueryStats | None:
        """Statistics from the most recent query() call."""
        return self._last_query_stats

    @property
    def store(self) -> MemoryStore:
        """Direct access to the underlying memory store (for introspection)."""
        return self._store

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self, name: str) -> None:
        if name in self._known_collections:
            return
        self._store.create_collection(name, self._vector_size, self._distance)
        self._store.create_payload_index(name, "hilbert_id")
        self._store.create_payload_index(name, "timestamp_ms")
        self._store.create_payload_index(name, "scale_level")
        self._known_collections.add(name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, state: WorldState) -> str:
        """Insert a single WorldState. Input is not mutated."""
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

        payload = _state_to_payload(state, hid)

        # Causal linking
        prev_id = self._find_latest_predecessor(col, state.scene_id, state.timestamp_ms)
        if prev_id is not None:
            payload["prev_state_id"] = prev_id
            self._store.set_payload(col, prev_id, {"next_state_id": point_id})

        self._store.upsert(col, [{"id": point_id, "vector": state.vector, "payload": payload}])
        return point_id

    def insert_batch(self, states: list[WorldState]) -> list[str]:
        """Insert a batch with intra-batch causal linking. Input is not mutated."""
        id_by_index: dict[int, str] = {}
        scene_chains: dict[str, str] = {}
        groups: dict[str, list[dict]] = {}

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

            payload = _state_to_payload(state, hid)

            if state.scene_id and state.scene_id in scene_chains:
                payload["prev_state_id"] = scene_chains[state.scene_id]
            if state.scene_id:
                scene_chains[state.scene_id] = point_id

            groups.setdefault(col, []).append(
                {"id": point_id, "vector": state.vector, "payload": payload}
            )

        for col, points in groups.items():
            self._store.upsert(col, points)

        # Patch next links
        for col, points in groups.items():
            for point in points:
                prev_id = point["payload"].get("prev_state_id")
                if prev_id:
                    self._store.set_payload(col, prev_id, {"next_state_id": point["id"]})

        return [id_by_index[i] for i in range(len(states))]

    # ------------------------------------------------------------------
    # Read
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
        """Search with Hilbert pre-filtering, temporal sharding, and decay.

        After each call, inspect :attr:`last_query_stats` for diagnostics.
        """
        t_start = time.perf_counter()
        stats = QueryStats()

        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            epochs = epochs_in_range(start_ms, end_ms, self._epoch_size_ms)
        else:
            epochs = self._list_active_epochs()

        collections = [
            collection_name(e) for e in epochs if collection_name(e) in self._known_collections
        ]
        stats.shards_searched = len(collections)

        # Build filter dict for MemoryStore
        payload_filter: dict = {}

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
            if not hids:
                stats.elapsed_ms = (time.perf_counter() - t_start) * 1000
                self._last_query_stats = stats
                return []
            payload_filter["hilbert_id"] = {"any": hids}
            stats.hilbert_ids_in_filter = len(hids)

        if time_window_ms is not None:
            payload_filter["timestamp_ms"] = {"gte": start_ms, "lte": end_ms}

        if _extra_payload_filter:
            payload_filter.update(_extra_payload_filter)

        # Search across shards
        all_results: list[dict] = []
        for col in collections:
            hits = self._store.search(
                collection=col,
                query_vector=vector,
                limit=limit,
                payload_filter=payload_filter if payload_filter else None,
            )
            stats.total_candidates += len(hits)
            for hit in hits:
                all_results.append(
                    {
                        "score": hit["score"],
                        "timestamp_ms": hit["payload"].get("timestamp_ms", 0),
                        "payload": hit["payload"],
                        "vector": hit["vector"],
                        "id": hit["id"],
                    }
                )

        # Decay and re-rank
        now_ms = int(time.time() * 1000)
        if self._decay_lambda > 0 and all_results:
            apply_decay(all_results, now_ms, self._decay_lambda)
            stats.decay_applied = True
        all_results = all_results[:limit]

        stats.elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._last_query_stats = stats

        return [_payload_to_state(r["payload"], r["id"], r["vector"]) for r in all_results]

    def predict_and_retrieve(
        self,
        context_vector: list[float],
        predictor_fn: Callable[[list[float]], list[float]],
        future_horizon_ms: int = 1000,
        limit: int = 5,
    ) -> list[WorldState]:
        """Predict-then-retrieve using the local backend."""
        predicted = predictor_fn(context_vector)
        now_ms = int(time.time() * 1000)
        return self.query(
            vector=predicted,
            time_window_ms=(now_ms, now_ms + future_horizon_ms),
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
        _SCALE_ORDER = ("sequence", "frame", "patch")
        best: list[WorldState] = []
        for scale in _SCALE_ORDER:
            results = self.query(
                vector=vector,
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window_ms,
                limit=limit * 3,
                _extra_payload_filter={"scale_level": scale},
            )
            if results:
                best = results
        return best[:limit]

    def get_trajectory(
        self,
        state_id: str,
        steps_back: int = 10,
        steps_forward: int = 10,
    ) -> list[WorldState]:
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
        epoch_start = ep * self._epoch_size_ms
        offset = timestamp_ms - epoch_start
        return min(1.0, max(0.0, offset / self._epoch_size_ms))

    def _get_state_by_id(self, state_id: str) -> WorldState | None:
        for col in list(self._known_collections):
            results = self._store.retrieve(col, [state_id])
            if results:
                r = results[0]
                return _payload_to_state(r["payload"], r["id"], r["vector"])
        return None

    def _find_latest_predecessor(
        self, collection: str, scene_id: str, before_ms: int
    ) -> str | None:
        if not scene_id:
            return None
        results = self._store.scroll(
            collection=collection,
            payload_filter={
                "scene_id": scene_id,
                "timestamp_ms": {"lt": before_ms},
            },
            limit=1,
            order_by="timestamp_ms",
        )
        if results:
            return str(results[0]["id"])
        return None

    def _list_active_epochs(self) -> list[int]:
        epochs: list[int] = []
        for col in self._known_collections:
            if col.startswith("engram_"):
                try:
                    epochs.append(int(col.split("_", 1)[1]))
                except ValueError:
                    pass
        return sorted(epochs) if epochs else [0]


# ------------------------------------------------------------------
# Shared payload helpers
# ------------------------------------------------------------------


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
