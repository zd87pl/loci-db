"""Systems under test for wm_bench.

Every system implements the :class:`SystemUnderTest` protocol so the tasks
in :mod:`benchmarks.wm_bench.tasks` can be pointed at any store:

- ``brute_force`` — numpy exact search over everything ever inserted.
  The recall oracle: its top-k IS the ground truth.
- ``loci_local`` — :class:`~loci.local_client.LocalLociClient` (in-memory
  backend, decay disabled for determinism).
- ``loci_local_consolidated`` — LocalLociClient with a
  :class:`~loci.temporal.consolidation.ConsolidationPolicy` (the
  flight-recorder configuration for the recall-vs-age task).
- ``loci_qdrant_memory`` — :class:`~loci.client.LociClient` over
  ``QdrantClient(":memory:")`` (qdrant-client's server-free local engine,
  wired the same way as tests/test_qdrant_integration.py).
- ``naive_qdrant`` — plain qdrant-client ``:memory:`` collection with
  float-range payload filters and no LOCI anywhere.  The honest baseline.

Result identity: synthetic streams have strictly increasing, unique
``timestamp_ms``, so hits are compared across systems by timestamp (store-
assigned IDs differ per system).  Consolidated summaries additionally carry
their source time range so coverage-style recall can credit them.
"""

from __future__ import annotations

import math
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from unittest import mock

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from loci.client import LociClient
from loci.local_client import LocalLociClient
from loci.schema import WorldState
from loci.temporal.consolidation import (
    ConsolidationPolicy,
    data_collection_name,
    summary_collection_name,
)

from .datasets import TrajectoryPoint

DEFAULT_EPOCH_SIZE_MS = 5000

# The qdrant-client local engine warns that payload indexes are no-ops;
# irrelevant to correctness and noisy in benchmark output.
_LOCAL_INDEX_WARNING = r"Payload indexes have no effect in the local Qdrant.*"


@dataclass(frozen=True)
class QueryHit:
    """One retrieval result, in system-independent identity terms."""

    timestamp_ms: int
    scene_id: str
    score: float
    is_summary: bool = False
    t_min_ms: int | None = None
    t_max_ms: int | None = None

    def covers(self, timestamp_ms: int, scene_id: str) -> bool:
        """True when this hit accounts for the given raw point.

        A raw hit covers exactly its own timestamp; a consolidated summary
        covers any timestamp inside its source range for its scene.
        """
        if not self.is_summary:
            return self.timestamp_ms == timestamp_ms
        if self.scene_id != scene_id:
            return False
        lo = self.t_min_ms if self.t_min_ms is not None else self.timestamp_ms
        hi = self.t_max_ms if self.t_max_ms is not None else self.timestamp_ms
        return lo <= timestamp_ms <= hi


@runtime_checkable
class SystemUnderTest(Protocol):
    """Protocol each benchmarked memory system implements."""

    name: str
    trajectory_method: str

    def setup(self, vector_size: int, distance: str = "cosine") -> None: ...

    def insert(self, point: TrajectoryPoint) -> None: ...

    def insert_many(self, points: Sequence[TrajectoryPoint]) -> None: ...

    def query(
        self,
        vector: Sequence[float],
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
    ) -> list[QueryHit]: ...

    def predict_novelty(self, vector: Sequence[float]) -> float | None:
        """Novelty in [0, 1] for a candidate observation; None = unsupported."""
        ...

    def get_trajectory(
        self, anchor_timestamp_ms: int, steps_back: int, steps_forward: int
    ) -> list[QueryHit] | None:
        """Ordered trajectory around an inserted point; None = unsupported."""
        ...

    def resident_points(self) -> int | None:
        """Points currently resident in the store; None = unknown."""
        ...

    def teardown(self) -> None: ...


def _clamped_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b)) / (na * nb)))


# ---------------------------------------------------------------------------
# BruteForce — the recall oracle
# ---------------------------------------------------------------------------


class BruteForceSystem:
    """Exact numpy search over the full stream.  Retains everything."""

    def __init__(self) -> None:
        self.name = "brute_force"
        self.trajectory_method = "array_slice"

    def setup(self, vector_size: int, distance: str = "cosine") -> None:
        if distance not in {"cosine", "dot", "euclidean"}:
            raise ValueError(f"unsupported distance {distance!r}")
        self._dim = vector_size
        self._distance = distance
        self._vectors: list[np.ndarray] = []
        self._timestamps: list[int] = []
        self._scenes: list[str] = []

    def insert(self, point: TrajectoryPoint) -> None:
        vec = np.asarray(point.embedding, dtype=np.float64)
        if self._distance == "cosine":
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        self._vectors.append(vec)
        self._timestamps.append(point.timestamp_ms)
        self._scenes.append(point.scene_id)

    def insert_many(self, points: Sequence[TrajectoryPoint]) -> None:
        for p in points:
            self.insert(p)

    def _scores(self, vector: Sequence[float]) -> np.ndarray:
        matrix = np.asarray(self._vectors)
        q = np.asarray(vector, dtype=np.float64)
        if self._distance == "cosine":
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
            return matrix @ q
        if self._distance == "dot":
            return matrix @ q
        return -np.linalg.norm(matrix - q[None, :], axis=1)  # euclidean, negated

    def query(
        self,
        vector: Sequence[float],
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
    ) -> list[QueryHit]:
        if not self._vectors:
            return []
        scores = self._scores(vector)
        ts = np.asarray(self._timestamps)
        idx = np.arange(len(scores))
        if time_window_ms is not None:
            lo, hi = time_window_ms
            mask = (ts >= lo) & (ts <= hi)
            idx = idx[mask]
        if idx.size == 0:
            return []
        order = idx[np.argsort(-scores[idx], kind="stable")][:limit]
        return [
            QueryHit(
                timestamp_ms=int(self._timestamps[i]),
                scene_id=self._scenes[i],
                score=float(scores[i]),
            )
            for i in order
        ]

    def predict_novelty(self, vector: Sequence[float]) -> float | None:
        if not self._vectors:
            return 1.0
        q = np.asarray(vector, dtype=np.float64)
        best = max(_clamped_cosine(q, v) for v in self._vectors)
        return 1.0 - best

    def get_trajectory(
        self, anchor_timestamp_ms: int, steps_back: int, steps_forward: int
    ) -> list[QueryHit] | None:
        try:
            anchor_idx = self._timestamps.index(anchor_timestamp_ms)
        except ValueError:
            return []
        scene = self._scenes[anchor_idx]
        scene_indices = [i for i, s in enumerate(self._scenes) if s == scene]
        scene_indices.sort(key=lambda i: self._timestamps[i])
        pos = scene_indices.index(anchor_idx)
        window = scene_indices[max(0, pos - steps_back) : pos + steps_forward + 1]
        return [
            QueryHit(timestamp_ms=self._timestamps[i], scene_id=self._scenes[i], score=0.0)
            for i in window
        ]

    def resident_points(self) -> int | None:
        return len(self._vectors)

    def teardown(self) -> None:
        self._vectors = []
        self._timestamps = []
        self._scenes = []


# ---------------------------------------------------------------------------
# LOCI systems
# ---------------------------------------------------------------------------


def _to_world_state(point: TrajectoryPoint) -> WorldState:
    return WorldState(
        x=point.x,
        y=point.y,
        z=point.z,
        timestamp_ms=point.timestamp_ms,
        vector=list(point.embedding),
        scene_id=point.scene_id,
    )


def _state_to_hit(state: WorldState, score: float = 0.0) -> QueryHit:
    meta = state.metadata or {}
    is_summary = bool(meta.get("consolidated"))
    return QueryHit(
        timestamp_ms=state.timestamp_ms,
        scene_id=state.scene_id,
        score=score,
        is_summary=is_summary,
        t_min_ms=meta.get("t_min_ms") if is_summary else None,
        t_max_ms=meta.get("t_max_ms") if is_summary else None,
    )


class _LociSystemBase:
    """Shared LOCI behaviour: WorldState conversion, novelty, trajectory."""

    name = "loci_base"
    trajectory_method = "causal_scene_scan"

    def _make_client(self, vector_size: int, distance: str):
        raise NotImplementedError

    def setup(self, vector_size: int, distance: str = "cosine") -> None:
        self._client = self._make_client(vector_size, distance)
        self._id_by_ts: dict[int, str] = {}

    def insert(self, point: TrajectoryPoint) -> None:
        state_id = self._client.insert(_to_world_state(point))
        self._id_by_ts[point.timestamp_ms] = state_id

    def insert_many(self, points: Sequence[TrajectoryPoint]) -> None:
        states = [_to_world_state(p) for p in points]
        ids = self._client.insert_batch(states)
        for p, state_id in zip(points, ids, strict=True):
            self._id_by_ts[p.timestamp_ms] = state_id

    def query(
        self,
        vector: Sequence[float],
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
    ) -> list[QueryHit]:
        scored = self._client.query_scored(
            list(vector), time_window_ms=time_window_ms, limit=limit
        )
        return [_state_to_hit(s.state, s.score) for s in scored]

    def predict_novelty(self, vector: Sequence[float]) -> float | None:
        # Exercise the shipped predict-then-retrieve path with an identity
        # predictor: novelty = 1 - best clamped cosine between the "predicted"
        # vector and any retrieved state's stored vector.  Uncalibrated, so
        # the score is absolute and deterministic.
        result = self._client.predict_and_retrieve(
            context_vector=list(vector),
            predictor_fn=lambda v: v,
            limit=5,
            return_prediction=True,
        )
        return float(result.prediction_novelty)

    def get_trajectory(
        self, anchor_timestamp_ms: int, steps_back: int, steps_forward: int
    ) -> list[QueryHit] | None:
        anchor_id = self._id_by_ts.get(anchor_timestamp_ms)
        if anchor_id is None:
            return []
        states = self._client.get_trajectory(
            anchor_id, steps_back=steps_back, steps_forward=steps_forward
        )
        return [_state_to_hit(s) for s in states]

    def resident_points(self) -> int | None:
        return None

    def teardown(self) -> None:
        self._id_by_ts = {}


class LociLocalSystem(_LociSystemBase):
    """LocalLociClient (in-memory backend, decay off for determinism)."""

    def __init__(
        self,
        *,
        epoch_size_ms: int = DEFAULT_EPOCH_SIZE_MS,
        consolidation_policy: ConsolidationPolicy | None = None,
        name: str = "loci_local",
    ) -> None:
        self.name = name
        self._epoch_size_ms = epoch_size_ms
        self._policy = consolidation_policy

    def _make_client(self, vector_size: int, distance: str) -> LocalLociClient:
        return LocalLociClient(
            vector_size=vector_size,
            epoch_size_ms=self._epoch_size_ms,
            decay_lambda=0.0,
            distance=distance,
            consolidation_policy=self._policy,
        )

    def resident_points(self) -> int | None:
        store = self._client.store
        return store.collection_count(data_collection_name()) + store.collection_count(
            summary_collection_name()
        )


class LociQdrantMemorySystem(_LociSystemBase):
    """LociClient over qdrant-client's ':memory:' local engine.

    Wiring mirrors tests/test_qdrant_integration.py: the real local engine is
    injected as the client's Qdrant connection.  LociClient natively handles
    the local engine's ValueError-on-missing-collection contract, so no
    get_collection shim is needed here.
    """

    def __init__(
        self,
        *,
        epoch_size_ms: int = DEFAULT_EPOCH_SIZE_MS,
        consolidation_policy: ConsolidationPolicy | None = None,
        name: str = "loci_qdrant_memory",
    ) -> None:
        self.name = name
        self._epoch_size_ms = epoch_size_ms
        self._policy = consolidation_policy

    def _make_client(self, vector_size: int, distance: str) -> LociClient:
        warnings.filterwarnings("ignore", message=_LOCAL_INDEX_WARNING)
        self._store = QdrantClient(location=":memory:")
        with mock.patch("loci.client.QdrantClient", return_value=self._store):
            return LociClient(
                qdrant_url="http://unused:6333",
                vector_size=vector_size,
                epoch_size_ms=self._epoch_size_ms,
                decay_lambda=0.0,
                distance=distance,
                consolidation_policy=self._policy,
            )

    def resident_points(self) -> int | None:
        total = 0
        for collection in (data_collection_name(), summary_collection_name()):
            if self._store.collection_exists(collection):
                total += int(self._store.count(collection).count)
        return total

    def teardown(self) -> None:
        super().teardown()
        self._client.close()


# ---------------------------------------------------------------------------
# Naive Qdrant baseline — no LOCI anywhere
# ---------------------------------------------------------------------------


class NaiveQdrantSystem:
    """Plain qdrant-client ':memory:' collection with float-range filters.

    What a competent engineer would build without LOCI: one collection,
    x/y/z/timestamp payload fields, Range conditions, HNSW (exact in the
    local engine).  No causal links, no consolidation, no novelty API —
    ``predict_novelty`` reports None (unsupported) and the trajectory task
    falls back to a payload-filtered scroll sorted by timestamp.
    """

    _COLLECTION = "naive_points"
    _DISTANCES = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclidean": Distance.EUCLID,
    }

    def __init__(self) -> None:
        self.name = "naive_qdrant"
        self.trajectory_method = "payload_scroll"

    def setup(self, vector_size: int, distance: str = "cosine") -> None:
        self._distance = distance
        self._store = QdrantClient(location=":memory:")
        self._store.create_collection(
            collection_name=self._COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=self._DISTANCES[distance]),
        )
        self._scene_by_ts: dict[int, str] = {}

    def _point_struct(self, point: TrajectoryPoint) -> PointStruct:
        return PointStruct(
            id=str(uuid.uuid4()),
            vector=list(point.embedding),
            payload={
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "timestamp_ms": point.timestamp_ms,
                "scene_id": point.scene_id,
            },
        )

    def insert(self, point: TrajectoryPoint) -> None:
        self._store.upsert(self._COLLECTION, points=[self._point_struct(point)])
        self._scene_by_ts[point.timestamp_ms] = point.scene_id

    def insert_many(self, points: Sequence[TrajectoryPoint]) -> None:
        self._store.upsert(self._COLLECTION, points=[self._point_struct(p) for p in points])
        for p in points:
            self._scene_by_ts[p.timestamp_ms] = p.scene_id

    def query(
        self,
        vector: Sequence[float],
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
    ) -> list[QueryHit]:
        query_filter = None
        if time_window_ms is not None:
            lo, hi = time_window_ms
            query_filter = Filter(
                must=[FieldCondition(key="timestamp_ms", range=Range(gte=lo, lte=hi))]
            )
        resp = self._store.query_points(
            collection_name=self._COLLECTION,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
        )
        hits = []
        for hit in resp.points:
            payload = hit.payload or {}
            score = float(hit.score)
            if self._distance == "euclidean":
                score = -score
            hits.append(
                QueryHit(
                    timestamp_ms=int(payload.get("timestamp_ms", 0)),
                    scene_id=str(payload.get("scene_id", "")),
                    score=score,
                )
            )
        return hits

    def predict_novelty(self, vector: Sequence[float]) -> float | None:
        return None  # no novelty API in the baseline; reported as unsupported

    def get_trajectory(
        self, anchor_timestamp_ms: int, steps_back: int, steps_forward: int
    ) -> list[QueryHit] | None:
        scene = self._scene_by_ts.get(anchor_timestamp_ms)
        if scene is None:
            return []
        collected = []
        offset = None
        scene_filter = Filter(
            must=[FieldCondition(key="scene_id", match=MatchValue(value=scene))]
        )
        while True:
            points, offset = self._store.scroll(
                self._COLLECTION, scroll_filter=scene_filter, limit=256, offset=offset
            )
            collected.extend(points)
            if not points or offset is None:
                break
        rows = sorted(
            (int((p.payload or {}).get("timestamp_ms", 0)) for p in collected),
        )
        if anchor_timestamp_ms not in rows:
            return []
        pos = rows.index(anchor_timestamp_ms)
        window = rows[max(0, pos - steps_back) : pos + steps_forward + 1]
        return [QueryHit(timestamp_ms=ts, scene_id=scene, score=0.0) for ts in window]

    def resident_points(self) -> int | None:
        return int(self._store.count(self._COLLECTION).count)

    def teardown(self) -> None:
        self._store.close()
        self._scene_by_ts = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SYSTEM_NAMES = [
    "brute_force",
    "loci_local",
    "loci_local_consolidated",
    "loci_qdrant_memory",
    "naive_qdrant",
]


def default_consolidation_policy() -> ConsolidationPolicy:
    """Flight-recorder policy used by ``loci_local_consolidated``.

    With 5s epochs: the newest ~30s stay raw; older epochs fold into coarse
    groups of 6 epochs, at most 4 summary states per scene per group.
    """
    return ConsolidationPolicy(raw_window_epochs=6, summary_epoch_ratio=6, max_states_per_scene=4)


def build_system(
    name: str,
    *,
    epoch_size_ms: int = DEFAULT_EPOCH_SIZE_MS,
    consolidation_policy: ConsolidationPolicy | None = None,
) -> SystemUnderTest:
    """Instantiate a system by registry name (setup() not yet called)."""
    if name == "brute_force":
        return BruteForceSystem()
    if name == "loci_local":
        return LociLocalSystem(epoch_size_ms=epoch_size_ms)
    if name == "loci_local_consolidated":
        policy = consolidation_policy or default_consolidation_policy()
        return LociLocalSystem(
            epoch_size_ms=epoch_size_ms,
            consolidation_policy=policy,
            name="loci_local_consolidated",
        )
    if name == "loci_qdrant_memory":
        return LociQdrantMemorySystem(epoch_size_ms=epoch_size_ms)
    if name == "naive_qdrant":
        return NaiveQdrantSystem()
    raise ValueError(f"unknown system {name!r}; known: {SYSTEM_NAMES}")


def novelty_supported(value: float | None) -> bool:
    return value is not None and math.isfinite(value)
