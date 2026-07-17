"""In-memory vector store backed by numpy.

Implements the same operations as Qdrant (insert, search, filter, retrieve)
using vectorised numpy similarity.  Designed for:
- Unit and integration tests without Docker
- Benchmarks that measure Loci's indexing overhead in isolation
- Rapid prototyping and demos
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np


@dataclass
class _Point:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class _Collection:
    name: str
    vector_size: int
    distance: str  # "cosine" | "dot" | "euclidean"
    points: dict[str, _Point] = field(default_factory=dict)
    payload_indices: set[str] = field(default_factory=set)


class MemoryStore:
    """In-memory vector store with the same semantics as Qdrant.

    Thread-safe for single-writer / multi-reader workloads (GIL-protected).
    For async usage, wrap calls in ``asyncio.to_thread``.
    """

    def __init__(self) -> None:
        self._collections: dict[str, _Collection] = {}

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def create_collection(self, name: str, vector_size: int, distance: str = "cosine") -> None:
        if name not in self._collections:
            self._collections[name] = _Collection(
                name=name, vector_size=vector_size, distance=distance
            )

    def collection_exists(self, name: str) -> bool:
        return name in self._collections

    def delete_collection(self, name: str) -> None:
        """Remove a collection and all its points."""
        self._collections.pop(name, None)

    def create_payload_index(self, collection: str, field_name: str) -> None:
        if collection in self._collections:
            self._collections[collection].payload_indices.add(field_name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, collection: str, points: list[dict]) -> None:
        """Insert or update points.

        Each dict must have ``id``, ``vector``, and ``payload`` keys.
        Vectors and payloads are copied on write so later caller-side
        mutation cannot corrupt stored points.

        Raises:
            ValueError: If any vector does not match the collection's
                configured ``vector_size``.
        """
        col = self._collections[collection]
        for p in points:
            vector = list(p["vector"])
            if len(vector) != col.vector_size:
                raise ValueError(
                    f"vector for point {p['id']!r} has dimension {len(vector)}, "
                    f"expected {col.vector_size} for collection {collection!r}"
                )
            col.points[p["id"]] = _Point(
                id=p["id"], vector=vector, payload=copy.deepcopy(p["payload"])
            )

    def set_payload(self, collection: str, point_id: str, payload: dict) -> None:
        col = self._collections[collection]
        if point_id in col.points:
            col.points[point_id].payload.update(copy.deepcopy(payload))

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_points(self, collection: str, ids: list[str]) -> int:
        """Delete points by id.  Returns the number actually removed."""
        col = self._collections.get(collection)
        if col is None:
            return 0
        return sum(1 for pid in ids if col.points.pop(pid, None) is not None)

    def delete_points_in_time_range(
        self,
        collection: str,
        start_ms: int,
        end_ms_exclusive: int,
        *,
        field: str = "timestamp_ms",
    ) -> int:
        """Delete points with ``start_ms <= payload[field] < end_ms_exclusive``.

        Points missing *field* are never deleted.  Returns the number of
        points removed (0 for a missing collection).
        """
        col = self._collections.get(collection)
        if col is None:
            return 0
        doomed = [
            pid
            for pid, p in col.points.items()
            if (value := p.payload.get(field)) is not None and start_ms <= value < end_ms_exclusive
        ]
        for pid in doomed:
            del col.points[pid]
        return len(doomed)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(self, collection: str, ids: list[str]) -> list[dict]:
        col = self._collections.get(collection)
        if col is None:
            return []
        results = []
        for pid in ids:
            if pid in col.points:
                p = col.points[pid]
                results.append(_point_to_dict(p))
        return results

    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        payload_filter: dict | None = None,
    ) -> list[dict]:
        """Brute-force ANN search with optional payload filtering.

        Args:
            collection: Collection name.
            query_vector: Query embedding.
            limit: Max results.
            payload_filter: Dict of ``{field: value}`` for exact match,
                or ``{field: {"gte": v, "lte": v}}`` for range,
                or ``{field: {"any": [...]}}`` for set membership.

        Returns:
            List of ``{"id", "vector", "payload", "score"}`` dicts,
            sorted by score descending.  Scores are always
            higher-is-better: euclidean distances are negated.
        """
        col = self._collections.get(collection)
        if col is None:
            return []

        candidates = list(col.points.values())
        if payload_filter:
            candidates = [p for p in candidates if _matches(p.payload, payload_filter)]

        if not candidates:
            return []

        scores = _batch_score(col.distance, query_vector, candidates)
        top_k = min(limit, len(candidates))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [_point_to_dict(candidates[i], score=float(scores[i])) for i in top_indices]

    def scroll(
        self,
        collection: str,
        payload_filter: dict | None = None,
        limit: int = 10,
        order_by: str | None = None,
    ) -> list[dict]:
        col = self._collections.get(collection)
        if col is None:
            return []

        if order_by is None:
            # Unordered scrolls stop as soon as `limit` matches are found,
            # so limit-1 probes (e.g. the consolidation staleness check)
            # stay cheap on filtered scans.
            results: list[dict] = []
            for p in col.points.values():
                if payload_filter and not _matches(p.payload, payload_filter):
                    continue
                results.append(_point_to_dict(p))
                if len(results) >= limit:
                    break
            return results

        candidates = list(col.points.values())
        if payload_filter:
            candidates = [p for p in candidates if _matches(p.payload, payload_filter)]
        candidates.sort(key=lambda p: p.payload.get(order_by, 0))
        return [_point_to_dict(p) for p in candidates[:limit]]

    @property
    def total_points(self) -> int:
        return sum(len(c.points) for c in self._collections.values())

    def collection_count(self, name: str) -> int:
        col = self._collections.get(name)
        return len(col.points) if col else 0

    def payload_value_range(self, collection: str, field: str) -> tuple[Any, Any] | None:
        """Return ``(min, max)`` of a payload field across a collection.

        Points missing *field* are skipped; returns ``None`` when nothing
        carries the field (or the collection does not exist).  Cheap stats
        helper — no point copies are made.
        """
        col = self._collections.get(collection)
        if col is None:
            return None
        values = [v for p in col.points.values() if (v := p.payload.get(field)) is not None]
        if not values:
            return None
        return min(values), max(values)


def _point_to_dict(p: _Point, score: float | None = None) -> dict:
    """Return a defensive copy of a stored point.

    Vectors and payloads are copied so callers cannot mutate stored state
    through returned results (and vice versa).
    """
    result: dict[str, Any] = {
        "id": p.id,
        "vector": list(p.vector),
        "payload": copy.deepcopy(p.payload),
    }
    if score is not None:
        result["score"] = score
    return result


# ------------------------------------------------------------------
# Distance functions
# ------------------------------------------------------------------


def _batch_score(distance: str, query_vector: list[float], candidates: list[_Point]) -> np.ndarray:
    """Compute similarity scores for all candidates at once using numpy."""
    q = np.asarray(query_vector, dtype=np.float64)
    mat = np.array([p.vector for p in candidates], dtype=np.float64)

    if distance == "cosine":
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return np.zeros(len(candidates))
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        return cast(np.ndarray, (mat @ q) / (norms * q_norm))
    elif distance == "dot":
        return mat @ q
    else:  # euclidean
        return cast(np.ndarray, -np.linalg.norm(mat - q, axis=1))


# ------------------------------------------------------------------
# Filter matching
# ------------------------------------------------------------------


def _matches(payload: dict, filters: dict) -> bool:
    """Check if a payload matches all filter conditions."""
    for key, condition in filters.items():
        value = payload.get(key)
        if isinstance(condition, dict):
            if "any" in condition:
                if value not in condition["any"]:
                    return False
            else:
                if "gte" in condition and (value is None or value < condition["gte"]):
                    return False
                if "lte" in condition and (value is None or value > condition["lte"]):
                    return False
                if "lt" in condition and (value is None or value >= condition["lt"]):
                    return False
                if "gt" in condition and (value is None or value <= condition["gt"]):
                    return False
        else:
            if value != condition:
                return False
    return True
