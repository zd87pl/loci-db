"""In-memory vector store backed by numpy.

Implements the same operations as Qdrant (insert, search, filter, retrieve)
using vectorised numpy similarity.  Designed for:
- Unit and integration tests without Docker
- Benchmarks that measure Loci's indexing overhead in isolation
- Rapid prototyping and demos
"""

from __future__ import annotations

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
        """
        col = self._collections[collection]
        for p in points:
            col.points[p["id"]] = _Point(id=p["id"], vector=p["vector"], payload=dict(p["payload"]))

    def set_payload(self, collection: str, point_id: str, payload: dict) -> None:
        col = self._collections[collection]
        if point_id in col.points:
            col.points[point_id].payload.update(payload)

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
                results.append({"id": p.id, "vector": p.vector, "payload": p.payload})
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
            sorted by score descending.
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

        return [
            {
                "id": candidates[i].id,
                "vector": candidates[i].vector,
                "payload": candidates[i].payload,
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

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

        candidates = list(col.points.values())
        if payload_filter:
            candidates = [p for p in candidates if _matches(p.payload, payload_filter)]

        if order_by:
            candidates.sort(key=lambda p: p.payload.get(order_by, 0))

        return [{"id": p.id, "vector": p.vector, "payload": p.payload} for p in candidates[:limit]]

    @property
    def total_points(self) -> int:
        return sum(len(c.points) for c in self._collections.values())

    def collection_count(self, name: str) -> int:
        col = self._collections.get(name)
        return len(col.points) if col else 0


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
