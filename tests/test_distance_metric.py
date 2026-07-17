"""Tests for configurable distance metric on sync, async, and local clients.

Score direction contract: ``query_scored`` returns HIGHER-IS-BETTER scores
for all metrics and backends. Qdrant reports raw euclidean distances
(smaller is better), so the clients negate them at the boundary.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from loci.async_client import AsyncLociClient
from loci.client import _DISTANCE_MAP, LociClient
from loci.local_client import LocalLociClient
from loci.schema import WorldState


def test_valid_distances():
    for name in ("cosine", "dot", "euclidean"):
        with patch("loci.client.QdrantClient"):
            client = LociClient(qdrant_url="http://fake:6333", distance=name)
            assert client._distance == _DISTANCE_MAP[name]


def test_invalid_distance_raises():
    with pytest.raises(ValueError, match="distance"), patch("loci.client.QdrantClient"):
        LociClient(qdrant_url="http://fake:6333", distance="hamming")


def test_collection_uses_configured_distance():
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import Distance

    with patch("loci.client.QdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance
        instance.get_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=httpx.Headers(),
        )

        client = LociClient(
            qdrant_url="http://fake:6333",
            vector_size=128,
            distance="dot",
        )
        client._ensure_collection("loci_0")

        create_call = instance.create_collection.call_args
        vectors_config = create_call.kwargs["vectors_config"]
        assert vectors_config.distance == Distance.DOT


# ---------------------------------------------------------------------------
# Euclidean score direction — sync/async clients negate raw Qdrant distances
# ---------------------------------------------------------------------------


def _euclid_hit(pid: str, distance: float) -> MagicMock:
    """A mocked Qdrant hit with a raw EUCLID score (SMALLER is better)."""
    hit = MagicMock()
    hit.score = distance
    hit.id = pid
    hit.vector = [1.0, 0.0, 0.0, 0.0]
    hit.payload = {
        "x": 0.5,
        "y": 0.5,
        "z": 0.5,
        "timestamp_ms": 10_000,
        "scene_id": "s1",
        "scale_level": "patch",
        "confidence": 1.0,
    }
    return hit


def _qdrant_euclid_hits() -> MagicMock:
    # Qdrant returns nearest-first: smallest distance first.
    qr = MagicMock()
    qr.points = [
        _euclid_hit("nearest", 0.5),
        _euclid_hit("middle", 2.0),
        _euclid_hit("farthest", 5.0),
    ]
    return qr


def test_sync_euclidean_ranking_and_truncation():
    with patch("loci.client.QdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance
        instance.query_points.return_value = _qdrant_euclid_hits()

        client = LociClient(
            qdrant_url="http://fake:6333",
            vector_size=4,
            distance="euclidean",
            decay_lambda=0.0,
        )
        client._known_collections = {"loci_2"}
        client._discovered = True

        scored = client.query_scored(
            vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(10_000, 14_999), limit=2
        )

    # Nearest first, and [:limit] keeps the nearest points, not the farthest.
    assert [s.state.id for s in scored] == ["nearest", "middle"]
    # Higher-is-better convention: negated distances, descending.
    assert [s.score for s in scored] == [-0.5, -2.0]
    assert scored[0].score > scored[1].score


@pytest.mark.asyncio
async def test_async_euclidean_ranking_and_truncation():
    with patch("loci.async_client.AsyncQdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance
        instance.query_points = AsyncMock(return_value=_qdrant_euclid_hits())
        instance.get_collections = AsyncMock(side_effect=RuntimeError("not used"))

        client = AsyncLociClient(
            qdrant_url="http://fake:6333",
            vector_size=4,
            distance="euclidean",
            decay_lambda=0.0,
        )
        client._known_collections = {"loci_2"}
        client._discovered = True

        scored = await client.query_scored(
            vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(10_000, 14_999), limit=2
        )

    assert [s.state.id for s in scored] == ["nearest", "middle"]
    assert [s.score for s in scored] == [-0.5, -2.0]


def test_local_euclidean_ranking():
    """The memory backend already returns negative distances (higher-is-better)."""
    client = LocalLociClient(vector_size=4, distance="euclidean", decay_lambda=0.0)

    def _state(ts: int, vector: list[float]) -> WorldState:
        return WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=ts, vector=vector, scene_id="")

    client.insert(_state(1000, [1.0, 0.0, 0.0, 0.0]))  # distance 0 → nearest
    client.insert(_state(1001, [3.0, 0.0, 0.0, 0.0]))  # distance 2
    client.insert(_state(1002, [9.0, 0.0, 0.0, 0.0]))  # distance 8 → farthest

    scored = client.query_scored(vector=[1.0, 0.0, 0.0, 0.0], limit=2)

    assert [s.state.vector[0] for s in scored] == [1.0, 3.0]
    assert scored[0].score == 0.0
    assert scored[1].score == -2.0
    assert scored[0].score > scored[1].score
