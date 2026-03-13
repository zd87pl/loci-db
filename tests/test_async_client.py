"""Tests for AsyncEngramClient with mocked async Qdrant backend."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from engram.async_client import AsyncEngramClient
from engram.schema import WorldState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_async_qdrant():
    """Patch AsyncQdrantClient so no real Qdrant is needed."""
    with patch("engram.async_client.AsyncQdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance

        from qdrant_client.http.exceptions import UnexpectedResponse

        resp_404 = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=httpx.Headers(),
        )

        instance.get_collection = AsyncMock(side_effect=resp_404)
        instance.create_collection = AsyncMock()
        instance.create_payload_index = AsyncMock()
        instance.upsert = AsyncMock()
        instance.search = AsyncMock(return_value=[])
        instance.scroll = AsyncMock(return_value=([], None))
        instance.set_payload = AsyncMock()
        instance.retrieve = AsyncMock(return_value=[])
        instance.close = AsyncMock()

        yield instance


@pytest.fixture()
def async_client(mock_async_qdrant):
    return AsyncEngramClient(
        qdrant_url="http://fake:6333",
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=4,
        decay_lambda=0.0,
        distance="cosine",
    )


def _make_state(**overrides) -> WorldState:
    defaults = dict(
        x=0.5, y=0.5, z=0.5, timestamp_ms=10_000,
        vector=[1.0, 2.0, 3.0, 4.0], scene_id="test_scene",
    )
    defaults.update(overrides)
    return WorldState(**defaults)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_collection_creates_on_404(async_client, mock_async_qdrant):
    await async_client._ensure_collection("engram_0")
    mock_async_qdrant.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_collection_idempotent(async_client, mock_async_qdrant):
    await async_client._ensure_collection("engram_0")
    mock_async_qdrant.reset_mock()

    await async_client._ensure_collection("engram_0")
    mock_async_qdrant.get_collection.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_collection_propagates_500(mock_async_qdrant):
    from qdrant_client.http.exceptions import UnexpectedResponse

    mock_async_qdrant.get_collection = AsyncMock(
        side_effect=UnexpectedResponse(
            status_code=500, reason_phrase="Internal",
            content=b"", headers=httpx.Headers(),
        )
    )
    client = AsyncEngramClient.__new__(AsyncEngramClient)
    client._qdrant = mock_async_qdrant
    client._vector_size = 4
    client._distance = MagicMock()
    client._known_collections = set()
    client._collection_locks = {}

    with pytest.raises(UnexpectedResponse):
        await client._ensure_collection("engram_0")


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_returns_id(async_client, mock_async_qdrant):
    result = await async_client.insert(_make_state())
    assert isinstance(result, str)
    assert len(result) == 32


@pytest.mark.asyncio
async def test_insert_does_not_mutate(async_client, mock_async_qdrant):
    state = _make_state()
    original_id = state.id
    await async_client.insert(state)
    assert state.id == original_id


@pytest.mark.asyncio
async def test_insert_routes_to_correct_collection(async_client, mock_async_qdrant):
    await async_client.insert(_make_state(timestamp_ms=10_000))
    upsert_call = mock_async_qdrant.upsert.call_args
    assert upsert_call.kwargs["collection_name"] == "engram_2"


# ---------------------------------------------------------------------------
# Insert batch with causal linking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_batch_returns_correct_count(async_client, mock_async_qdrant):
    states = [_make_state(timestamp_ms=10_000 + i * 50) for i in range(5)]
    ids = await async_client.insert_batch(states)
    assert len(ids) == 5
    assert len(set(ids)) == 5


@pytest.mark.asyncio
async def test_insert_batch_causal_links(async_client, mock_async_qdrant):
    """States in the same scene should be causally linked within a batch."""
    states = [
        _make_state(timestamp_ms=10_000, scene_id="scene_a"),
        _make_state(timestamp_ms=10_050, scene_id="scene_a"),
        _make_state(timestamp_ms=10_100, scene_id="scene_a"),
    ]
    await async_client.insert_batch(states)

    # Verify that upsert was called and points have prev_state_id set
    upsert_calls = mock_async_qdrant.upsert.call_args_list
    all_points = []
    for call in upsert_calls:
        all_points.extend(call.kwargs["points"])

    # Sort by timestamp to check linking
    all_points.sort(key=lambda p: p.payload["timestamp_ms"])
    assert all_points[0].payload.get("prev_state_id") is None
    assert all_points[1].payload.get("prev_state_id") == all_points[0].id
    assert all_points[2].payload.get("prev_state_id") == all_points[1].id


@pytest.mark.asyncio
async def test_insert_batch_separate_scenes(async_client, mock_async_qdrant):
    """States in different scenes should NOT be linked to each other."""
    states = [
        _make_state(timestamp_ms=10_000, scene_id="scene_a"),
        _make_state(timestamp_ms=10_050, scene_id="scene_b"),
    ]
    await async_client.insert_batch(states)

    upsert_calls = mock_async_qdrant.upsert.call_args_list
    all_points = []
    for call in upsert_calls:
        all_points.extend(call.kwargs["points"])

    # Neither should have a prev_state_id
    for point in all_points:
        assert point.payload.get("prev_state_id") is None


# ---------------------------------------------------------------------------
# Query — parallel fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_returns_empty_when_no_collections(async_client, mock_async_qdrant):
    results = await async_client.query(vector=[1.0, 2.0, 3.0, 4.0])
    assert results == []


@pytest.mark.asyncio
async def test_query_with_vectors(async_client, mock_async_qdrant):
    await async_client.insert(_make_state())

    hit = MagicMock()
    hit.score = 0.95
    hit.id = "abc123"
    hit.vector = [1.0, 2.0, 3.0, 4.0]
    hit.payload = {
        "x": 0.5, "y": 0.5, "z": 0.5,
        "timestamp_ms": 10_000,
        "scene_id": "s1",
        "scale_level": "patch",
        "confidence": 1.0,
    }
    mock_async_qdrant.search = AsyncMock(return_value=[hit])

    results = await async_client.query(
        vector=[1.0, 2.0, 3.0, 4.0],
        time_window_ms=(10_000, 15_000),
    )
    assert len(results) == 1
    assert results[0].vector == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.asyncio
async def test_query_parallel_fanout(async_client, mock_async_qdrant):
    """Searches across multiple epochs should happen via asyncio.gather."""
    # Create states in two different epochs
    await async_client.insert(_make_state(timestamp_ms=3000))  # epoch 0
    await async_client.insert(_make_state(timestamp_ms=8000))  # epoch 1
    mock_async_qdrant.search = AsyncMock(return_value=[])

    await async_client.query(
        vector=[1.0, 2.0, 3.0, 4.0],
        time_window_ms=(0, 10_000),
    )
    # Both collections should have been searched
    assert mock_async_qdrant.search.call_count == 2


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------


def test_invalid_distance_raises():
    with pytest.raises(ValueError, match="distance"):
        AsyncEngramClient(
            qdrant_url="http://fake:6333",
            distance="manhattan",
        )


# ---------------------------------------------------------------------------
# Predict and retrieve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_predict_and_retrieve(async_client, mock_async_qdrant):
    import time as _time

    now_ms = int(_time.time() * 1000)
    await async_client.insert(_make_state(timestamp_ms=now_ms))
    mock_async_qdrant.search = AsyncMock(return_value=[])

    predicted = [9.0, 8.0, 7.0, 6.0]
    predictor = MagicMock(return_value=predicted)

    await async_client.predict_and_retrieve(
        context_vector=[1.0, 2.0, 3.0, 4.0],
        predictor_fn=predictor,
        future_horizon_ms=2000,
    )

    predictor.assert_called_once_with([1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager(mock_async_qdrant):
    async with AsyncEngramClient(
        qdrant_url="http://fake:6333", vector_size=4
    ) as client:
        assert isinstance(client, AsyncEngramClient)
    mock_async_qdrant.close.assert_called_once()
