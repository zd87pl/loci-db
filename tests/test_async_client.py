"""Tests for AsyncLociClient with mocked async Qdrant backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from loci.async_client import AsyncLociClient
from loci.retrieval.predict import PredictRetrieveResult
from loci.schema import WorldState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_async_qdrant():
    """Patch AsyncQdrantClient so no real Qdrant is needed."""
    with patch("loci.async_client.AsyncQdrantClient") as MockCls:
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
        _empty_qr = MagicMock()
        _empty_qr.points = []
        instance.query_points = AsyncMock(return_value=_empty_qr)
        instance.scroll = AsyncMock(return_value=([], None))
        instance.set_payload = AsyncMock()
        instance.retrieve = AsyncMock(return_value=[])
        instance.close = AsyncMock()

        yield instance


@pytest.fixture()
def async_client(mock_async_qdrant):
    return AsyncLociClient(
        qdrant_url="http://fake:6333",
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=4,
        decay_lambda=0.0,
        distance="cosine",
    )


def _make_state(**overrides) -> WorldState:
    defaults = dict(
        x=0.5,
        y=0.5,
        z=0.5,
        timestamp_ms=10_000,
        vector=[1.0, 2.0, 3.0, 4.0],
        scene_id="test_scene",
    )
    defaults.update(overrides)
    return WorldState(**defaults)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_collection_creates_on_404(async_client, mock_async_qdrant):
    await async_client._ensure_collection("loci_0")
    mock_async_qdrant.create_collection.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_collection_idempotent(async_client, mock_async_qdrant):
    await async_client._ensure_collection("loci_0")
    mock_async_qdrant.reset_mock()

    await async_client._ensure_collection("loci_0")
    mock_async_qdrant.get_collection.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_collection_propagates_500(mock_async_qdrant):
    from qdrant_client.http.exceptions import UnexpectedResponse

    mock_async_qdrant.get_collection = AsyncMock(
        side_effect=UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal",
            content=b"",
            headers=httpx.Headers(),
        )
    )
    client = AsyncLociClient.__new__(AsyncLociClient)
    client._qdrant = mock_async_qdrant
    client._vector_size = 4
    client._distance = MagicMock()
    client._known_collections = set()
    client._collection_locks = {}

    with pytest.raises(UnexpectedResponse):
        await client._ensure_collection("loci_0")


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
    assert upsert_call.kwargs["collection_name"] == "loci_2"


@pytest.mark.asyncio
async def test_find_latest_predecessor_paginates_scroll_results(
    async_client, mock_async_qdrant
):
    async_client._known_collections = {"loci_0"}

    page_1 = []
    for i in range(256):
        point = MagicMock()
        point.id = f"p{i}"
        page_1.append(point)

    page_2 = []
    for i in range(256, 300):
        point = MagicMock()
        point.id = f"p{i}"
        page_2.append(point)

    mock_async_qdrant.scroll.side_effect = [
        (page_1, "page-2"),
        (page_2, None),
    ]

    predecessor = await async_client._find_latest_predecessor("scene_a", 20_000)

    assert predecessor == ("p299", "loci_0")
    assert mock_async_qdrant.scroll.call_args_list[1].kwargs["offset"] == "page-2"


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
        "x": 0.5,
        "y": 0.5,
        "z": 0.5,
        "timestamp_ms": 10_000,
        "scene_id": "s1",
        "scale_level": "patch",
        "confidence": 1.0,
    }
    qr = MagicMock()
    qr.points = [hit]
    mock_async_qdrant.query_points = AsyncMock(return_value=qr)

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
    _empty_qr = MagicMock()
    _empty_qr.points = []
    mock_async_qdrant.query_points = AsyncMock(return_value=_empty_qr)

    await async_client.query(
        vector=[1.0, 2.0, 3.0, 4.0],
        time_window_ms=(0, 10_000),
    )
    # Both collections should have been searched
    assert mock_async_qdrant.query_points.call_count == 2


@pytest.mark.asyncio
async def test_query_applies_exact_post_filter(async_client, mock_async_qdrant):
    await async_client.insert(_make_state())

    inside = MagicMock()
    inside.score = 0.8
    inside.id = "inside"
    inside.vector = [1.0, 0.0, 0.0, 0.0]
    inside.payload = {
        "x": 0.02,
        "y": 0.02,
        "z": 0.02,
        "timestamp_ms": 10_000,
        "scene_id": "s1",
        "scale_level": "patch",
        "confidence": 1.0,
    }

    outside = MagicMock()
    outside.score = 0.99
    outside.id = "outside"
    outside.vector = [1.0, 0.0, 0.0, 0.0]
    outside.payload = {
        "x": 0.08,
        "y": 0.08,
        "z": 0.08,
        "timestamp_ms": 10_000,
        "scene_id": "s1",
        "scale_level": "patch",
        "confidence": 1.0,
    }

    qr = MagicMock()
    qr.points = [outside, inside]
    mock_async_qdrant.query_points = AsyncMock(return_value=qr)

    results = await async_client.query(
        vector=[1.0, 0.0, 0.0, 0.0],
        spatial_bounds={
            "x_min": 0.0,
            "x_max": 0.03,
            "y_min": 0.0,
            "y_max": 0.03,
            "z_min": 0.0,
            "z_max": 0.03,
        },
        time_window_ms=(9_000, 11_000),
        limit=5,
    )

    assert [result.id for result in results] == ["inside"]


@pytest.mark.asyncio
async def test_adaptive_query_uses_finer_hilbert_field(mock_async_qdrant):
    from loci.spatial.adaptive import AdaptiveResolution

    async_client = AsyncLociClient(
        qdrant_url="http://fake:6333",
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=4,
        decay_lambda=0.0,
        adaptive=True,
    )
    async_client._adaptive = AdaptiveResolution(base_order=4, max_order=12, density_threshold=3)

    for i in range(10):
        await async_client.insert(_make_state(timestamp_ms=1000 + i * 10))

    mock_async_qdrant.query_points.reset_mock()
    _empty_qr = MagicMock()
    _empty_qr.points = []
    mock_async_qdrant.query_points = AsyncMock(return_value=_empty_qr)

    await async_client.query(
        vector=[1.0, 2.0, 3.0, 4.0],
        spatial_bounds={
            "x_min": 0.49,
            "x_max": 0.51,
            "y_min": 0.49,
            "y_max": 0.51,
            "z_min": 0.49,
            "z_max": 0.51,
        },
        time_window_ms=(1000, 1100),
        limit=5,
    )

    filt = mock_async_qdrant.query_points.call_args.kwargs["query_filter"]
    keys = {condition.key for condition in filt.must}
    assert "hilbert_r8" in keys


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------


def test_invalid_distance_raises():
    with pytest.raises(ValueError, match="distance"):
        AsyncLociClient(
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
    _empty_qr = MagicMock()
    _empty_qr.points = []
    mock_async_qdrant.query_points = AsyncMock(return_value=_empty_qr)

    predicted = [9.0, 8.0, 7.0, 6.0]
    predictor = MagicMock(return_value=predicted)

    await async_client.predict_and_retrieve(
        context_vector=[1.0, 2.0, 3.0, 4.0],
        predictor_fn=predictor,
        future_horizon_ms=2000,
    )

    predictor.assert_called_once_with([1.0, 2.0, 3.0, 4.0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_position, return_prediction, expected_predicted_vector, expected_spatial_bounds",
    [
        (
            (0.5, 0.5, 0.5),
            False,
            None,
            {
                "x_min": 0.3,
                "x_max": 0.7,
                "y_min": 0.3,
                "y_max": 0.7,
                "z_min": 0.3,
                "z_max": 0.7,
            },
        ),
        (None, True, [9.0, 8.0, 7.0, 6.0], None),
    ],
)
async def test_predict_and_retrieve_extended_path(
    async_client,
    current_position,
    return_prediction,
    expected_predicted_vector,
    expected_spatial_bounds,
):
    import time as _time

    now_ms = int(_time.time() * 1000)
    hit = _make_state(timestamp_ms=now_ms + 500)
    async_client.query = AsyncMock(return_value=[hit])

    predicted = [9.0, 8.0, 7.0, 6.0]
    predictor = MagicMock(return_value=predicted)

    with patch("loci.async_client.time.time", return_value=now_ms / 1000.0):
        result = await async_client.predict_and_retrieve(
            context_vector=[1.0, 2.0, 3.0, 4.0],
            predictor_fn=predictor,
            future_horizon_ms=2000,
            limit=3,
            current_position=current_position,
            spatial_search_radius=0.2,
            return_prediction=return_prediction,
        )

    assert isinstance(result, PredictRetrieveResult)
    assert result.results == [hit]
    assert result.predicted_vector == expected_predicted_vector
    assert result.retrieval_latency_ms >= 0.0
    assert result.predictor_call_ms >= 0.0
    predictor.assert_called_once_with([1.0, 2.0, 3.0, 4.0])
    async_client.query.assert_awaited_once_with(
        vector=predicted,
        spatial_bounds=expected_spatial_bounds,
        time_window_ms=(now_ms, now_ms + 2000),
        limit=6,
    )


@pytest.mark.asyncio
async def test_insert_batch_patches_cross_epoch_next_link(async_client, mock_async_qdrant):
    states = [
        _make_state(timestamp_ms=4_900, scene_id="scene_a"),
        _make_state(timestamp_ms=5_100, scene_id="scene_a"),
    ]

    await async_client.insert_batch(states)

    call_kwargs = mock_async_qdrant.set_payload.call_args.kwargs
    assert call_kwargs["collection_name"] == "loci_0"


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager(mock_async_qdrant):
    async with AsyncLociClient(qdrant_url="http://fake:6333", vector_size=4) as client:
        assert isinstance(client, AsyncLociClient)
    mock_async_qdrant.close.assert_called_once()
