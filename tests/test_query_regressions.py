"""Exercise ranking and filtering through each real backend implementation."""

from __future__ import annotations

import inspect
import time

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, QdrantClient

from loci import AsyncLociClient, LocalLociClient, LociClient, WorldState
from loci.cloud_transport import CloudModeUnsupportedError


async def _call(method, *args, **kwargs):
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest_asyncio.fixture(params=["local", "qdrant", "async_qdrant"])
async def client(request, monkeypatch):
    options = {"vector_size": 2, "distance": "euclidean", "decay_lambda": 0.0}
    backend = None
    if request.param == "local":
        result = LocalLociClient(**options)
    elif request.param == "qdrant":
        backend = QdrantClient(":memory:")
        monkeypatch.setattr("loci.client.QdrantClient", lambda **_: backend)
        result = LociClient("http://unused", **options)
    else:
        backend = AsyncQdrantClient(":memory:")
        monkeypatch.setattr("loci.async_client.AsyncQdrantClient", lambda **_: backend)
        result = AsyncLociClient("http://unused", **options)
    yield result
    if backend is not None:
        await _call(backend.close)


def _state(vector, timestamp_ms=1000, x=0.5, confidence=1.0):
    return WorldState(
        x=x,
        y=0.5,
        z=0.5,
        timestamp_ms=timestamp_ms,
        vector=vector,
        confidence=confidence,
    )


@pytest.mark.asyncio
async def test_euclidean_nearest_first_across_shards(client):
    ids = await _call(
        client.insert_batch,
        [_state([5.0, 0.0]), _state([1.0, 0.0], 6000), _state([3.0, 0.0], 11000)],
    )

    hits = await _call(client.query_scored, [1.0, 0.0], limit=3)

    assert [hit.state.id for hit in hits] == [ids[1], ids[2], ids[0]]
    assert [hit.score for hit in hits] == pytest.approx([0.0, -2.0, -4.0])
    assert [hit.decayed_score for hit in hits] == pytest.approx([1.0, 1 / 3, 1 / 5])


@pytest.mark.asyncio
async def test_euclidean_decay_penalizes_old_matches(client):
    now_ms = int(time.time() * 1000)
    client._decay_lambda = 0.001
    ids = await _call(
        client.insert_batch,
        [_state([2.0, 0.0], now_ms - 10000), _state([2.0, 0.0], now_ms)],
    )

    hits = await _call(client.query_scored, [1.0, 0.0], limit=2)

    assert [hit.state.id for hit in hits] == [ids[1], ids[0]]
    assert hits[0].decayed_score > hits[1].decayed_score
    assert [hit.score for hit in hits] == [-1.0, -1.0]


@pytest.mark.asyncio
async def test_confidence_filter_precedes_top_k(client):
    states = [_state([1.0, 0.0], confidence=0.1) for _ in range(8)]
    states.append(_state([2.0, 0.0], confidence=0.9))
    ids = await _call(client.insert_batch, states)

    hits = await _call(client.query, [1.0, 0.0], limit=1, min_confidence=0.8)

    assert [hit.id for hit in hits] == [ids[-1]]


@pytest.mark.asyncio
async def test_exact_spatial_filter_precedes_top_k(client):
    # More false positives than the threefold overfetch, all in a covering
    # Hilbert cell. The valid match must survive backend candidate selection.
    states = [_state([1.0, 0.0], x=0.52) for _ in range(8)]
    states.append(_state([2.0, 0.0]))
    ids = await _call(client.insert_batch, states)

    hits = await _call(
        client.query,
        [1.0, 0.0],
        spatial_bounds={
            "x_min": 0.49,
            "x_max": 0.51,
            "y_min": 0.49,
            "y_max": 0.51,
            "z_min": 0.49,
            "z_max": 0.51,
        },
        limit=1,
    )

    assert [hit.id for hit in hits] == [ids[-1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp_filter", [1001, [999, 1001]])
async def test_exact_ranges_and_extra_predicates_are_conjoined(client, timestamp_filter):
    ids = await _call(
        client.insert_batch,
        [_state([1.0, 0.0], timestamp_ms=ts) for ts in (999, 1000, 1001)],
    )

    hits = await _call(
        client.query,
        [1.0, 0.0],
        time_window_ms=(1000, 2000),
        _extra_payload_filter={"timestamp_ms": timestamp_filter},
        limit=1,
    )

    assert [hit.id for hit in hits] == [ids[-1]]


@pytest.mark.asyncio
async def test_backend_query_failures_are_not_reported_as_empty_results(client, monkeypatch):
    await _call(client.insert, _state([1.0, 0.0]))

    def fail(*args, **kwargs):
        raise ValueError("Required payload index is missing")

    async def fail_async(*args, **kwargs):
        fail()

    if isinstance(client, LocalLociClient):
        monkeypatch.setattr(client.store, "search", fail)
    elif isinstance(client, AsyncLociClient):
        monkeypatch.setattr(client._qdrant, "query_points", fail_async)
    else:
        monkeypatch.setattr(client._qdrant, "query_points", fail)

    with pytest.raises(ValueError, match="Required payload index"):
        await _call(client.query, [1.0, 0.0])


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [LociClient, AsyncLociClient])
@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("query_scored", ([1.0, 0.0],)),
        ("insert_batch", ([],)),
        ("predict_and_retrieve", ([1.0, 0.0], lambda vector: vector)),
        ("funnel_query", ([1.0, 0.0],)),
        ("get_trajectory", ("missing-id",)),
        ("get_causal_context", ("missing-id",)),
    ],
)
async def test_cloud_unsupported_operations_fail_explicitly(client_type, operation, args):
    client = client_type(base_url="https://example.com", api_key="test")
    with pytest.raises(CloudModeUnsupportedError, match=operation):
        await _call(getattr(client, operation), *args)
