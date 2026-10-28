"""Exercise ranking and filtering through each real backend implementation.

Every test runs against the local in-memory client and against the sync and
async Qdrant clients backed by qdrant-client's embedded ``:memory:`` engine,
so the three clients are held to identical ranking and filtering semantics.
"""

from __future__ import annotations

import inspect
import time

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, QdrantClient

from loci import AsyncLociClient, LocalLociClient, LociClient, WorldState
from loci.cloud_transport import CloudModeUnsupportedError
from loci.temporal.decay import apply_decay


async def _call(method, *args, **kwargs):
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest_asyncio.fixture(params=["local", "qdrant", "async_qdrant"])
async def client(request, monkeypatch):
    options = {"vector_size": 2, "distance": "euclidean", "decay_lambda": 0.0}
    # No retry sleeps: the failure tests raise non-transient errors anyway,
    # but keep the suite fast if that ever changes.
    qdrant_options = {**options, "max_retries": 0}
    backend = None
    if request.param == "local":
        result = LocalLociClient(**options)
    elif request.param == "qdrant":
        backend = QdrantClient(":memory:")
        monkeypatch.setattr("loci.client.QdrantClient", lambda **_: backend)
        result = LociClient("http://unused", **qdrant_options)
    else:
        backend = AsyncQdrantClient(":memory:")
        monkeypatch.setattr("loci.async_client.AsyncQdrantClient", lambda **_: backend)
        result = AsyncLociClient("http://unused", **qdrant_options)
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
async def test_euclidean_nearest_first_across_epochs(client):
    # Three states in three logical epochs (epoch_size_ms=5000): ranking must
    # be nearest-first regardless of age, with raw score == -distance and
    # decayed score == 1 / (1 + distance) at decay_lambda == 0.
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


def test_apply_decay_euclidean_older_ranks_lower():
    """Unit-level twin of the client test: decay on a negative raw score
    must not promote the older hit (it would with plain multiplication)."""
    results = [
        {"score": -1.0, "timestamp_ms": 0},  # older
        {"score": -1.0, "timestamp_ms": 10_000},  # newer
    ]
    apply_decay(results, now_ms=10_000, lambda_=1e-3, distance="euclidean")

    assert [r["timestamp_ms"] for r in results] == [10_000, 0]
    assert results[0]["decayed_score"] == pytest.approx(0.5)
    assert all(0.0 < r["decayed_score"] <= 1.0 for r in results)
    assert [r["score"] for r in results] == [-1.0, -1.0]  # raw scores untouched


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
    # The only state satisfying BOTH the time window and the extra predicate
    # (ts == 1001) is deliberately the farthest from the query vector.  If a
    # same-key extra predicate replaced the window instead of narrowing it,
    # the nearer decoys at ts == 999 would take the limit=1 slot and the
    # exact post-filter would then empty the result.
    decoys = [_state([1.0, 0.0], timestamp_ms=999) for _ in range(4)]
    in_window_decoy = _state([1.0, 0.0], timestamp_ms=1000)
    target = _state([0.9, 0.0], timestamp_ms=1001)
    ids = await _call(client.insert_batch, [*decoys, in_window_decoy, target])

    hits = await _call(
        client.query,
        [1.0, 0.0],
        time_window_ms=(1000, 2000),
        _extra_payload_filter={"timestamp_ms": timestamp_filter},
        limit=1,
    )

    assert [hit.id for hit in hits] == [ids[-1]]


@pytest.mark.asyncio
async def test_inverted_window_and_out_of_range_confidence_select_nothing(client):
    """Degenerate predicates yield an empty result, not an exception: the REST
    servers forward caller-supplied windows unchecked, and this matches the
    pre-0.4.0 behaviour of all three clients."""
    await _call(client.insert, _state([1.0, 0.0], timestamp_ms=1500, confidence=0.9))

    assert await _call(client.query, [1.0, 0.0], time_window_ms=(1000, 2000)) != []
    assert await _call(client.query, [1.0, 0.0], time_window_ms=(2000, 1000)) == []
    assert await _call(client.query, [1.0, 0.0], min_confidence=1.5) == []


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
    ("operation", "args", "kwargs", "match"),
    [
        # insert_batch, query_scored, get_trajectory and get_causal_context have cloud
        # endpoints (tests/test_cloud_conformance.py); internal filters do not.
        ("query_scored", ([1.0, 0.0],), {"_epoch_ids": {1}}, "advanced filtering"),
        # The legacy predict_and_retrieve path (plain query on the predicted
        # vector) is supported in cloud mode; the scored path is not.
        (
            "predict_and_retrieve",
            ([1.0, 0.0], lambda vector: vector),
            {"current_position": (0.5, 0.5, 0.5)},
            "predict_and_retrieve",
        ),
        ("funnel_query", ([1.0, 0.0],), {}, "funnel_query"),
    ],
)
async def test_cloud_unsupported_operations_fail_explicitly(
    client_type, operation, args, kwargs, match
):
    client = client_type(base_url="https://example.com", api_key="test")
    with pytest.raises(CloudModeUnsupportedError, match=match):
        await _call(getattr(client, operation), *args, **kwargs)
