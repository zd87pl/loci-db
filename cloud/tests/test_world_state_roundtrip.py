"""Exercise the SDK, real HTTP API models, and in-memory Qdrant together."""

from __future__ import annotations

import inspect
from io import BytesIO

import httpx
import pytest
from qdrant_client import QdrantClient

from loci import AsyncLociClient, LociClient, WorldState


@pytest.fixture
def memory_backend(client, monkeypatch):
    import server as srv

    qdrant = QdrantClient(":memory:")
    with monkeypatch.context() as constructors:
        constructors.setattr("loci.client.QdrantClient", lambda **kwargs: qdrant)
        backend = LociClient(
            "http://unused",
            vector_size=srv.VECTOR_SIZE,
            collection_prefix="test_ns_abc_",
            decay_lambda=0.0,
        )
    monkeypatch.setitem(srv._clients, "test_ns_abc", backend)
    yield backend
    qdrant.close()


async def _resolve(value):
    return await value if inspect.isawaitable(value) else value


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_world_state_roundtrip_and_confidence_filter(
    mode, app, client, memory_backend, monkeypatch
):
    """Both transports preserve fields, generated links, and tenant routing."""
    import server as srv

    if mode == "sync":
        # Exercise urllib request construction and JSON serialization without a
        # socket. FastAPI still handles validation, auth, persistence and output.
        def _urlopen(request, timeout=None):
            response = client.request(
                request.method,
                request.full_url,
                content=request.data,
                headers=dict(request.header_items()),
            )
            response.raise_for_status()
            return BytesIO(response.content)

        monkeypatch.setattr("urllib.request.urlopen", _urlopen)
        sdk = LociClient(base_url="http://testserver", api_key="loci_test")
    else:
        sdk = AsyncLociClient(base_url="http://testserver", api_key="loci_test")
        sdk._cloud._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            headers={"Authorization": "Bearer loci_test"},
        )

    vector = [1.0] + [0.0] * (srv.VECTOR_SIZE - 1)
    metadata = {
        "source": {"camera": "front", "labels": ["obstacle"], "optional": None},
        # Reserved-looking keys remain ordinary nested metadata. They must not
        # overwrite state fields or select another tenant's client/collection.
        "namespace": "other",
        "scene_id": "other-scene",
        "x": 0.9,
        "confidence": 0.0,
    }
    state = WorldState(
        x=0.25,
        y=0.5,
        z=0.75,
        timestamp_ms=1000,
        vector=vector,
        scene_id="robot-episode",
        scale_level="sequence",
        confidence=0.85,
        metadata=metadata,
        prev_state_id="external-predecessor",
        next_state_id="external-successor",
        id="caller-id-is-not-used",
    )
    try:
        first_id = await _resolve(sdk.insert(state))
        hits = await _resolve(sdk.query(vector))
        assert len(hits) == 1
        first = hits[0]
        assert first.id == first_id != state.id
        assert (first.x, first.y, first.z, first.timestamp_ms) == (0.25, 0.5, 0.75, 1000)
        assert first.vector == vector
        assert first.scene_id == "robot-episode"
        assert first.scale_level == "sequence"
        assert first.confidence == 0.85
        assert first.metadata == metadata
        assert first.prev_state_id == "external-predecessor"
        assert first.next_state_id == "external-successor"

        # A later insert in the same scene updates causal links across epochs.
        later = WorldState(
            x=0.25,
            y=0.5,
            z=0.75,
            timestamp_ms=6000,
            vector=vector,
            scene_id=state.scene_id,
            scale_level="frame",
            confidence=0.4,
        )
        later_id = await _resolve(sdk.insert(later))
        hits = await _resolve(sdk.query(vector))
        by_id = {hit.id: hit for hit in hits}
        assert set(by_id) == {first_id, later_id}
        assert by_id[first_id].next_state_id == later_id
        assert by_id[later_id].prev_state_id == first_id

        filtered = await _resolve(sdk.query(vector, min_confidence=0.7))
        assert [hit.id for hit in filtered] == [first_id]
        assert filtered[0].metadata == metadata

        without_vectors = await _resolve(sdk._cloud.query(vector, include_vectors=False))
        assert all(hit.vector == [] for hit in without_vectors)
        assert next(hit for hit in without_vectors if hit.id == first_id).metadata == metadata
        assert all(
            collection.name.startswith("test_ns_abc_loci_")
            for collection in memory_backend._qdrant.get_collections().collections
        )
    finally:
        if mode == "async":
            await sdk._cloud.close()


def test_insert_rejects_oversized_nested_metadata(client, monkeypatch):
    import server as srv

    monkeypatch.setattr(srv, "MAX_METADATA_BYTES", 256)
    response = client.post(
        "/insert",
        json={
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "timestamp_ms": 1000,
            "vector": [0.1] * srv.VECTOR_SIZE,
            "scene_id": "s",
            "metadata": {"nested": {"description": "x" * 256}},
        },
    )
    assert response.status_code == 422
    assert "metadata exceeds 256 bytes" in str(response.json()["detail"])


@pytest.mark.parametrize("min_confidence", [-0.1, 1.1])
def test_query_rejects_invalid_confidence_filter(client, min_confidence):
    import server as srv

    response = client.post(
        "/query",
        json={"vector": [0.1] * srv.VECTOR_SIZE, "min_confidence": min_confidence},
    )
    assert response.status_code == 422
