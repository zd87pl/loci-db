"""Regression tests for readers sharing an evolving collection catalog."""

from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from loci.async_client import AsyncLociClient
from loci.client import LociClient


def _listing(*names):
    return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in names])


def _point(point_id, timestamp_ms):
    return PointStruct(
        id=point_id,
        vector=[1.0, 0.0],
        payload={
            "x": 0.5,
            "y": 0.5,
            "z": 0.5,
            "timestamp_ms": timestamp_ms,
            "scene_id": "shared-scene",
            "scale_level": "frame",
        },
    )


def test_reader_sees_external_additions_and_deletions():
    with closing(QdrantClient(":memory:")) as backend:
        with patch("loci.client.QdrantClient", return_value=backend):
            reader = LociClient(qdrant_url="http://unused", vector_size=2, decay_lambda=0)
        assert reader.query([1.0, 0.0]) == []
        for epoch in (0, 1):
            backend.create_collection(
                f"loci_{epoch}", vectors_config=VectorParams(size=2, distance=Distance.COSINE)
            )
            backend.upsert(f"loci_{epoch}", [_point(epoch + 1, epoch * 5000)])
            assert len(reader.query([1.0, 0.0])) == epoch + 1
        backend.delete_collection("loci_0")
        assert [hit.id for hit in reader.query([1.0, 0.0])] == ["2"]
        assert reader._known_collections == {"loci_1"}
        backend.delete_collection("loci_1")
        assert reader.query([1.0, 0.0]) == []
        assert reader._known_collections == set()


@pytest.mark.asyncio
async def test_async_reader_sees_external_additions_and_deletions():
    backend = AsyncQdrantClient(":memory:")
    try:
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            reader = AsyncLociClient(qdrant_url="http://unused", vector_size=2, decay_lambda=0)
        assert await reader.query([1.0, 0.0]) == []
        for epoch in (0, 1):
            await backend.create_collection(
                f"loci_{epoch}", vectors_config=VectorParams(size=2, distance=Distance.COSINE)
            )
            await backend.upsert(f"loci_{epoch}", [_point(epoch + 1, epoch * 5000)])
            assert len(await reader.query([1.0, 0.0])) == epoch + 1
        await backend.delete_collection("loci_0")
        assert [hit.id for hit in await reader.query([1.0, 0.0])] == ["2"]
        assert reader._known_collections == {"loci_1"}
        await backend.delete_collection("loci_1")
        assert await reader.query([1.0, 0.0]) == []
        assert reader._known_collections == set()
    finally:
        await backend.close()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_discovery_matches_only_canonical_epochs_in_own_namespace(async_mode):
    backend = MagicMock()
    response = _listing(
        "tenant_a_loci_1",
        "tenant_a_loci_2",
        "tenant_a_loci_backup",
        "tenant_a_loci_01",
        "tenant_a_loci_+3",
        "tenant_b_loci_4",
        "loci_5",
    )
    if async_mode:
        backend.get_collections = AsyncMock(return_value=response)
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            client = AsyncLociClient(qdrant_url="http://unused", collection_prefix="tenant_a_")
    else:
        backend.get_collections.return_value = response
        with patch("loci.client.QdrantClient", return_value=backend):
            client = LociClient(qdrant_url="http://unused", collection_prefix="tenant_a_")
    client._known_collections = {"tenant_a_loci_0"}
    if async_mode:
        await client._discover_collections()
    else:
        client._discover_collections()
    assert client._known_collections == {"tenant_a_loci_1", "tenant_a_loci_2"}


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_failed_discovery_preserves_cache_and_fails_the_query(async_mode):
    backend = MagicMock()
    if async_mode:
        backend.get_collections = AsyncMock(side_effect=ConnectionError("unavailable"))
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            client = AsyncLociClient(qdrant_url="http://unused", max_retries=0)
    else:
        backend.get_collections.side_effect = ConnectionError("unavailable")
        with patch("loci.client.QdrantClient", return_value=backend):
            client = LociClient(qdrant_url="http://unused", max_retries=0)
    client._known_collections = {"loci_0"}
    with pytest.raises(ConnectionError, match="unavailable"):
        if async_mode:
            await client.query([1.0, 0.0])
        else:
            client.query([1.0, 0.0])
    assert client._known_collections == {"loci_0"}


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_discovery_retries_transient_errors(async_mode):
    backend = MagicMock()
    outcomes = [ConnectionError("retry me"), _listing("loci_1")]
    if async_mode:
        backend.get_collections = AsyncMock(side_effect=outcomes)
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            client = AsyncLociClient(qdrant_url="http://unused", retry_backoff=0)
        await client._discover_collections()
    else:
        backend.get_collections.side_effect = outcomes
        with patch("loci.client.QdrantClient", return_value=backend):
            client = LociClient(qdrant_url="http://unused", retry_backoff=0)
        client._discover_collections()
    assert client._known_collections == {"loci_1"}
    assert backend.get_collections.call_count == 2


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_predecessor_discovery_includes_shards_from_other_writers(async_mode):
    backend = MagicMock()
    response = _listing("loci_0", "loci_1")
    if async_mode:
        backend.get_collections = AsyncMock(return_value=response)
        backend.scroll = AsyncMock(side_effect=[([], None), ([SimpleNamespace(id="prior")], None)])
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            client = AsyncLociClient(qdrant_url="http://unused")
    else:
        backend.get_collections.return_value = response
        backend.scroll.side_effect = [([], None), ([SimpleNamespace(id="prior")], None)]
        with patch("loci.client.QdrantClient", return_value=backend):
            client = LociClient(qdrant_url="http://unused")
    client._known_collections = {"loci_1"}
    if async_mode:
        predecessor = await client._find_latest_predecessor("shared-scene", 7000)
    else:
        predecessor = client._find_latest_predecessor("shared-scene", 7000)
    assert predecessor == ("prior", "loci_0")
    assert [call.kwargs["collection_name"] for call in backend.scroll.call_args_list] == [
        "loci_1",
        "loci_0",
    ]
