"""Regression tests for readers sharing the bounded collection set.

The storage layout is exactly two collections (``{prefix}loci_data`` and
``{prefix}loci_summary``), so there is no catalogue to discover.  What a
reader must still get right is the *lifecycle* of those collections when
another writer (or an operator) creates or deletes them underneath it:

* a reader constructed before any writer sees ``loci_data`` as soon as it
  exists (the not-found probe result is never cached);
* a collection deleted after the reader cached it as ready yields an empty
  result instead of an error, and the readiness cache is reset so the
  collection is re-probed (and found again once recreated).

Backend *outages* on the probe or search path are a different matter and
propagate; see ``test_probe_failure_propagates`` and the search-failure
tests in ``tests/test_client.py`` / ``tests/test_async_client.py``.
"""

from __future__ import annotations

from contextlib import closing
from unittest.mock import patch

import pytest
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from loci.async_client import AsyncLociClient
from loci.client import LociClient

DATA = "loci_data"


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


def _vectors():
    return VectorParams(size=2, distance=Distance.COSINE)


def test_reader_tracks_externally_created_and_deleted_data_collection():
    with closing(QdrantClient(":memory:")) as backend:
        with patch("loci.client.QdrantClient", return_value=backend):
            reader = LociClient(
                qdrant_url="http://unused", vector_size=2, decay_lambda=0, max_retries=0
            )

        # Nothing exists yet: empty, and nothing cached as ready.
        assert reader.query([1.0, 0.0]) == []
        assert DATA not in reader._collection_ready

        # An external writer creates the data collection and inserts.
        backend.create_collection(DATA, vectors_config=_vectors())
        backend.upsert(DATA, [_point(1, 0), _point(2, 5000)])
        assert {hit.id for hit in reader.query([1.0, 0.0])} == {"1", "2"}
        assert reader._collection_ready[DATA] is True

        # The collection is deleted out from under the reader: empty result,
        # no exception, readiness cache reset.
        backend.delete_collection(DATA)
        assert reader.query([1.0, 0.0]) == []
        assert DATA not in reader._collection_ready
        assert reader.query([1.0, 0.0]) == []  # re-probe finds nothing, still fine

        # Recreated by the external writer: visible again without a new client.
        backend.create_collection(DATA, vectors_config=_vectors())
        backend.upsert(DATA, [_point(3, 10_000)])
        assert [hit.id for hit in reader.query([1.0, 0.0])] == ["3"]
        assert reader._collection_ready[DATA] is True


@pytest.mark.asyncio
async def test_async_reader_tracks_externally_created_and_deleted_data_collection():
    backend = AsyncQdrantClient(":memory:")
    try:
        with patch("loci.async_client.AsyncQdrantClient", return_value=backend):
            reader = AsyncLociClient(
                qdrant_url="http://unused", vector_size=2, decay_lambda=0, max_retries=0
            )

        assert await reader.query([1.0, 0.0]) == []
        assert DATA not in reader._collection_ready

        await backend.create_collection(DATA, vectors_config=_vectors())
        await backend.upsert(DATA, [_point(1, 0), _point(2, 5000)])
        assert {hit.id for hit in await reader.query([1.0, 0.0])} == {"1", "2"}
        assert reader._collection_ready[DATA] is True

        await backend.delete_collection(DATA)
        assert await reader.query([1.0, 0.0]) == []
        assert DATA not in reader._collection_ready
        assert await reader.query([1.0, 0.0]) == []

        await backend.create_collection(DATA, vectors_config=_vectors())
        await backend.upsert(DATA, [_point(3, 10_000)])
        assert [hit.id for hit in await reader.query([1.0, 0.0])] == ["3"]
        assert reader._collection_ready[DATA] is True
    finally:
        await backend.close()


def test_sync_reader_in_own_namespace_ignores_other_tenants():
    """Prefixed clients only ever touch ``{prefix}loci_data``/``{prefix}loci_summary``."""
    with closing(QdrantClient(":memory:")) as backend:
        backend.create_collection("tenant_b_loci_data", vectors_config=_vectors())
        backend.upsert("tenant_b_loci_data", [_point(9, 0)])
        backend.create_collection("loci_data", vectors_config=_vectors())
        backend.upsert("loci_data", [_point(8, 0)])

        with patch("loci.client.QdrantClient", return_value=backend):
            reader = LociClient(
                qdrant_url="http://unused",
                vector_size=2,
                decay_lambda=0,
                collection_prefix="tenant_a_",
                max_retries=0,
            )

        assert reader.query([1.0, 0.0]) == []
        assert reader._collection_ready == {}
        assert {c.name for c in backend.get_collections().collections} == {
            "tenant_b_loci_data",
            "loci_data",
        }
