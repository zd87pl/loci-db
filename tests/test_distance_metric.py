"""Tests for configurable distance metric on sync client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from engram.client import _DISTANCE_MAP, EngramClient


def test_valid_distances():
    for name in ("cosine", "dot", "euclidean"):
        with patch("engram.client.QdrantClient"):
            client = EngramClient(qdrant_url="http://fake:6333", distance=name)
            assert client._distance == _DISTANCE_MAP[name]


def test_invalid_distance_raises():
    with pytest.raises(ValueError, match="distance"):
        with patch("engram.client.QdrantClient"):
            EngramClient(qdrant_url="http://fake:6333", distance="hamming")


def test_collection_uses_configured_distance():
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import Distance

    with patch("engram.client.QdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance
        instance.get_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=httpx.Headers(),
        )

        client = EngramClient(
            qdrant_url="http://fake:6333",
            vector_size=128,
            distance="dot",
        )
        client._ensure_collection("engram_0")

        create_call = instance.create_collection.call_args
        vectors_config = create_call.kwargs["vectors_config"]
        assert vectors_config.distance == Distance.DOT
