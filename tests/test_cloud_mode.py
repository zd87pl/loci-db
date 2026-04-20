"""Unit tests for SDK cloud mode (LociClient + AsyncLociClient).

The HTTP layer is stubbed — we don't need a live server to verify that the
client routes ``insert``/``query`` through the cloud transport, builds the
correct request payload, and maps responses back to :class:`WorldState`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from loci import AsyncLociClient, LociClient, WorldState
from loci.cloud_transport import (
    CloudModeUnsupportedError,
    CloudTransport,
)

# ── Sync client ────────────────────────────────────────────────────────────


def _make_sync_client() -> tuple[LociClient, CloudTransport]:
    client = LociClient(base_url="https://api.example.com", api_key="loci_x", vector_size=4)
    assert client._cloud is not None
    return client, client._cloud


def test_sync_requires_qdrant_url_without_base_url():
    with pytest.raises(ValueError, match="qdrant_url is required"):
        LociClient(vector_size=4)


def test_sync_requires_api_key_for_cloud_mode():
    with pytest.raises(ValueError, match="cloud mode requires api_key"):
        LociClient(base_url="https://api.example.com", vector_size=4)


def test_sync_insert_routes_through_cloud():
    client, transport = _make_sync_client()
    transport._request = MagicMock(return_value={"id": "abc123"})

    state = WorldState(x=0.1, y=0.2, z=0.3, timestamp_ms=1000, vector=[0.0] * 4, scene_id="s")
    result_id = client.insert(state)

    assert result_id == "abc123"
    transport._request.assert_called_once()
    method, path, body = transport._request.call_args.args
    assert method == "POST"
    assert path == "/insert"
    assert body["x"] == 0.1
    assert body["y"] == 0.2
    assert body["z"] == 0.3
    assert body["scene_id"] == "s"


def test_sync_query_routes_through_cloud():
    client, transport = _make_sync_client()
    transport._request = MagicMock(
        return_value={
            "results": [
                {
                    "id": "p1",
                    "x": 0.5,
                    "y": 0.5,
                    "z": 0.5,
                    "timestamp_ms": 123,
                    "scene_id": "s",
                }
            ]
        }
    )

    hits = client.query(
        vector=[0.0] * 4,
        spatial_bounds={
            "x_min": 0.1,
            "x_max": 0.9,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
        },
        time_window_ms=(1000, 2000),
        limit=3,
    )

    assert len(hits) == 1
    assert hits[0].id == "p1"
    assert hits[0].x == 0.5

    body = transport._request.call_args.args[2]
    assert body["vector"] == [0.0] * 4
    assert body["limit"] == 3
    assert body["x_min"] == 0.1
    assert body["x_max"] == 0.9
    assert body["time_start_ms"] == 1000
    assert body["time_end_ms"] == 2000


def test_sync_query_unsupported_options_raise():
    client, _ = _make_sync_client()
    with pytest.raises(CloudModeUnsupportedError):
        client.query(vector=[0.0] * 4, _extra_payload_filter={"scene_id": "x"})
    with pytest.raises(CloudModeUnsupportedError):
        client.query(vector=[0.0] * 4, _epoch_ids={1})
    with pytest.raises(CloudModeUnsupportedError):
        client.query(vector=[0.0] * 4, min_confidence=0.5)


def test_sync_local_mode_unchanged_shape():
    """Local-mode construction must not regress: qdrant_url is accepted positionally."""
    # We only care that __init__ works without hitting Qdrant; construct with an
    # unreachable URL and do not call any method that would hit the wire.
    client = LociClient("http://localhost:6333", vector_size=4)
    assert client._cloud is None


# ── Async client ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_requires_qdrant_url_without_base_url():
    with pytest.raises(ValueError, match="qdrant_url is required"):
        AsyncLociClient(vector_size=4)


@pytest.mark.asyncio
async def test_async_requires_api_key_for_cloud_mode():
    with pytest.raises(ValueError, match="cloud mode requires api_key"):
        AsyncLociClient(base_url="https://api.example.com", vector_size=4)


@pytest.mark.asyncio
async def test_async_insert_routes_through_cloud():
    client = AsyncLociClient(base_url="https://api.example.com", api_key="loci_x", vector_size=4)
    assert client._cloud is not None

    async def _fake_request(method, path, body=None):
        assert method == "POST"
        assert path == "/insert"
        assert body["scene_id"] == "async_s"
        return {"id": "async-id"}

    client._cloud._request = _fake_request  # type: ignore[assignment]

    state = WorldState(x=0.1, y=0.2, z=0.3, timestamp_ms=2000, vector=[0.0] * 4, scene_id="async_s")
    result_id = await client.insert(state)
    assert result_id == "async-id"


@pytest.mark.asyncio
async def test_async_query_routes_through_cloud():
    client = AsyncLociClient(base_url="https://api.example.com", api_key="loci_x", vector_size=4)

    async def _fake_request(method, path, body=None):
        assert path == "/query"
        return {
            "results": [
                {"id": "pA", "x": 0.1, "y": 0.2, "z": 0.3, "timestamp_ms": 9, "scene_id": "s"}
            ]
        }

    client._cloud._request = _fake_request  # type: ignore[assignment]

    hits = await client.query(vector=[0.0] * 4, limit=1)
    assert len(hits) == 1
    assert hits[0].id == "pA"
