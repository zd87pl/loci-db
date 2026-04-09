"""Integration tests for the LOCI REST API server (server.py).

Uses httpx + FastAPI TestClient so no real Qdrant instance is needed — the
LociClient is monkey-patched at the module level before the app is imported.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_SIZE = 4  # small dimension for tests


def _make_client() -> TestClient:
    """Return a TestClient with server module freshly imported at VECTOR_SIZE=4."""
    # Force re-import so VECTOR_SIZE env var takes effect even if already cached
    if "server" in sys.modules:
        del sys.modules["server"]

    mock_loci_client = MagicMock()
    mock_loci_client.return_value.insert.return_value = "test-uuid-1234"
    mock_loci_client.return_value.query.return_value = []

    with (
        patch.dict("os.environ", {"LOCI_VECTOR_SIZE": str(VECTOR_SIZE)}),
        patch("loci.LociClient", mock_loci_client),
    ):
        import server as srv  # noqa: PLC0415

        importlib.reload(srv)
        return TestClient(srv.app), srv, mock_loci_client.return_value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self) -> None:
        client, _, _ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "vector_size" in data


class TestInsert:
    def test_insert_valid(self) -> None:
        client, srv, mock = _make_client()
        payload: dict[str, Any] = {
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "timestamp_ms": 1000,
            "vector": [0.1, 0.2, 0.3, 0.4],
            "scene_id": "scene-a",
        }
        resp = client.post("/insert", json=payload)
        assert resp.status_code == 200
        assert resp.json() == {"id": "test-uuid-1234"}
        mock.insert.assert_called_once()

    def test_insert_wrong_vector_size(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "x": 0.1,
            "y": 0.2,
            "z": 0.3,
            "timestamp_ms": 1000,
            "vector": [0.1, 0.2],  # wrong length
            "scene_id": "scene-a",
        }
        resp = client.post("/insert", json=payload)
        assert resp.status_code == 400
        assert "dimensions" in resp.json()["detail"]

    def test_insert_metadata_isolation(self) -> None:
        """Successive requests must not share metadata across instances."""
        client, _, _ = _make_client()
        base = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "timestamp_ms": 0,
            "vector": [0.0] * VECTOR_SIZE,
            "scene_id": "s",
        }
        resp1 = client.post("/insert", json={**base, "metadata": {"key": "val"}})
        resp2 = client.post("/insert", json=base)  # no metadata
        assert resp1.status_code == 200
        assert resp2.status_code == 200


class TestQuery:
    def test_query_valid(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.1, 0.2, 0.3, 0.4],
            "limit": 5,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 200
        assert "results" in resp.json()

    def test_query_wrong_vector_size(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {"vector": [0.1, 0.2]}
        resp = client.post("/query", json=payload)
        assert resp.status_code == 400
        assert "dimensions" in resp.json()["detail"]

    def test_query_limit_too_large(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "limit": 9999,  # exceeds max of 1000
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_query_limit_zero_invalid(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "limit": 0,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 422

    def test_query_negative_limit_invalid(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "limit": -1,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 422

    def test_query_with_time_window(self) -> None:
        client, _, mock = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "time_start_ms": 0,
            "time_end_ms": 5000,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 200
        call_kwargs = mock.query.call_args.kwargs
        assert call_kwargs["time_window_ms"] == (0, 5000)
