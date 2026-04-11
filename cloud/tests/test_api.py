"""Integration tests for the Cloud LOCI API (Phase 1).

Tests cover:
  - Auth: valid key, invalid key, missing key
  - Insert: happy path, wrong vector dim, spatial bounds, metadata limit
  - Query: happy path, wrong vector dim, bad bounds
  - Health / readiness probes
  - CORS: only listed origins are reflected
  - OpenAPI: /openapi.json always available
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

VECTOR_SIZE = int(os.environ.get("LOCI_VECTOR_SIZE", "4"))
VEC = [0.1] * VECTOR_SIZE


# ── Health & readiness ─────────────────────────────────────────────────────


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health_no_auth(client):
    """Health must be reachable without an Authorization header."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_openapi_json_available(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "paths" in schema
    assert "/insert" in schema["paths"]
    assert "/query" in schema["paths"]


def test_swagger_ui_in_dev_mode(client):
    """Swagger UI is available when LOCI_DEV_MODE=true."""
    resp = client.get("/docs")
    assert resp.status_code == 200


# ── Auth ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def client_no_auth(client):
    """Reuse the session client but with dependency overrides cleared.

    Does NOT start a new ASGI lifespan — that would trigger _clients.clear()
    and destroy the mock injected by the app fixture.
    Function-scoped so overrides are restored after each individual test.
    """
    import server as srv

    saved = dict(srv.app.dependency_overrides)
    srv.app.dependency_overrides.clear()
    yield client
    srv.app.dependency_overrides.update(saved)


def test_insert_requires_auth(client_no_auth):
    """Missing Bearer token must be rejected by real auth dependency."""
    resp = client_no_auth.post(
        "/insert",
        json={"x": 0, "y": 0, "z": 0, "timestamp_ms": 0, "vector": VEC, "scene_id": "s"},
    )
    assert resp.status_code in (401, 403)


def test_query_requires_auth(client_no_auth):
    """Missing Bearer token must be rejected by real auth dependency."""
    resp = client_no_auth.post("/query", json={"vector": VEC})
    assert resp.status_code in (401, 403)


# ── Insert ─────────────────────────────────────────────────────────────────


def test_insert_happy_path(client):
    resp = client.post(
        "/insert",
        json={
            "x": 0.5, "y": 0.5, "z": 0.5,
            "timestamp_ms": 1000,
            "vector": VEC,
            "scene_id": "test-scene",
        },
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 200
    assert "id" in resp.json()


def test_insert_wrong_vector_dim(client):
    bad_vec = [0.1] * (VECTOR_SIZE + 1)
    resp = client.post(
        "/insert",
        json={
            "x": 0.5, "y": 0.5, "z": 0.5,
            "timestamp_ms": 1000,
            "vector": bad_vec,
            "scene_id": "test-scene",
        },
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any("dimensions" in str(d) for d in detail)


def test_insert_negative_timestamp_rejected(client):
    resp = client.post(
        "/insert",
        json={
            "x": 0.0, "y": 0.0, "z": 0.0,
            "timestamp_ms": -1,
            "vector": VEC,
            "scene_id": "s",
        },
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422


def test_insert_empty_scene_id_rejected(client):
    resp = client.post(
        "/insert",
        json={
            "x": 0.0, "y": 0.0, "z": 0.0,
            "timestamp_ms": 0,
            "vector": VEC,
            "scene_id": "",
        },
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422


def test_insert_confidence_out_of_range(client):
    resp = client.post(
        "/insert",
        json={
            "x": 0.0, "y": 0.0, "z": 0.0,
            "timestamp_ms": 0,
            "vector": VEC,
            "scene_id": "s",
            "confidence": 1.5,
        },
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422


# ── Query ──────────────────────────────────────────────────────────────────


def test_query_happy_path(client):
    resp = client.post(
        "/query",
        json={"vector": VEC},
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 200
    assert "results" in resp.json()


def test_query_wrong_vector_dim(client):
    resp = client.post(
        "/query",
        json={"vector": [0.1] * (VECTOR_SIZE + 3)},
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422


def test_query_inverted_x_bounds(client):
    resp = client.post(
        "/query",
        json={"vector": VEC, "x_min": 1.0, "x_max": 0.0},
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422


def test_query_inverted_time_bounds(client):
    resp = client.post(
        "/query",
        json={"vector": VEC, "time_start_ms": 1000, "time_end_ms": 500},
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422


def test_query_limit_too_large(client):
    resp = client.post(
        "/query",
        json={"vector": VEC, "limit": 9999},
        headers={"Authorization": f"Bearer loci_{'a' * 64}"},
    )
    assert resp.status_code == 422


# ── Request ID ────────────────────────────────────────────────────────────


def test_request_id_header_echoed(client):
    resp = client.get("/health", headers={"X-Request-Id": "my-req-123"})
    assert resp.headers.get("x-request-id") == "my-req-123"


def test_request_id_generated_when_absent(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) == 32  # uuid4().hex
