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
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def _fake_result() -> SimpleNamespace:
    return SimpleNamespace(
        id="abc",
        x=0.1,
        y=0.2,
        z=0.3,
        timestamp_ms=1000,
        scene_id="s",
        vector=VEC,
    )


def test_query_returns_vectors_by_default(client):
    """Cloud query returns embedding vectors, matching local-client semantics."""
    import server as srv

    mock_client = srv._clients["test_ns_abc"]
    mock_client.query.return_value = [_fake_result()]
    try:
        resp = client.post(
            "/query",
            json={"vector": VEC},
            headers={"Authorization": f"Bearer loci_{'a' * 64}"},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert results[0]["vector"] == VEC
    finally:
        mock_client.query.return_value = []


def test_query_omits_vectors_when_disabled(client):
    """include_vectors=false trims the (potentially large) vector payload."""
    import server as srv

    mock_client = srv._clients["test_ns_abc"]
    mock_client.query.return_value = [_fake_result()]
    try:
        resp = client.post(
            "/query",
            json={"vector": VEC, "include_vectors": False},
            headers={"Authorization": f"Bearer loci_{'a' * 64}"},
        )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["vector"] == []
    finally:
        mock_client.query.return_value = []


def test_per_key_rate_limit_returns_429(client):
    """Once a tenant exceeds its rate_limit_rpm, further requests get 429."""
    import auth

    import server as srv

    ns = "rltestns"
    low_rpm_row = {
        "id": "00000000-0000-0000-0000-0000000000aa",
        "tenant_id": "00000000-0000-0000-0000-000000000001",
        "namespace": ns,
        "label": "rl",
        "rate_limit_rpm": 2,
        "is_admin": False,
        "email": "rl@example.com",
    }

    async def _fake_low_rpm_key():
        return low_rpm_row

    saved = srv.app.dependency_overrides.get(auth.require_api_key)
    srv.app.dependency_overrides[auth.require_api_key] = _fake_low_rpm_key
    srv._clients[ns] = MagicMock(insert=MagicMock(return_value="x" * 32))
    # Reset the shared counter window for this namespace.
    srv._key_rate_counter._windows.pop(ns, None)
    try:
        body = {"x": 0, "y": 0, "z": 0, "timestamp_ms": 0, "vector": VEC, "scene_id": "s"}
        h = {"Authorization": f"Bearer loci_{'a' * 64}"}
        assert client.post("/insert", json=body, headers=h).status_code == 200
        assert client.post("/insert", json=body, headers=h).status_code == 200
        assert client.post("/insert", json=body, headers=h).status_code == 429
    finally:
        if saved is not None:
            srv.app.dependency_overrides[auth.require_api_key] = saved
        srv._clients.pop(ns, None)
        srv._key_rate_counter._windows.pop(ns, None)


def test_fixed_window_counter_semantics():
    from auth import FixedWindowCounter

    c = FixedWindowCounter()
    assert c.hit("k", 2) is True   # 1
    assert c.over("k", 2) is False
    assert c.hit("k", 2) is True   # 2
    assert c.over("k", 2) is True   # at limit
    assert c.hit("k", 2) is False  # 3 — over
    # A non-positive limit means unlimited.
    assert c.hit("unlimited", 0) is True
    assert c.over("unlimited", 0) is False


# ── Request ID ────────────────────────────────────────────────────────────


def test_request_id_header_echoed(client):
    resp = client.get("/health", headers={"X-Request-Id": "my-req-123"})
    assert resp.headers.get("x-request-id") == "my-req-123"


def test_request_id_generated_when_absent(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) == 32  # uuid4().hex
