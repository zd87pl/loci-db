"""Integration tests for the Phase 3 admin API key endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ───────────────────────────────────────────────────────────────


def _fake_pool(conn: MagicMock) -> MagicMock:
    """Build an asyncpg-style pool whose .acquire() yields *conn*."""
    pool = MagicMock()

    class _AcquireCM:
        async def __aenter__(self_inner):
            return conn

        async def __aexit__(self_inner, *exc):
            return False

    pool.acquire = MagicMock(return_value=_AcquireCM())
    return pool


def _fake_transaction() -> Any:
    tx = MagicMock()

    class _TxCM:
        async def __aenter__(self_inner):
            return tx

        async def __aexit__(self_inner, *exc):
            return False

    return _TxCM()


@pytest.fixture()
def admin_client(app):
    """TestClient with the admin-auth dependency forced to an admin key."""
    import auth
    import server as srv

    async def _fake_admin():
        return {
            "id": "00000000-0000-0000-0000-000000000099",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "namespace": "admin_ns",
            "label": "admin",
            "rate_limit_rpm": 6000,
            "is_admin": True,
            "email": "admin@example.com",
        }

    saved = dict(srv.app.dependency_overrides)
    srv.app.dependency_overrides[auth.require_admin_api_key] = _fake_admin
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()
    srv.app.dependency_overrides.update(saved)


@pytest.fixture()
def non_admin_client(app):
    """TestClient where the admin dependency raises 403 like the real one does."""
    import auth
    import server as srv
    from fastapi import HTTPException

    async def _fake_admin_rejected():
        raise HTTPException(status_code=403, detail="Admin API key required")

    saved = dict(srv.app.dependency_overrides)
    srv.app.dependency_overrides[auth.require_admin_api_key] = _fake_admin_rejected
    yield TestClient(srv.app)
    srv.app.dependency_overrides.clear()
    srv.app.dependency_overrides.update(saved)


# ── /admin/keys POST ───────────────────────────────────────────────────────


def test_create_key_requires_admin(non_admin_client):
    resp = non_admin_client.post(
        "/admin/keys",
        json={"tenant_email": "x@y", "namespace": "ns_create_a"},
    )
    assert resp.status_code == 403


def test_create_key_happy_path(admin_client, monkeypatch):
    import server as srv

    tenant_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    key_id = uuid.UUID("22222222-2222-2222-2222-222222222222")

    conn = MagicMock()
    conn.transaction = MagicMock(return_value=_fake_transaction())
    conn.fetchrow = AsyncMock(side_effect=[{"id": tenant_id}, {"id": key_id}])
    pool = _fake_pool(conn)

    async def _fake_get_pool():
        return pool

    monkeypatch.setattr(srv, "get_pool", _fake_get_pool)

    resp = admin_client.post(
        "/admin/keys",
        json={
            "tenant_email": "partner@example.com",
            "tenant_name": "Partner",
            "namespace": "partner_prod",
            "label": "prod",
            "rate_limit_rpm": 1200,
            "is_admin": False,
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["key_id"] == str(key_id)
    assert body["tenant_id"] == str(tenant_id)
    assert body["namespace"] == "partner_prod"
    assert body["is_admin"] is False
    assert body["raw_key"].startswith("loci_")
    assert len(body["raw_key"]) == len("loci_") + 64


def test_create_key_rejects_bad_namespace(admin_client):
    resp = admin_client.post(
        "/admin/keys",
        json={"tenant_email": "x@y.z", "namespace": "Bad Namespace!"},
    )
    assert resp.status_code == 422


def test_create_key_conflict_on_duplicate_namespace(admin_client, monkeypatch):
    import server as srv

    conn = MagicMock()
    conn.transaction = MagicMock(return_value=_fake_transaction())
    conn.fetchrow = AsyncMock(
        side_effect=[
            {"id": uuid.UUID("11111111-1111-1111-1111-111111111111")},
            asyncpg.UniqueViolationError("duplicate namespace"),
        ]
    )
    pool = _fake_pool(conn)

    async def _fake_get_pool():
        return pool

    monkeypatch.setattr(srv, "get_pool", _fake_get_pool)

    resp = admin_client.post(
        "/admin/keys",
        json={"tenant_email": "x@y.z", "namespace": "taken_ns"},
    )
    assert resp.status_code == 409


# ── /admin/keys GET ────────────────────────────────────────────────────────


def test_list_keys_requires_admin(non_admin_client):
    resp = non_admin_client.get("/admin/keys")
    assert resp.status_code == 403


def test_list_keys_happy_path(admin_client, monkeypatch):
    import server as srv

    now = datetime.now(timezone.utc)
    row = {
        "id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
        "tenant_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "prefix": "loci_abcdef1",
        "namespace": "partner_prod",
        "label": "prod",
        "rate_limit_rpm": 1200,
        "is_admin": False,
        "revoked": False,
        "last_used_at": None,
        "created_at": now,
    }
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[row])
    pool = _fake_pool(conn)

    async def _fake_get_pool():
        return pool

    monkeypatch.setattr(srv, "get_pool", _fake_get_pool)

    resp = admin_client.get("/admin/keys")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["keys"]) == 1
    assert body["keys"][0]["namespace"] == "partner_prod"
    assert body["keys"][0]["id"] == str(row["id"])


def test_list_keys_rejects_bad_tenant_id(admin_client):
    resp = admin_client.get("/admin/keys?tenant_id=not-a-uuid")
    assert resp.status_code == 422


# ── /admin/keys/{id} DELETE ────────────────────────────────────────────────


def test_revoke_key_requires_admin(non_admin_client):
    resp = non_admin_client.delete(
        "/admin/keys/22222222-2222-2222-2222-222222222222"
    )
    assert resp.status_code == 403


def test_revoke_key_happy_path(admin_client, monkeypatch):
    import server as srv

    key_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"id": key_id})
    pool = _fake_pool(conn)

    async def _fake_get_pool():
        return pool

    monkeypatch.setattr(srv, "get_pool", _fake_get_pool)

    resp = admin_client.delete(f"/admin/keys/{key_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["key_id"] == str(key_id)
    assert body["revoked"] is True


def test_revoke_key_not_found(admin_client, monkeypatch):
    import server as srv

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=None)
    pool = _fake_pool(conn)

    async def _fake_get_pool():
        return pool

    monkeypatch.setattr(srv, "get_pool", _fake_get_pool)

    resp = admin_client.delete("/admin/keys/22222222-2222-2222-2222-222222222222")
    assert resp.status_code == 404


def test_revoke_key_rejects_bad_id(admin_client):
    resp = admin_client.delete("/admin/keys/not-a-uuid")
    assert resp.status_code == 422
