"""Unit tests for cloud/api/auth.py — exercised for real, not dependency-overridden.

Covers require_api_key (valid key, unknown key, brute-force throttle), the
_client_ip trust-chain preference order, and FixedWindowCounter window
rollover semantics.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

RAW_KEY = "loci_" + "a" * 64
CLIENT_HOST = "203.0.113.7"

_KEY_ROW: dict[str, Any] = {
    "id": "00000000-0000-0000-0000-000000000042",
    "tenant_id": "00000000-0000-0000-0000-000000000001",
    "namespace": "authns",
    "label": "auth test",
    "rate_limit_rpm": 600,
    "is_admin": False,
    "email": "auth@example.com",
}


# ── Helpers ────────────────────────────────────────────────────────────────


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


def _make_request(
    headers: dict[str, str] | None = None,
    client_host: str | None = CLIENT_HOST,
) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/insert",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_host, 4242) if client_host else None,
    }
    return Request(scope)


def _creds(raw_key: str = RAW_KEY) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=raw_key)


@pytest.fixture()
def fresh_counter(monkeypatch):
    """Isolate the module-level fail counter and freeze its window."""
    import auth

    monkeypatch.setattr(auth.FixedWindowCounter, "_window", staticmethod(lambda: 424242))
    counter = auth.FixedWindowCounter()
    monkeypatch.setattr(auth, "_auth_fail_counter", counter)
    return counter


def _patch_pool(monkeypatch, conn: MagicMock) -> None:
    import auth

    pool = _fake_pool(conn)

    async def _fake_get_pool():
        return pool

    monkeypatch.setattr(auth, "get_pool", _fake_get_pool)


# ── _client_ip preference order ────────────────────────────────────────────


def test_client_ip_prefers_fly_client_ip_header():
    import auth

    req = _make_request(headers={"Fly-Client-IP": "198.51.100.9", "X-Forwarded-For": "6.6.6.6"})
    assert auth._client_ip(req) == "198.51.100.9"


def test_client_ip_falls_back_to_request_client_host():
    import auth

    assert auth._client_ip(_make_request()) == CLIENT_HOST


def test_client_ip_unknown_when_no_client():
    import auth

    assert auth._client_ip(_make_request(client_host=None)) == "unknown"


# ── require_api_key ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_require_api_key_valid_returns_row_and_touches_last_used(monkeypatch, fresh_counter):
    import auth

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=_KEY_ROW)
    conn.execute = AsyncMock()
    _patch_pool(monkeypatch, conn)

    row = await auth.require_api_key(_make_request(), _creds())

    assert row == _KEY_ROW
    # Lookup used the SHA-256 hash of the raw key, never the raw key itself.
    assert conn.fetchrow.await_args.args[1] == hashlib.sha256(RAW_KEY.encode()).hexdigest()
    # last_used_at update issued for the matched key id.
    conn.execute.assert_awaited_once()
    assert "last_used_at" in conn.execute.await_args.args[0]
    assert conn.execute.await_args.args[1] == _KEY_ROW["id"]
    # A successful auth must not count against the failure throttle.
    assert fresh_counter._windows == {}


@pytest.mark.asyncio
async def test_require_api_key_unknown_key_401_and_counts_failure(monkeypatch, fresh_counter):
    import auth

    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    _patch_pool(monkeypatch, conn)

    with pytest.raises(HTTPException) as excinfo:
        await auth.require_api_key(_make_request(), _creds("loci_" + "b" * 64))

    assert excinfo.value.status_code == 401
    assert fresh_counter._windows[CLIENT_HOST][1] == 1
    conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_require_api_key_throttles_at_limit_with_retry_after(monkeypatch, fresh_counter):
    import auth

    monkeypatch.setattr(auth, "AUTH_FAIL_RPM", 3)
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    _patch_pool(monkeypatch, conn)

    for _ in range(3):
        with pytest.raises(HTTPException) as excinfo:
            await auth.require_api_key(_make_request(), _creds("loci_" + "c" * 64))
        assert excinfo.value.status_code == 401

    with pytest.raises(HTTPException) as excinfo:
        await auth.require_api_key(_make_request(), _creds("loci_" + "c" * 64))

    assert excinfo.value.status_code == 429
    assert excinfo.value.headers["Retry-After"] == "60"
    # The throttled request never reached the database.
    assert conn.fetchrow.await_count == 3


@pytest.mark.asyncio
async def test_require_api_key_throttle_is_per_ip(monkeypatch, fresh_counter):
    import auth

    monkeypatch.setattr(auth, "AUTH_FAIL_RPM", 1)
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock()
    _patch_pool(monkeypatch, conn)

    with pytest.raises(HTTPException):
        await auth.require_api_key(
            _make_request(headers={"Fly-Client-IP": "198.51.100.1"}),
            _creds("loci_" + "d" * 64),
        )
    with pytest.raises(HTTPException) as excinfo:
        await auth.require_api_key(
            _make_request(headers={"Fly-Client-IP": "198.51.100.1"}),
            _creds("loci_" + "d" * 64),
        )
    assert excinfo.value.status_code == 429

    # A different client IP still gets a real (401) answer.
    with pytest.raises(HTTPException) as excinfo:
        await auth.require_api_key(
            _make_request(headers={"Fly-Client-IP": "198.51.100.2"}),
            _creds("loci_" + "d" * 64),
        )
    assert excinfo.value.status_code == 401


# ── FixedWindowCounter window rollover ────────────────────────────────────


def test_fixed_window_counter_hit_resets_on_rollover(monkeypatch):
    from auth import FixedWindowCounter

    now = [100]
    monkeypatch.setattr(FixedWindowCounter, "_window", staticmethod(lambda: now[0]))
    c = FixedWindowCounter()

    assert c.hit("k", 1) is True
    assert c.hit("k", 1) is False  # over within the same window

    now[0] = 101  # minute rolls over
    assert c.hit("k", 1) is True  # counter starts fresh


def test_fixed_window_counter_over_ignores_stale_window(monkeypatch):
    from auth import FixedWindowCounter

    now = [200]
    monkeypatch.setattr(FixedWindowCounter, "_window", staticmethod(lambda: now[0]))
    c = FixedWindowCounter()

    c.hit("k", 1)
    c.hit("k", 1)
    assert c.over("k", 1) is True

    now[0] = 201
    assert c.over("k", 1) is False  # stale entry from the old window is ignored
