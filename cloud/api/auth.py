"""API key authentication against Supabase Postgres.

Keys are stored as SHA-256 hashes — raw keys never touch the database.
Raw key format: ``loci_<64 hex chars>`` (e.g. ``loci_abc123...``).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Annotated, Any

import asyncpg
from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

DATABASE_URL: str = os.environ["DATABASE_URL"]

# Max failed auth attempts per client IP per minute before we stop hitting the
# database and return 429. Blunts API-key brute-forcing.
AUTH_FAIL_RPM: int = int(os.environ.get("LOCI_AUTH_FAIL_RPM", "60"))

_pool: asyncpg.Pool | None = None

_bearer = HTTPBearer(auto_error=True)


class FixedWindowCounter:
    """Thread-safe in-process fixed-window rate counter.

    Counts events per key within a one-minute wall-clock window. This is a
    per-process counter: in a multi-instance deployment each instance keeps its
    own window, so the effective global limit is ``limit * num_instances``.
    Sufficient for the single-region private beta; swap for a shared store
    (e.g. Redis) before scaling horizontally.
    """

    _MAX_KEYS = 50_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> [window_index, count]
        self._windows: dict[str, list[int]] = {}

    @staticmethod
    def _window() -> int:
        return int(time.time()) // 60

    def hit(self, key: str, limit: int) -> bool:
        """Increment ``key``'s counter; return True if still within ``limit``.

        A non-positive ``limit`` means "unlimited" and always returns True.
        """
        if limit <= 0:
            return True
        window = self._window()
        with self._lock:
            if len(self._windows) > self._MAX_KEYS:
                # Drop entries from elapsed windows to bound memory. If a flood of
                # distinct keys within the *current* window still leaves us over
                # the cap, reset entirely — a safety valve to bound memory/CPU.
                self._windows = {k: v for k, v in self._windows.items() if v[0] == window}
                if len(self._windows) > self._MAX_KEYS:
                    self._windows.clear()
            entry = self._windows.get(key)
            if entry is None or entry[0] != window:
                entry = [window, 0]
            entry[1] += 1
            self._windows[key] = entry
            return entry[1] <= limit

    def over(self, key: str, limit: int) -> bool:
        """Return True if ``key`` is already at/over ``limit`` (no increment)."""
        if limit <= 0:
            return False
        window = self._window()
        with self._lock:
            entry = self._windows.get(key)
            if entry is None or entry[0] != window:
                return False
            return entry[1] >= limit


# Per-IP counter for *failed* auth attempts (brute-force throttle).
_auth_fail_counter = FixedWindowCounter()


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer)],
) -> dict[str, Any]:
    """FastAPI dependency — validates Bearer token against Supabase.

    Returns the api_keys row dict on success, raises 401 on failure. Repeated
    failures from the same client IP are throttled with 429 to blunt brute force.
    """
    ip = _client_ip(request)
    if _auth_fail_counter.over(ip, AUTH_FAIL_RPM):
        raise HTTPException(
            status_code=429,
            detail="Too many failed authentication attempts",
            headers={"Retry-After": "60"},
        )

    raw_key = credentials.credentials
    key_hash = _hash_key(raw_key)

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ak.id, ak.tenant_id, ak.namespace, ak.label,
                   ak.rate_limit_rpm, ak.is_admin, t.email
            FROM api_keys ak
            JOIN tenants t ON t.id = ak.tenant_id
            WHERE ak.key_hash = $1 AND ak.revoked = false
            """,
            key_hash,
        )
        if row is None:
            _auth_fail_counter.hit(ip, AUTH_FAIL_RPM)
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        await conn.execute(
            "UPDATE api_keys SET last_used_at = now() WHERE id = $1",
            row["id"],
        )

    return dict(row)


async def require_admin_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer)],
) -> dict[str, Any]:
    """FastAPI dependency — validates Bearer token AND requires is_admin.

    Returns the api_keys row dict on success. Raises 401 for invalid keys and
    403 for non-admin keys.
    """
    row = await require_api_key(request, credentials)
    if not row.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin API key required")
    return row
