"""API key authentication against Supabase Postgres.

Keys are stored as SHA-256 hashes — raw keys never touch the database.
Raw key format: ``loci_<64 hex chars>`` (e.g. ``loci_abc123...``).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import asyncpg
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

DATABASE_URL: str = os.environ["DATABASE_URL"]

_pool: asyncpg.Pool | None = None

_bearer = HTTPBearer(auto_error=True)


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
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> dict[str, Any]:
    """FastAPI dependency — validates Bearer token against Supabase.

    Returns the api_keys row dict on success, raises 401 on failure.
    """
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
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")
        await conn.execute(
            "UPDATE api_keys SET last_used_at = now() WHERE id = $1",
            row["id"],
        )

    return dict(row)


async def require_admin_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> dict[str, Any]:
    """FastAPI dependency — validates Bearer token AND requires is_admin.

    Returns the api_keys row dict on success. Raises 401 for invalid keys and
    403 for non-admin keys.
    """
    row = await require_api_key(credentials)
    if not row.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin API key required")
    return row
