"""Shared fixtures for cloud API integration tests.

Tests run against a local FastAPI test client with mocked Qdrant and Supabase.
Set LOCI_INTEGRATION=true to run against live staging services (requires env vars).
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ── Path setup — must happen before any local imports ─────────────────────
# cloud/api is not a package; add it to sys.path so `auth` and `server` resolve.
_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api"))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

# ── Env vars — must be set before server module is imported ───────────────
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-qdrant-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("LOCI_VECTOR_SIZE", "4")   # tiny vectors for speed
os.environ.setdefault("LOCI_DEV_MODE", "true")

VECTOR_SIZE = int(os.environ["LOCI_VECTOR_SIZE"])
TEST_NAMESPACE = "test_ns_abc"

_MOCK_KEY_ROW: dict[str, Any] = {
    "id": "00000000-0000-0000-0000-000000000002",
    "tenant_id": "00000000-0000-0000-0000-000000000001",
    "namespace": TEST_NAMESPACE,
    "label": "test key",
    "rate_limit_rpm": 600,
    "email": "test@example.com",
}


@pytest.fixture(scope="session")
def app():
    """Return the FastAPI app with auth and LociClient dependencies mocked out."""
    import auth
    import server as srv

    # ── Override auth dependency — bypasses real DB lookup ────────────────
    # FastAPI resolves dependencies by identity; override the exact function
    # that _auth_with_state depends on.
    async def _fake_require_api_key():
        return _MOCK_KEY_ROW

    srv.app.dependency_overrides[auth.require_api_key] = _fake_require_api_key

    # ── Pre-populate the per-namespace client cache with a mock ───────────
    mock_client = MagicMock()
    mock_client.insert.return_value = "deadbeef" * 4
    mock_client.query.return_value = []
    srv._clients[TEST_NAMESPACE] = mock_client

    yield srv.app

    # Teardown
    srv.app.dependency_overrides.clear()
    srv._clients.clear()


@pytest.fixture(scope="session")
def client(app):
    with TestClient(app) as c:
        yield c
