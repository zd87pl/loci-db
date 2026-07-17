"""Cloud LOCI API — Phase 1 hardened FastAPI server for Fly.io deployment.

Endpoints:
    GET  /health          — unauthenticated liveness probe
    GET  /ready           — readiness probe (checks Qdrant + Supabase)
    GET  /openapi.json    — OpenAPI schema (always available)
    GET  /docs            — Swagger UI (dev mode only, LOCI_DEV_MODE=true)
    POST /insert          — insert a world-state vector (auth + rate-limited)
    POST /query           — spatiotemporal vector search (auth + rate-limited)

Environment variables (set as Fly.io secrets — never committed to git):
    DATABASE_URL          Supabase Postgres connection string
    QDRANT_URL            Qdrant Cloud cluster URL
    QDRANT_API_KEY        Qdrant Cloud API token
    LOCI_VECTOR_SIZE      Embedding dimension (default: 512)
    LOCI_EPOCH_SIZE_MS    Temporal epoch length in ms (default: 5000)
    LOCI_DISTANCE         Qdrant distance metric (default: cosine)
    LOCI_CORS_ORIGINS     Comma-separated allowed origins (default: none)
    LOCI_DEV_MODE         Enable Swagger/ReDoc UI when "true" (default: false)
    LOCI_MAX_METADATA_BYTES  Max metadata payload size in bytes (default: 16384)
    LOCI_MAX_BODY_BYTES   Max request body size in bytes (default: 5MB)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

import asyncpg
from auth import (
    FixedWindowCounter,
    close_pool,
    get_pool,
    require_admin_api_key,
    require_api_key,
)
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from pythonjsonlogger import jsonlogger

from loci import LociClient, WorldState

# ── Structured JSON logging ────────────────────────────────────────────────

_handler = logging.StreamHandler()
_handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logging.root.addHandler(_handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("loci.api")

# ── Config ────────────────────────────────────────────────────────────────

QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]
VECTOR_SIZE: int = int(os.environ.get("LOCI_VECTOR_SIZE", "512"))
EPOCH_SIZE_MS: int = int(os.environ.get("LOCI_EPOCH_SIZE_MS", "5000"))
DISTANCE: str = os.environ.get("LOCI_DISTANCE", "cosine")
DEV_MODE: bool = os.environ.get("LOCI_DEV_MODE", "").lower() == "true"
MAX_METADATA_BYTES: int = int(os.environ.get("LOCI_MAX_METADATA_BYTES", str(16 * 1024)))
MAX_BODY_BYTES: int = int(os.environ.get("LOCI_MAX_BODY_BYTES", str(5 * 1024 * 1024)))
DEFAULT_RPM: int = int(os.environ.get("LOCI_DEFAULT_RPM", "600"))

# Hard ceiling on vector list length at the schema level. The real dimension
# check is VECTOR_SIZE, but that runs after the whole list is parsed; this cap
# (together with the body-size middleware) bounds worst-case parse work.
MAX_VECTOR_ITEMS: int = max(8192, VECTOR_SIZE)

_raw_origins = os.environ.get("LOCI_CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── Rate limiting ─────────────────────────────────────────────────────────
# Per-key throttle: each authenticated request counts against the tenant's
# namespace, enforced at the tenant's own ``rate_limit_rpm`` (read from the key
# row at auth time). This replaces the old slowapi decorator, which applied a
# single hardcoded limit to everyone and ignored per-key limits entirely.

_key_rate_counter = FixedWindowCounter()

# ── Per-namespace LociClient cache ────────────────────────────────────────

_clients: dict[str, LociClient] = {}
# /insert and /query are sync path operations, so FastAPI runs them in a
# threadpool — guard the lazy cache against a concurrent check-then-create that
# would build duplicate clients for the same namespace.
_clients_lock = threading.Lock()


def _get_client(namespace: str) -> LociClient:
    if QDRANT_URL is None or QDRANT_API_KEY is None:
        raise RuntimeError("Server configuration not validated; call _validate_config() first")
    client = _clients.get(namespace)
    if client is None:
        with _clients_lock:
            client = _clients.get(namespace)
            if client is None:
                client = LociClient(
                    QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    vector_size=VECTOR_SIZE,
                    epoch_size_ms=EPOCH_SIZE_MS,
                    distance=DISTANCE,
                    collection_prefix=f"{namespace}_",
                )
                _clients[namespace] = client
    return client


# ── App lifecycle ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    yield
    await close_pool()
    _clients.clear()


app = FastAPI(
    title="LOCI Cloud API",
    description="Authenticated 4D spatiotemporal vector database API",
    version="0.2.0",
    docs_url="/docs" if DEV_MODE else None,
    redoc_url="/redoc" if DEV_MODE else None,
    lifespan=lifespan,
)

# CORS: read strictly from env var, no wildcards.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Body size limit middleware ─────────────────────────────────────────────


class BodySizeLimitMiddleware:
    """Pure-ASGI middleware enforcing ``LOCI_MAX_BODY_BYTES``.

    Two layers of defence:
      1. Requests declaring ``Content-Length`` above the cap are rejected with
         413 before any body bytes are read.
      2. Bodies streamed without (or with a lying) ``Content-Length`` are
         counted as chunks arrive; the moment the running total exceeds the cap
         a 413 is sent and the application is handed an ``http.disconnect`` —
         the oversized payload is never fully buffered or JSON-parsed. (This
         cannot be signalled with an exception: FastAPI wraps body-read errors
         into a generic 400, so the receive wrapper responds directly and any
         response the aborted app still produces is swallowed.)
    """

    def __init__(self, app, max_body_bytes: int) -> None:  # noqa: ANN001
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = -1
                if declared > self.max_body_bytes:
                    await self._reject(send)
                    return
                break

        received = 0
        responded = False
        response_started = False

        async def counting_receive():  # noqa: ANN202
            nonlocal received, responded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    if not responded and not response_started:
                        responded = True
                        await self._reject(send)
                    # Starve the app of further body bytes; it will abort.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message) -> None:  # noqa: ANN001
            nonlocal response_started
            if responded:
                return  # 413 already sent; drop the aborted app's own response
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, counting_receive, guarded_send)

    async def _reject(self, send) -> None:  # noqa: ANN001
        body = json.dumps({"detail": "Request body too large"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# Starlette builds the middleware stack in reverse addition order, so this
# wraps CORS and everything inward — oversized bodies are rejected before any
# routing or parsing work. The request-ID middleware below is added later
# (further out), so 413 responses still get logged with a request ID.
app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=MAX_BODY_BYTES)


# ── Request ID middleware ──────────────────────────────────────────────────


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # noqa: ANN001
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    request.state.request_id = rid
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    response.headers["X-Request-Id"] = rid
    logger.info(
        "request",
        extra={
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
        },
    )
    return response


# ── Request / response models ──────────────────────────────────────────────


class InsertRequest(BaseModel):
    # Bounds mirror WorldState's normalised-coordinate contract ([0, 1]) so
    # invalid input fails validation as 422 instead of surfacing as a 500 when
    # the WorldState constructor rejects it.
    x: float = Field(..., ge=0.0, le=1.0, description="X spatial coordinate (normalised)")
    y: float = Field(..., ge=0.0, le=1.0, description="Y spatial coordinate (normalised)")
    z: float = Field(..., ge=0.0, le=1.0, description="Z spatial coordinate (normalised)")
    timestamp_ms: int = Field(..., ge=0, description="Unix timestamp in milliseconds")
    vector: list[float] = Field(
        ...,
        max_length=MAX_VECTOR_ITEMS,
        description=f"Embedding vector ({VECTOR_SIZE} dims)",
    )
    scene_id: str = Field(..., min_length=1, max_length=256, description="Scene identifier")
    scale_level: str = Field("patch", max_length=64, description="Spatial scale level")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            f"Arbitrary key/value payload stored with the vector and returned "
            f"verbatim by /query (max {MAX_METADATA_BYTES} bytes JSON-serialized)"
        ),
    )

    @field_validator("vector")
    @classmethod
    def check_vector_dims(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_SIZE:
            raise ValueError(f"vector must have {VECTOR_SIZE} dimensions, got {len(v)}")
        return v

    @field_validator("metadata")
    @classmethod
    def check_metadata_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            blob = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        if len(blob.encode("utf-8")) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata exceeds {MAX_METADATA_BYTES} bytes JSON-serialized")
        return v


class QueryRequest(BaseModel):
    vector: list[float] = Field(
        ...,
        max_length=MAX_VECTOR_ITEMS,
        description=f"Query vector ({VECTOR_SIZE} dims)",
    )
    x_min: float = Field(0.0, ge=-1e9, le=1e9)
    x_max: float = Field(1.0, ge=-1e9, le=1e9)
    y_min: float = Field(0.0, ge=-1e9, le=1e9)
    y_max: float = Field(1.0, ge=-1e9, le=1e9)
    z_min: float = Field(0.0, ge=-1e9, le=1e9)
    z_max: float = Field(1.0, ge=-1e9, le=1e9)
    time_start_ms: int | None = Field(None, ge=0)
    time_end_ms: int | None = Field(None, ge=0)
    limit: int = Field(10, ge=1, le=1000)
    overlap_factor: float = Field(1.0, ge=0.1, le=10.0)
    include_vectors: bool = Field(
        True, description="Return the stored embedding vector with each result"
    )

    @field_validator("vector")
    @classmethod
    def check_vector_dims(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_SIZE:
            raise ValueError(f"vector must have {VECTOR_SIZE} dimensions, got {len(v)}")
        return v

    @model_validator(mode="after")
    def check_spatial_bounds(self) -> QueryRequest:
        if self.x_min > self.x_max:
            raise ValueError("x_min must be <= x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be <= y_max")
        if self.z_min > self.z_max:
            raise ValueError("z_min must be <= z_max")
        if (
            self.time_start_ms is not None
            and self.time_end_ms is not None
            and self.time_start_ms > self.time_end_ms
        ):
            raise ValueError("time_start_ms must be <= time_end_ms")
        return self


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    qdrant: str
    supabase: str


class InsertResponse(BaseModel):
    id: str


class QueryResult(BaseModel):
    id: str
    x: float
    y: float
    z: float
    timestamp_ms: int
    scene_id: str
    scale_level: str = "patch"
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] = Field(
        default_factory=list,
        description="Embedding vector; empty unless include_vectors was set",
    )


class QueryResponse(BaseModel):
    results: list[QueryResult]


# ── Auth dependency that also stamps namespace on request.state ───────────


async def _auth_with_state(
    request: Request,
    key_row: Annotated[dict[str, Any], Depends(require_api_key)],
) -> dict[str, Any]:
    namespace = key_row["namespace"]
    rpm = key_row.get("rate_limit_rpm") or DEFAULT_RPM
    request.state.namespace = namespace
    request.state.rate_limit_rpm = rpm
    if not _key_rate_counter.hit(namespace, rpm):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )
    return key_row


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness probe — no auth required."""
    return HealthResponse(status="ok", version=app.version)


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
async def ready():
    """Readiness probe — checks Qdrant and Supabase connectivity."""
    qdrant_ok = False
    supabase_ok = False

    # Check Qdrant. The client construction + call are blocking, so run them in
    # a threadpool to avoid stalling the event loop. Log only the exception type
    # — the message can embed the connection URL/API key.
    def _ping_qdrant() -> None:
        from qdrant_client import QdrantClient as _QC

        _QC(url=QDRANT_URL, api_key=QDRANT_API_KEY).get_collections()

    try:
        await run_in_threadpool(_ping_qdrant)
        qdrant_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("qdrant_not_ready", extra={"error_type": type(exc).__name__})

    # Check Supabase (Postgres)
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        supabase_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("supabase_not_ready", extra={"error_type": type(exc).__name__})

    overall = "ok" if (qdrant_ok and supabase_ok) else "degraded"
    status_code = 200 if overall == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content=ReadyResponse(
            status=overall,
            qdrant="ok" if qdrant_ok else "error",
            supabase="ok" if supabase_ok else "error",
        ).model_dump(),
    )


@app.post("/insert", response_model=InsertResponse, tags=["data"])
def insert(
    req: InsertRequest,
    key_row: Annotated[dict[str, Any], Depends(_auth_with_state)],
):
    """Insert a world-state vector. Requires valid API key.

    Defined as a sync endpoint so FastAPI runs the blocking Qdrant call in a
    worker thread rather than on the event loop.
    """
    namespace = key_row["namespace"]
    try:
        state = WorldState(
            x=req.x,
            y=req.y,
            z=req.z,
            timestamp_ms=req.timestamp_ms,
            vector=req.vector,
            scene_id=req.scene_id,
            scale_level=req.scale_level,
            confidence=req.confidence,
            metadata=req.metadata,
        )
    except ValueError as exc:
        # WorldState enforces invariants Pydantic doesn't fully mirror (e.g.
        # scale_level ∈ {patch, frame, sequence}) — report as a validation
        # error, not a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state_id = _get_client(namespace).insert(state)
    return InsertResponse(id=state_id)


@app.post("/query", response_model=QueryResponse, tags=["data"])
def query(
    req: QueryRequest,
    key_row: Annotated[dict[str, Any], Depends(_auth_with_state)],
):
    """Spatiotemporal vector search. Requires valid API key.

    Defined as a sync endpoint so FastAPI runs the blocking Qdrant call in a
    worker thread rather than on the event loop.
    """
    namespace = key_row["namespace"]
    time_window = None
    if req.time_start_ms is not None and req.time_end_ms is not None:
        time_window = (req.time_start_ms, req.time_end_ms)

    results = _get_client(namespace).query(
        vector=req.vector,
        spatial_bounds={
            "x_min": req.x_min,
            "x_max": req.x_max,
            "y_min": req.y_min,
            "y_max": req.y_max,
            "z_min": req.z_min,
            "z_max": req.z_max,
        },
        time_window_ms=time_window,
        limit=req.limit,
        overlap_factor=req.overlap_factor,
    )

    return QueryResponse(
        results=[
            QueryResult(
                id=r.id,
                x=r.x,
                y=r.y,
                z=r.z,
                timestamp_ms=r.timestamp_ms,
                scene_id=r.scene_id,
                scale_level=r.scale_level,
                confidence=r.confidence,
                metadata=r.metadata,
                vector=list(r.vector) if req.include_vectors else [],
            )
            for r in results
        ]
    )


# ── Admin: API key management ─────────────────────────────────────────────

# Namespaces must NOT contain underscores. Qdrant collections are named
# ``{namespace}_loci_{epoch}`` and discovery matches on the ``{namespace}_loci_``
# prefix; if underscores were allowed, namespace "foo" would match collections
# belonging to namespace "foo_loci" (prefix "foo_loci_") — a cross-tenant read.
# Restricting to lowercase alphanumerics makes the separator unambiguous.
_NAMESPACE_RE = re.compile(r"^[a-z0-9]{3,64}$")


def _generate_raw_key() -> str:
    return "loci_" + secrets.token_hex(32)


def _hash_raw_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class CreateKeyRequest(BaseModel):
    tenant_email: str = Field(..., min_length=3, max_length=320)
    tenant_name: str | None = Field(None, max_length=256)
    namespace: str = Field(
        ...,
        description="Qdrant collection prefix (3-64 lowercase letters and digits)",
    )
    label: str | None = Field(None, max_length=128)
    rate_limit_rpm: int | None = Field(None, ge=1, le=100_000)
    is_admin: bool = Field(False, description="Grant admin privileges to this key")

    @field_validator("namespace")
    @classmethod
    def check_namespace(cls, v: str) -> str:
        if not _NAMESPACE_RE.match(v):
            raise ValueError(
                "namespace must be 3-64 chars of lowercase letters and digits (no underscores)"
            )
        return v


class CreateKeyResponse(BaseModel):
    key_id: str
    raw_key: str = Field(..., description="Shown once; store securely")
    prefix: str
    tenant_id: str
    namespace: str
    is_admin: bool


class KeyInfo(BaseModel):
    id: str
    tenant_id: str
    prefix: str
    namespace: str
    label: str | None
    rate_limit_rpm: int | None
    is_admin: bool
    revoked: bool
    last_used_at: str | None
    created_at: str


class ListKeysResponse(BaseModel):
    keys: list[KeyInfo]


class RevokeKeyResponse(BaseModel):
    key_id: str
    revoked: bool


@app.post(
    "/admin/keys",
    response_model=CreateKeyResponse,
    tags=["admin"],
    status_code=201,
)
async def admin_create_key(
    req: CreateKeyRequest,
    _admin: Annotated[dict[str, Any], Depends(require_admin_api_key)],
):
    """Create a new API key. Requires an admin key.

    Creates or reuses a tenant by email, then inserts a new api_keys row and
    returns the raw key value exactly once.
    """
    raw_key = _generate_raw_key()
    key_hash = _hash_raw_key(raw_key)
    prefix = raw_key[:12]

    pool = await get_pool()
    async with pool.acquire() as conn, conn.transaction():
        tenant_row = await conn.fetchrow(
            """
            INSERT INTO tenants (name, email, tier)
            VALUES ($1, $2, 'pro')
            ON CONFLICT (email) DO UPDATE
                SET name = COALESCE(EXCLUDED.name, tenants.name)
            RETURNING id
            """,
            req.tenant_name or req.tenant_email,
            req.tenant_email,
        )
        tenant_id = tenant_row["id"]

        try:
            key_row = await conn.fetchrow(
                """
                INSERT INTO api_keys
                    (tenant_id, key_hash, prefix, namespace, label,
                     rate_limit_rpm, is_admin)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                tenant_id,
                key_hash,
                prefix,
                req.namespace,
                req.label,
                req.rate_limit_rpm,
                req.is_admin,
            )
        except asyncpg.UniqueViolationError as exc:
            raise HTTPException(
                status_code=409,
                detail="namespace already in use",
            ) from exc

    return CreateKeyResponse(
        key_id=str(key_row["id"]),
        raw_key=raw_key,
        prefix=prefix,
        tenant_id=str(tenant_id),
        namespace=req.namespace,
        is_admin=req.is_admin,
    )


@app.get(
    "/admin/keys",
    response_model=ListKeysResponse,
    tags=["admin"],
)
async def admin_list_keys(
    _admin: Annotated[dict[str, Any], Depends(require_admin_api_key)],
    tenant_id: str | None = None,
    include_revoked: bool = False,
):
    """List API keys, optionally filtered by tenant_id. Requires admin key."""
    try:
        tenant_uuid = uuid.UUID(tenant_id) if tenant_id else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="tenant_id must be a UUID") from exc

    query = [
        "SELECT id, tenant_id, prefix, namespace, label, rate_limit_rpm,",
        "       is_admin, revoked, last_used_at, created_at",
        "FROM api_keys",
    ]
    where: list[str] = []
    params: list[Any] = []
    if tenant_uuid is not None:
        params.append(tenant_uuid)
        where.append(f"tenant_id = ${len(params)}")
    if not include_revoked:
        where.append("revoked = false")
    if where:
        query.append("WHERE " + " AND ".join(where))
    query.append("ORDER BY created_at DESC LIMIT 500")

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("\n".join(query), *params)

    return ListKeysResponse(
        keys=[
            KeyInfo(
                id=str(r["id"]),
                tenant_id=str(r["tenant_id"]),
                prefix=r["prefix"],
                namespace=r["namespace"],
                label=r["label"],
                rate_limit_rpm=r["rate_limit_rpm"],
                is_admin=r["is_admin"],
                revoked=r["revoked"],
                last_used_at=r["last_used_at"].isoformat() if r["last_used_at"] else None,
                created_at=r["created_at"].isoformat(),
            )
            for r in rows
        ]
    )


@app.delete(
    "/admin/keys/{key_id}",
    response_model=RevokeKeyResponse,
    tags=["admin"],
)
async def admin_revoke_key(
    key_id: str,
    _admin: Annotated[dict[str, Any], Depends(require_admin_api_key)],
):
    """Revoke an API key. Idempotent — returns 404 only for unknown IDs."""
    try:
        key_uuid = uuid.UUID(key_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="key_id must be a UUID") from exc

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE api_keys SET revoked = true WHERE id = $1 RETURNING id",
            key_uuid,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="key not found")

    return RevokeKeyResponse(key_id=str(row["id"]), revoked=True)
