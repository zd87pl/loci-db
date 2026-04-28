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
    LOCI_MAX_METADATA_BYTES  Max metadata payload size in bytes (default: 4096)
    LOCI_MAX_BODY_BYTES   Max request body size in bytes (default: 5MB)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from pythonjsonlogger import jsonlogger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from auth import close_pool, get_pool, require_admin_api_key, require_api_key
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
MAX_METADATA_BYTES: int = int(os.environ.get("LOCI_MAX_METADATA_BYTES", "4096"))
MAX_BODY_BYTES: int = int(os.environ.get("LOCI_MAX_BODY_BYTES", str(5 * 1024 * 1024)))
DEFAULT_RPM: int = int(os.environ.get("LOCI_DEFAULT_RPM", "600"))
_DEFAULT_RATE_LIMIT: str = f"{DEFAULT_RPM}/minute"

_raw_origins = os.environ.get("LOCI_CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── Rate limiting ─────────────────────────────────────────────────────────
# Key function: use the authenticated tenant's namespace (set in request.state by auth).
# Falls back to remote IP for unauthenticated paths.


def _rate_key(request: Request) -> str:
    ns = getattr(request.state, "namespace", None)
    return ns if ns else get_remote_address(request)


limiter = Limiter(key_func=_rate_key)

# ── Per-namespace LociClient cache ────────────────────────────────────────

_clients: dict[str, LociClient] = {}


def _get_client(namespace: str) -> LociClient:
    if QDRANT_URL is None or QDRANT_API_KEY is None:
        raise RuntimeError("Server configuration not validated; call _validate_config() first")
    if namespace not in _clients:
        _clients[namespace] = LociClient(
            QDRANT_URL,
            api_key=QDRANT_API_KEY,
            vector_size=VECTOR_SIZE,
            epoch_size_ms=EPOCH_SIZE_MS,
            distance=DISTANCE,
            collection_prefix=f"{namespace}_",
        )
    return _clients[namespace]


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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: read strictly from env var, no wildcards.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


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
    x: float = Field(..., ge=-1e9, le=1e9, description="X spatial coordinate")
    y: float = Field(..., ge=-1e9, le=1e9, description="Y spatial coordinate")
    z: float = Field(..., ge=-1e9, le=1e9, description="Z spatial coordinate")
    timestamp_ms: int = Field(..., ge=0, description="Unix timestamp in milliseconds")
    vector: list[float] = Field(..., description=f"Embedding vector ({VECTOR_SIZE} dims)")
    scene_id: str = Field(..., min_length=1, max_length=256, description="Scene identifier")
    scale_level: str = Field("patch", max_length=64, description="Spatial scale level")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")

    @field_validator("vector")
    @classmethod
    def check_vector_dims(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_SIZE:
            raise ValueError(f"vector must have {VECTOR_SIZE} dimensions, got {len(v)}")
        return v

    @model_validator(mode="after")
    def check_metadata_size(self) -> "InsertRequest":
        import json
        meta = json.dumps({"scene_id": self.scene_id, "scale_level": self.scale_level})
        if len(meta.encode()) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata exceeds {MAX_METADATA_BYTES} bytes")
        return self


class QueryRequest(BaseModel):
    vector: list[float] = Field(..., description=f"Query vector ({VECTOR_SIZE} dims)")
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

    @field_validator("vector")
    @classmethod
    def check_vector_dims(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_SIZE:
            raise ValueError(f"vector must have {VECTOR_SIZE} dimensions, got {len(v)}")
        return v

    @model_validator(mode="after")
    def check_spatial_bounds(self) -> "QueryRequest":
        if self.x_min > self.x_max:
            raise ValueError("x_min must be <= x_max")
        if self.y_min > self.y_max:
            raise ValueError("y_min must be <= y_max")
        if self.z_min > self.z_max:
            raise ValueError("z_min must be <= z_max")
        if self.time_start_ms is not None and self.time_end_ms is not None:
            if self.time_start_ms > self.time_end_ms:
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


class QueryResponse(BaseModel):
    results: list[QueryResult]


# ── Auth dependency that also stamps namespace on request.state ───────────


async def _auth_with_state(
    request: Request,
    key_row: Annotated[dict[str, Any], Depends(require_api_key)],
) -> dict[str, Any]:
    request.state.namespace = key_row["namespace"]
    request.state.rate_limit_rpm = key_row.get("rate_limit_rpm") or 60
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

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient as _QC
        _qc = _QC(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        _qc.get_collections()
        qdrant_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("qdrant_not_ready", extra={"error": str(exc)})

    # Check Supabase (Postgres)
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        supabase_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("supabase_not_ready", extra={"error": str(exc)})

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
@limiter.limit(_DEFAULT_RATE_LIMIT)
async def insert(
    request: Request,
    req: InsertRequest,
    key_row: Annotated[dict[str, Any], Depends(_auth_with_state)],
):
    """Insert a world-state vector. Requires valid API key."""
    namespace = key_row["namespace"]
    state = WorldState(
        x=req.x,
        y=req.y,
        z=req.z,
        timestamp_ms=req.timestamp_ms,
        vector=req.vector,
        scene_id=req.scene_id,
        scale_level=req.scale_level,
        confidence=req.confidence,
    )
    state_id = _get_client(namespace).insert(state)
    return InsertResponse(id=state_id)


@app.post("/query", response_model=QueryResponse, tags=["data"])
@limiter.limit(_DEFAULT_RATE_LIMIT)
async def query(
    request: Request,
    req: QueryRequest,
    key_row: Annotated[dict[str, Any], Depends(_auth_with_state)],
):
    """Spatiotemporal vector search. Requires valid API key."""
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
            )
            for r in results
        ]
    )


# ── Admin: API key management ─────────────────────────────────────────────

_NAMESPACE_RE = re.compile(r"^[a-z0-9_]{3,64}$")


def _generate_raw_key() -> str:
    return "loci_" + secrets.token_hex(32)


def _hash_raw_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


class CreateKeyRequest(BaseModel):
    tenant_email: str = Field(..., min_length=3, max_length=320)
    tenant_name: str | None = Field(None, max_length=256)
    namespace: str = Field(..., description="Qdrant collection prefix (lowercase alnum + underscores)")
    label: str | None = Field(None, max_length=128)
    rate_limit_rpm: int | None = Field(None, ge=1, le=100_000)
    is_admin: bool = Field(False, description="Grant admin privileges to this key")

    @field_validator("namespace")
    @classmethod
    def check_namespace(cls, v: str) -> str:
        if not _NAMESPACE_RE.match(v):
            raise ValueError(
                "namespace must be 3-64 chars of lowercase letters, digits, or underscores"
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
    async with pool.acquire() as conn:
        async with conn.transaction():
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
