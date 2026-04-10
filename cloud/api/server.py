"""Cloud LOCI API — authenticated FastAPI server for Fly.io deployment.

Endpoints:
    GET  /health          — unauthenticated liveness probe
    POST /insert          — insert a world-state vector (auth required)
    POST /query           — spatiotemporal vector search (auth required)

Environment variables (set as Fly.io secrets — never committed to git):
    DATABASE_URL          Supabase Postgres connection string
    QDRANT_URL            Qdrant Cloud cluster URL
    QDRANT_API_KEY        Qdrant Cloud API token
    LOCI_VECTOR_SIZE      Embedding dimension (default: 512)
    LOCI_EPOCH_SIZE_MS    Temporal epoch length in ms (default: 5000)
    LOCI_DISTANCE         Qdrant distance metric (default: cosine)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth import close_pool, require_api_key
from loci import LociClient, WorldState

# ── Config ────────────────────────────────────────────────────────────────

QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]
VECTOR_SIZE: int = int(os.environ.get("LOCI_VECTOR_SIZE", "512"))
EPOCH_SIZE_MS: int = int(os.environ.get("LOCI_EPOCH_SIZE_MS", "5000"))
DISTANCE: str = os.environ.get("LOCI_DISTANCE", "cosine")

# ── App lifecycle ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    yield
    await close_pool()


app = FastAPI(
    title="LOCI Cloud API",
    description="Authenticated 4D spatiotemporal vector database — private PoC",
    version="0.1.0",
    docs_url=None,   # disable public Swagger UI
    redoc_url=None,  # disable public ReDoc
    lifespan=lifespan,
)

# CORS: no wildcard — only explicitly listed origins may call this API.
# For the PoC there are no browser clients, so we allow nothing by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── LOCI client ───────────────────────────────────────────────────────────

_client: LociClient | None = None


def _get_client() -> LociClient:
    global _client
    if _client is None:
        _client = LociClient(
            QDRANT_URL,
            api_key=QDRANT_API_KEY,
            vector_size=VECTOR_SIZE,
            epoch_size_ms=EPOCH_SIZE_MS,
            distance=DISTANCE,
        )
    return _client


# ── Request / response models ──────────────────────────────────────────────


class InsertRequest(BaseModel):
    x: float
    y: float
    z: float
    timestamp_ms: int
    vector: list[float]
    scene_id: str
    scale_level: str = "patch"
    metadata: dict[str, Any] = {}


class QueryRequest(BaseModel):
    vector: list[float]
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0
    time_start_ms: int | None = None
    time_end_ms: int | None = None
    limit: int = 10
    overlap_factor: float = 1.0


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Liveness probe — no auth required."""
    return {"status": "ok", "version": app.version}


@app.post("/insert")
async def insert(
    req: InsertRequest,
    _key: dict = Depends(require_api_key),
):
    """Insert a world-state vector. Requires valid API key."""
    if len(req.vector) != VECTOR_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Vector must have {VECTOR_SIZE} dimensions, got {len(req.vector)}",
        )
    state = WorldState(
        x=req.x,
        y=req.y,
        z=req.z,
        timestamp_ms=req.timestamp_ms,
        vector=req.vector,
        scene_id=req.scene_id,
        scale_level=req.scale_level,
        metadata=req.metadata,
    )
    state_id = _get_client().insert(state)
    return {"id": state_id}


@app.post("/query")
async def query(
    req: QueryRequest,
    _key: dict = Depends(require_api_key),
):
    """Spatiotemporal vector search. Requires valid API key."""
    time_window = None
    if req.time_start_ms is not None and req.time_end_ms is not None:
        time_window = (req.time_start_ms, req.time_end_ms)

    results = _get_client().query(
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

    return {
        "results": [
            {
                "id": r.id,
                "x": r.x,
                "y": r.y,
                "z": r.z,
                "timestamp_ms": r.timestamp_ms,
                "scene_id": r.scene_id,
                "score": r.score,
            }
            for r in results
        ]
    }
