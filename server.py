"""Minimal LOCI REST API server for researcher use.

Wraps LociClient with a FastAPI HTTP layer so researchers can insert and
query world states without writing Python.  Connect to the running server
at http://localhost:8000.

Environment variables:
    QDRANT_URL          Qdrant base URL (default: http://qdrant:6333)
    LOCI_VECTOR_SIZE    Embedding dimension (default: 512)
    LOCI_EPOCH_SIZE_MS  Temporal epoch length in ms (default: 5000)
    LOCI_DISTANCE       Qdrant distance metric (default: cosine)
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from loci import LociClient, WorldState

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
VECTOR_SIZE = int(os.environ.get("LOCI_VECTOR_SIZE", "512"))
EPOCH_SIZE_MS = int(os.environ.get("LOCI_EPOCH_SIZE_MS", "5000"))
DISTANCE = os.environ.get("LOCI_DISTANCE", "cosine")

app = FastAPI(title="LOCI API", description="4D spatiotemporal vector database")

_client: LociClient | None = None


def get_client() -> LociClient:
    global _client
    if _client is None:
        _client = LociClient(
            QDRANT_URL,
            vector_size=VECTOR_SIZE,
            epoch_size_ms=EPOCH_SIZE_MS,
            distance=DISTANCE,
        )
    return _client


# ── Models ────────────────────────────────────────────────────────────────


class InsertRequest(BaseModel):
    x: float
    y: float
    z: float
    timestamp_ms: int
    vector: list[float]
    scene_id: str
    scale_level: str = "patch"
    metadata: dict[str, Any] = Field(default_factory=dict)


_MAX_LIMIT = 1_000


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
    limit: int = Field(default=10, ge=1, le=_MAX_LIMIT)
    overlap_factor: float = Field(default=1.0, ge=0.0, le=10.0)


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "qdrant_url": QDRANT_URL, "vector_size": VECTOR_SIZE}


@app.post("/insert")
def insert(req: InsertRequest):
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
    )
    state_id = get_client().insert(state)
    return {"id": state_id}


@app.post("/query")
def query(req: QueryRequest):
    if len(req.vector) != VECTOR_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Vector must have {VECTOR_SIZE} dimensions, got {len(req.vector)}",
        )
    time_window = None
    if req.time_start_ms is not None and req.time_end_ms is not None:
        time_window = (req.time_start_ms, req.time_end_ms)

    results = get_client().query(
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
