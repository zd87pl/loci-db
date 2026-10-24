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
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from loci import LociClient, WorldState

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
VECTOR_SIZE = int(os.environ.get("LOCI_VECTOR_SIZE", "512"))
EPOCH_SIZE_MS = int(os.environ.get("LOCI_EPOCH_SIZE_MS", "5000"))
DISTANCE = os.environ.get("LOCI_DISTANCE", "cosine")

app = FastAPI(title="LOCI API", description="4D spatiotemporal vector database")

_client: LociClient | None = None
# /insert and /query are sync path operations, so FastAPI runs them in a
# threadpool — guard the lazy singleton against concurrent first requests
# building duplicate clients.
_client_lock = threading.Lock()


def get_client() -> LociClient:
    global _client
    if _client is None:
        with _client_lock:
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
    # Bounds mirror WorldState's normalised-coordinate contract ([0, 1]) so
    # invalid input fails validation as 422 instead of surfacing as a 500 when
    # the WorldState constructor rejects it. Mirrors the cloud API validators.
    x: float = Field(
        ..., ge=0.0, le=1.0, allow_inf_nan=False, description="X spatial coordinate (normalised)"
    )
    y: float = Field(
        ..., ge=0.0, le=1.0, allow_inf_nan=False, description="Y spatial coordinate (normalised)"
    )
    z: float = Field(
        ..., ge=0.0, le=1.0, allow_inf_nan=False, description="Z spatial coordinate (normalised)"
    )
    timestamp_ms: int = Field(..., ge=0, description="Unix timestamp in milliseconds")
    vector: list[float] = Field(..., description=f"Embedding vector ({VECTOR_SIZE} dims)")
    scene_id: str
    scale_level: Literal["patch", "frame", "sequence"] = "patch"
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, allow_inf_nan=False, description="Detection confidence"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def check_vector_dims(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_SIZE:
            raise ValueError(f"vector must have {VECTOR_SIZE} dimensions, got {len(v)}")
        return v


_MAX_LIMIT = 1_000

# Upper bound used to close a start-only time window. Well past any plausible
# millisecond timestamp while staying inside Qdrant's signed-64-bit range.
_MAX_TIME_MS = 2**62

_BOUND_FIELD = {
    "default": None,
    "ge": 0.0,
    "le": 1.0,
    "allow_inf_nan": False,
}


class QueryRequest(BaseModel):
    vector: list[float] = Field(..., description=f"Query vector ({VECTOR_SIZE} dims)")
    # Spatial bounds are optional. Leave ALL six unset for an unfiltered
    # (whole-space) search; set a subset and the rest default to the full
    # [0, 1] extent on that axis.
    x_min: float | None = Field(**_BOUND_FIELD, description="Min X bound (default 0.0)")
    x_max: float | None = Field(**_BOUND_FIELD, description="Max X bound (default 1.0)")
    y_min: float | None = Field(**_BOUND_FIELD, description="Min Y bound (default 0.0)")
    y_max: float | None = Field(**_BOUND_FIELD, description="Max Y bound (default 1.0)")
    z_min: float | None = Field(**_BOUND_FIELD, description="Min Z bound (default 0.0)")
    z_max: float | None = Field(**_BOUND_FIELD, description="Max Z bound (default 1.0)")
    time_start_ms: int | None = Field(
        None,
        ge=0,
        description=(
            "Window start (inclusive, ms). May be set without time_end_ms for a "
            "half-open 'everything since' window."
        ),
    )
    time_end_ms: int | None = Field(
        None,
        ge=0,
        description=(
            "Window end (inclusive, ms). May be set without time_start_ms for a "
            "half-open 'everything until' window."
        ),
    )
    limit: int = Field(default=10, ge=1, le=_MAX_LIMIT)
    overlap_factor: float = Field(default=1.0, ge=0.0, le=10.0)

    @field_validator("vector")
    @classmethod
    def check_vector_dims(cls, v: list[float]) -> list[float]:
        if len(v) != VECTOR_SIZE:
            raise ValueError(f"vector must have {VECTOR_SIZE} dimensions, got {len(v)}")
        return v

    @model_validator(mode="after")
    def check_bounds_ordering(self) -> QueryRequest:
        for axis in ("x", "y", "z"):
            lo = getattr(self, f"{axis}_min")
            hi = getattr(self, f"{axis}_max")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"{axis}_min must be <= {axis}_max")
        if (
            self.time_start_ms is not None
            and self.time_end_ms is not None
            and self.time_start_ms > self.time_end_ms
        ):
            raise ValueError("time_start_ms must be <= time_end_ms")
        return self

    def spatial_bounds(self) -> dict[str, float] | None:
        """Resolve the six bound fields into a LociClient spatial filter.

        Returns None (no spatial filter) when no bound is set, or when the
        explicit bounds cover the full unit box — filtering on the full box
        would force enumeration of every spatial bucket for no selectivity.
        """
        raw = (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)
        if all(b is None for b in raw):
            return None
        defaults = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        resolved = tuple(b if b is not None else d for b, d in zip(raw, defaults, strict=True))
        if resolved == defaults:  # explicit full unit box == no filter
            return None
        keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        return dict(zip(keys, resolved, strict=True))

    def time_window(self) -> tuple[int, int] | None:
        """Resolve start/end into a client time window, supporting half-open ends."""
        if self.time_start_ms is None and self.time_end_ms is None:
            return None
        start = self.time_start_ms if self.time_start_ms is not None else 0
        end = self.time_end_ms if self.time_end_ms is not None else _MAX_TIME_MS
        return (start, end)


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    # Deliberately omits internal connection details (e.g. the Qdrant URL) —
    # this endpoint may be exposed to untrusted callers.
    return {"status": "ok", "vector_size": VECTOR_SIZE}


@app.post("/insert")
def insert(req: InsertRequest):
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
        # Belt and braces: WorldState enforces its own invariants; report any
        # miss in the Pydantic mirror above as a validation error, not a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state_id = get_client().insert(state)
    return {"id": state_id}


@app.post("/query")
def query(req: QueryRequest):
    results = get_client().query_scored(
        vector=req.vector,
        spatial_bounds=req.spatial_bounds(),
        time_window_ms=req.time_window(),
        limit=req.limit,
        overlap_factor=req.overlap_factor,
    )

    return {
        "results": [
            {
                "id": r.state.id,
                "x": r.state.x,
                "y": r.state.y,
                "z": r.state.z,
                "timestamp_ms": r.state.timestamp_ms,
                "scene_id": r.state.scene_id,
                "metadata": r.state.metadata,
                "score": r.score,
                "decayed_score": r.decayed_score,
            }
            for r in results
        ]
    }
