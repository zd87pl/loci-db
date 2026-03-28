"""FastAPI application for the LOCI warehouse robot demo."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .simulation import Simulation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sim = Simulation()
_sim_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    sim.running = False


app = FastAPI(title="LOCI Demo — Warehouse Robot Memory", lifespan=lifespan)


# ── Request/Response models ─────────────────────────────────────────────


class SpatialQueryReq(BaseModel):
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    time_start_ms: int | None = None
    time_end_ms: int | None = None
    limit: int = 20


class SimilarQueryReq(BaseModel):
    x: int
    y: int
    radius: int = 5
    time_start_ms: int | None = None
    time_end_ms: int | None = None
    limit: int = 5


class PredictQueryReq(BaseModel):
    steps_ahead: int = 10


class AnomalyReq(BaseModel):
    x: int
    y: int


# ── Static files & index ────────────────────────────────────────────────

import os

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_static_dir, "index.html"))


# Mount static after the root route so "/" takes precedence
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Health ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "memory_count": sim.memory_count}


# ── Simulation control ──────────────────────────────────────────────────


@app.post("/api/simulation/start")
async def start_simulation():
    global _sim_task
    if sim.running:
        return {"status": "already_running"}
    _sim_task = asyncio.create_task(sim.run_loop())
    return {"status": "started"}


@app.post("/api/simulation/stop")
async def stop_simulation():
    sim.running = False
    return {"status": "stopped"}


@app.post("/api/simulation/reset")
async def reset_simulation():
    sim.running = False
    await asyncio.sleep(0.1)
    sim.reset()
    return {"status": "reset"}


@app.get("/api/simulation/state")
async def simulation_state():
    return {
        "robot": {"x": sim.robot_x, "y": sim.robot_y},
        "running": sim.running,
        "tick": sim.tick_count,
        "elapsed_ms": sim.elapsed_ms,
        "memory_count": sim.memory_count,
        "anomalies": [{"x": a[0], "y": a[1]} for a in sim.anomalies],
    }


@app.get("/api/warehouse")
async def warehouse_layout():
    return {"objects": sim.get_warehouse_layout(), "width": 20, "height": 20}


# ── Query endpoints ─────────────────────────────────────────────────────


@app.post("/api/query/spatial")
async def query_spatial(req: SpatialQueryReq):
    if sim.memory_count == 0:
        return {"results": [], "stats": {}, "message": "No memories stored yet. Start the simulation first."}

    # Use the robot's current embedding as query vector
    visible_keys = sim.get_visible_object_keys(sim.robot_x, sim.robot_y)
    from .embeddings import generate_embedding

    query_vec = generate_embedding(
        (req.x_min + req.x_max) // 2,
        (req.y_min + req.y_max) // 2,
        visible_keys,
    )

    # Normalize grid coords to [0,1]
    spatial_bounds = {
        "x_min": max(0.0, req.x_min / 19.0),
        "x_max": min(1.0, req.x_max / 19.0),
        "y_min": max(0.0, req.y_min / 19.0),
        "y_max": min(1.0, req.y_max / 19.0),
        "z_min": 0.0,
        "z_max": 1.0,
    }

    time_window = None
    if req.time_start_ms is not None and req.time_end_ms is not None:
        time_window = (req.time_start_ms, req.time_end_ms)

    results = sim.client.query(
        vector=query_vec,
        spatial_bounds=spatial_bounds,
        time_window_ms=time_window,
        limit=req.limit,
    )

    stats = sim.client.last_query_stats
    total_points = sim.memory_count

    return {
        "results": [
            {
                "id": r.id,
                "x": round(r.x * 19),
                "y": round(r.y * 19),
                "timestamp_ms": r.timestamp_ms,
                "scene_id": r.scene_id,
            }
            for r in results
        ],
        "stats": {
            "shards_searched": stats.shards_searched if stats else 0,
            "total_candidates": stats.total_candidates if stats else 0,
            "hilbert_ids_in_filter": stats.hilbert_ids_in_filter if stats else 0,
            "elapsed_ms": round(stats.elapsed_ms, 2) if stats else 0,
            "total_points": total_points,
        },
    }


@app.post("/api/query/similar")
async def query_similar(req: SimilarQueryReq):
    if sim.memory_count == 0:
        return {"anchor": None, "results": [], "message": "No memories stored yet."}

    from .embeddings import generate_embedding

    # Generate embedding for the clicked point
    visible_keys = sim.get_visible_object_keys(req.x, req.y)
    click_vec = generate_embedding(req.x, req.y, visible_keys)

    # First find the nearest memory to this click point (narrow spatial)
    nx = req.x / 19.0
    ny = req.y / 19.0
    narrow_bounds = {
        "x_min": max(0.0, nx - 0.1),
        "x_max": min(1.0, nx + 0.1),
        "y_min": max(0.0, ny - 0.1),
        "y_max": min(1.0, ny + 0.1),
        "z_min": 0.0,
        "z_max": 1.0,
    }

    anchor_results = sim.client.query(
        vector=click_vec,
        spatial_bounds=narrow_bounds,
        limit=1,
    )

    if not anchor_results:
        # Try without spatial bounds
        anchor_results = sim.client.query(vector=click_vec, limit=1)

    if not anchor_results:
        return {"anchor": None, "results": [], "message": "No nearby memories found."}

    anchor = anchor_results[0]

    # Now find similar memories within radius
    radius_norm = req.radius / 19.0
    wide_bounds = {
        "x_min": max(0.0, anchor.x - radius_norm),
        "x_max": min(1.0, anchor.x + radius_norm),
        "y_min": max(0.0, anchor.y - radius_norm),
        "y_max": min(1.0, anchor.y + radius_norm),
        "z_min": 0.0,
        "z_max": 1.0,
    }

    time_window = None
    if req.time_start_ms is not None and req.time_end_ms is not None:
        time_window = (req.time_start_ms, req.time_end_ms)

    similar = sim.client.query(
        vector=anchor.vector,
        spatial_bounds=wide_bounds,
        time_window_ms=time_window,
        limit=req.limit + 1,  # +1 because anchor might be in results
    )

    # Remove anchor from results if present
    similar = [r for r in similar if r.id != anchor.id][: req.limit]

    stats = sim.client.last_query_stats

    return {
        "anchor": {
            "id": anchor.id,
            "x": round(anchor.x * 19),
            "y": round(anchor.y * 19),
            "timestamp_ms": anchor.timestamp_ms,
        },
        "results": [
            {
                "id": r.id,
                "x": round(r.x * 19),
                "y": round(r.y * 19),
                "timestamp_ms": r.timestamp_ms,
            }
            for r in similar
        ],
        "stats": {
            "shards_searched": stats.shards_searched if stats else 0,
            "total_candidates": stats.total_candidates if stats else 0,
            "elapsed_ms": round(stats.elapsed_ms, 2) if stats else 0,
        },
    }


@app.post("/api/query/predict")
async def query_predict(req: PredictQueryReq):
    if sim.memory_count < 3:
        return {
            "novelty": None,
            "results": [],
            "message": "Need at least 3 memories for prediction. Let the simulation run a bit.",
        }

    if not sim.recent_embeddings:
        return {"novelty": None, "results": [], "message": "No recent observations."}

    context_vec = sim.recent_embeddings[-1]
    predictor_fn = sim.make_predictor(req.steps_ahead)

    # Current position normalized
    cx = sim.robot_x / 19.0
    cy = sim.robot_y / 19.0

    result = sim.client.predict_and_retrieve(
        context_vector=context_vec,
        predictor_fn=predictor_fn,
        future_horizon_ms=req.steps_ahead * 500,
        current_position=(cx, cy, 0.5),
        spatial_search_radius=0.3,
        limit=5,
        alpha=0.7,
        return_prediction=True,
    )

    # Predict where robot will be in N steps
    route_idx = sim.route_idx
    predicted_x, predicted_y = sim.robot_x, sim.robot_y
    for _ in range(req.steps_ahead):
        predicted_x, predicted_y = sim.patrol_route[route_idx % len(sim.patrol_route)]
        route_idx = (route_idx + 1) % len(sim.patrol_route)

    return {
        "novelty": round(result.prediction_novelty, 3),
        "predicted_position": {"x": predicted_x, "y": predicted_y},
        "current_position": {"x": sim.robot_x, "y": sim.robot_y},
        "predictor_ms": round(result.predictor_call_ms, 2),
        "retrieval_ms": round(result.retrieval_latency_ms, 2),
        "results": [
            {
                "id": r.id,
                "x": round(r.x * 19),
                "y": round(r.y * 19),
                "timestamp_ms": r.timestamp_ms,
            }
            for r in result.results
        ],
    }


@app.post("/api/anomaly/place")
async def place_anomaly(req: AnomalyReq):
    if 0 <= req.x < 20 and 0 <= req.y < 20:
        sim.anomalies.append((req.x, req.y))
        # Also add to warehouse_grid so observations pick it up
        sim.warehouse_grid[(req.x, req.y)] = "anomaly"
        return {"status": "placed", "x": req.x, "y": req.y, "total_anomalies": len(sim.anomalies)}
    return {"status": "error", "message": "Position out of bounds"}


@app.get("/api/stats")
async def stats():
    store = sim.client.store
    collections = [name for name in store._collections]
    return {
        "memory_count": sim.memory_count,
        "epoch_count": len(collections),
        "collections": collections,
        "tick_count": sim.tick_count,
        "elapsed_ms": sim.elapsed_ms,
        "anomaly_count": len(sim.anomalies),
    }


@app.get("/api/heatmap")
async def heatmap():
    """Return memory density per grid cell."""
    grid = [[0] * 20 for _ in range(20)]
    store = sim.client.store
    for col in store._collections.values():
        for point in col.points.values():
            gx = round(point.payload.get("x", 0) * 19)
            gy = round(point.payload.get("y", 0) * 19)
            gx = max(0, min(19, gx))
            gy = max(0, min(19, gy))
            grid[gy][gx] += 1
    return {"grid": grid, "max_density": max(max(row) for row in grid) if sim.memory_count > 0 else 0}


# ── WebSocket ────────────────────────────────────────────────────────────


@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    await ws.accept()
    sim.subscribers.add(ws)
    try:
        while True:
            # Keep connection alive; handle client messages if needed
            data = await ws.receive_text()
            # Could handle client commands here
    except WebSocketDisconnect:
        sim.subscribers.discard(ws)
    except Exception:
        sim.subscribers.discard(ws)
