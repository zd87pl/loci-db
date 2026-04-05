"""FastAPI application for the LOCI warehouse robot demo."""

from __future__ import annotations

import asyncio
import logging
import os
import time
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
    time_start_s: float = 0
    time_end_s: float = 9999
    limit: int = 20


class SimilarQueryReq(BaseModel):
    x: int
    y: int
    radius: int = 5
    limit: int = 5


class PredictQueryReq(BaseModel):
    steps_ahead: int = 10


class AnomalyReq(BaseModel):
    x: int
    y: int


# ── Static files & index ────────────────────────────────────────────────

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_static_dir, "index.html"))


app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Health ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    demo = sim.get_demo_status()
    return {
        "status": "ok",
        "memory_count": sim.memory_count,
        "running": sim.running,
        "demo_phase": demo["phase"],
        "demo_summary": demo["summary"],
        "route": demo["route"],
        "readiness": {"predict": demo["predict"], "anomaly": demo["anomaly"]},
    }


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
    global _sim_task
    sim.running = False
    if _sim_task and not _sim_task.done():
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass
    _sim_task = None
    sim.reset()
    return {"status": "reset"}


@app.get("/api/simulation/state")
async def simulation_state():
    demo = sim.get_demo_status()
    return {
        "robot": {"x": sim.robot_x, "y": sim.robot_y},
        "running": sim.running,
        "tick": sim.tick_count,
        "elapsed_ms": sim.elapsed_ms,
        "memory_count": sim.memory_count,
        "start_time_ms": sim.start_time_ms,
        "anomalies": [{"x": a[0], "y": a[1]} for a in sim.anomalies],
        "route": demo["route"],
        "guide": demo,
    }


@app.get("/api/demo/status")
async def demo_status():
    return sim.get_demo_status()


@app.get("/api/warehouse")
async def warehouse_layout():
    return {"objects": sim.get_warehouse_layout(), "width": 20, "height": 20}


# ── Query endpoints ─────────────────────────────────────────────────────


@app.post("/api/query/spatial")
async def query_spatial(req: SpatialQueryReq):
    if sim.memory_count == 0:
        return {
            "results": [],
            "stats": {},
            "message": "No memories stored yet. Start the simulation first.",
        }

    from .embeddings import generate_embedding

    now_ms = int(time.time() * 1000)
    query_logs = []

    # Generate embedding for the center of the bounding box
    center_x = (req.x_min + req.x_max) // 2
    center_y = (req.y_min + req.y_max) // 2
    visible_keys = sim.get_visible_object_keys(center_x, center_y)
    query_vec = generate_embedding(center_x, center_y, visible_keys)

    query_logs.append({
        "ts": now_ms, "tag": "QUERY",
        "msg": f"Spatial search: bbox ({req.x_min},{req.y_min})→({req.x_max},{req.y_max}), time {req.time_start_s}s–{req.time_end_s}s",
    })
    query_logs.append({
        "ts": now_ms, "tag": "EMBED",
        "msg": f"Query vector from center ({center_x},{center_y}) + {len(visible_keys)} visible obj",
    })

    # Normalize grid coords to [0,1]
    spatial_bounds = {
        "x_min": max(0.0, req.x_min / 19.0),
        "x_max": min(1.0, req.x_max / 19.0),
        "y_min": max(0.0, req.y_min / 19.0),
        "y_max": min(1.0, req.y_max / 19.0),
        "z_min": 0.0,
        "z_max": 1.0,
    }

    # Convert seconds-from-start to absolute timestamps
    time_window = None
    if sim.start_time_ms > 0:
        t_start = sim.start_time_ms + int(req.time_start_s * 1000)
        t_end = sim.start_time_ms + int(req.time_end_s * 1000)
        time_window = (t_start, t_end)

    results = sim.client.query(
        vector=query_vec,
        spatial_bounds=spatial_bounds,
        time_window_ms=time_window,
        limit=req.limit,
    )

    stats = sim.client.last_query_stats
    total_points = sim.memory_count

    if stats:
        query_logs.append({
            "ts": now_ms, "tag": "HILBERT",
            "msg": f"Expanded bbox to {stats.hilbert_ids_in_filter} Hilbert buckets at r4",
        })
        query_logs.append({
            "ts": now_ms, "tag": "SHARD",
            "msg": f"Scanned {stats.shards_searched} epoch shards, {stats.total_candidates} candidates",
        })
        query_logs.append({
            "ts": now_ms, "tag": "RANK",
            "msg": f"Decay-ranked → {len(results)} results in {round(stats.elapsed_ms, 2)}ms",
        })

    return {
        "results": [
            {
                "id": r.id,
                "x": round(r.x * 19),
                "y": round(r.y * 19),
                "timestamp_ms": r.timestamp_ms,
                "elapsed_s": round((r.timestamp_ms - sim.start_time_ms) / 1000, 1)
                if sim.start_time_ms
                else 0,
            }
            for r in results
        ],
        "guide": sim.get_demo_status(),
        "inference_log": query_logs,
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

    now_ms = int(time.time() * 1000)
    query_logs = []

    visible_keys = sim.get_visible_object_keys(req.x, req.y)
    click_vec = generate_embedding(req.x, req.y, visible_keys)

    query_logs.append({
        "ts": now_ms, "tag": "QUERY",
        "msg": f"Similarity search: anchor ({req.x},{req.y}), radius {req.radius} cells",
    })
    query_logs.append({
        "ts": now_ms, "tag": "EMBED",
        "msg": f"Query vector from ({req.x},{req.y}) + {len(visible_keys)} visible obj",
    })

    # Find nearest memory to click point
    nx = req.x / 19.0
    ny = req.y / 19.0
    narrow_bounds = {
        "x_min": max(0.0, nx - 0.15),
        "x_max": min(1.0, nx + 0.15),
        "y_min": max(0.0, ny - 0.15),
        "y_max": min(1.0, ny + 0.15),
        "z_min": 0.0,
        "z_max": 1.0,
    }

    anchor_results = sim.client.query(vector=click_vec, spatial_bounds=narrow_bounds, limit=1)
    if not anchor_results:
        anchor_results = sim.client.query(vector=click_vec, limit=1)
    if not anchor_results:
        return {"anchor": None, "results": [], "message": "No memories found near that location."}

    anchor = anchor_results[0]

    query_logs.append({
        "ts": now_ms, "tag": "SHARD",
        "msg": f"Anchor found: id={anchor.id[:8]}… at ({round(anchor.x*19)},{round(anchor.y*19)})",
    })

    # Find similar memories within radius
    radius_norm = req.radius / 19.0
    wide_bounds = {
        "x_min": max(0.0, anchor.x - radius_norm),
        "x_max": min(1.0, anchor.x + radius_norm),
        "y_min": max(0.0, anchor.y - radius_norm),
        "y_max": min(1.0, anchor.y + radius_norm),
        "z_min": 0.0,
        "z_max": 1.0,
    }

    similar = sim.client.query(
        vector=anchor.vector, spatial_bounds=wide_bounds, limit=req.limit + 1
    )
    similar = [r for r in similar if r.id != anchor.id][: req.limit]

    stats = sim.client.last_query_stats

    if stats:
        query_logs.append({
            "ts": now_ms, "tag": "RANK",
            "msg": f"Found {len(similar)} similar moments from {stats.total_candidates} candidates in {round(stats.elapsed_ms, 2)}ms",
        })

    return {
        "anchor": {
            "id": anchor.id,
            "x": round(anchor.x * 19),
            "y": round(anchor.y * 19),
            "timestamp_ms": anchor.timestamp_ms,
            "elapsed_s": round((anchor.timestamp_ms - sim.start_time_ms) / 1000, 1)
            if sim.start_time_ms
            else 0,
        },
        "results": [
            {
                "id": r.id,
                "x": round(r.x * 19),
                "y": round(r.y * 19),
                "timestamp_ms": r.timestamp_ms,
                "elapsed_s": round((r.timestamp_ms - sim.start_time_ms) / 1000, 1)
                if sim.start_time_ms
                else 0,
            }
            for r in similar
        ],
        "guide": sim.get_demo_status(),
        "inference_log": query_logs,
        "stats": {
            "shards_searched": stats.shards_searched if stats else 0,
            "total_candidates": stats.total_candidates if stats else 0,
            "elapsed_ms": round(stats.elapsed_ms, 2) if stats else 0,
        },
    }


@app.post("/api/query/predict")
async def query_predict(req: PredictQueryReq):
    demo = sim.get_demo_status()
    min_memories = demo["predict"]["minimum_memories"]
    if sim.memory_count < min_memories:
        return {
            "novelty": None,
            "results": [],
            "message": demo["predict"]["message"],
            "guide": demo,
        }

    if not sim.recent_embeddings:
        return {
            "novelty": None,
            "results": [],
            "message": "No recent observations.",
            "guide": demo,
        }

    now_ms = int(time.time() * 1000)
    query_logs = []

    context_vec = sim.recent_embeddings[-1]
    predictor_fn = sim.make_predictor(req.steps_ahead)
    route = sim.get_route_status()

    cx = sim.robot_x / 19.0
    cy = sim.robot_y / 19.0

    query_logs.append({
        "ts": now_ms, "tag": "PREDICT",
        "msg": f"Linear extrapolation from last 3 embeddings, {req.steps_ahead} steps ahead",
    })

    result = sim.client.predict_and_retrieve(
        context_vector=context_vec,
        predictor_fn=predictor_fn,
        future_horizon_ms=req.steps_ahead * sim.tick_interval_ms,
        current_position=(cx, cy, 0.5),
        spatial_search_radius=0.3,
        limit=5,
        alpha=0.7,
        return_prediction=True,
    )

    query_logs.append({
        "ts": now_ms, "tag": "EMBED",
        "msg": f"Predicted 128-d future vector in {round(result.predictor_call_ms, 2)}ms",
    })
    query_logs.append({
        "ts": now_ms, "tag": "SHARD",
        "msg": f"Searched memory for matches within {req.steps_ahead * sim.tick_interval_ms}ms horizon",
    })

    novelty = round(result.prediction_novelty, 3)
    novelty_label = "expected" if novelty < 0.3 else ("partly new" if novelty < 0.7 else "surprising!")
    query_logs.append({
        "ts": now_ms, "tag": "NOVELTY",
        "msg": f"Score: {novelty} ({novelty_label}) — {len(result.results)} memory matches in {round(result.retrieval_latency_ms, 2)}ms",
    })

    # Calculate predicted position from patrol route
    route_idx = sim.route_idx
    predicted_x, predicted_y = sim.robot_x, sim.robot_y
    # Also build the predicted path for visualization
    predicted_path = [{"x": sim.robot_x, "y": sim.robot_y}]
    for _ in range(req.steps_ahead):
        predicted_x, predicted_y = sim.patrol_route[route_idx % len(sim.patrol_route)]
        route_idx = (route_idx + 1) % len(sim.patrol_route)
        predicted_path.append({"x": predicted_x, "y": predicted_y})

    return {
        "novelty": novelty,
        "predicted_position": {"x": predicted_x, "y": predicted_y},
        "predicted_path": predicted_path,
        "current_position": {"x": sim.robot_x, "y": sim.robot_y},
        "route": route,
        "guide": demo,
        "inference_log": query_logs,
        "predictor_ms": round(result.predictor_call_ms, 2),
        "retrieval_ms": round(result.retrieval_latency_ms, 2),
        "results": [
            {
                "id": r.id,
                "x": round(r.x * 19),
                "y": round(r.y * 19),
                "timestamp_ms": r.timestamp_ms,
                "elapsed_s": round((r.timestamp_ms - sim.start_time_ms) / 1000, 1)
                if sim.start_time_ms
                else 0,
            }
            for r in result.results
        ],
    }


@app.post("/api/anomaly/place")
async def place_anomaly(req: AnomalyReq):
    x = max(0, min(19, req.x))
    y = max(0, min(19, req.y))
    sim.anomalies.append((x, y))
    sim.warehouse_grid[(x, y)] = "anomaly"
    demo = sim.get_demo_status()
    return {
        "status": "placed",
        "x": x,
        "y": y,
        "total_anomalies": len(sim.anomalies),
        "guide": demo,
    }


@app.get("/api/stats")
async def stats():
    store = sim.client.store
    collections = list(store._collections.keys())
    demo = sim.get_demo_status()
    return {
        "memory_count": sim.memory_count,
        "epoch_count": len(collections),
        "collections": collections,
        "tick_count": sim.tick_count,
        "elapsed_ms": sim.elapsed_ms,
        "anomaly_count": len(sim.anomalies),
        "demo_phase": demo["phase"],
        "demo_summary": demo["summary"],
        "route": demo["route"],
        "predict_ready": demo["predict"]["ready"],
        "anomaly_ready": demo["anomaly"]["ready"],
    }


# Cached heatmap
_heatmap_cache: dict = {"tick": -1, "data": None}


@app.get("/api/heatmap")
async def heatmap():
    """Return memory density per grid cell (cached per tick)."""
    if _heatmap_cache["tick"] == sim.tick_count and _heatmap_cache["data"]:
        return _heatmap_cache["data"]

    grid = [[0] * 20 for _ in range(20)]
    store = sim.client.store
    for col in store._collections.values():
        for point in col.points.values():
            gx = round(point.payload.get("x", 0) * 19)
            gy = round(point.payload.get("y", 0) * 19)
            gx = max(0, min(19, gx))
            gy = max(0, min(19, gy))
            grid[gy][gx] += 1

    max_val = 0
    for row in grid:
        for v in row:
            if v > max_val:
                max_val = v

    data = {"grid": grid, "max_density": max_val}
    _heatmap_cache["tick"] = sim.tick_count
    _heatmap_cache["data"] = data
    return data


# ── WebSocket ────────────────────────────────────────────────────────────


@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    await ws.accept()
    sim.subscribers.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        sim.subscribers.discard(ws)
    except Exception:
        sim.subscribers.discard(ws)
