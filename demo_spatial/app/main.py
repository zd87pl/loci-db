"""FastAPI application for the LOCI-DB Spatial Memory Assistant.

This is the Phase A demo backend: a stateful AI assistant that helps
blind users track where they left objects using spatial episodic memory.

Architecture:
  Browser/client → POST /api/ingest/frame  (camera frames)
                 → POST /api/voice/query   (voice queries)
                 → WebSocket /ws           (real-time tracking feed)
                 → GET /api/objects/*      (spatial query API)

Environment variables:
  OPENAI_API_KEY   — enables Whisper STT, GPT-4o mini VLM, OpenAI TTS
  VLM_PROVIDER     — "openai" (default) | "gemini"
  GOOGLE_API_KEY   — enables Gemini VLM
  TTS_ENGINE       — "openai" (default) | "piper" | "edge"
  WHISPER_LOCAL    — "1" to use local faster-whisper instead of API
  PIPER_MODEL_PATH — path to .onnx Piper model file (if TTS_ENGINE=piper)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import random
import socket
import struct
from contextlib import asynccontextmanager

import segno
from fastapi import FastAPI, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .scene_ingestion import SceneIngestion
from .spatial_memory import SpatialMemory
from .vlm_client import VLMClient
from .voice_pipeline import VoicePipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

memory = SpatialMemory(epoch_size_ms=5000)
vlm_client = VLMClient()
ingestion = SceneIngestion(memory, vlm_client=vlm_client, use_vlm_fallback=True)
voice = VoicePipeline(memory, vlm_client=vlm_client)

_ws_subscribers: set[WebSocket] = set()

# In-memory store for the latest uploaded LiDAR point cloud
_point_cloud: dict = {"points": [], "count": 0, "filename": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Spatial Memory Assistant starting — VLM: %s, TTS: %s, STT: %s",
        vlm_client.is_available,
        voice.tts_engine,
        voice.stt_available,
    )
    yield
    await ingestion.stop_capture()
    logger.info("Spatial Memory Assistant stopped")


app = FastAPI(
    title="LOCI-DB Spatial Memory Assistant",
    description=(
        "Stateful AI assistant for the blind — tracks object locations "
        "using 4D spatiotemporal episodic memory."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tightened in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def index():
    idx = os.path.join(_static_dir, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx, headers={"Cache-Control": "no-cache"})
    return {"message": "LOCI Spatial Memory Assistant API", "docs": "/docs"}


@app.get("/scanner")
async def scanner():
    """Mobile-optimized 3D scanner page for iPhone (camera + LiDAR)."""
    scanner_path = os.path.join(_static_dir, "scanner.html")
    if os.path.exists(scanner_path):
        return FileResponse(scanner_path, headers={"Cache-Control": "no-cache"})
    return {"error": "scanner.html not found"}


@app.get("/api/scanner-qr")
async def scanner_qr(url: str = Query(..., description="Scanner URL to encode")):
    """Generate a scannable QR code PNG for the mobile scanner URL."""
    qr = segno.make(url, error="l")
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=5, border=2)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/local-ip")
async def local_ip():
    """Return this machine's local network IP for cross-device access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "localhost"
    return {"ip": ip}



# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class FrameRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG frame (data URI or raw)")
    timestamp_ms: int | None = Field(None, description="Override frame timestamp")
    use_vlm: bool | None = Field(None, description="Force VLM usage for this frame")


class TextQueryRequest(BaseModel):
    text: str = Field(..., description="Natural language query text")


class CaptureStartRequest(BaseModel):
    camera_index: int = Field(0, description="Camera device index")
    fps: float = Field(2.0, description="Capture rate in frames per second", ge=0.1, le=30.0)


class Frame3DRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG/PNG frame (data URI or raw)")
    timestamp_ms: int | None = Field(None, description="Override frame timestamp")
    use_vlm: bool | None = Field(None, description="Force VLM usage for this frame")
    has_lidar: bool = Field(False, description="Whether LiDAR depth data is available")
    depth_samples: dict[str, float] | None = Field(
        None,
        description="Depth samples from LiDAR at grid positions (e.g. 'mc': 1.23 meters)",
    )


class RegionQueryRequest(BaseModel):
    x_min: float = Field(0.0, ge=0.0, le=1.0)
    x_max: float = Field(1.0, ge=0.0, le=1.0)
    y_min: float = Field(0.0, ge=0.0, le=1.0)
    y_max: float = Field(1.0, ge=0.0, le=1.0)
    limit: int = Field(20, ge=1, le=100)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "observation_count": memory.observation_count,
        "tracked_objects": len(memory.tracked_labels),
        "capturing": ingestion.is_capturing,
        "frame_count": ingestion.frame_count,
        "pipeline": voice.status(),
        "vlm_available": vlm_client.is_available,
    }


# ---------------------------------------------------------------------------
# Scene ingestion endpoints
# ---------------------------------------------------------------------------

@app.post("/api/ingest/frame")
async def ingest_frame(req: FrameRequest):
    """Submit a camera frame for object detection and spatial indexing.

    The frontend should call this at ~2 fps while the camera is active.
    Returns the list of detected objects and their normalized positions.
    """
    detections = await ingestion.process_frame_b64(
        req.image_b64,
        timestamp_ms=req.timestamp_ms,
        use_vlm=req.use_vlm,
    )
    result = {
        "detections": [d.to_dict() for d in detections],
        "frame_count": ingestion.frame_count,
        "observation_count": memory.observation_count,
    }
    # Broadcast to WebSocket subscribers
    if _ws_subscribers:
        msg = json.dumps({"type": "detections", **result})
        dead = set()
        for ws in list(_ws_subscribers):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _ws_subscribers.difference_update(dead)
    return result


@app.post("/api/ingest/frame3d")
async def ingest_frame_3d(req: Frame3DRequest):
    """Submit a camera frame with optional LiDAR depth data for 3D spatial indexing.

    Used by the mobile scanner (/scanner) to send frames from iPhone camera + LiDAR.
    Depth samples are a dict mapping grid positions (tl, tc, tr, ml, mc, mr, bl, bc, br)
    to depth values in meters. Each detection is matched to the nearest depth sample.
    """
    detections = await ingestion.process_frame_b64(
        req.image_b64,
        timestamp_ms=req.timestamp_ms,
        use_vlm=req.use_vlm,
        depth_samples=req.depth_samples if req.has_lidar else None,
    )
    result = {
        "detections": [d.to_dict() for d in detections],
        "frame_count": ingestion.frame_count,
        "observation_count": memory.observation_count,
        "has_lidar": req.has_lidar,
    }
    # Broadcast to WebSocket subscribers
    if _ws_subscribers:
        msg = json.dumps({"type": "detections", **result})
        dead = set()
        for ws in list(_ws_subscribers):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _ws_subscribers.difference_update(dead)
    return result


@app.post("/api/ingest/frame/upload")
async def ingest_frame_upload(file: UploadFile):
    """Upload a camera frame as multipart form data (alternative to base64)."""
    image_bytes = await file.read()
    detections = await ingestion.process_frame(image_bytes)
    return {
        "detections": [d.to_dict() for d in detections],
        "frame_count": ingestion.frame_count,
        "observation_count": memory.observation_count,
    }


@app.post("/api/ingest/capture/start")
async def start_capture(req: CaptureStartRequest):
    """Start server-side webcam capture (requires camera attached to server)."""
    if ingestion.is_capturing:
        return {"status": "already_capturing", "frame_count": ingestion.frame_count}
    await ingestion.start_capture(camera_index=req.camera_index, fps=req.fps)
    return {"status": "started", "camera_index": req.camera_index, "fps": req.fps}


@app.post("/api/ingest/capture/stop")
async def stop_capture():
    """Stop server-side webcam capture."""
    await ingestion.stop_capture()
    return {"status": "stopped", "frame_count": ingestion.frame_count}


@app.get("/api/ingest/status")
async def ingestion_status():
    return ingestion.status()


# ---------------------------------------------------------------------------
# Spatial query endpoints
# ---------------------------------------------------------------------------

@app.get("/api/objects")
async def list_objects():
    """Return the latest known position of every tracked object."""
    objects = memory.current_objects()
    return {
        "objects": [o.to_dict() for o in objects],
        "count": len(objects),
        "stats": memory.stats(),
    }


@app.get("/api/objects/{label}/location")
async def object_location(label: str, limit: int = 5):
    """Find the most recent observations of a specific object.

    Uses LOCI-DB semantic + spatial search to retrieve the best matches.
    """
    results = memory.where_is(label, limit=limit)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Object '{label}' not found in spatial memory.",
        )
    return {
        "label": label,
        "results": [o.to_dict() for o in results],
        "best_match": results[0].to_dict(),
    }


@app.get("/api/objects/{label}/history")
async def object_history(label: str, limit: int = 20):
    """Return full temporal history of where an object has been seen."""
    history = memory.history(label, limit=limit)
    return {
        "label": label,
        "observations": [o.to_dict() for o in history],
        "count": len(history),
    }


@app.post("/api/objects/region")
async def objects_in_region(req: RegionQueryRequest):
    """Find all objects observed in a spatial region (normalized [0,1] coords)."""
    results = memory.objects_in_region(
        req.x_min, req.x_max, req.y_min, req.y_max, limit=req.limit
    )
    return {
        "region": {"x_min": req.x_min, "x_max": req.x_max, "y_min": req.y_min, "y_max": req.y_max},
        "objects": [o.to_dict() for o in results],
        "count": len(results),
    }


@app.get("/api/objects/changes/recent")
async def recent_changes(window_seconds: float = 30.0):
    """Objects that have been observed in the last N seconds."""
    recent = memory.recent_changes(window_seconds=window_seconds)
    return {
        "window_seconds": window_seconds,
        "objects": [o.to_dict() for o in recent],
        "count": len(recent),
    }


# ---------------------------------------------------------------------------
# Voice pipeline endpoints
# ---------------------------------------------------------------------------

@app.post("/api/voice/query")
async def voice_query(file: UploadFile, language: str = "en", audio_response: bool = True):
    """Process a voice query: audio → transcribe → understand → answer → audio.

    Accepts audio in any format Whisper supports: WebM, WAV, MP3, M4A.
    Returns JSON with transcription + answer text, plus MP3 audio if requested.

    For binary audio response use: GET /api/voice/query/audio with the same params.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    result = await voice.handle_audio_query(
        audio_bytes,
        language=language,
        synthesize_response=audio_response,
    )
    return {
        "transcription": result.transcription,
        "intent": {
            "kind": result.intent.kind,
            "object_name": result.intent.object_name,
        },
        "answer": result.answer_text,
        "has_audio": len(result.answer_audio) > 0,
        "audio_b64": _b64(result.answer_audio) if result.answer_audio else None,
        "latency_ms": result.latency_ms,
    }


@app.post("/api/voice/text-query")
async def text_query(req: TextQueryRequest):
    """Process a text query and return a spoken-style answer + TTS audio.

    Useful for testing without a microphone, or for frontend debugging.
    """
    result = await voice.handle_text_query(req.text)
    return {
        "transcription": result.transcription,
        "intent": {
            "kind": result.intent.kind,
            "object_name": result.intent.object_name,
        },
        "answer": result.answer_text,
        "has_audio": len(result.answer_audio) > 0,
        "audio_b64": _b64(result.answer_audio) if result.answer_audio else None,
        "latency_ms": result.latency_ms,
    }


@app.post("/api/voice/tts")
async def text_to_speech(req: TextQueryRequest):
    """Convert text directly to audio bytes (MP3).

    Returns binary audio/mpeg response for direct browser playback.
    """
    audio = await voice._tts.synthesize(req.text)
    if not audio:
        raise HTTPException(status_code=503, detail="TTS synthesis failed — check TTS_ENGINE config")
    return Response(content=audio, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# WebSocket — real-time object tracking feed
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_tracking(ws: WebSocket):
    """WebSocket stream for real-time object detections and memory updates.

    The server pushes messages on:
      - New frame processed (type: "detections")
      - Periodic heartbeat with all current objects (type: "snapshot")

    Message format:
      { "type": "detections", "detections": [...], "frame_count": N }
      { "type": "snapshot", "objects": [...], "observation_count": N }
    """
    await ws.accept()
    _ws_subscribers.add(ws)
    try:
        # Send initial snapshot
        snapshot = {
            "type": "snapshot",
            "objects": [o.to_dict() for o in memory.current_objects()],
            "observation_count": memory.observation_count,
        }
        await ws.send_text(json.dumps(snapshot))

        # Periodic snapshot every 5s + keep-alive
        while True:
            await asyncio.sleep(5)
            snapshot = {
                "type": "snapshot",
                "objects": [o.to_dict() for o in memory.current_objects()],
                "observation_count": memory.observation_count,
                "stats": memory.stats(),
            }
            await ws.send_text(json.dumps(snapshot))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket error: %s", e)
    finally:
        _ws_subscribers.discard(ws)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def stats():
    return {
        "memory": memory.stats(),
        "ingestion": ingestion.status(),
        "voice": voice.status(),
        "ws_subscribers": len(_ws_subscribers),
    }


# ---------------------------------------------------------------------------
# LiDAR point cloud upload (from Scaniverse / Polycam / 3D Scanner App)
# ---------------------------------------------------------------------------

def _parse_ply(data: bytes) -> list[dict]:
    """Parse a PLY file (ASCII or binary little-endian) and return point list.

    Each point dict has: x, y, z (float) and optionally r, g, b (0-255 int).
    Returns an empty list on any parse failure.
    """
    try:
        # Split header from data
        if b"end_header" not in data:
            return []
        header_raw, body = data.split(b"end_header\n", 1)
        header = header_raw.decode("utf-8", errors="replace")

        lines = header.splitlines()
        fmt = "ascii"
        vertex_count = 0
        props: list[tuple[str, str]] = []  # (name, type)
        in_vertex = False

        for line in lines:
            line = line.strip()
            if line.startswith("format"):
                parts = line.split()
                if "binary_little_endian" in line:
                    fmt = "binary_le"
                elif "binary_big_endian" in line:
                    fmt = "binary_be"
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                if len(parts) >= 3:
                    props.append((parts[-1], parts[1]))  # (name, type)

        if vertex_count == 0 or not props:
            return []

        # Determine field indices
        prop_names = [p[0] for p in props]
        prop_types = [p[1] for p in props]

        def _idx(candidates):
            for c in candidates:
                if c in prop_names:
                    return prop_names.index(c)
            return -1

        ix = _idx(["x"])
        iy = _idx(["y"])
        iz = _idx(["z"])
        if ix < 0 or iy < 0 or iz < 0:
            return []

        ir = _idx(["red", "r", "diffuse_red"])
        ig = _idx(["green", "g", "diffuse_green"])
        ib = _idx(["blue", "b", "diffuse_blue"])

        points: list[dict] = []

        if fmt == "ascii":
            text_lines = body.decode("utf-8", errors="replace").splitlines()
            for i, tl in enumerate(text_lines):
                if i >= vertex_count:
                    break
                vals = tl.strip().split()
                if len(vals) < max(ix, iy, iz) + 1:
                    continue
                pt: dict = {
                    "x": float(vals[ix]),
                    "y": float(vals[iy]),
                    "z": float(vals[iz]),
                }
                if ir >= 0 and len(vals) > ir:
                    pt["r"] = int(float(vals[ir]))
                    pt["g"] = int(float(vals[ig]))
                    pt["b"] = int(float(vals[ib]))
                points.append(pt)

        else:  # binary
            _TYPE_FMT = {
                "float": "f", "float32": "f",
                "double": "d", "float64": "d",
                "int": "i", "int32": "i",
                "uint": "I", "uint32": "I",
                "short": "h", "int16": "h",
                "ushort": "H", "uint16": "H",
                "char": "b", "int8": "b",
                "uchar": "B", "uint8": "B",
            }
            endian = "<" if fmt == "binary_le" else ">"
            row_fmt = endian + "".join(_TYPE_FMT.get(t, "f") for _, t in props)
            row_size = struct.calcsize(row_fmt)
            for i in range(vertex_count):
                offset = i * row_size
                if offset + row_size > len(body):
                    break
                vals = struct.unpack_from(row_fmt, body, offset)
                pt: dict = {
                    "x": float(vals[ix]),
                    "y": float(vals[iy]),
                    "z": float(vals[iz]),
                }
                if ir >= 0:
                    pt["r"] = int(vals[ir]) & 0xFF
                    pt["g"] = int(vals[ig]) & 0xFF
                    pt["b"] = int(vals[ib]) & 0xFF
                points.append(pt)

        return points

    except Exception as exc:
        logger.warning("PLY parse error: %s", exc)
        return []


@app.post("/api/ingest/pointcloud")
async def upload_pointcloud(file: UploadFile):
    """Upload a LiDAR point cloud (PLY format) from Scaniverse, Polycam, or 3D Scanner App.

    The point cloud is stored in memory and exposed via GET /api/pointcloud/latest
    for real-time Three.js visualization on the dashboard.

    Supported exports:
      - Scaniverse (free): Share → Export → PLY
      - Polycam (free tier): Export → Point Cloud → PLY
      - 3D Scanner App (free): Export → Point Cloud (PLY)
    """
    global _point_cloud
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Parse PLY
    points = await asyncio.get_event_loop().run_in_executor(None, _parse_ply, data)
    if not points:
        raise HTTPException(
            status_code=422,
            detail="Could not parse PLY file. Make sure to export as PLY from your scanning app.",
        )

    _point_cloud = {
        "points": points,
        "count": len(points),
        "filename": file.filename or "scan.ply",
    }
    logger.info("Point cloud uploaded: %d points from %s", len(points), file.filename)
    return {
        "count": len(points),
        "filename": file.filename,
        "has_color": "r" in points[0] if points else False,
    }


@app.get("/api/pointcloud/latest")
async def get_pointcloud(max_points: int = Query(5000, ge=100, le=50000)):
    """Return the latest uploaded point cloud (downsampled for web visualization).

    Returns up to max_points points (default 5000) randomly sampled from the full cloud.
    Each point: { x, y, z } and optionally { r, g, b } (0-255).
    """
    pts = _point_cloud["points"]
    if not pts:
        return {"count": 0, "points": [], "filename": None}
    # Random downsample for web
    sample = random.sample(pts, min(max_points, len(pts))) if len(pts) > max_points else pts
    return {
        "count": _point_cloud["count"],
        "sampled": len(sample),
        "filename": _point_cloud["filename"],
        "has_color": "r" in pts[0] if pts else False,
        "points": sample,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()
