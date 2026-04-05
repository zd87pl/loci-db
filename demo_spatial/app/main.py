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
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
        return FileResponse(idx)
    return {"message": "LOCI Spatial Memory Assistant API", "docs": "/docs"}


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
# Helpers
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()
