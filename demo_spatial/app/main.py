"""FastAPI application for the LOCI-DB Spatial Memory Assistant.

This is the Phase A demo backend: a stateful AI assistant that helps
blind users track where they left objects using spatial episodic memory.

Architecture:
  Browser/client → POST /api/ingest/frame  (camera frames)
                 → POST /api/voice/query   (voice queries)
                 → WebSocket /ws           (real-time tracking feed)
                 → GET /api/objects/*      (spatial query API)
  Handheld (Pi 5 + OAK-D Lite, design docs/HANDHELD_DEMO_DESIGN.md §4.11):
    handheld-runtime thread → SceneIngestion.process_detections → SpatialMemory
                 → POST /api/query        (text-first answers, TTS on request)
                 → GET /phone             (mobile page)
                 → /api/rooms /api/anchor /api/objects?room= /api/objects/changes
                 → /api/handheld/status /api/handheld/snapshot /api/memory/clear
                 → WebSocket /ws pushes "detections" (≤ LOCI_WS_PUSH_HZ) and "anchor" (1 Hz)

Environment variables:
  OPENAI_API_KEY   — enables Whisper STT, GPT-4o mini VLM, OpenAI TTS
  VLM_PROVIDER     — "openai" (default) | "gemini"
  GOOGLE_API_KEY   — enables Gemini VLM
  TTS_ENGINE       — "openai" (default) | "piper" | "edge"
  WHISPER_LOCAL    — "1" to use local faster-whisper instead of API
  PIPER_MODEL_PATH — path to .onnx Piper model file (if TTS_ENGINE=piper)
  CORS_ORIGINS     — comma-separated allowed origins (default: "*" for local dev;
                     set to your domain in production, e.g. "https://example.com")
  LOCI_*           — handheld knobs (design §4.12): LOCI_DETECTION_SOURCE none|sim|oak|replay,
                     LOCI_DATA_DIR, LOCI_PERSIST, LOCI_ROOMS_FILE, LOCI_LABEL_EMBEDDER,
                     LOCI_WS_PUSH_HZ, LOCI_VOICE none|headset, ... (see deploy/handheld/env.example)
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import random
import socket
import struct
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .handheld import (
    HandheldConfig,
    HandheldRuntime,
    VoiceLoopHandle,
    answer_text,
    build_handheld,
    camera_only_pose,
    load_rooms,
    make_embedder_safe,
    start_voice_loop,
    voice_enabled,
)
from .persistence import MemoryPersistence
from .room_anchor import RoomLocalizer
from .scene_ingestion import SceneIngestion
from .spatial_memory import SpatialMemory
from .vlm_client import VLMClient, sanitize_label
from .voice_pipeline import VoicePipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App state (construction order of design §4.11)
# ---------------------------------------------------------------------------

cfg = HandheldConfig.from_env()
_handheld_errors: dict[str, str] = {}

rooms = load_rooms(cfg)
_embedder, _embedder_error = make_embedder_safe(cfg)
if _embedder_error:
    _handheld_errors["embedder"] = _embedder_error
persistence: MemoryPersistence | None = (
    MemoryPersistence(
        cfg.data_dir,
        snapshot_interval_s=cfg.snapshot_interval_s,
        journal_fsync_s=cfg.journal_fsync_s,
        max_records=cfg.max_records,
    )
    if cfg.persist
    else None
)
memory = SpatialMemory(
    epoch_size_ms=5000,
    embedder=_embedder,
    rooms=rooms,
    dedup_radius_m=cfg.dedup_radius_m,
    moved_threshold_m=cfg.moved_threshold_m,
    persistence=persistence,
)
vlm_client = VLMClient()
ingestion = SceneIngestion(
    memory,
    vlm_client=vlm_client,
    use_vlm_fallback=True,
    consensus_refire_ms=cfg.consensus_refire_ms,
    consensus_radius_m=cfg.consensus_radius_m,
)
runtime: HandheldRuntime | None
localizer: RoomLocalizer | None
try:
    runtime, localizer = build_handheld(memory, ingestion, cfg)
except Exception as _exc:  # a bad LOCI_DETECTION_SOURCE must not take the API down
    runtime, localizer = None, None
    _handheld_errors["source"] = f"{type(_exc).__name__}: {_exc}"
    logger.error("handheld source disabled: %s", _handheld_errors["source"])
voice = VoicePipeline(
    memory,
    vlm_client=vlm_client,
    pose_provider=lambda: localizer.current() if localizer is not None else None,
)

_ws_subscribers: set[WebSocket] = set()
_restore_report: Any = None
_voice_loop: VoiceLoopHandle | None = None
_push_task: asyncio.Task | None = None
_started_at_ms = int(time.time() * 1000)

# In-memory store for the latest uploaded LiDAR point cloud
_point_cloud: dict = {"points": [], "count": 0, "filename": None}


def _anchor_dict() -> dict:
    """``CameraPose.to_dict()`` of the current pose (camera-only without a localizer)."""
    if localizer is not None:
        try:
            return localizer.current().to_dict()
        except Exception as exc:  # never let a pose error break a status call
            logger.debug("localizer.current failed: %s", exc)
    if runtime is not None:
        return runtime.current_pose().to_dict()
    return camera_only_pose().to_dict()


def _source_state() -> str | None:
    if runtime is None:
        return None
    return str(runtime.source_status().get("state"))


async def _broadcast(msg: dict) -> None:
    """Fan one JSON message out to every WebSocket subscriber, dropping dead sockets."""
    if not _ws_subscribers:
        return
    text = json.dumps(msg)
    dead: set[WebSocket] = set()
    for ws in list(_ws_subscribers):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    _ws_subscribers.difference_update(dead)


async def _live_push_loop() -> None:
    """Poll the runtime at ≤ ``LOCI_WS_PUSH_HZ`` and push ``detections`` / ``anchor`` (D17).

    Runs on the asyncio loop only; the runtime thread never touches asyncio.
    """
    period = 1.0 / max(float(cfg.ws_push_hz), 0.2)
    last_frames = -1
    last_anchor = 0.0
    while True:
        await asyncio.sleep(period)
        if not _ws_subscribers:
            continue
        try:
            if runtime is not None and runtime.frames != last_frames:
                frame = runtime.latest_frame
                if frame is not None:
                    last_frames = runtime.frames
                    await _broadcast(
                        {
                            "type": "detections",
                            "source": frame.source,
                            "timestamp_ms": frame.timestamp_ms,
                            "seq": frame.seq,
                            "detections": runtime.live_detections(),
                            "frame_count": ingestion.frame_count,
                            "observation_count": memory.observation_count,
                            "anchor": _anchor_dict(),
                        }
                    )
            now = time.monotonic()
            if now - last_anchor >= 1.0:
                last_anchor = now
                await _broadcast(
                    {"type": "anchor", **_anchor_dict(), "source_state": _source_state()}
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("live push failed: %s", exc)


def _start_voice() -> VoiceLoopHandle | None:
    if not voice_enabled():
        return None
    try:
        return start_voice_loop(
            lambda text: answer_text(voice, memory, text)["answer"], voice._stt, voice._tts
        )
    except Exception as exc:
        _handheld_errors["voice"] = f"{type(exc).__name__}: {exc}"
        logger.error("headset voice loop not started: %s", _handheld_errors["voice"])
        return None


def _voice_status() -> dict | None:
    if _voice_loop is not None:
        return _voice_loop.status()
    if "voice" in _handheld_errors:
        return {"running": False, "error": _handheld_errors["voice"]}
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _restore_report, _voice_loop, _push_task
    try:
        _restore_report = memory.restore()
    except Exception as exc:
        _handheld_errors["restore"] = f"{type(exc).__name__}: {exc}"
        logger.exception("memory restore failed: %s", exc)
    memory.set_vocabulary(runtime.source.classes if runtime is not None else ingestion.classes)
    if runtime is not None:
        runtime.start()
    _push_task = asyncio.create_task(_live_push_loop())
    _voice_loop = _start_voice()
    logger.info(
        "Spatial Memory Assistant starting — source: %s, rooms: %s, embedder: %s, "
        "restored: %s, VLM: %s, TTS: %s, STT: %s, voice: %s",
        cfg.detection_source,
        ",".join(rooms.names()),
        memory.embedder.name,
        _restore_report.records if _restore_report is not None else "n/a",
        vlm_client.is_available,
        voice.tts_engine,
        voice.stt_available,
        "on" if _voice_loop is not None else "off",
    )
    yield
    if _voice_loop is not None:
        _voice_loop.stop()
    if _push_task is not None:
        _push_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await _push_task
    if runtime is not None:
        runtime.stop()
    await ingestion.stop_capture()
    try:
        memory.close()
    except Exception as exc:
        logger.exception("memory close failed: %s", exc)
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

_cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
_cors_origins: list[str] = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
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


@app.get("/phone")
async def phone():
    """Mobile-first, offline page for the handheld demo (typed questions, aria-live answers)."""
    phone_path = os.path.join(_static_dir, "phone.html")
    if os.path.exists(phone_path):
        return FileResponse(phone_path, headers={"Cache-Control": "no-cache"})
    raise HTTPException(status_code=404, detail="phone.html not found")


@app.get("/api/scanner-qr")
async def scanner_qr(url: str = Query(..., description="Scanner URL to encode")):
    """Generate a scannable QR code PNG for the mobile scanner URL."""
    import segno  # optional dependency (demo_spatial/requirements.txt), not in the dev extras

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


class QueryRequest(BaseModel):
    text: str = Field(..., description="Natural language question, e.g. 'where is my mug'")
    tts: bool = Field(False, description="Also synthesize the answer (audio_b64 + audio_mime)")
    room: str | None = Field(None, description="Room to assume when the question names none")


class ConfirmRequest(BaseModel):
    confirm: bool = Field(False, description="Must be true to wipe the memory")


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
        "handheld": {
            "detection_source": cfg.detection_source,
            "source_state": _source_state(),
            "voice": _voice_loop is not None,
        },
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
    await _broadcast({"type": "detections", **result})
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
    await _broadcast({"type": "detections", **result})
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
# Detection class configuration (YOLO-World open-vocabulary)
# ---------------------------------------------------------------------------


class DetectionClassesRequest(BaseModel):
    classes: list[str] = Field(
        ...,
        description="List of object class names for open-vocabulary detection",
        min_length=1,
    )


@app.get("/api/detection/classes")
async def get_detection_classes():
    """Return the current open-vocabulary detection class list."""
    return {
        "classes": ingestion.classes,
        "count": len(ingestion.classes),
        "open_vocabulary": ingestion._is_world_model,
        "model": ingestion.status()["yolo_model"],
    }


@app.put("/api/detection/classes")
async def set_detection_classes(req: DetectionClassesRequest):
    """Update the open-vocabulary detection class list at runtime.

    Allows users to add custom objects to track (e.g. "insulin pen", "AirPods case").
    Only effective when using a YOLO-World model.
    """
    if not ingestion._is_world_model:
        raise HTTPException(
            status_code=400,
            detail="Class configuration requires a YOLO-World model. Set YOLO_MODEL env var.",
        )
    # Server-side hygiene: strip control chars, cap length, drop empties
    sanitized = [s for s in (sanitize_label(c) for c in req.classes) if s]
    if not sanitized:
        raise HTTPException(
            status_code=400,
            detail="No valid class names after sanitization.",
        )
    ingestion.set_classes(sanitized)
    return {
        "classes": ingestion.classes,
        "count": len(ingestion.classes),
    }


# ---------------------------------------------------------------------------
# Spatial query endpoints
# ---------------------------------------------------------------------------


@app.get("/api/objects")
async def list_objects(room: str | None = Query(None, description="Only this room")):
    """Return the latest known position of every tracked object (optionally per room)."""
    objects = memory.current_objects(room or None)
    return {
        "objects": [o.to_dict() for o in objects],
        "count": len(objects),
        "stats": memory.stats(),
        "room": room or None,
    }


@app.get("/api/objects/changes")
async def objects_changes(
    room: str | None = Query(None, description="Room name (all rooms when omitted)"),
    window_seconds: float = Query(600.0, gt=0.0, le=86400.0),
):
    """New / moved / gone / seen objects inside the window (design §7.5, D24)."""
    changes = memory.changes(room or None, window_seconds)
    return {
        "room": room or None,
        "window_seconds": window_seconds,
        "changes": [c.to_dict() for c in changes],
        "count": len(changes),
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
    results = memory.objects_in_region(req.x_min, req.x_max, req.y_min, req.y_max, limit=req.limit)
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
    return _voice_result_json(result)


@app.post("/api/voice/text-query")
async def text_query(req: TextQueryRequest):
    """Process a text query and return a spoken-style answer + TTS audio.

    Useful for testing without a microphone, or for frontend debugging.
    """
    result = await voice.handle_text_query(req.text)
    return _voice_result_json(result)


@app.post("/api/voice/tts")
async def text_to_speech(req: TextQueryRequest):
    """Convert text directly to audio bytes.

    The media type is sniffed from the bytes (``audio/wav`` for Piper, ``audio/mpeg``
    for OpenAI/edge) so the browser can play it directly.
    """
    audio = await voice._tts.synthesize(req.text)
    if not audio:
        raise HTTPException(
            status_code=503, detail="TTS synthesis failed — check TTS_ENGINE config"
        )
    return Response(content=audio, media_type=VoicePipeline.audio_mime(audio))


def _voice_result_json(result: VoicePipeline.QueryResult) -> dict:
    """Legacy voice response keys plus additive ``intent.room``, ``resolution``, ``audio_mime``."""
    audio = result.answer_audio
    return {
        "transcription": result.transcription,
        "intent": result.intent.to_dict(),
        "answer": result.answer_text,
        "has_audio": len(audio) > 0,
        "audio_b64": _b64(audio) if audio else None,
        "audio_mime": VoicePipeline.audio_mime(audio) if audio else None,
        "resolution": result.resolution.to_dict() if result.resolution is not None else None,
        "latency_ms": result.latency_ms,
    }


@app.post("/api/query")
async def query(req: QueryRequest):
    """Text-first question answering (design D22): TTS only when ``tts`` is true.

    Returns ``answer``, ``intent`` ({kind, object_name, room}), ``resolution``
    (label/how/score/suggestions or null), ``results`` (sightings used), ``changes``,
    ``latency_ms`` and ``audio_b64``/``audio_mime`` (null unless ``tts``).
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty question")
    result = answer_text(voice, memory, text, room=req.room)
    audio = b""
    if req.tts:
        try:
            audio = await voice._tts.synthesize(result["answer"])
        except Exception as exc:
            logger.warning("TTS failed for /api/query: %s", exc)
            audio = b""
    result["audio_b64"] = _b64(audio) if audio else None
    result["audio_mime"] = VoicePipeline.audio_mime(audio) if audio else None
    return result


# ---------------------------------------------------------------------------
# Handheld endpoints (rooms, anchor, status, persistence)
# ---------------------------------------------------------------------------


@app.get("/api/anchor")
async def anchor():
    """Current camera pose in the room frame (``CameraPose.to_dict()``)."""
    return _anchor_dict()


@app.get("/api/rooms")
async def list_rooms():
    """Configured rooms with the tags registered so far and the current pose."""
    loc_status = localizer.status() if localizer is not None else {}
    registered = loc_status.get("tags_registered", {}) or {}
    last_anchored = loc_status.get("last_anchored_ms", {}) or {}
    out = []
    for spec in rooms.rooms.values():
        d = spec.to_dict()
        d["tags_registered"] = list(registered.get(spec.name, [int(spec.anchor_tag)]))
        d["last_anchored_ms"] = last_anchored.get(spec.name)
        out.append(d)
    return {
        "rooms": out,
        "current": _anchor_dict() if (localizer is not None or runtime is not None) else None,
        "tag_size_m": float(rooms.tag_size_m),
        "tag_family": rooms.tag_family,
        "rooms_seen": memory.rooms_seen,
    }


@app.get("/api/handheld/status")
async def handheld_status():
    """Everything an operator needs: config, runtime/source, memory, restore, voice."""
    return {
        "config": cfg.to_dict(),
        "runtime": runtime.status() if runtime is not None else None,
        "memory": memory.stats(),
        "restore": _restore_report.to_dict() if _restore_report is not None else None,
        "started_at_ms": _started_at_ms,
        "voice": _voice_status(),
        "errors": _errors_dict(),
        "anchor": _anchor_dict(),
        "ws_subscribers": len(_ws_subscribers),
    }


def _errors_dict() -> dict[str, str]:
    """Startup errors plus live conditions an operator must see (D14: silent
    non-persistence is the one failure the status page must never hide)."""
    errors = dict(_handheld_errors)
    p = getattr(memory, "persistence", None)
    if p is not None and _restore_report is not None and getattr(p, "enabled", False):
        st = p.status()
        if not st.get("running"):
            errors.setdefault("persistence", f"writer not running ({st.get('errors', 0)} errors)")
        elif st.get("errors"):
            errors.setdefault(
                "persistence", f"{st['errors']} write error(s), see memory.persistence"
            )
    return errors


@app.post("/api/handheld/snapshot")
async def handheld_snapshot():
    """Write a memory snapshot now (``memory.flush()``)."""
    try:
        await asyncio.get_running_loop().run_in_executor(None, memory.flush)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"snapshot failed: {exc}") from exc
    return {
        "status": "ok",
        "persistence": memory.persistence.status() if memory.persistence is not None else None,
    }


@app.post("/api/memory/clear")
async def memory_clear(req: ConfirmRequest):
    """Forget everything (points, indices, persisted files). Requires ``{"confirm": true}``."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail='Send {"confirm": true} to clear the memory')
    await asyncio.get_running_loop().run_in_executor(None, memory.clear)
    return {"status": "cleared", "observation_count": memory.observation_count}


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
    Handheld additions (design §4.11): snapshots carry "rooms_seen" and "anchor";
    the live push loop sends "detections" (with "source", "seq", "anchor") when the
    runtime processed a frame and {"type": "anchor", ...CameraPose, "source_state"} 1 Hz.
    """
    await ws.accept()
    _ws_subscribers.add(ws)
    try:
        # Send initial snapshot
        snapshot = {
            "type": "snapshot",
            "objects": [o.to_dict() for o in memory.current_objects()],
            "observation_count": memory.observation_count,
            "rooms_seen": memory.rooms_seen,
            "anchor": _anchor_dict(),
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
                "rooms_seen": memory.rooms_seen,
                "anchor": _anchor_dict(),
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
        "handheld": runtime.status() if runtime is not None else None,
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
        [p[1] for p in props]

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
                "float": "f",
                "float32": "f",
                "double": "d",
                "float64": "d",
                "int": "i",
                "int32": "i",
                "uint": "I",
                "uint32": "I",
                "short": "h",
                "int16": "h",
                "ushort": "H",
                "uint16": "H",
                "char": "b",
                "int8": "b",
                "uchar": "B",
                "uint8": "B",
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
