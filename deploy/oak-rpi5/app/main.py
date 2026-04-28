"""LOCI OAK-D Lite + Raspberry Pi 5 assistive demo — main entry point.

Runs the full assistive pipeline:
  1. OAK-D Lite captures RGB + stereo depth + on-device YOLO detections
  2. Detections pass through temporal consensus and are stored in LOCI memory
  3. Voice queries (via Bluetooth headphones) search spatial memory
  4. Spoken responses describe object locations in natural language

Can run in two modes:
  - Headless (default): voice-only via Bluetooth headphones
  - Server: FastAPI with REST/WebSocket endpoints for debugging

Usage:
  python -m app.main                    # headless voice loop
  python -m app.main --server           # FastAPI debug server
  python -m app.main --server --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path

import yaml

logger = logging.getLogger("loci.oak_demo")

# Resolve project root for config and models
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load config.yaml, falling back to defaults."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Headless voice loop (primary demo mode)
# ---------------------------------------------------------------------------

async def run_headless(config: dict) -> None:
    """Run the assistive demo in headless mode (voice-only, no server)."""
    from .audio_io import AudioIO
    from .oak_pipeline import OakPipeline
    from .scene_ingestion import DEFAULT_CLASSES, SceneIngestion
    from .spatial_memory import SpatialMemory
    from .voice_pipeline import VoicePipeline

    # Initialize components
    classes = config.get("detection", {}).get("classes", DEFAULT_CLASSES)
    blob_path = config.get("oak", {}).get("blob_path", str(PROJECT_ROOT / "models" / "yolov8n.blob"))
    rgb_fps = config.get("oak", {}).get("rgb_fps", 8.0)
    nn_confidence = config.get("detection", {}).get("confidence", 0.5)

    memory = SpatialMemory(
        epoch_size_ms=config.get("loci", {}).get("epoch_size_ms", 5000),
    )
    ingestion = SceneIngestion(memory, classes=classes)
    voice = VoicePipeline(memory)
    audio = AudioIO()

    oak = OakPipeline(
        blob_path=blob_path,
        classes=classes,
        rgb_fps=rgb_fps,
        nn_confidence_threshold=nn_confidence,
    )

    # Start OAK-D camera
    logger.info("Starting OAK-D Lite pipeline...")
    try:
        oak.start()
    except Exception as e:
        logger.error("Failed to start OAK-D: %s", e)
        logger.info("Continuing without camera — voice queries will work but no new detections")
        oak = None

    # Audio feedback: system ready
    ready_msg = "LOCI is ready. I'm watching for objects. Ask me where something is."
    ready_audio = await voice._tts.synthesize(ready_msg)
    await audio.play(ready_audio)

    # Run camera ingestion and voice loop concurrently
    shutdown = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    camera_task = asyncio.create_task(_camera_loop(oak, ingestion, shutdown))
    voice_task = asyncio.create_task(_voice_loop(voice, audio, shutdown))

    await shutdown.wait()

    camera_task.cancel()
    voice_task.cancel()

    # Cleanup
    if oak is not None:
        oak.stop()
    audio.cleanup()

    logger.info(
        "Session complete. Tracked %d objects, processed %d frames.",
        len(memory.tracked_labels),
        ingestion.frame_count,
    )


async def _camera_loop(oak, ingestion, shutdown: asyncio.Event) -> None:
    """Background task: continuously process OAK-D frames."""
    if oak is None:
        return
    try:
        loop = asyncio.get_event_loop()
        while not shutdown.is_set():
            frame = await loop.run_in_executor(None, _get_next_frame, oak)
            if frame is not None:
                ingestion.process_oak_frame(frame)
    except asyncio.CancelledError:
        pass


def _get_next_frame(oak):
    """Get the next frame from the OAK pipeline (blocking, runs in executor)."""
    for frame in oak.frames():
        return frame
    return None


async def _voice_loop(voice, audio, shutdown: asyncio.Event) -> None:
    """Background task: listen for voice queries and respond."""
    logger.info("Voice loop started — listening for queries via Bluetooth mic")
    try:
        while not shutdown.is_set():
            # Record until silence
            audio_bytes = await audio.record_until_silence()
            if not audio_bytes or len(audio_bytes) < 1000:
                continue

            # Process voice query
            result = await voice.handle_audio_query(audio_bytes)
            if result.transcription:
                logger.info(
                    "Query: '%s' -> %s (%.0fms)",
                    result.transcription,
                    result.intent.kind,
                    result.latency_ms,
                )

            # Play response
            if result.answer_audio:
                await audio.play(result.answer_audio)
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# FastAPI server mode (debug/demo)
# ---------------------------------------------------------------------------

def create_server_app(config: dict):
    """Create the FastAPI application for debug/demo mode."""
    from fastapi import FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware

    from .oak_pipeline import OakPipeline
    from .scene_ingestion import DEFAULT_CLASSES, SceneIngestion
    from .spatial_memory import SpatialMemory
    from .voice_pipeline import VoicePipeline

    classes = config.get("detection", {}).get("classes", DEFAULT_CLASSES)
    blob_path = config.get("oak", {}).get("blob_path", str(PROJECT_ROOT / "models" / "yolov8n.blob"))

    memory = SpatialMemory(
        epoch_size_ms=config.get("loci", {}).get("epoch_size_ms", 5000),
    )
    ingestion = SceneIngestion(memory, classes=classes)
    voice = VoicePipeline(memory)
    oak = OakPipeline(blob_path=blob_path, classes=classes)

    _oak_task = None
    _shutdown = asyncio.Event()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        nonlocal _oak_task
        try:
            oak.start()
            _oak_task = asyncio.create_task(_camera_loop(oak, ingestion, _shutdown))
            logger.info("OAK-D pipeline started in server mode")
        except Exception as e:
            logger.warning("OAK-D not available: %s", e)
        yield
        _shutdown.set()
        if _oak_task:
            _oak_task.cancel()
        oak.stop()

    app = FastAPI(title="LOCI OAK-D Lite Demo", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "observation_count": memory.observation_count,
            "tracked_objects": len(memory.tracked_labels),
            "frame_count": ingestion.frame_count,
            "oak_running": oak.is_running,
            "tts_engine": voice.tts_engine,
            "stt_available": voice.stt_available,
        }

    @app.get("/api/objects")
    async def list_objects():
        return [o.to_dict() for o in memory.current_objects()]

    @app.get("/api/objects/{label}/location")
    async def object_location(label: str):
        results = memory.where_is(label, limit=1)
        if not results:
            return {"error": f"Object '{label}' not found"}
        return results[0].to_dict()

    @app.get("/api/objects/{label}/history")
    async def object_history(label: str, limit: int = 10):
        return [o.to_dict() for o in memory.history(label, limit=limit)]

    @app.post("/api/voice/text-query")
    async def text_query(text: str):
        result = await voice.handle_text_query(text)
        return {
            "transcription": result.transcription,
            "intent": {"kind": result.intent.kind, "object": result.intent.object_name},
            "answer": result.answer_text,
            "latency_ms": result.latency_ms,
        }

    @app.post("/api/voice/query")
    async def voice_query(audio: bytes):
        result = await voice.handle_audio_query(audio)
        return Response(
            content=result.answer_audio,
            media_type="audio/wav",
            headers={
                "X-Transcription": result.transcription,
                "X-Answer": result.answer_text,
                "X-Latency-Ms": str(result.latency_ms),
            },
        )

    @app.get("/api/stats")
    async def stats():
        return {
            "memory": memory.stats(),
            "ingestion": ingestion.status(),
            "voice": voice.status(),
            "oak": oak.status(),
        }

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LOCI OAK-D Lite assistive demo")
    parser.add_argument("--server", action="store_true", help="Run in FastAPI server mode")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()

    if args.server:
        import uvicorn
        app = create_server_app(config)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        asyncio.run(run_headless(config))


if __name__ == "__main__":
    main()
