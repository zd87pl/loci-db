"""FastAPI routes for the ESP32 thin client (``/api/esp32/*`` and the ``/esp32`` page).

Wire-up from ``main.py`` is one call::

    esp32_hub = install_esp32_routes(
        app, ingestion=ingestion, memory=memory, voice=voice, subscribers=_ws_subscribers
    )

Protocol (full description in ``docs/ESP32_THIN_CLIENT_DEMO.md``):

* board → host  ``POST /api/esp32/frame`` — raw JPEG body (``image/jpeg``); optional headers
  ``X-Device-Id``, ``X-Seq``, ``X-Device-Ms``, ``X-Rssi``. The JSON response carries
  ``labels``, ``audio_pending`` and ``next_ms`` (the interval the host wants).
* board → host  ``GET /api/esp32/audio/next?device=&wait=`` — next clip for the speaker as
  ``audio/L16; rate=16000; channels=1`` (raw PCM16 mono), or ``204`` when nothing waits.
* board → host  ``POST /api/esp32/voice?rate=16000`` — one push-to-talk utterance as raw
  PCM16 (or any WAV/MP3/WebM); the spoken answer is queued for the board.
* anyone → host ``POST /api/esp32/ask`` ``{"text"}`` — typed question, answer spoken on the
  board; ``POST /api/esp32/say`` ``{"text"}`` — speak arbitrary text.
* debugging     ``GET /api/esp32/status``, ``/api/esp32/frame.jpg``, ``/api/esp32/stream.mjpg``,
  and the page at ``/esp32``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .esp32_source import Esp32Config, Esp32Hub, HostYoloDetector
from .esp32_voice import Esp32VoiceBridge
from .pcm_audio import decode_to_pcm16, pcm16_duration_s, resample_pcm16, sniff_mime

logger = logging.getLogger(__name__)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


class AskRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Question, e.g. 'where is my cup'")
    device: str | None = Field(None, description="Board to speak the answer on")
    speak: bool = Field(True, description="Queue the spoken answer for the board")


class SayRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to speak on the board")
    device: str | None = None


def _header_int(request: Request, name: str) -> int | None:
    raw = request.headers.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def install_esp32_routes(
    app: FastAPI,
    *,
    ingestion: Any,
    memory: Any,
    voice: Any,
    subscribers: Iterable[Any] | None = None,
    cfg: Esp32Config | None = None,
    hub: Esp32Hub | None = None,
    bridge: Esp32VoiceBridge | None = None,
) -> Esp32Hub:
    """Mount the ESP32 endpoints on ``app`` and return the hub (also on ``app.state``).

    ``subscribers`` is the demo's live WebSocket set: processed frames are fanned out to
    it as ``{"type": "detections", "source": "esp32", ...}`` so the browser page draws
    the board's detections like its own camera's. ``hub``/``bridge`` are injectable for
    tests (fake detector, fake STT/TTS); by default a :class:`HostYoloDetector` and the
    pipeline's own STT/TTS are used.
    """
    cfg = cfg or Esp32Config.from_env()
    if hub is None:
        hub = Esp32Hub(ingestion, cfg)
    if bridge is None:
        bridge = Esp32VoiceBridge(voice, rate=cfg.audio_rate, gain=cfg.audio_gain)
    router = APIRouter(tags=["esp32"])
    rate = cfg.audio_rate

    async def _broadcast(msg: dict) -> None:
        if not subscribers:
            return
        text = json.dumps(msg)
        dead = []
        for ws in list(subscribers):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        if dead and hasattr(subscribers, "discard"):
            for ws in dead:
                subscribers.discard(ws)  # type: ignore[union-attr]

    def _device(request: Request, device: str | None) -> str:
        return device or request.headers.get("x-device-id") or cfg.default_device

    # -- frames ------------------------------------------------------------------

    @router.post("/api/esp32/frame")
    async def esp32_frame(
        request: Request,
        device: str | None = Query(None, description="Board id (or X-Device-Id header)"),
        seq: int | None = Query(None, description="Frame counter (or X-Seq header)"),
    ):
        """One JPEG frame from the board; returns labels and whether audio is waiting."""
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file") or form.get("image") or form.get("frame")
            body = await upload.read() if upload is not None and hasattr(upload, "read") else b""
        else:
            body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty frame")
        device_id = _device(request, device)
        try:
            result = await hub.ingest_frame(
                body,
                device_id=device_id,
                seq=seq if seq is not None else _header_int(request, "x-seq"),
                device_ms=_header_int(request, "x-device-ms"),
                rssi=_header_int(request, "x-rssi"),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if not result.dropped:
            await _broadcast(
                {
                    "type": "detections",
                    "source": "esp32",
                    "device": device_id,
                    "timestamp_ms": result.timestamp_ms,
                    "seq": result.seq,
                    "detections": [d.to_dict() for d in result.detections],
                    "frame_count": getattr(ingestion, "frame_count", None),
                    "observation_count": getattr(memory, "observation_count", None),
                }
            )
        return result.to_dict()

    @router.get("/api/esp32/frame.jpg")
    async def esp32_frame_jpg(device: str | None = None):
        """Latest JPEG received from the board (for checking framing and orientation)."""
        jpeg = hub.latest_jpeg(device)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no frame received yet")
        return Response(
            content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @router.get("/api/esp32/stream.mjpg")
    async def esp32_stream(device: str | None = None, fps: float = Query(5.0, gt=0, le=15)):
        """MJPEG preview of the board's frames, as received (before rotation)."""
        device_id = device or cfg.default_device
        period = 1.0 / fps

        async def gen():
            last: bytes | None = None
            while True:
                jpeg = hub.latest_jpeg(device_id)
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(period)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    # -- audio -------------------------------------------------------------------

    @router.get("/api/esp32/audio/next")
    async def esp32_audio_next(
        device: str | None = None,
        wait: float = Query(0.0, ge=0.0, le=25.0, description="Long-poll seconds"),
    ):
        """Next queued PCM clip for the board's speaker, or 204 when nothing is waiting."""
        device_id = device or cfg.default_device
        pcm = await hub.next_audio(device_id, wait_s=wait)
        if pcm is None:
            return Response(status_code=204)
        return Response(
            content=pcm,
            media_type=f"audio/L16; rate={rate}; channels=1",
            headers={
                "X-Audio-Rate": str(rate),
                "X-Audio-Channels": "1",
                "X-Audio-Format": "s16le",
                "X-Audio-Seconds": f"{pcm16_duration_s(pcm, rate):.2f}",
                "X-Audio-Queued": str(hub.pending_audio(device_id)),
                "Cache-Control": "no-store",
            },
        )

    @router.post("/api/esp32/say")
    async def esp32_say(req: SayRequest):
        """Speak ``text`` on the board (TTS → PCM → queue)."""
        pcm = await bridge.say(req.text)
        if not pcm:
            raise HTTPException(status_code=503, detail=f"TTS unavailable: {bridge.last_error}")
        device_id = req.device or cfg.default_device
        queued = hub.queue_audio(pcm, device_id)
        return {
            "status": "queued",
            "device": device_id,
            "audio_seconds": round(pcm16_duration_s(pcm, rate), 2),
            "queue_len": queued,
        }

    # -- questions ---------------------------------------------------------------

    @router.post("/api/esp32/ask")
    async def esp32_ask(req: AskRequest):
        """Typed question answered from spatial memory; the answer is spoken on the board."""
        answer = await bridge.answer_text(req.text, speak=req.speak)
        device_id = req.device or cfg.default_device
        queued = (
            hub.queue_audio(answer.pcm, device_id) if answer.pcm else hub.pending_audio(device_id)
        )
        return {
            **answer.to_dict(),
            "device": device_id,
            "audio_queued": bool(answer.pcm),
            "queue_len": queued,
        }

    @router.post("/api/esp32/voice")
    async def esp32_voice(
        request: Request,
        device: str | None = None,
        rate_in: int = Query(16000, alias="rate", ge=8000, le=48000),
        language: str = "en",
        speak: bool = True,
    ):
        """One push-to-talk utterance (raw PCM16 mono at ``rate``, or any WAV/MP3/WebM)."""
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty audio")
        device_id = _device(request, device)
        mime = sniff_mime(body)
        try:
            if mime == "application/octet-stream":
                pcm = resample_pcm16(body, rate_in, rate) if rate_in != rate else body
            else:
                pcm = decode_to_pcm16(body, rate)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"undecodable audio: {e}") from e
        answer = await bridge.answer_pcm(pcm, language=language, speak=speak)
        queued = (
            hub.queue_audio(answer.pcm, device_id) if answer.pcm else hub.pending_audio(device_id)
        )
        return {
            **answer.to_dict(),
            "device": device_id,
            "seconds_in": round(pcm16_duration_s(pcm, rate), 2),
            "audio_queued": bool(answer.pcm),
            "queue_len": queued,
        }

    # -- status and page ---------------------------------------------------------

    @router.get("/api/esp32/status")
    async def esp32_status():
        return {
            **hub.status(),
            "voice": bridge.status(),
            "memory": {
                "observation_count": getattr(memory, "observation_count", None),
                "tracked_labels": list(getattr(memory, "tracked_labels", []) or []),
            },
        }

    @router.get("/esp32")
    async def esp32_page():
        page = os.path.join(_STATIC_DIR, "esp32.html")
        if os.path.exists(page):
            return FileResponse(page, headers={"Cache-Control": "no-cache"})
        return {"message": "ESP32 thin client", "status": "/api/esp32/status"}

    app.include_router(router)
    app.state.esp32_hub = hub
    app.state.esp32_voice = bridge

    detector = hub.detector
    if cfg.warmup and isinstance(detector, HostYoloDetector):
        threading.Thread(target=detector.load, name="esp32-yolo-warmup", daemon=True).start()
    logger.info(
        "ESP32 thin-client routes mounted (model=%s, max_fps=%s, audio=%d Hz)",
        cfg.yolo_model,
        cfg.max_fps,
        cfg.audio_rate,
    )
    return hub
