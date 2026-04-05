"""Scene ingestion pipeline: camera frames → object detections → LOCI-DB.

Supports two modes:
  1. Frame push (default): frontend sends frames via HTTP/WebSocket
  2. Server-side capture: background thread reads from local webcam

Detection is done with YOLOv8-nano (ultralytics). Falls back to the
VLM client for low-confidence or ambiguous scenes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .spatial_memory import SpatialMemory
    from .vlm_client import VLMClient

# Minimum YOLO confidence to trust a detection without VLM fallback
_YOLO_CONF_THRESHOLD = 0.45
# If fewer than this many objects are detected, try VLM enrichment
_MIN_DETECTIONS_FOR_NO_VLM = 1


@dataclass
class Detection:
    """A single object detection from a camera frame."""

    label: str
    cx: float           # normalized [0, 1] — center x in frame
    cy: float           # normalized [0, 1] — center y in frame
    width: float        # normalized bbox width
    height: float       # normalized bbox height
    confidence: float
    source: str = "yolo"   # "yolo" | "vlm"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


class SceneIngestion:
    """Processes camera frames, extracts object detections, stores in LOCI-DB.

    Usage::

        ingestion = SceneIngestion(memory, vlm_client)
        detections = await ingestion.process_frame(jpeg_bytes)
    """

    def __init__(
        self,
        memory: "SpatialMemory",
        vlm_client: "VLMClient | None" = None,
        use_vlm_fallback: bool = True,
    ) -> None:
        self._memory = memory
        self._vlm = vlm_client
        self._use_vlm_fallback = use_vlm_fallback
        self._model = None   # lazy-loaded YOLO model
        self._server_capture_task: asyncio.Task | None = None
        self._capturing = False
        self.last_detections: list[Detection] = []
        self.frame_count = 0

    def _load_yolo(self):
        """Lazy-load YOLOv8 nano model (downloads ~6 MB on first run)."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolo11n.pt")
            logger.info("YOLO model loaded")
        except ImportError:
            logger.warning("ultralytics not installed — YOLO detection disabled")
            self._model = None

    def _run_yolo(self, image_bytes: bytes) -> list[Detection]:
        """Run YOLO inference on raw image bytes. Returns list of detections."""
        if self._model is None:
            return []
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = img.size

            results = self._model.predict(
                source=np.array(img),
                conf=_YOLO_CONF_THRESHOLD,
                verbose=False,
                stream=False,
            )
            detections: list[Detection] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    label = result.names[cls_id]
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    detections.append(Detection(
                        label=label, cx=cx, cy=cy,
                        width=bw, height=bh,
                        confidence=conf, source="yolo",
                    ))
            return detections
        except Exception as e:
            logger.error("YOLO inference error: %s", e)
            return []

    async def process_frame(
        self,
        image_bytes: bytes,
        timestamp_ms: int | None = None,
        use_vlm: bool | None = None,
    ) -> list[Detection]:
        """Process a single camera frame and store detections in LOCI-DB.

        Args:
            image_bytes: Raw JPEG/PNG image data.
            timestamp_ms: Override timestamp (defaults to now).
            use_vlm: Override VLM fallback for this frame.

        Returns:
            List of all detections (YOLO + optional VLM-enriched).
        """
        self._load_yolo()
        ts = timestamp_ms or int(time.time() * 1000)

        # Step 1: YOLO detection
        yolo_detections = await asyncio.get_event_loop().run_in_executor(
            None, self._run_yolo, image_bytes
        )

        # Step 2: VLM fallback if few/no YOLO detections and VLM is available
        vlm_detections: list[Detection] = []
        should_use_vlm = use_vlm if use_vlm is not None else self._use_vlm_fallback
        if (
            should_use_vlm
            and self._vlm is not None
            and len(yolo_detections) < _MIN_DETECTIONS_FOR_NO_VLM
        ):
            try:
                vlm_results = await self._vlm.describe_scene(image_bytes)
                for obj in vlm_results:
                    vlm_detections.append(Detection(
                        label=obj["label"],
                        cx=obj.get("cx", 0.5),
                        cy=obj.get("cy", 0.5),
                        width=obj.get("width", 0.1),
                        height=obj.get("height", 0.1),
                        confidence=obj.get("confidence", 0.7),
                        source="vlm",
                    ))
            except Exception as e:
                logger.warning("VLM scene description failed: %s", e)

        all_detections = yolo_detections + vlm_detections

        # Step 3: store each detection in spatial memory
        for det in all_detections:
            try:
                self._memory.observe(
                    label=det.label,
                    cx=det.cx,
                    cy=det.cy,
                    confidence=det.confidence,
                    timestamp_ms=ts,
                )
            except Exception as e:
                logger.error("Failed to store detection %s: %s", det.label, e)

        self.last_detections = all_detections
        self.frame_count += 1
        return all_detections

    async def process_frame_b64(self, b64_image: str, **kwargs) -> list[Detection]:
        """Convenience method for base64-encoded images from the browser."""
        # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_image)
        return await self.process_frame(image_bytes, **kwargs)

    # ------------------------------------------------------------------
    # Server-side webcam capture (optional)
    # ------------------------------------------------------------------

    async def start_capture(self, camera_index: int = 0, fps: float = 2.0) -> None:
        """Start background webcam capture loop."""
        if self._capturing:
            return
        self._capturing = True
        self._server_capture_task = asyncio.create_task(
            self._capture_loop(camera_index, fps)
        )
        logger.info("Server-side camera capture started (camera=%d, fps=%.1f)", camera_index, fps)

    async def stop_capture(self) -> None:
        """Stop background webcam capture."""
        self._capturing = False
        if self._server_capture_task and not self._server_capture_task.done():
            self._server_capture_task.cancel()
            try:
                await self._server_capture_task
            except asyncio.CancelledError:
                pass
        self._server_capture_task = None
        logger.info("Server-side camera capture stopped")

    async def _capture_loop(self, camera_index: int, fps: float) -> None:
        interval = 1.0 / fps
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python not installed — server-side capture unavailable")
            self._capturing = False
            return

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Could not open camera %d", camera_index)
            self._capturing = False
            return

        try:
            while self._capturing:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera frame read failed, retrying...")
                    await asyncio.sleep(0.1)
                    continue
                # Encode to JPEG bytes
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    await self.process_frame(bytes(buf))
                await asyncio.sleep(interval)
        finally:
            cap.release()

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    def status(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "capturing": self._capturing,
            "last_detections": [d.to_dict() for d in self.last_detections],
            "yolo_available": self._model is not None,
            "vlm_available": self._vlm is not None,
        }
