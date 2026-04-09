"""Scene ingestion pipeline: camera frames → object detections → LOCI-DB.

Supports two modes:
  1. Frame push (default): frontend sends frames via HTTP/WebSocket
  2. Server-side capture: background thread reads from local webcam

Detection uses YOLO-World (open-vocabulary) by default, falling back to
YOLOv8-nano when YOLO_MODEL is overridden. The VLM client provides
enrichment for low-confidence or ambiguous scenes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .spatial_memory import SpatialMemory
    from .vlm_client import VLMClient

# Minimum YOLO confidence to trust a detection without VLM fallback
_YOLO_CONF_THRESHOLD = 0.65
# If fewer than this many objects are detected, try VLM enrichment
_MIN_DETECTIONS_FOR_NO_VLM = 1

# YOLO-World model (open-vocabulary). Override with YOLO_MODEL env var.
_YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8l-worldv2.pt")

# Default open-vocabulary class list for assistive use.
# Users can update at runtime via PUT /api/detection/classes.
DEFAULT_CLASSES: list[str] = [
    "person", "keys", "wallet", "phone", "laptop", "cup", "bottle",
    "bag", "backpack", "glasses", "remote control", "charger",
    "headphones", "AirPods", "watch", "book", "pen", "medicine",
    "cane", "shoe", "jacket", "umbrella", "food container",
    "water bottle", "mug", "plate", "bowl", "fork", "knife", "spoon",
    "chair", "table", "door", "cat", "dog",
]


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
    depth_m: float | None = None  # LiDAR depth in meters (None if unavailable)

    def to_dict(self) -> dict:
        result = {
            "label": self.label,
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }
        if self.depth_m is not None:
            result["depth_m"] = round(self.depth_m, 3)
        return result


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
        classes: list[str] | None = None,
        consensus_window_ms: int = 1000,
        consensus_min_count: int = 2,
        consensus_iou_threshold: float = 0.4,
    ) -> None:
        self._memory = memory
        self._vlm = vlm_client
        self._use_vlm_fallback = use_vlm_fallback
        self._model = None   # lazy-loaded YOLO model
        self._classes: list[str] = list(classes or DEFAULT_CLASSES)
        self._is_world_model = "world" in _YOLO_MODEL.lower()
        self._server_capture_task: asyncio.Task | None = None
        self._capturing = False
        self.last_detections: list[Detection] = []
        self.frame_count = 0
        # Temporal consensus buffer: per-label deque of (cx, cy, confidence, timestamp_ms)
        self._consensus_buffer: dict[str, deque[tuple[float, float, float, int]]] = {}
        self.consensus_window_ms = consensus_window_ms
        self.consensus_min_count = consensus_min_count
        self.consensus_iou_threshold = consensus_iou_threshold

    def _load_yolo(self):
        """Lazy-load YOLO model. Uses YOLO-World (open-vocab) by default."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(_YOLO_MODEL)
            if self._is_world_model:
                self._model.set_classes(self._classes)
                logger.info(
                    "YOLO-World loaded (%s) with %d classes",
                    _YOLO_MODEL, len(self._classes),
                )
            else:
                logger.info("YOLO model loaded (%s)", _YOLO_MODEL)
        except ImportError:
            logger.warning("ultralytics not installed — YOLO detection disabled")
            self._model = None

    @property
    def classes(self) -> list[str]:
        """Return the current open-vocabulary class list."""
        return list(self._classes)

    def set_classes(self, classes: list[str]) -> None:
        """Update the open-vocabulary class list at runtime.

        Only effective when using a YOLO-World model.
        """
        self._classes = list(classes)
        if self._model is not None and self._is_world_model:
            self._model.set_classes(self._classes)
            logger.info("YOLO-World classes updated: %d classes", len(self._classes))

    def _run_yolo(self, image_bytes: bytes, conf_threshold: float = _YOLO_CONF_THRESHOLD) -> list[Detection]:
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
                conf=conf_threshold,
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

    @staticmethod
    def _iou(cx1: float, cy1: float, cx2: float, cy2: float, box_size: float = 0.1) -> float:
        """Compute IoU between two boxes using a fixed-size proxy bbox (cx, cy, w, h)."""
        half = box_size / 2
        ax1, ay1, ax2, ay2 = cx1 - half, cy1 - half, cx1 + half, cy1 + half
        bx1, by1, bx2, by2 = cx2 - half, cy2 - half, cx2 + half, cy2 + half
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        box_area = box_size * box_size
        union_area = 2 * box_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _update_consensus(
        self,
        label: str,
        cx: float,
        cy: float,
        confidence: float,
        timestamp_ms: int,
        depth_m: float | None,
    ) -> None:
        """Buffer a detection and call observe() when temporal consensus is reached.

        A detection is considered the same object as a buffered entry when:
        - same label (keyed by label)
        - IoU > consensus_iou_threshold (spatial overlap)
        - within consensus_window_ms (temporal window)

        On reaching consensus_min_count confirmations, observe() is called with
        the averaged cx/cy and the maximum confidence of all matching entries.
        """
        if label not in self._consensus_buffer:
            self._consensus_buffer[label] = deque()

        buf = self._consensus_buffer[label]

        # Evict entries outside the rolling window
        cutoff = timestamp_ms - self.consensus_window_ms
        while buf and buf[0][3] < cutoff:
            buf.popleft()

        # Find entries that spatially overlap with the new detection
        matching_idxs = [
            i for i, (bcx, bcy, _, _ts) in enumerate(buf)
            if self._iou(cx, cy, bcx, bcy) > self.consensus_iou_threshold
        ]

        if len(matching_idxs) + 1 >= self.consensus_min_count:
            # Consensus reached — compute averaged position and max confidence
            matching_entries = [buf[i] for i in matching_idxs]
            all_cx = [e[0] for e in matching_entries] + [cx]
            all_cy = [e[1] for e in matching_entries] + [cy]
            all_conf = [e[2] for e in matching_entries] + [confidence]
            avg_cx = sum(all_cx) / len(all_cx)
            avg_cy = sum(all_cy) / len(all_cy)
            max_conf = max(all_conf)
            try:
                self._memory.observe(
                    label=label,
                    cx=avg_cx,
                    cy=avg_cy,
                    confidence=max_conf,
                    timestamp_ms=timestamp_ms,
                    depth_m=depth_m,
                )
            except Exception as e:
                logger.error("Failed to store consensus detection %s: %s", label, e)
        else:
            # Not enough confirmations yet — buffer this detection
            buf.append((cx, cy, confidence, timestamp_ms))

    @staticmethod
    def _lookup_depth(
        cx: float, cy: float, depth_samples: dict[str, float] | None
    ) -> float | None:
        """Estimate depth at (cx, cy) by interpolating the nearest LiDAR grid sample.

        The depth_samples dict maps 3x3 grid keys (tl, tc, tr, ml, mc, mr, bl, bc, br)
        to depth values in meters. We pick the nearest sample to the detection center.
        """
        if not depth_samples:
            return None
        grid_positions = {
            "tl": (0.25, 0.25), "tc": (0.5, 0.25), "tr": (0.75, 0.25),
            "ml": (0.25, 0.5),  "mc": (0.5, 0.5),  "mr": (0.75, 0.5),
            "bl": (0.25, 0.75), "bc": (0.5, 0.75), "br": (0.75, 0.75),
        }
        best_key = None
        best_dist = float("inf")
        for key, (gx, gy) in grid_positions.items():
            if key in depth_samples:
                dist = (cx - gx) ** 2 + (cy - gy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_key = key
        return depth_samples.get(best_key) if best_key else None

    async def process_frame(
        self,
        image_bytes: bytes,
        timestamp_ms: int | None = None,
        use_vlm: bool | None = None,
        depth_samples: dict[str, float] | None = None,
    ) -> list[Detection]:
        """Process a single camera frame and store detections in LOCI-DB.

        Args:
            image_bytes: Raw JPEG/PNG image data.
            timestamp_ms: Override timestamp (defaults to now).
            use_vlm: Override VLM fallback for this frame.
            depth_samples: Optional LiDAR depth grid from the 3D scanner.

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
                # Stage 2 — run YOLO without threshold for corroboration
                raw_yolo = await asyncio.get_event_loop().run_in_executor(
                    None, self._run_yolo, image_bytes, 0.01
                )
                raw_by_label: dict[str, list[Detection]] = {}
                for d in raw_yolo:
                    raw_by_label.setdefault(d.label, []).append(d)

                for obj in vlm_results:
                    lbl = obj["label"]
                    raw_conf = obj.get("confidence", 0.50)  # Stage 1/3: VLM-provided or fallback
                    width = obj.get("width", 0.3)
                    height = obj.get("height", 0.3)
                    vlm_cx = obj.get("cx", 0.5)
                    vlm_cy = obj.get("cy", 0.5)

                    # Stage 2: apply skepticism multiplier when YOLO sees nothing for this label
                    yolo_corroborated = any(
                        self._iou(vlm_cx, vlm_cy, d.cx, d.cy) > 0.3
                        for d in raw_by_label.get(lbl, [])
                    )
                    if not yolo_corroborated:
                        raw_conf *= 0.6

                    vlm_detections.append(Detection(
                        label=lbl,
                        cx=vlm_cx,
                        cy=vlm_cy,
                        width=width,
                        height=height,
                        confidence=raw_conf,
                        source="vlm",
                    ))
            except Exception as e:
                logger.warning("VLM scene description failed: %s", e)

        all_detections = yolo_detections + vlm_detections

        # Step 2.5: Assign depth from LiDAR samples to each detection
        if depth_samples:
            for det in all_detections:
                det.depth_m = self._lookup_depth(det.cx, det.cy, depth_samples)

        # Step 3: run each detection through the temporal consensus buffer
        for det in all_detections:
            self._update_consensus(
                label=det.label,
                cx=det.cx,
                cy=det.cy,
                confidence=det.confidence,
                timestamp_ms=ts,
                depth_m=det.depth_m,
            )

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
            "yolo_model": _YOLO_MODEL,
            "open_vocabulary": self._is_world_model,
            "classes": self._classes if self._is_world_model else None,
            "vlm_available": self._vlm is not None,
        }
