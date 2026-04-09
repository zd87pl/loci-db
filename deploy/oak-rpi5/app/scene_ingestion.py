"""Scene ingestion for OAK-D Lite: detections -> temporal consensus -> LOCI-DB.

Processes OakFrame outputs from the DepthAI pipeline, applies temporal
consensus buffering to reduce false positives, and stores confirmed
observations in LOCI spatial memory.

Two modes:
  1. On-device NN: Detections come pre-computed from OAK-D Myriad X VPU
  2. Host-side fallback: Runs YOLO on RPi5 CPU when NN blob is unavailable
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .oak_pipeline import OakDetection, OakFrame
    from .spatial_memory import SpatialMemory

# Confidence threshold for host-side YOLO fallback
_HOST_YOLO_CONF = float(os.environ.get("YOLO_CONFIDENCE", "0.5"))
_HOST_YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

# Default assistive object classes for OAK-D Lite demo
DEFAULT_CLASSES: list[str] = [
    "person", "keys", "wallet", "phone", "laptop", "cup", "bottle",
    "bag", "backpack", "glasses", "remote control", "charger",
    "headphones", "watch", "book", "pen", "medicine",
    "cane", "shoe", "jacket", "umbrella", "water bottle",
    "mug", "plate", "bowl", "fork", "knife", "spoon",
    "chair", "table", "door", "cat", "dog",
]


@dataclass
class Detection:
    """A single object detection (unified from OAK NN or host YOLO)."""

    label: str
    cx: float           # normalized [0, 1]
    cy: float           # normalized [0, 1]
    width: float
    height: float
    confidence: float
    source: str = "oak_nn"  # "oak_nn" | "host_yolo"
    depth_m: float | None = None

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
    """Processes OAK-D frames and stores detections in LOCI spatial memory.

    Usage::

        ingestion = SceneIngestion(memory, classes=DEFAULT_CLASSES)
        detections = ingestion.process_oak_frame(oak_frame)
    """

    def __init__(
        self,
        memory: "SpatialMemory",
        classes: list[str] | None = None,
        consensus_window_ms: int = 1000,
        consensus_min_count: int = 2,
        consensus_iou_threshold: float = 0.4,
    ) -> None:
        self._memory = memory
        self._classes: list[str] = list(classes or DEFAULT_CLASSES)
        self._host_model = None  # lazy-loaded for fallback
        self.last_detections: list[Detection] = []
        self.frame_count = 0
        # Temporal consensus buffer: per-label deque of (cx, cy, conf, ts, depth_m)
        self._consensus_buffer: dict[str, deque[tuple[float, float, float, int, float | None]]] = {}
        self.consensus_window_ms = consensus_window_ms
        self.consensus_min_count = consensus_min_count
        self.consensus_iou_threshold = consensus_iou_threshold

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def set_classes(self, classes: list[str]) -> None:
        """Update the detection class list at runtime."""
        self._classes = list(classes)
        logger.info("Detection classes updated: %d classes", len(self._classes))

    def process_oak_frame(self, frame: "OakFrame") -> list[Detection]:
        """Process a frame from the OAK-D pipeline.

        If the frame has on-device NN detections, uses those directly.
        Otherwise falls back to host-side YOLO inference.
        """
        if frame.detections:
            detections = [
                Detection(
                    label=d.label, cx=d.cx, cy=d.cy,
                    width=d.width, height=d.height,
                    confidence=d.confidence,
                    source="oak_nn",
                    depth_m=d.depth_m,
                )
                for d in frame.detections
            ]
        else:
            detections = self._run_host_yolo(frame.rgb, frame.depth_map)

        # Run through temporal consensus buffer
        for det in detections:
            self._update_consensus(
                label=det.label,
                cx=det.cx,
                cy=det.cy,
                confidence=det.confidence,
                timestamp_ms=frame.timestamp_ms,
                depth_m=det.depth_m,
            )

        self.last_detections = detections
        self.frame_count += 1
        return detections

    def _run_host_yolo(
        self,
        rgb_frame: np.ndarray,
        depth_map: np.ndarray | None,
    ) -> list[Detection]:
        """Fallback: run YOLO on RPi5 CPU when no on-device NN is available."""
        if self._host_model is None:
            try:
                from ultralytics import YOLO
                self._host_model = YOLO(_HOST_YOLO_MODEL)
                if "world" in _HOST_YOLO_MODEL.lower():
                    self._host_model.set_classes(self._classes)
                logger.info("Host-side YOLO loaded: %s", _HOST_YOLO_MODEL)
            except ImportError:
                logger.error("ultralytics not installed — no detection available")
                return []

        try:
            h, w = rgb_frame.shape[:2]
            results = self._host_model.predict(
                source=rgb_frame,
                conf=_HOST_YOLO_CONF,
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

                    # Look up depth at detection center
                    depth_m = None
                    if depth_map is not None:
                        depth_m = self._depth_at_center(depth_map, cx, cy)

                    detections.append(Detection(
                        label=label, cx=cx, cy=cy,
                        width=bw, height=bh,
                        confidence=conf, source="host_yolo",
                        depth_m=depth_m,
                    ))
            return detections
        except Exception as e:
            logger.error("Host YOLO inference error: %s", e)
            return []

    @staticmethod
    def _depth_at_center(
        depth_map: np.ndarray,
        cx: float,
        cy: float,
    ) -> float | None:
        """Look up median depth at a normalized point from stereo depth map."""
        h, w = depth_map.shape[:2]
        px, py = int(cx * w), int(cy * h)
        k = 3
        y0, y1 = max(0, py - k), min(h, py + k + 1)
        x0, x1 = max(0, px - k), min(w, px + k + 1)
        patch = depth_map[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None
        depth_m = float(np.median(valid)) / 1000.0
        if depth_m < 0.3 or depth_m > 4.0:
            return None
        return round(depth_m, 3)

    @staticmethod
    def _iou(cx1: float, cy1: float, cx2: float, cy2: float, box_size: float = 0.1) -> float:
        """IoU between two proxy bounding boxes."""
        half = box_size / 2
        ax1, ay1, ax2, ay2 = cx1 - half, cy1 - half, cx1 + half, cy1 + half
        bx1, by1, bx2, by2 = cx2 - half, cy2 - half, cx2 + half, cy2 + half
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h
        box_area = box_size * box_size
        union = 2 * box_area - inter_area
        return inter_area / union if union > 0 else 0.0

    def _update_consensus(
        self,
        label: str,
        cx: float,
        cy: float,
        confidence: float,
        timestamp_ms: int,
        depth_m: float | None,
    ) -> None:
        """Buffer detection and store to LOCI when temporal consensus is reached."""
        if label not in self._consensus_buffer:
            self._consensus_buffer[label] = deque()

        buf = self._consensus_buffer[label]

        # Evict old entries
        cutoff = timestamp_ms - self.consensus_window_ms
        while buf and buf[0][3] < cutoff:
            buf.popleft()

        # Find spatial matches in the buffer
        matching_idxs = [
            i for i, (bcx, bcy, _, _ts, _d) in enumerate(buf)
            if self._iou(cx, cy, bcx, bcy) > self.consensus_iou_threshold
        ]

        if len(matching_idxs) + 1 >= self.consensus_min_count:
            # Consensus reached — average position, max confidence
            entries = [buf[i] for i in matching_idxs]
            all_cx = [e[0] for e in entries] + [cx]
            all_cy = [e[1] for e in entries] + [cy]
            all_conf = [e[2] for e in entries] + [confidence]
            # Use latest non-None depth
            depths = [e[4] for e in entries if e[4] is not None]
            if depth_m is not None:
                depths.append(depth_m)
            avg_depth = depths[-1] if depths else None

            try:
                self._memory.observe(
                    label=label,
                    cx=sum(all_cx) / len(all_cx),
                    cy=sum(all_cy) / len(all_cy),
                    confidence=max(all_conf),
                    timestamp_ms=timestamp_ms,
                    depth_m=avg_depth,
                )
            except Exception as e:
                logger.error("Failed to store consensus detection %s: %s", label, e)
        else:
            buf.append((cx, cy, confidence, timestamp_ms, depth_m))

    def status(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "last_detections": [d.to_dict() for d in self.last_detections],
            "host_yolo_available": self._host_model is not None,
            "host_yolo_model": _HOST_YOLO_MODEL,
            "classes": self._classes,
        }
