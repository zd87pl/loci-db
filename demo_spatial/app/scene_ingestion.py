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
import contextlib
import inspect
import io
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .spatial_memory import SpatialMemory
    from .vlm_client import VLMClient

# A point in metres: camera frame (OpenCV: +x right, +y down, +z forward) or room frame.
Vec3 = tuple[float, float, float]

# Keyword arguments of ``SpatialMemory.observe`` that only the handheld (metric)
# path adds on top of the legacy ``label, cx, cy, confidence, timestamp_ms, depth_m``.
_METRIC_OBSERVE_KEYS = (
    "xyz_cam_m",
    "pos_room_m",
    "room",
    "anchored",
    "anchor_confidence",
    "source",
)

# Minimum YOLO confidence to trust a detection without VLM fallback
_YOLO_CONF_THRESHOLD = 0.65
# If fewer than this many objects are detected, try VLM enrichment
_MIN_DETECTIONS_FOR_NO_VLM = 1
# Stage 2 (ADR-5): skepticism multiplier applied to VLM confidence when no
# raw YOLO detection with the same label corroborates the VLM position
_VLM_UNCORROBORATED_PENALTY = 0.6
# Minimum IoU between the VLM position and a same-label YOLO detection
# for the YOLO detection to count as corroboration
_VLM_CORROBORATION_IOU = 0.3

# YOLO-World model (open-vocabulary). Override with YOLO_MODEL env var.
_YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8l-worldv2.pt")

# Default open-vocabulary class list for assistive use.
# Users can update at runtime via PUT /api/detection/classes.
DEFAULT_CLASSES: list[str] = [
    "person",
    "keys",
    "wallet",
    "phone",
    "laptop",
    "cup",
    "bottle",
    "bag",
    "backpack",
    "glasses",
    "remote control",
    "charger",
    "headphones",
    "AirPods",
    "watch",
    "book",
    "pen",
    "medicine",
    "cane",
    "shoe",
    "jacket",
    "umbrella",
    "food container",
    "water bottle",
    "mug",
    "plate",
    "bowl",
    "fork",
    "knife",
    "spoon",
    "chair",
    "table",
    "door",
    "cat",
    "dog",
]


@dataclass
class Detection:
    """A single object detection from a camera frame."""

    label: str
    cx: float  # normalized [0, 1] — center x in frame
    cy: float  # normalized [0, 1] — center y in frame
    width: float  # normalized bbox width
    height: float  # normalized bbox height
    confidence: float
    source: str = "yolo"  # "yolo" | "vlm" | "oak" | "sim" | "replay"
    depth_m: float | None = None  # LiDAR/stereo depth in meters (None if unavailable)
    # --- handheld (metric) fields; all optional so legacy callers are unaffected ---
    xyz_cam_m: Vec3 | None = None  # camera frame, metres (OpenCV axes); None = no depth
    pos_room_m: Vec3 | None = None  # room frame, metres (set by the runtime when anchored)
    room: str | None = None  # room name when the camera pose is known
    frame: str = "image"  # "image" | "camera" | "room"
    anchored: bool = False  # True while a room tag was accepted <= hold_s ago
    anchor_confidence: float = 0.0  # 1 - age/hold_s clipped to [0, 1]; 0 when not anchored

    def __post_init__(self) -> None:
        # Normalise metric points to plain float tuples (JSON-safe, hashable) and
        # derive the legacy depth from the camera-frame z when it was not given.
        if self.xyz_cam_m is not None:
            self.xyz_cam_m = _as_vec3(self.xyz_cam_m)
        if self.pos_room_m is not None:
            self.pos_room_m = _as_vec3(self.pos_room_m)
        if self.depth_m is None and self.xyz_cam_m is not None:
            self.depth_m = self.xyz_cam_m[2]

    @property
    def distance_m(self) -> float | None:
        """Euclidean range from the camera in metres, or None without depth."""
        if self.xyz_cam_m is None:
            return None
        return math.sqrt(sum(v * v for v in self.xyz_cam_m))

    @property
    def is_metric(self) -> bool:
        """True when this detection carries any handheld (metric) information."""
        return (
            self.frame != "image"
            or self.xyz_cam_m is not None
            or self.pos_room_m is not None
            or self.room is not None
        )

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
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
        # The keys below appear only for metric detections so the legacy browser
        # payload (and every existing WS message shape) stays byte-identical.
        if self.xyz_cam_m is not None:
            result["xyz_cam_m"] = [round(v, 3) for v in self.xyz_cam_m]
            distance = self.distance_m
            if distance is not None:
                result["distance_m"] = round(distance, 3)
        if self.pos_room_m is not None:
            result["pos_room_m"] = [round(v, 3) for v in self.pos_room_m]
        if self.room is not None:
            result["room"] = self.room
        if self.is_metric:
            result["frame"] = self.frame
            result["anchored"] = self.anchored
            result["anchor_confidence"] = round(self.anchor_confidence, 3)
        return result


def _as_vec3(p: Sequence[float]) -> Vec3:
    """Coerce any 3-sequence (tuple, list, numpy row) into a tuple of floats."""
    x, y, z = p
    return (float(x), float(y), float(z))


def _dist3(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _mean_vec3(points: Sequence[Vec3]) -> Vec3 | None:
    if not points:
        return None
    n = float(len(points))
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


@dataclass(slots=True)
class _BufEntry:
    """One buffered detection awaiting temporal consensus."""

    cx: float
    cy: float
    confidence: float
    timestamp_ms: int
    depth_m: float | None = None
    xyz_cam_m: Vec3 | None = None
    pos_room_m: Vec3 | None = None
    anchored: bool = False
    anchor_confidence: float = 0.0
    source: str = "yolo"


# Consensus buffer key: (label, frame, room or "").
_BufKey = tuple[str, str, str]


class SceneIngestion:
    """Processes camera frames, extracts object detections, stores in LOCI-DB.

    Usage::

        ingestion = SceneIngestion(memory, vlm_client)
        detections = await ingestion.process_frame(jpeg_bytes)
    """

    def __init__(
        self,
        memory: SpatialMemory,
        vlm_client: VLMClient | None = None,
        use_vlm_fallback: bool = True,
        classes: list[str] | None = None,
        consensus_window_ms: int = 1000,
        consensus_min_count: int = 2,
        consensus_iou_threshold: float = 0.4,
        *,
        consensus_radius_m: float = 0.3,
        consensus_refire_ms: int = 0,
    ) -> None:
        self._memory = memory
        self._vlm = vlm_client
        self._use_vlm_fallback = use_vlm_fallback
        self._model = None  # lazy-loaded YOLO model
        self._classes: list[str] = list(classes or DEFAULT_CLASSES)
        self._is_world_model = "world" in _YOLO_MODEL.lower()
        self._server_capture_task: asyncio.Task | None = None
        self._capturing = False
        self.last_detections: list[Detection] = []
        self.frame_count = 0
        # Temporal consensus buffer keyed by (label, frame, room or ""); entries are
        # time-ordered _BufEntry records. Guarded by a lock because the handheld
        # runtime thread (process_detections) and the asyncio loop (process_frame)
        # may both feed it.
        self._consensus_buffer: dict[_BufKey, deque[_BufEntry]] = {}
        self._consensus_lock = threading.Lock()
        self._last_fire: dict[_BufKey, int] = {}
        self._consensus_fires = 0
        self._consensus_suppressed = 0
        self._observe_accepts_metric: bool | None = None  # resolved on first fire
        self.consensus_window_ms = consensus_window_ms
        self.consensus_min_count = consensus_min_count
        self.consensus_iou_threshold = consensus_iou_threshold
        # Metric match radius (metres) used when both entries carry room or camera metres.
        self.consensus_radius_m = consensus_radius_m
        # Minimum gap between two fires of the same key (0 = legacy: fire whenever
        # min_count is reached). Entries keep accumulating during the hold-off so
        # the eventual fire averages more samples.
        self.consensus_refire_ms = consensus_refire_ms

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
                    _YOLO_MODEL,
                    len(self._classes),
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

    def _run_yolo(
        self, image_bytes: bytes, conf_threshold: float = _YOLO_CONF_THRESHOLD
    ) -> list[Detection]:
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
                    detections.append(
                        Detection(
                            label=label,
                            cx=cx,
                            cy=cy,
                            width=bw,
                            height=bh,
                            confidence=conf,
                            source="yolo",
                        )
                    )
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

    def _calibrate_vlm_confidence(
        self,
        label: str,
        cx: float,
        cy: float,
        raw_conf: float,
        yolo_detections: list[Detection],
    ) -> float:
        """Stage 2 (ADR-5): apply the skepticism multiplier to VLM confidence.

        A VLM detection is corroborated when a raw YOLO detection with the
        same label overlaps it (IoU > ``_VLM_CORROBORATION_IOU``). When no
        YOLO detection corroborates, the VLM confidence is multiplied by
        ``_VLM_UNCORROBORATED_PENALTY``.
        """
        yolo_corroborated = any(
            d.label == label and self._iou(cx, cy, d.cx, d.cy) > _VLM_CORROBORATION_IOU
            for d in yolo_detections
        )
        if yolo_corroborated:
            return raw_conf
        return raw_conf * _VLM_UNCORROBORATED_PENALTY

    def _entries_match(self, a: _BufEntry, b: _BufEntry) -> bool:
        """Same physical object? Metric distance when both carry metres, else IoU proxy.

        Room metres win over camera metres (the camera moves between frames, the
        room does not); the image-box IoU proxy is the legacy fallback and the only
        option when one side has no depth.
        """
        if a.pos_room_m is not None and b.pos_room_m is not None:
            return _dist3(a.pos_room_m, b.pos_room_m) <= self.consensus_radius_m
        if a.xyz_cam_m is not None and b.xyz_cam_m is not None:
            return _dist3(a.xyz_cam_m, b.xyz_cam_m) <= self.consensus_radius_m
        return self._iou(a.cx, a.cy, b.cx, b.cy) > self.consensus_iou_threshold

    @staticmethod
    def _consensus_payload(
        label: str,
        entries: list[_BufEntry],
        *,
        room: str | None,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        """Keyword arguments for ``SpatialMemory.observe`` from the agreeing entries.

        ``entries`` are the matched buffer entries followed by the triggering
        detection (newest last).
        """
        n = float(len(entries))
        newest = entries[-1]
        depth_m = newest.depth_m
        if depth_m is None:
            depth_m = next(
                (e.depth_m for e in reversed(entries) if e.depth_m is not None),
                None,
            )
        return {
            "label": label,
            "cx": sum(e.cx for e in entries) / n,
            "cy": sum(e.cy for e in entries) / n,
            "confidence": max(e.confidence for e in entries),
            "timestamp_ms": timestamp_ms,
            "depth_m": depth_m,
            "xyz_cam_m": _mean_vec3([e.xyz_cam_m for e in entries if e.xyz_cam_m is not None]),
            "pos_room_m": _mean_vec3([e.pos_room_m for e in entries if e.pos_room_m is not None]),
            "room": room,
            "anchored": all(e.anchored for e in entries),
            "anchor_confidence": sum(e.anchor_confidence for e in entries) / n,
            "source": newest.source,
        }

    def _observe(self, payload: dict[str, Any]) -> None:
        """Call ``memory.observe(**payload)``; errors are logged, never raised.

        The metric keywords are dropped (once, with a warning) when the attached
        memory still has the legacy ``observe`` signature, so a stale
        ``SpatialMemory`` degrades to image-frame storage instead of failing.
        """
        if self._observe_accepts_metric is None:
            self._observe_accepts_metric = self._probe_observe_signature()
        if not self._observe_accepts_metric:
            payload = {k: v for k, v in payload.items() if k not in _METRIC_OBSERVE_KEYS}
        try:
            self._memory.observe(**payload)
        except Exception as e:
            logger.error("Failed to store consensus detection %s: %s", payload.get("label"), e)

    def _probe_observe_signature(self) -> bool:
        try:
            params = inspect.signature(self._memory.observe).parameters
        except (TypeError, ValueError):
            return True  # cannot introspect (builtin/C callable): assume the contract
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return True
        if all(k in params for k in _METRIC_OBSERVE_KEYS):
            return True
        logger.warning(
            "SpatialMemory.observe lacks the metric keywords %s; "
            "consensus fires will store image-frame observations only",
            [k for k in _METRIC_OBSERVE_KEYS if k not in params],
        )
        return False

    def _update_consensus(
        self,
        label: str,
        cx: float,
        cy: float,
        confidence: float,
        timestamp_ms: int,
        depth_m: float | None,
        *,
        xyz_cam_m: Vec3 | None = None,
        pos_room_m: Vec3 | None = None,
        room: str | None = None,
        frame: str = "image",
        anchored: bool = False,
        anchor_confidence: float = 0.0,
        source: str = "yolo",
    ) -> None:
        """Buffer a detection and call observe() when temporal consensus is reached.

        Buffers are keyed by ``(label, frame, room)`` so image-frame, camera-frame and
        per-room observations never vote for each other. Within a key, a detection
        is the same object as a buffered entry when (see ``_entries_match``):
        - both have room metres  -> distance <= consensus_radius_m
        - both have camera metres -> distance <= consensus_radius_m
        - otherwise               -> IoU of the image boxes > consensus_iou_threshold
        and the entry is within ``consensus_window_ms``.

        On ``consensus_min_count`` agreeing samples, ``observe()`` is called with the
        averaged box centre / metres, the maximum confidence, the newest depth,
        ``anchored = all(...)`` and the mean anchor confidence; the agreeing entries
        are CONSUMED, so the next fire needs ``min_count`` fresh samples again.
        While a key is inside its ``consensus_refire_ms`` hold-off, samples keep
        accumulating (averaging continues) and the fire is deferred.
        """
        key: _BufKey = (label, frame, room or "")
        new = _BufEntry(
            cx=cx,
            cy=cy,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            depth_m=depth_m,
            xyz_cam_m=None if xyz_cam_m is None else _as_vec3(xyz_cam_m),
            pos_room_m=None if pos_room_m is None else _as_vec3(pos_room_m),
            anchored=anchored,
            anchor_confidence=anchor_confidence,
            source=source,
        )

        with self._consensus_lock:
            buf = self._consensus_buffer.get(key)
            if buf is None:
                buf = deque()
                self._consensus_buffer[key] = buf

            # Evict entries outside the rolling window (entries are time-ordered)
            cutoff = timestamp_ms - self.consensus_window_ms
            while buf and buf[0].timestamp_ms < cutoff:
                buf.popleft()

            matches = [e for e in buf if self._entries_match(e, new)]

            if len(matches) + 1 < self.consensus_min_count:
                # Not enough confirmations yet — buffer this detection
                buf.append(new)
                return

            last_fire = self._last_fire.get(key)
            if last_fire is not None and timestamp_ms - last_fire < self.consensus_refire_ms:
                # Hold-off: keep sampling, fire later with a longer average
                self._consensus_suppressed += 1
                buf.append(new)
                return

            # Consensus reached — consume the agreeing entries and fire
            matched_ids = {id(e) for e in matches}
            remaining = [e for e in buf if id(e) not in matched_ids]
            buf.clear()
            buf.extend(remaining)
            self._last_fire[key] = timestamp_ms
            self._consensus_fires += 1
            payload = self._consensus_payload(
                label, [*matches, new], room=room, timestamp_ms=timestamp_ms
            )

        # observe() may take the memory lock and hit the store: keep it outside ours
        self._observe(payload)

    def process_detections(
        self, detections: list[Detection], timestamp_ms: int | None = None
    ) -> list[Detection]:
        """Feed pre-made (typically metric) detections through temporal consensus.

        Synchronous counterpart of :meth:`process_frame` for sources that already
        provide labelled boxes with metres (OAK-D, simulator, replay). No YOLO, no
        VLM: each detection goes straight into the consensus buffer and reaches
        ``SpatialMemory.observe`` once enough frames agree. Safe to call from the
        handheld runtime thread.

        Args:
            detections: Detections for one frame, with ``frame``/``room``/
                ``pos_room_m`` already filled in by the runtime.
            timestamp_ms: Host Unix time of the frame (defaults to now).

        Returns:
            The same detections (also kept in ``last_detections``).
        """
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        dets = list(detections)
        for det in dets:
            self._update_consensus(
                label=det.label,
                cx=det.cx,
                cy=det.cy,
                confidence=det.confidence,
                timestamp_ms=ts,
                depth_m=det.depth_m,
                xyz_cam_m=det.xyz_cam_m,
                pos_room_m=det.pos_room_m,
                room=det.room,
                frame=det.frame,
                anchored=det.anchored,
                anchor_confidence=det.anchor_confidence,
                source=det.source,
            )
        self.last_detections = dets
        self.frame_count += 1
        return dets

    @property
    def consensus_stats(self) -> dict:
        """Live counters of the consensus buffer (surfaced in the handheld status)."""
        with self._consensus_lock:
            return {
                "buffer_entries": sum(len(b) for b in self._consensus_buffer.values()),
                "fires": self._consensus_fires,
                "suppressed_by_holdoff": self._consensus_suppressed,
            }

    @staticmethod
    def _lookup_depth(cx: float, cy: float, depth_samples: dict[str, float] | None) -> float | None:
        """Estimate depth at (cx, cy) by interpolating the nearest LiDAR grid sample.

        The depth_samples dict maps 3x3 grid keys (tl, tc, tr, ml, mc, mr, bl, bc, br)
        to depth values in meters. We pick the nearest sample to the detection center.
        """
        if not depth_samples:
            return None
        grid_positions = {
            "tl": (0.25, 0.25),
            "tc": (0.5, 0.25),
            "tr": (0.75, 0.25),
            "ml": (0.25, 0.5),
            "mc": (0.5, 0.5),
            "mr": (0.75, 0.5),
            "bl": (0.25, 0.75),
            "bc": (0.5, 0.75),
            "br": (0.75, 0.75),
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

                for obj in vlm_results:
                    lbl = obj["label"]
                    raw_conf = obj.get("confidence", 0.50)  # Stage 1/3: VLM-provided or fallback
                    width = obj.get("width", 0.3)
                    height = obj.get("height", 0.3)
                    vlm_cx = obj.get("cx", 0.5)
                    vlm_cy = obj.get("cy", 0.5)

                    # Stage 2: skepticism multiplier when YOLO sees nothing for this label
                    final_conf = self._calibrate_vlm_confidence(
                        lbl, vlm_cx, vlm_cy, raw_conf, raw_yolo
                    )

                    vlm_detections.append(
                        Detection(
                            label=lbl,
                            cx=vlm_cx,
                            cy=vlm_cy,
                            width=width,
                            height=height,
                            confidence=final_conf,
                            source="vlm",
                        )
                    )
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
        self._server_capture_task = asyncio.create_task(self._capture_loop(camera_index, fps))
        logger.info("Server-side camera capture started (camera=%d, fps=%.1f)", camera_index, fps)

    async def stop_capture(self) -> None:
        """Stop background webcam capture."""
        self._capturing = False
        if self._server_capture_task and not self._server_capture_task.done():
            self._server_capture_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_capture_task
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
            "consensus": {
                **self.consensus_stats,
                "window_ms": self.consensus_window_ms,
                "min_count": self.consensus_min_count,
                "iou_threshold": self.consensus_iou_threshold,
                "radius_m": self.consensus_radius_m,
                "refire_ms": self.consensus_refire_ms,
            },
        }
