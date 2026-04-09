"""DepthAI pipeline for OAK-D Lite: RGB + stereo depth + on-device neural network.

Creates a DepthAI pipeline that runs YOLOv8-nano on the Myriad X VPU for
object detection, while streaming aligned RGB and stereo depth frames to
the host (Raspberry Pi 5).

The pipeline operates in two modes:
  1. On-device NN: YOLO runs on OAK-D's Myriad X VPU (preferred, ~15 FPS)
  2. Host-side NN: YOLO runs on RPi5 CPU via ultralytics (fallback, ~2 FPS)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

# Default blob path for the pre-converted YOLOv8-nano OpenVINO model
_DEFAULT_BLOB_PATH = Path(__file__).parent.parent / "models" / "yolov8n.blob"

# Detection input resolution for the neural network
_NN_WIDTH = 416
_NN_HEIGHT = 416

# Stereo depth parameters
_MONO_RESOLUTION_KEY = "THE_400_P"  # 640x400 stereo pair
_DEPTH_CONFIDENCE_THRESHOLD = 200   # 0-255, higher = more confident
_DEPTH_MEDIAN_FILTER_KEY = "KERNEL_7x7"

# Depth clamp range (meters)
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 4.0


@dataclass
class OakDetection:
    """A single detection from the OAK-D pipeline."""

    label: str
    label_id: int
    cx: float           # normalized [0, 1]
    cy: float           # normalized [0, 1]
    width: float        # normalized bbox width
    height: float       # normalized bbox height
    confidence: float
    depth_m: float | None = None  # stereo depth at detection center (meters)


@dataclass
class OakFrame:
    """A single frame output from the OAK-D pipeline."""

    rgb: np.ndarray             # HxWx3 BGR image
    depth_map: np.ndarray | None  # HxW depth in millimeters (uint16)
    detections: list[OakDetection] = field(default_factory=list)
    timestamp_ms: int = 0
    frame_seq: int = 0


class OakPipeline:
    """Manages the DepthAI pipeline for OAK-D Lite.

    Usage::

        pipeline = OakPipeline(blob_path="models/yolov8n.blob")
        pipeline.start()
        for frame in pipeline.frames():
            # frame.rgb, frame.depth_map, frame.detections
            ...
        pipeline.stop()
    """

    def __init__(
        self,
        blob_path: str | Path | None = None,
        classes: list[str] | None = None,
        rgb_fps: float = 10.0,
        nn_confidence_threshold: float = 0.5,
        use_depth: bool = True,
        sync_nn: bool = True,
    ) -> None:
        self._blob_path = Path(blob_path) if blob_path else _DEFAULT_BLOB_PATH
        self._classes = classes or []
        self._rgb_fps = rgb_fps
        self._nn_conf = nn_confidence_threshold
        self._use_depth = use_depth
        self._sync_nn = sync_nn
        self._device = None
        self._pipeline = None
        self._running = False

    def _build_pipeline(self):
        """Construct the DepthAI pipeline graph."""
        import depthai as dai

        pipeline = dai.Pipeline()

        # --- RGB camera ---
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(_NN_WIDTH, _NN_HEIGHT)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(self._rgb_fps)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # Full-resolution RGB output for display/storage
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.video.link(xout_rgb.input)

        # --- Neural network (on-device YOLO) ---
        if self._blob_path.exists():
            nn = pipeline.create(dai.node.YoloDetectionNetwork)
            nn.setBlobPath(str(self._blob_path))
            nn.setConfidenceThreshold(self._nn_conf)
            nn.setNumClasses(len(self._classes) if self._classes else 80)
            nn.setCoordinateSize(4)
            nn.setAnchors([])  # YOLOv8 is anchor-free
            nn.setAnchorMasks({})
            nn.setIouThreshold(0.5)
            nn.input.setBlocking(False)
            nn.input.setQueueSize(1)

            cam_rgb.preview.link(nn.input)

            xout_nn = pipeline.create(dai.node.XLinkOut)
            xout_nn.setStreamName("nn")
            nn.out.link(xout_nn.input)

            # Passthrough for syncing NN detections with frames
            if self._sync_nn:
                xout_nn_passthrough = pipeline.create(dai.node.XLinkOut)
                xout_nn_passthrough.setStreamName("nn_passthrough")
                nn.passthrough.link(xout_nn_passthrough.input)

            self._has_nn = True
            logger.info("On-device NN enabled: %s", self._blob_path)
        else:
            self._has_nn = False
            logger.warning(
                "NN blob not found at %s — falling back to host-side inference",
                self._blob_path,
            )

        # --- Stereo depth ---
        if self._use_depth:
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_left.setResolution(
                getattr(dai.MonoCameraProperties.SensorResolution, _MONO_RESOLUTION_KEY)
            )
            mono_left.setCamera("left")
            mono_left.setFps(self._rgb_fps)

            mono_right = pipeline.create(dai.node.MonoCamera)
            mono_right.setResolution(
                getattr(dai.MonoCameraProperties.SensorResolution, _MONO_RESOLUTION_KEY)
            )
            mono_right.setCamera("right")
            mono_right.setFps(self._rgb_fps)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to RGB
            stereo.initialConfig.setConfidenceThreshold(_DEPTH_CONFIDENCE_THRESHOLD)
            stereo.initialConfig.setMedianFilter(
                getattr(dai.MedianFilter, _DEPTH_MEDIAN_FILTER_KEY)
            )
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)

            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

            logger.info("Stereo depth enabled (aligned to RGB)")

        self._pipeline = pipeline
        return pipeline

    def start(self) -> None:
        """Start the OAK-D pipeline and connect to the device."""
        import depthai as dai

        if self._running:
            return

        pipeline = self._build_pipeline()
        self._device = dai.Device(pipeline)
        self._running = True

        device_name = self._device.getDeviceName()
        logger.info("OAK-D device connected: %s", device_name)

    def stop(self) -> None:
        """Stop the pipeline and release the device."""
        self._running = False
        if self._device is not None:
            self._device.close()
            self._device = None
            logger.info("OAK-D device released")

    def frames(self) -> Generator[OakFrame, None, None]:
        """Yield frames from the OAK-D pipeline.

        Each frame includes RGB image, optional depth map, and NN detections.
        Blocks until a new frame is available.
        """
        if not self._running or self._device is None:
            return

        q_rgb = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = self._device.getOutputQueue("depth", maxSize=4, blocking=False) if self._use_depth else None
        q_nn = self._device.getOutputQueue("nn", maxSize=4, blocking=False) if self._has_nn else None

        seq = 0
        while self._running:
            # Get RGB frame
            in_rgb = q_rgb.tryGet()
            if in_rgb is None:
                time.sleep(0.001)
                continue

            rgb_frame = in_rgb.getCvFrame()
            timestamp_ms = int(time.time() * 1000)

            # Get depth map (aligned to RGB)
            depth_map = None
            if q_depth is not None:
                in_depth = q_depth.tryGet()
                if in_depth is not None:
                    depth_map = in_depth.getFrame()  # uint16, millimeters

            # Get NN detections
            detections = []
            if q_nn is not None:
                in_nn = q_nn.tryGet()
                if in_nn is not None:
                    detections = self._parse_detections(in_nn, rgb_frame.shape, depth_map)

            seq += 1
            yield OakFrame(
                rgb=rgb_frame,
                depth_map=depth_map,
                detections=detections,
                timestamp_ms=timestamp_ms,
                frame_seq=seq,
            )

    def _parse_detections(
        self,
        nn_data,
        rgb_shape: tuple[int, ...],
        depth_map: np.ndarray | None,
    ) -> list[OakDetection]:
        """Parse DepthAI detection network output into OakDetection objects."""
        h, w = rgb_shape[:2]
        results = []

        for detection in nn_data.detections:
            label_id = detection.label
            label = self._classes[label_id] if label_id < len(self._classes) else f"class_{label_id}"
            confidence = detection.confidence

            # DepthAI gives normalized bounding box coords
            x_min = max(0.0, detection.xmin)
            y_min = max(0.0, detection.ymin)
            x_max = min(1.0, detection.xmax)
            y_max = min(1.0, detection.ymax)

            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            bw = x_max - x_min
            bh = y_max - y_min

            # Look up stereo depth at detection center
            depth_m = None
            if depth_map is not None:
                depth_m = self._depth_at_point(depth_map, cx, cy)

            results.append(OakDetection(
                label=label,
                label_id=label_id,
                cx=cx, cy=cy,
                width=bw, height=bh,
                confidence=confidence,
                depth_m=depth_m,
            ))

        return results

    @staticmethod
    def _depth_at_point(
        depth_map: np.ndarray,
        cx: float,
        cy: float,
        kernel: int = 5,
    ) -> float | None:
        """Get median depth at a normalized point, clamped to usable range.

        Uses a small kernel around the center point to reduce noise from
        stereo matching artifacts.
        """
        h, w = depth_map.shape[:2]
        px = int(cx * w)
        py = int(cy * h)

        # Extract kernel-sized patch around the center
        half = kernel // 2
        y0 = max(0, py - half)
        y1 = min(h, py + half + 1)
        x0 = max(0, px - half)
        x1 = min(w, px + half + 1)

        patch = depth_map[y0:y1, x0:x1]
        # Filter out zero/invalid depth values
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None

        depth_mm = float(np.median(valid))
        depth_m = depth_mm / 1000.0

        # Clamp to usable range
        if depth_m < DEPTH_MIN_M or depth_m > DEPTH_MAX_M:
            return None

        return round(depth_m, 3)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_nn(self) -> bool:
        return self._has_nn if hasattr(self, "_has_nn") else False

    def status(self) -> dict:
        return {
            "running": self._running,
            "has_nn": self.has_nn,
            "blob_path": str(self._blob_path),
            "use_depth": self._use_depth,
            "rgb_fps": self._rgb_fps,
            "nn_confidence": self._nn_conf,
            "classes": self._classes,
        }
