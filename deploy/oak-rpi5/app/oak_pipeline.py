"""OAK-D Lite camera pipeline for the RPi5 proactive assistant.

Wraps the DepthAI SDK to produce a stream of (detections, depth_map) frames
that feed into ProactiveEngine.

Hardware: Luxonis OAK-D Lite connected via USB3 to Raspberry Pi 5.
VPU: Intel Myriad X (on-device YOLO inference, no host GPU needed).

When DepthAI is not installed (CI, dev laptop), the module can still be
imported; constructing OakPipeline will raise ``OakUnavailable``.

Sim mode (``OakSimPipeline``): generates synthetic detection sequences
suitable for exercising the full proactive pipeline on a desk.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OakUnavailable(RuntimeError):
    """Raised when DepthAI hardware or library is not available."""


# ---------------------------------------------------------------------------
# Data model — mirrors what ProactiveEngine expects
# ---------------------------------------------------------------------------


@dataclass
class OakDetection:
    """One bounding-box detection from the OAK-D Lite pipeline."""

    label: str
    label_id: int
    cx: float          # normalised [0, 1] horizontal centre
    cy: float          # normalised [0, 1] vertical centre
    width: float       # normalised bounding-box width
    height: float      # normalised bounding-box height
    confidence: float  # [0, 1]
    depth_m: float | None = None  # metres; None if no valid spatial data


@dataclass
class OakFrame:
    """One complete camera frame with detections and optional depth map."""

    detections: list[OakDetection]
    depth_map: np.ndarray | None    # HxW uint16 in mm; None when unavailable
    fps: float
    frame_id: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


# ---------------------------------------------------------------------------
# YOLO class map (COCO 80-class subset used on OAK-D Lite blob)
# ---------------------------------------------------------------------------

_COCO_CLASSES: dict[int, str] = {
    0: "person",    1: "bicycle",     2: "car",
    3: "motorcycle", 5: "bus",        7: "truck",
    9: "traffic light", 11: "fire hydrant", 15: "cat",
    16: "dog",
    # Extend with custom classes as needed
}


def _label_for_id(label_id: int) -> str:
    return _COCO_CLASSES.get(label_id, f"object_{label_id}")


# ---------------------------------------------------------------------------
# Hardware pipeline
# ---------------------------------------------------------------------------

_DEPTHAI_AVAILABLE: bool | None = None  # lazy-checked on first construction


def _check_depthai() -> None:
    global _DEPTHAI_AVAILABLE
    if _DEPTHAI_AVAILABLE is None:
        try:
            import depthai  # noqa: F401
            _DEPTHAI_AVAILABLE = True
        except ImportError:
            _DEPTHAI_AVAILABLE = False
    if not _DEPTHAI_AVAILABLE:
        raise OakUnavailable(
            "DepthAI library not found. Install via: pip install depthai\n"
            "Use OakSimPipeline for development without hardware."
        )


class OakPipeline:
    """Hardware OAK-D Lite pipeline.

    Initialises the DepthAI pipeline with:
      - StereoDepth for metric depth estimation
      - YOLO detection network running on the Myriad X VPU
      - Spatial detection node to attach depth to each bounding box

    Usage::

        pipeline = OakPipeline(model_blob_path="/path/to/yolov8n_oak.blob")
        pipeline.start()
        try:
            while True:
                frame = pipeline.next_frame(timeout_ms=100)
                if frame:
                    alerts = engine.process(frame.detections, frame.depth_map)
        finally:
            pipeline.stop()
    """

    # Spatial detection limits (metres)
    DEPTH_LOWER_THRESHOLD_MM = 100    # 0.1 m minimum depth
    DEPTH_UPPER_THRESHOLD_MM = 8000   # 8 m maximum depth

    def __init__(
        self,
        model_blob_path: str,
        confidence_threshold: float = 0.45,
        target_fps: int = 30,
    ) -> None:
        _check_depthai()
        self._blob_path = model_blob_path
        self._conf_thresh = confidence_threshold
        self._target_fps = target_fps
        self._device = None
        self._queue = None
        self._frame_id = 0
        self._fps_estimator = _FPSEstimator()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Build the DepthAI pipeline and connect to the OAK-D Lite."""
        import depthai as dai

        pipeline = dai.Pipeline()

        # Color camera (used for YOLO input; preview 416x416 for YOLOv8n)
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(416, 416)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(self._target_fps)

        # Stereo depth
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(416, 416)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # Spatial detection network
        spatial_net = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        spatial_net.setBlobPath(self._blob_path)
        spatial_net.setConfidenceThreshold(self._conf_thresh)
        spatial_net.input.setBlocking(False)
        spatial_net.setBoundingBoxScaleFactor(0.5)
        spatial_net.setDepthLowerThreshold(self.DEPTH_LOWER_THRESHOLD_MM)
        spatial_net.setDepthUpperThreshold(self.DEPTH_UPPER_THRESHOLD_MM)

        cam.preview.link(spatial_net.input)
        stereo.depth.link(spatial_net.inputDepth)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("spatial_detections")
        spatial_net.out.link(xout.input)

        self._device = dai.Device(pipeline)
        self._queue = self._device.getOutputQueue(
            name="spatial_detections",
            maxSize=4,
            blocking=False,
        )
        logger.info(
            "OakPipeline started — device: %s, blob: %s, fps: %d",
            self._device.getMxId(),
            self._blob_path,
            self._target_fps,
        )

    def stop(self) -> None:
        """Close the DepthAI device connection."""
        if self._device is not None:
            self._device.close()
            self._device = None
            self._queue = None
        logger.info("OakPipeline stopped")

    # ------------------------------------------------------------------
    # Frame retrieval
    # ------------------------------------------------------------------

    def next_frame(self, timeout_ms: int = 100) -> OakFrame | None:
        """Block up to *timeout_ms* for the next detection frame.

        Returns None if no frame arrives within the timeout.
        """
        if self._queue is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

        msg = self._queue.tryGet()
        if msg is None:
            return None

        self._frame_id += 1
        fps = self._fps_estimator.tick()
        dets: list[OakDetection] = []

        for d in msg.detections:
            cx = (d.xmin + d.xmax) / 2.0
            cy = (d.ymin + d.ymax) / 2.0
            w = d.xmax - d.xmin
            h = d.ymax - d.ymin

            # spatialCoordinates.z is in mm; convert to metres
            depth_m = d.spatialCoordinates.z / 1000.0 if d.spatialCoordinates.z > 0 else None

            dets.append(OakDetection(
                label=_label_for_id(d.label),
                label_id=d.label,
                cx=float(cx),
                cy=float(cy),
                width=float(w),
                height=float(h),
                confidence=float(d.confidence),
                depth_m=depth_m,
            ))

        return OakFrame(
            detections=dets,
            depth_map=None,  # spatial detections encode depth per object
            fps=fps,
            frame_id=self._frame_id,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "OakPipeline":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Sim pipeline — desk-scenario synthetic detections for development
# ---------------------------------------------------------------------------

_SIM_SCENARIOS: list[dict] = [
    # (label, cx, cy, start_depth_m, end_depth_m) — depth interpolates over time
    {"label": "person",   "cx": 0.5,  "cy": 0.5,  "depth_start": 3.0, "depth_end": 0.8},
    {"label": "door",     "cx": 0.55, "cy": 0.45, "depth_start": 1.2, "depth_end": 1.2},
    {"label": "car",      "cx": 0.2,  "cy": 0.5,  "depth_start": 2.5, "depth_end": 1.2},
    {"label": "stairs",   "cx": 0.5,  "cy": 0.6,  "depth_start": 2.0, "depth_end": 0.4},
    {"label": "dog",      "cx": 0.7,  "cy": 0.55, "depth_start": 2.8, "depth_end": 1.4},
]


class OakSimPipeline:
    """Synthetic detection source for development without OAK-D Lite hardware.

    Cycles through a set of desk-scale hazard scenarios and emits frames at
    a configurable rate.  Not thread-safe — call from a single thread.
    """

    def __init__(
        self,
        fps: float = 15.0,
        scenario_duration_s: float = 8.0,
        loop: bool = True,
    ) -> None:
        self._fps = fps
        self._scenario_duration = scenario_duration_s
        self._loop = loop
        self._frame_id = 0
        self._started_at: float | None = None
        self._fps_estimator = _FPSEstimator()
        self._frame_interval = 1.0 / fps

    def start(self) -> None:
        self._started_at = time.monotonic()
        logger.info("OakSimPipeline started (fps=%.1f)", self._fps)

    def stop(self) -> None:
        logger.info("OakSimPipeline stopped after %d frames", self._frame_id)

    def next_frame(self, timeout_ms: int = 100) -> OakFrame | None:
        """Return the next simulated frame, rate-limited to target FPS."""
        if self._started_at is None:
            raise RuntimeError("Pipeline not started. Call start() first.")

        elapsed = time.monotonic() - self._started_at
        expected_frames = int(elapsed / self._frame_interval)

        # Rate-limit: only emit when a new frame is due
        if self._frame_id >= expected_frames:
            time.sleep(min(self._frame_interval / 2, timeout_ms / 1000.0))
            return None

        self._frame_id += 1
        fps = self._fps_estimator.tick()

        # Determine which scenario is active
        total_duration = len(_SIM_SCENARIOS) * self._scenario_duration
        if not self._loop and elapsed > total_duration:
            return None

        scenario_time = elapsed % total_duration
        scenario_idx = int(scenario_time / self._scenario_duration) % len(_SIM_SCENARIOS)
        t = (scenario_time % self._scenario_duration) / self._scenario_duration  # 0→1

        sc = _SIM_SCENARIOS[scenario_idx]
        depth = sc["depth_start"] + (sc["depth_end"] - sc["depth_start"]) * t

        # Add a small amount of noise to simulate real sensor readings
        depth = max(0.1, depth + random.gauss(0, 0.03))
        cx = sc["cx"] + random.gauss(0, 0.01)
        cy = sc["cy"] + random.gauss(0, 0.01)

        det = OakDetection(
            label=sc["label"],
            label_id=0,
            cx=max(0.0, min(1.0, cx)),
            cy=max(0.0, min(1.0, cy)),
            width=0.12,
            height=0.18,
            confidence=0.88 + random.gauss(0, 0.02),
            depth_m=depth,
        )

        return OakFrame(
            detections=[det],
            depth_map=None,
            fps=fps,
            frame_id=self._frame_id,
        )

    def __enter__(self) -> "OakSimPipeline":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# FPS estimator (rolling window)
# ---------------------------------------------------------------------------


class _FPSEstimator:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span
