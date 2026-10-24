"""OAK-D Lite detection source on DepthAI v3 (design §4.3, hardware facts verified live).

The only module that imports ``depthai``, and only inside functions.  The pipeline
built here is exactly the configuration that streamed from this OAK-D-LITE-AF:
``StereoDepth`` preset ``FAST_DENSITY`` aligned to ``CAM_A`` with
``setPostProcessingHardwareResources(2, 2)`` (``DEFAULT``/``HIGH_DETAIL`` do not fit
next to the 8-shave YOLOv6-nano blob), ``SpatialDetectionNetwork`` on the zoo model
``yolov6-nano``, and an optional second RGB output for the AprilTag stream.

Spatial coordinates come back in millimetres, +x right, +y down, +z forward; a box
without valid depth is reported as exact zeros and becomes ``xyz_cam_m=None``.
After any failure the device must return to ``X_LINK_UNBOOTED`` before it can be
reopened; :func:`wait_for_unbooted` polls for that and the supervisor inside
:meth:`OakDetectionSource.events` reconnects forever with a 1 -> 5 s backoff.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .detection_source import (
    COCO80,
    DetectionFrame,
    HostClock,
    JsonlRecorder,
    SourceEvent,
    SourceStatus,
    SpatialDetection,
    TagFrame,
    depthai_xyz_to_cam,
    env_flag,
)

logger = logging.getLogger(__name__)

SPATIAL_ALGOS: dict[str, str] = {
    "median": "MEDIAN",
    "average": "AVERAGE",
    "mean": "MEAN",
    "min": "MIN",
    "max": "MAX",
    "mode": "MODE",
}

# Every ``dai.*`` attribute path this module touches; ``tests/test_oak_source_offline.py``
# resolves each one against the installed depthai so an API rename fails loudly offline.
DAI_NAMES_USED: tuple[str, ...] = (
    "Pipeline",
    "Pipeline.start",
    "Pipeline.stop",
    "Pipeline.isRunning",
    "Device",
    "Device.getAllAvailableDevices",
    "Device.close",
    "Device.getUsbSpeed",
    "Device.getDeviceName",
    "Device.getDeviceId",
    "DeviceInfo.state",
    "XLinkDeviceState.X_LINK_UNBOOTED",
    "UsbSpeed.HIGH",
    "CameraBoardSocket.CAM_A",
    "CameraBoardSocket.CAM_B",
    "CameraBoardSocket.CAM_C",
    "node.Camera.build",
    "node.Camera.requestOutput",
    "node.StereoDepth.build",
    "node.StereoDepth.PresetMode.FAST_DENSITY",
    "node.StereoDepth.PresetMode.DEFAULT",
    "node.StereoDepth.PresetMode.HIGH_DETAIL",
    "node.StereoDepth.setDepthAlign",
    "node.StereoDepth.setPostProcessingHardwareResources",
    "node.StereoDepth.setLeftRightCheck",
    "node.StereoDepth.setExtendedDisparity",
    "NNArchive",
    "NNModelDescription",
    "node.SpatialDetectionNetwork.build",
    "node.SpatialDetectionNetwork.input",
    "node.SpatialDetectionNetwork.out",
    "node.SpatialDetectionNetwork.getClasses",
    "node.SpatialDetectionNetwork.setConfidenceThreshold",
    "node.SpatialDetectionNetwork.setBoundingBoxScaleFactor",
    "node.SpatialDetectionNetwork.setDepthLowerThreshold",
    "node.SpatialDetectionNetwork.setDepthUpperThreshold",
    "node.SpatialDetectionNetwork.setSpatialCalculationAlgorithm",
    "SpatialLocationCalculatorAlgorithm.MEDIAN",
    "SpatialLocationCalculatorAlgorithm.AVERAGE",
    "SpatialLocationCalculatorAlgorithm.MEAN",
    "SpatialLocationCalculatorAlgorithm.MIN",
    "SpatialLocationCalculatorAlgorithm.MAX",
    "SpatialLocationCalculatorAlgorithm.MODE",
    "Node.Input.setBlocking",
    "Node.Output.createOutputQueue",
    "MessageQueue.tryGet",
    "ImgFrame.Type.NV12",
    "ImgFrame.getFrame",
    "ImgFrame.getCvFrame",
    "ImgFrame.getTransformation",
    "ImgFrame.getTimestamp",
    "ImgFrame.getWidth",
    "ImgFrame.getHeight",
    "ImgTransformation.getIntrinsicMatrix",
    "ImgTransformation.getDistortionCoefficients",
    "ImgTransformation.getSize",
    "SpatialImgDetections.detections",
    "SpatialImgDetections.getTimestamp",
    "SpatialImgDetection.label",
    "SpatialImgDetection.confidence",
    "SpatialImgDetection.xmin",
    "SpatialImgDetection.ymin",
    "SpatialImgDetection.xmax",
    "SpatialImgDetection.ymax",
    "SpatialImgDetection.spatialCoordinates",
    "Point3f.x",
    "Point3f.y",
    "Point3f.z",
)


class StallError(RuntimeError):
    """Raised by the supervisor when the running pipeline stops delivering detections."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OakConfig:
    fps: float = 10.0
    model: str = "yolov6-nano"
    usb2: bool = False
    axes: str = "rdf"
    stereo_preset: str = "FAST_DENSITY"
    pp_shaves: int = 2
    lr_check: bool = False
    extended_disparity: bool = False
    confidence: float = 0.5
    bbox_scale: float = 0.5
    depth_min_mm: int = 200
    depth_max_mm: int = 6000
    spatial_algo: str = "median"
    tag_stream: tuple[int, int, float] | None = (1280, 800, 5.0)
    tag_undistort: bool = False
    stall_timeout_s: float = 5.0
    reconnect_backoff_s: tuple[float, float] = (1.0, 5.0)

    def __post_init__(self) -> None:
        self.axes = self.axes.lower()
        self.stereo_preset = self.stereo_preset.upper()
        self.spatial_algo = self.spatial_algo.lower()
        if self.spatial_algo != "default" and self.spatial_algo not in SPATIAL_ALGOS:
            raise ValueError(
                f"unknown spatial algorithm {self.spatial_algo!r}; "
                f"expected default|{'|'.join(SPATIAL_ALGOS)}"
            )
        if self.pp_shaves < 0:
            raise ValueError("pp_shaves must be >= 0")
        if self.depth_min_mm < 0 or self.depth_max_mm <= self.depth_min_mm:
            raise ValueError("need 0 <= depth_min_mm < depth_max_mm")

    @staticmethod
    def parse_tag_stream(s: str) -> tuple[int, int, float] | None:
        """``"1280x800@5"`` -> ``(1280, 800, 5.0)``; ``"off"``/``""`` -> ``None``."""
        txt = (s or "").strip().lower()
        if txt in {"", "off", "none", "0", "false", "no"}:
            return None
        size, _, fps_txt = txt.partition("@")
        w_txt, sep, h_txt = size.partition("x")
        if not sep:
            raise ValueError(f"bad tag stream {s!r}; expected WxH@FPS or off")
        try:
            w, h = int(w_txt), int(h_txt)
            fps = float(fps_txt) if fps_txt else 5.0
        except ValueError as exc:
            raise ValueError(f"bad tag stream {s!r}; expected WxH@FPS or off") from exc
        if w <= 0 or h <= 0 or fps <= 0:
            raise ValueError(f"bad tag stream {s!r}; sizes and fps must be positive")
        return (w, h, fps)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OakConfig:
        env = os.environ if env is None else env

        def get(key: str, default: str) -> str:
            raw = env.get(key)
            return default if raw is None or raw.strip() == "" else raw.strip()

        return cls(
            fps=float(get("LOCI_OAK_FPS", "10")),
            model=get("LOCI_OAK_MODEL", "yolov6-nano"),
            usb2=env_flag(env, "LOCI_OAK_USB2", False),
            axes=get("LOCI_OAK_AXES", "rdf"),
            stereo_preset=get("LOCI_OAK_STEREO_PRESET", "FAST_DENSITY"),
            pp_shaves=int(get("LOCI_OAK_PP_SHAVES", "2")),
            lr_check=env_flag(env, "LOCI_OAK_STEREO_LRC", False),
            extended_disparity=env_flag(env, "LOCI_OAK_STEREO_EXTENDED", False),
            confidence=float(get("LOCI_OAK_CONF", "0.5")),
            bbox_scale=float(get("LOCI_OAK_BBOX_SCALE", "0.5")),
            depth_min_mm=int(float(get("LOCI_OAK_DEPTH_MIN_MM", "200"))),
            depth_max_mm=int(float(get("LOCI_OAK_DEPTH_MAX_MM", "6000"))),
            spatial_algo=get("LOCI_OAK_SPATIAL_ALGO", "median"),
            tag_stream=cls.parse_tag_stream(get("LOCI_TAG_STREAM", "1280x800@5")),
            tag_undistort=env_flag(env, "LOCI_TAG_UNDISTORT", False),
            stall_timeout_s=float(get("LOCI_OAK_STALL_S", "5")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "model": self.model,
            "usb2": self.usb2,
            "axes": self.axes,
            "stereo_preset": self.stereo_preset,
            "pp_shaves": self.pp_shaves,
            "lr_check": self.lr_check,
            "extended_disparity": self.extended_disparity,
            "confidence": self.confidence,
            "bbox_scale": self.bbox_scale,
            "depth_min_mm": self.depth_min_mm,
            "depth_max_mm": self.depth_max_mm,
            "spatial_algo": self.spatial_algo,
            "tag_stream": None if self.tag_stream is None else list(self.tag_stream),
            "tag_undistort": self.tag_undistort,
            "stall_timeout_s": self.stall_timeout_s,
            "reconnect_backoff_s": list(self.reconnect_backoff_s),
        }


@dataclass
class PipelineHandles:
    pipeline: Any
    device: Any
    q_det: Any
    q_tag: Any | None
    labels: list[str]
    tag_size: tuple[int, int] | None
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline construction (pure function of the ``dai`` module, testable with a fake)
# ---------------------------------------------------------------------------


def _enum_member(enum_cls: Any, name: str) -> Any:
    """Look an enum member up by name (pybind11 enums are not subscriptable)."""
    members = getattr(enum_cls, "__members__", None)
    if members is not None and name in members:
        return members[name]
    try:
        return getattr(enum_cls, name)
    except AttributeError as exc:
        known = sorted(members) if members else [m for m in dir(enum_cls) if m.isupper()]
        raise ValueError(f"{name!r} is not a member of {enum_cls}; known: {known}") from exc


def resolve_model(dai: Any, model: str) -> Any:
    """``dai.NNArchive`` for an existing ``*.tar.xz`` path, else a zoo ``NNModelDescription``."""
    if model.endswith(".tar.xz") and os.path.isfile(model):
        return dai.NNArchive(model)
    return dai.NNModelDescription(model, platform="RVC2")


def build_pipeline(dai: Any, device: Any, cfg: OakConfig) -> PipelineHandles:
    """Build the live-verified pipeline on ``device`` (does not start it)."""
    pipeline = dai.Pipeline(device)
    socket = dai.CameraBoardSocket
    cam = pipeline.create(dai.node.Camera).build(socket.CAM_A, sensorFps=cfg.fps)
    left = pipeline.create(dai.node.Camera).build(socket.CAM_B, sensorFps=cfg.fps)
    right = pipeline.create(dai.node.Camera).build(socket.CAM_C, sensorFps=cfg.fps)

    preset = _enum_member(dai.node.StereoDepth.PresetMode, cfg.stereo_preset)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left.requestOutput((640, 400)),
        right=right.requestOutput((640, 400)),
        presetMode=preset,
    )
    stereo.setDepthAlign(socket.CAM_A)
    stereo.setPostProcessingHardwareResources(cfg.pp_shaves, cfg.pp_shaves)
    if cfg.lr_check:
        stereo.setLeftRightCheck(True)
    if cfg.extended_disparity:
        stereo.setExtendedDisparity(True)

    model = resolve_model(dai, cfg.model)
    sdn = pipeline.create(dai.node.SpatialDetectionNetwork).build(cam, stereo, model, fps=cfg.fps)
    sdn.input.setBlocking(False)
    sdn.setConfidenceThreshold(cfg.confidence)
    sdn.setBoundingBoxScaleFactor(cfg.bbox_scale)
    sdn.setDepthLowerThreshold(int(cfg.depth_min_mm))
    sdn.setDepthUpperThreshold(int(cfg.depth_max_mm))
    if cfg.spatial_algo != "default":
        algo = _enum_member(dai.SpatialLocationCalculatorAlgorithm, SPATIAL_ALGOS[cfg.spatial_algo])
        sdn.setSpatialCalculationAlgorithm(algo)
    labels = [str(x) for x in sdn.getClasses()]
    q_det = sdn.out.createOutputQueue(maxSize=4, blocking=False)

    q_tag = None
    tag_size: tuple[int, int] | None = None
    if cfg.tag_stream is not None:
        w, h, tag_fps = cfg.tag_stream
        kwargs: dict[str, Any] = {"type": dai.ImgFrame.Type.NV12, "fps": float(tag_fps)}
        if cfg.tag_undistort:
            kwargs["enableUndistortion"] = True
        tag_out = cam.requestOutput((int(w), int(h)), **kwargs)
        q_tag = tag_out.createOutputQueue(maxSize=2, blocking=False)
        tag_size = (int(w), int(h))

    return PipelineHandles(
        pipeline=pipeline,
        device=device,
        q_det=q_det,
        q_tag=q_tag,
        labels=labels,
        tag_size=tag_size,
        extras={"cam": cam, "stereo": stereo, "sdn": sdn},
    )


# ---------------------------------------------------------------------------
# Message conversion (pure)
# ---------------------------------------------------------------------------


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def spatial_to_detections(msg: Any, labels: list[str], cfg: OakConfig) -> list[SpatialDetection]:
    """Convert a ``SpatialImgDetections`` message; ``z <= 0`` (no depth) -> ``xyz_cam_m=None``."""
    out: list[SpatialDetection] = []
    for d in msg.detections:
        idx = int(d.label)
        label = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
        sc = d.spatialCoordinates
        xyz = depthai_xyz_to_cam((sc.x, sc.y, sc.z), cfg.axes)
        xmin, xmax = sorted((_clip01(d.xmin), _clip01(d.xmax)))
        ymin, ymax = sorted((_clip01(d.ymin), _clip01(d.ymax)))
        out.append(
            SpatialDetection(
                label=str(label).lower(),
                label_id=idx,
                confidence=_clip01(float(d.confidence)),
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                xyz_cam_m=xyz,
            )
        )
    return out


def message_device_ms(msg: Any) -> int | None:
    """Milliseconds of ``msg.getTimestamp()`` (a ``timedelta`` on the steady clock) or None."""
    get = getattr(msg, "getTimestamp", None)
    if get is None:
        return None
    try:
        td = get()
    except Exception:  # pragma: no cover - defensive against binding quirks
        return None
    total = getattr(td, "total_seconds", None)
    if total is not None:
        return int(round(total() * 1000.0))
    try:
        return int(round(float(td) * 1000.0))
    except (TypeError, ValueError):
        return None


def tag_frame_from_img(frame: Any, cfg: OakConfig, timestamp_ms: int) -> TagFrame:
    """Luma plane of an NV12 ``ImgFrame`` plus the intrinsics of that exact output."""
    tf = frame.getTransformation()
    K = np.asarray(tf.getIntrinsicMatrix(), dtype=np.float64).reshape(3, 3)
    dist: np.ndarray | None = None
    if not cfg.tag_undistort:
        coeffs = np.asarray(tf.getDistortionCoefficients(), dtype=np.float64).reshape(-1)
        dist = coeffs if coeffs.size else None
    width, height = _frame_size(frame, tf)
    raw = np.asarray(frame.getFrame())
    if raw.ndim == 2 and raw.shape == (height * 3 // 2, width):
        gray = np.ascontiguousarray(raw[:height])
    elif raw.ndim == 2 and raw.shape == (height, width):
        gray = np.ascontiguousarray(raw)
    else:
        import cv2

        gray = cv2.cvtColor(frame.getCvFrame(), cv2.COLOR_BGR2GRAY)
    return TagFrame(
        timestamp_ms=int(timestamp_ms),
        width=int(width),
        height=int(height),
        K=K,
        dist=dist,
        gray=gray,
        tags=None,
    )


def _frame_size(frame: Any, tf: Any) -> tuple[int, int]:
    get_w, get_h = getattr(frame, "getWidth", None), getattr(frame, "getHeight", None)
    if get_w is not None and get_h is not None:
        return int(get_w()), int(get_h())
    w, h = tf.getSize()
    return int(w), int(h)


# ---------------------------------------------------------------------------
# Device state helpers
# ---------------------------------------------------------------------------


def _is_unbooted(dai: Any, info: Any) -> bool:
    state = getattr(info, "state", None)
    target = getattr(getattr(dai, "XLinkDeviceState", None), "X_LINK_UNBOOTED", None)
    if target is not None and state == target:
        return True
    return "UNBOOTED" in str(state)


def wait_for_unbooted(
    dai: Any,
    timeout_s: float = 45.0,
    poll_s: float = 1.0,
    stop: threading.Event | None = None,
) -> bool:
    """Poll ``dai.Device.getAllAvailableDevices()`` until one device is ``X_LINK_UNBOOTED``."""
    deadline = time.monotonic() + float(timeout_s)
    while True:
        try:
            infos = dai.Device.getAllAvailableDevices()
        except Exception as exc:  # the XLink layer throws while the watchdog resets the device
            logger.debug("getAllAvailableDevices failed: %s", exc)
            infos = []
        if any(_is_unbooted(dai, info) for info in infos):
            return True
        if time.monotonic() >= deadline:
            return False
        if stop is not None:
            if stop.wait(poll_s):
                return False
        else:
            time.sleep(poll_s)


def _usb_speed_name(device: Any) -> str | None:
    get = getattr(device, "getUsbSpeed", None)
    if get is None:
        return None
    try:
        speed = get()
    except Exception:
        return None
    return str(getattr(speed, "name", speed)).rsplit(".", 1)[-1]


def _device_name(device: Any) -> str | None:
    for attr in ("getDeviceName", "getDeviceId", "getMxId"):
        get = getattr(device, attr, None)
        if get is None:
            continue
        try:
            value = get()
        except Exception as exc:
            logger.debug("%s() failed: %s", attr, exc)
            continue
        if value:
            return str(value)
    return None


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------


class OakDetectionSource:
    """``DetectionSource`` over the OAK-D with a reconnect supervisor and stall watchdog.

    ``opener`` (injectable for tests) returns un-started :class:`PipelineHandles`;
    ``events()`` starts the pipeline, polls both queues with ``tryGet()`` + 2 ms
    sleeps, and on any exception closes everything, waits for ``X_LINK_UNBOOTED``
    (``wait_fn``), sleeps the backoff and retries until :meth:`stop` is called.
    """

    name = "oak"

    def __init__(
        self,
        cfg: OakConfig,
        recorder: JsonlRecorder | None = None,
        clock: HostClock | None = None,
        opener: Callable[[], PipelineHandles] | None = None,
        *,
        wait_fn: Callable[[threading.Event], bool] | None = None,
        unbooted_timeout_s: float = 45.0,
    ) -> None:
        self.cfg = cfg
        self.recorder = recorder
        self.clock = clock or HostClock()
        # COCO-80 until the device reports the model's own labels (sdn.getClasses()):
        # the resolver vocabulary must never be empty while the camera is still booting.
        self.classes: list[str] = list(COCO80)
        self.decode_errors = 0
        self._opener = opener
        self._wait_fn = wait_fn
        self._unbooted_timeout_s = float(unbooted_timeout_s)
        self._dai: Any = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = "starting"
        self._error: str | None = None
        self._frames = 0
        self._reconnects = 0
        self._usb_speed: str | None = None
        self._device_name: str | None = None
        self._last_frame_mono: float | None = None
        self._fps = 0.0
        self.states_seen: list[str] = ["starting"]

    # -- protocol -------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._set_state("starting")

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> SourceStatus:
        with self._lock:
            age = None
            if self._last_frame_mono is not None:
                age = max(0.0, time.monotonic() - self._last_frame_mono)
            return SourceStatus(
                state=self._state,
                fps=round(self._fps, 2),
                frames=self._frames,
                reconnects=self._reconnects,
                last_frame_age_s=age,
                usb_speed=self._usb_speed,
                device=self._device_name,
                error=self._error,
            )

    # -- internals ------------------------------------------------------------------

    def _note_decode_error(self, what: str, exc: Exception) -> None:
        """A single message that cannot be converted is dropped, never a camera loss."""
        with self._lock:
            self.decode_errors += 1
            n = self.decode_errors
        if n <= 3:
            logger.exception("OAK %s conversion failed (dropping message): %s", what, exc)
        elif n % 100 == 0:
            logger.warning("OAK %s conversion failed %d times (dropping messages)", what, n)

    def _record_quiet(self, ev: SourceEvent) -> None:
        if self.recorder is None:
            return
        try:
            self.recorder.write(ev)
        except Exception as exc:
            logger.error("OAK recorder failed; recording disabled: %s", exc)
            self.recorder = None

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            if state != self._state:
                self.states_seen.append(state)
            self._state = state
            if error is not None:
                self._error = error or None

    def _import_dai(self) -> Any:
        if self._dai is None:
            import depthai as dai

            self._dai = dai
        return self._dai

    def _default_open(self) -> PipelineHandles:
        dai = self._import_dai()
        device = dai.Device(dai.UsbSpeed.HIGH) if self.cfg.usb2 else dai.Device()
        try:
            return build_pipeline(dai, device, self.cfg)
        except Exception:
            self._close_quietly(None, device)
            raise

    def _open(self) -> PipelineHandles:
        return self._opener() if self._opener is not None else self._default_open()

    def _wait_for_device(self) -> bool:
        if self._wait_fn is not None:
            return self._wait_fn(self._stop)
        if self._opener is not None:
            return True  # injected opener: nothing to poll
        try:
            dai = self._import_dai()
        except Exception as exc:  # depthai missing: nothing to wait for, backoff still applies
            logger.warning("depthai unavailable: %s", exc)
            return False
        return wait_for_unbooted(dai, timeout_s=self._unbooted_timeout_s, stop=self._stop)

    @staticmethod
    def _close_quietly(pipeline: Any, device: Any) -> None:
        for obj, meth in ((pipeline, "stop"), (device, "close")):
            fn = getattr(obj, meth, None) if obj is not None else None
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:
                logger.debug("%s() during close failed: %s", meth, exc)

    def _note_frame(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._frames += 1
            if self._last_frame_mono is not None:
                dt = now - self._last_frame_mono
                inst = 1.0 / dt if dt > 1e-6 else 1000.0
                self._fps = inst if self._fps == 0.0 else 0.8 * self._fps + 0.2 * inst
            self._last_frame_mono = now

    def events(self) -> Iterator[SourceEvent]:
        lo, hi = self.cfg.reconnect_backoff_s
        backoff = float(lo)
        seq = 0
        try:
            yield from self._supervise(seq, lo, hi, backoff)
        finally:
            # also reached when the consumer abandons the generator mid-stream
            self._set_state("stopped")

    def _supervise(self, seq: int, lo: float, hi: float, backoff: float) -> Iterator[SourceEvent]:
        while not self._stop.is_set():
            handles: PipelineHandles | None = None
            try:
                self._set_state("starting")
                handles = self._open()
                self.classes = list(handles.labels)
                with self._lock:
                    self._usb_speed = _usb_speed_name(handles.device)
                    self._device_name = _device_name(handles.device)
                handles.pipeline.start()
                self.clock.reset()
                self._set_state("running", error="")
                backoff = float(lo)
                last_det = time.monotonic()
                last_ts_ms = int(time.time() * 1000)
                while not self._stop.is_set():
                    got = False
                    msg = handles.q_det.tryGet()
                    if msg is not None:
                        got = True
                        last_det = time.monotonic()
                        arrival_ms = int(time.time() * 1000)
                        device_ms = message_device_ms(msg)
                        ts_ms = (
                            self.clock.to_unix_ms(device_ms, arrival_ms)
                            if device_ms is not None
                            else arrival_ms
                        )
                        last_ts_ms = ts_ms
                        try:
                            dets = spatial_to_detections(msg, handles.labels, self.cfg)
                        except Exception as exc:
                            self._note_decode_error("detection", exc)
                            continue
                        frame = DetectionFrame(
                            timestamp_ms=ts_ms,
                            seq=seq,
                            detections=dets,
                            source=self.name,
                            device_ts_ms=device_ms,
                        )
                        seq += 1
                        self._note_frame()
                        self._record_quiet(frame)
                        yield frame
                    if handles.q_tag is not None:
                        img = handles.q_tag.tryGet()
                        if img is not None:
                            got = True
                            arrival_ms = int(time.time() * 1000)
                            device_ms = message_device_ms(img)
                            ts_ms = (
                                self.clock.to_unix_ms(device_ms, arrival_ms)
                                if device_ms is not None
                                else max(arrival_ms, last_ts_ms)
                            )
                            try:
                                tag_frame = tag_frame_from_img(img, self.cfg, ts_ms)
                            except Exception as exc:
                                self._note_decode_error("tag frame", exc)
                                continue
                            self._record_quiet(tag_frame)
                            yield tag_frame
                    if got:
                        continue
                    running = handles.pipeline.isRunning()
                    if not running:
                        raise RuntimeError("pipeline stopped running (device lost?)")
                    if time.monotonic() - last_det > self.cfg.stall_timeout_s:
                        raise StallError(
                            f"no detections for {self.cfg.stall_timeout_s:.1f}s while running"
                        )
                    time.sleep(0.002)
            except Exception as exc:
                if self._stop.is_set():
                    break
                with self._lock:
                    self._reconnects += 1
                self._set_state("waiting_for_camera", error=f"{type(exc).__name__}: {exc}")
                logger.warning("OAK source lost (%s); reconnecting", exc)
            finally:
                if handles is not None:
                    self._close_quietly(handles.pipeline, handles.device)
                    handles = None
            if self._stop.is_set():
                break
            self._wait_for_device()
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2.0, float(hi))


__all__ = [
    "DAI_NAMES_USED",
    "SPATIAL_ALGOS",
    "OakConfig",
    "OakDetectionSource",
    "PipelineHandles",
    "StallError",
    "build_pipeline",
    "message_device_ms",
    "resolve_model",
    "spatial_to_detections",
    "tag_frame_from_img",
    "wait_for_unbooted",
]
