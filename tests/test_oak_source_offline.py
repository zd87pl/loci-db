"""Offline tests for ``demo_spatial/app/oak_source.py`` (T2).

Everything runs against a recording fake ``dai`` module; no camera is ever opened.
The last test resolves every ``dai.*`` name the module uses against the installed
depthai and is skipped where depthai is absent.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.detection_source import DetectionFrame, HostClock, JsonlRecorder, TagFrame  # noqa: E402
from app.oak_source import (  # noqa: E402
    DAI_NAMES_USED,
    OakConfig,
    OakDetectionSource,
    PipelineHandles,
    StallError,
    build_pipeline,
    message_device_ms,
    spatial_to_detections,
    tag_frame_from_img,
    wait_for_unbooted,
)

# ---------------------------------------------------------------------------
# Recording fake of the depthai v3 surface used by build_pipeline
# ---------------------------------------------------------------------------

Call = tuple[str, str, tuple, dict]


class _Enum:
    def __init__(self, name: str, *members: str) -> None:
        self.__members__: dict[str, str] = {m: f"{name}.{m}" for m in members}
        for m, v in self.__members__.items():
            setattr(self, m, v)


class _Rec:
    kind = "node"

    def __init__(self, calls: list[Call]) -> None:
        self._calls = calls

    def _rec(self, name: str, *args: Any, **kwargs: Any) -> None:
        self._calls.append((self.kind, name, args, kwargs))


class FakeQueue:
    def __init__(self, items: list[Any] | None = None) -> None:
        self.items = list(items or [])

    def tryGet(self) -> Any:  # noqa: N802 - depthai spelling
        return self.items.pop(0) if self.items else None


class FakeOutput(_Rec):
    kind = "output"

    def __init__(self, calls: list[Call], owner: str) -> None:
        super().__init__(calls)
        self.owner = owner
        self.queue = FakeQueue()

    def createOutputQueue(self, maxSize: int = 16, blocking: bool = False) -> FakeQueue:  # noqa: N802,N803
        self._rec("createOutputQueue", self.owner, maxSize=maxSize, blocking=blocking)
        return self.queue


class FakeInput(_Rec):
    kind = "input"

    def setBlocking(self, flag: bool) -> None:  # noqa: N802
        self._rec("setBlocking", flag)


class FakeCamera(_Rec):
    kind = "Camera"

    def build(self, socket: Any, sensorFps: float | None = None) -> FakeCamera:  # noqa: N803
        self.socket = socket
        self._rec("build", socket, sensorFps=sensorFps)
        return self

    def requestOutput(self, size: tuple[int, int], **kwargs: Any) -> FakeOutput:  # noqa: N802
        self._rec("requestOutput", self.socket, size, **kwargs)
        return FakeOutput(self._calls, f"{self.socket}:{size}")


class FakeStereo(_Rec):
    kind = "StereoDepth"
    PresetMode = _Enum(
        "PresetMode", "ACCURACY", "DEFAULT", "DENSITY", "FACE", "FAST_ACCURACY", "FAST_DENSITY",
        "HIGH_DETAIL", "ROBOTICS",
    )  # fmt: skip

    def build(self, left: Any, right: Any, presetMode: Any) -> FakeStereo:  # noqa: N803
        self._rec("build", left=left, right=right, presetMode=presetMode)
        return self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("set"):
            return lambda *a, **k: self._rec(name, *a, **k)
        raise AttributeError(name)


class FakeSDN(_Rec):
    kind = "SpatialDetectionNetwork"
    classes = ["person", "bicycle", "car", "cup"]

    def __init__(self, calls: list[Call]) -> None:
        super().__init__(calls)
        self.input = FakeInput(calls)
        self.out = FakeOutput(calls, "sdn.out")

    def build(self, cam: Any, depth: Any, model: Any, fps: float | None = None) -> FakeSDN:
        self._rec("build", cam, depth, model, fps=fps)
        return self

    def getClasses(self) -> list[str]:  # noqa: N802
        self._rec("getClasses")
        return list(self.classes)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("set"):
            return lambda *a, **k: self._rec(name, *a, **k)
        raise AttributeError(name)


class FakePipeline:
    def __init__(self, device: Any, calls: list[Call]) -> None:
        self.device = device
        self.calls = calls
        self.started = False
        self.stopped = False
        calls.append(("Pipeline", "__init__", (device,), {}))

    def create(self, cls: type) -> Any:
        return cls(self.calls)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def isRunning(self) -> bool:  # noqa: N802
        return self.started and not self.stopped


class FakeModelDescription:
    def __init__(self, model: str, platform: str = "") -> None:
        self.model, self.platform = model, platform


class FakeArchive:
    def __init__(self, path: str) -> None:
        self.path = path


def make_fake_dai(calls: list[Call], available: list[Any] | None = None) -> Any:
    def _pipeline(device: Any) -> FakePipeline:
        return FakePipeline(device, calls)

    return types.SimpleNamespace(
        Pipeline=_pipeline,
        node=types.SimpleNamespace(
            Camera=FakeCamera, StereoDepth=FakeStereo, SpatialDetectionNetwork=FakeSDN
        ),
        CameraBoardSocket=_Enum("CameraBoardSocket", "CAM_A", "CAM_B", "CAM_C"),
        ImgFrame=types.SimpleNamespace(Type=_Enum("Type", "NV12", "BGR888p")),
        SpatialLocationCalculatorAlgorithm=_Enum(
            "Algo", "AVERAGE", "MEAN", "MIN", "MAX", "MODE", "MEDIAN"
        ),
        NNModelDescription=FakeModelDescription,
        NNArchive=FakeArchive,
        XLinkDeviceState=_Enum("XLinkDeviceState", "X_LINK_UNBOOTED", "X_LINK_BOOTED"),
        UsbSpeed=_Enum("UsbSpeed", "HIGH", "SUPER"),
        Device=types.SimpleNamespace(getAllAvailableDevices=lambda: list(available or [])),
    )


def _calls(calls: list[Call], kind: str, name: str) -> list[Call]:
    return [c for c in calls if c[0] == kind and c[1] == name]


# ---------------------------------------------------------------------------
# OakConfig
# ---------------------------------------------------------------------------


def test_oak_config_defaults_match_the_verified_pipeline() -> None:
    cfg = OakConfig.from_env({})
    assert cfg == OakConfig()
    assert cfg.stereo_preset == "FAST_DENSITY" and cfg.pp_shaves == 2
    assert cfg.lr_check is False and cfg.extended_disparity is False
    assert cfg.tag_stream == (1280, 800, 5.0) and cfg.tag_undistort is False
    assert cfg.depth_min_mm == 200 and cfg.depth_max_mm == 6000 and cfg.spatial_algo == "median"
    assert cfg.axes == "rdf" and cfg.model == "yolov6-nano" and cfg.usb2 is False


def test_oak_config_from_env_parses_every_key() -> None:
    env = {
        "LOCI_OAK_FPS": "15",
        "LOCI_OAK_MODEL": "/models/x.rvc2.tar.xz",
        "LOCI_OAK_USB2": "1",
        "LOCI_OAK_AXES": "RUF",
        "LOCI_OAK_STEREO_PRESET": "fast_accuracy",
        "LOCI_OAK_PP_SHAVES": "3",
        "LOCI_OAK_STEREO_LRC": "true",
        "LOCI_OAK_STEREO_EXTENDED": "yes",
        "LOCI_OAK_CONF": "0.4",
        "LOCI_OAK_BBOX_SCALE": "0.7",
        "LOCI_OAK_DEPTH_MIN_MM": "300",
        "LOCI_OAK_DEPTH_MAX_MM": "5000",
        "LOCI_OAK_SPATIAL_ALGO": "Average",
        "LOCI_TAG_STREAM": "640x400@10",
        "LOCI_TAG_UNDISTORT": "1",
        "LOCI_OAK_STALL_S": "2.5",
    }
    cfg = OakConfig.from_env(env)
    assert cfg.fps == 15.0 and cfg.model == "/models/x.rvc2.tar.xz" and cfg.usb2 is True
    assert cfg.axes == "ruf" and cfg.stereo_preset == "FAST_ACCURACY" and cfg.pp_shaves == 3
    assert cfg.lr_check is True and cfg.extended_disparity is True
    assert cfg.confidence == 0.4 and cfg.bbox_scale == 0.7
    assert cfg.depth_min_mm == 300 and cfg.depth_max_mm == 5000 and cfg.spatial_algo == "average"
    assert cfg.tag_stream == (640, 400, 10.0) and cfg.tag_undistort is True
    assert cfg.stall_timeout_s == 2.5
    assert cfg.to_dict()["tag_stream"] == [640, 400, 10.0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1280x800@5", (1280, 800, 5.0)),
        ("640x400@10", (640, 400, 10.0)),
        ("640X400", (640, 400, 5.0)),
        ("off", None),
        ("", None),
        ("none", None),
    ],
)
def test_parse_tag_stream(text: str, expected: tuple[int, int, float] | None) -> None:
    assert OakConfig.parse_tag_stream(text) == expected


@pytest.mark.parametrize("bad", ["1280", "axb@5", "1280x800@0", "0x800@5"])
def test_parse_tag_stream_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        OakConfig.parse_tag_stream(bad)


def test_oak_config_validation() -> None:
    with pytest.raises(ValueError, match="spatial algorithm"):
        OakConfig(spatial_algo="fancy")
    with pytest.raises(ValueError, match="depth"):
        OakConfig(depth_min_mm=500, depth_max_mm=400)


# ---------------------------------------------------------------------------
# build_pipeline against the recording fake
# ---------------------------------------------------------------------------


def test_build_pipeline_default_matches_hardware_facts() -> None:
    calls: list[Call] = []
    dai = make_fake_dai(calls)
    device = object()
    h = build_pipeline(dai, device, OakConfig())

    assert isinstance(h, PipelineHandles) and h.device is device
    assert calls[0] == ("Pipeline", "__init__", (device,), {})
    assert h.pipeline.started is False, "build_pipeline must not start the pipeline"

    cams = _calls(calls, "Camera", "build")
    assert [c[2][0] for c in cams] == [
        "CameraBoardSocket.CAM_A", "CameraBoardSocket.CAM_B", "CameraBoardSocket.CAM_C"
    ]  # fmt: skip
    assert all(c[3] == {"sensorFps": 10.0} for c in cams)

    mono = [c for c in _calls(calls, "Camera", "requestOutput") if c[2][1] == (640, 400)]
    assert [c[2][0] for c in mono] == ["CameraBoardSocket.CAM_B", "CameraBoardSocket.CAM_C"]
    assert all(c[3] == {} for c in mono)

    [stereo_build] = _calls(calls, "StereoDepth", "build")
    assert stereo_build[3]["presetMode"] == "PresetMode.FAST_DENSITY"
    assert stereo_build[3]["left"].owner == "CameraBoardSocket.CAM_B:(640, 400)"
    assert stereo_build[3]["right"].owner == "CameraBoardSocket.CAM_C:(640, 400)"
    assert _calls(calls, "StereoDepth", "setDepthAlign") == [
        ("StereoDepth", "setDepthAlign", ("CameraBoardSocket.CAM_A",), {})
    ]
    assert _calls(calls, "StereoDepth", "setPostProcessingHardwareResources") == [
        ("StereoDepth", "setPostProcessingHardwareResources", (2, 2), {})
    ]
    assert _calls(calls, "StereoDepth", "setLeftRightCheck") == []
    assert _calls(calls, "StereoDepth", "setExtendedDisparity") == []

    [sdn_build] = _calls(calls, "SpatialDetectionNetwork", "build")
    cam, stereo, model = sdn_build[2]
    assert isinstance(cam, FakeCamera) and cam.socket == "CameraBoardSocket.CAM_A"
    assert isinstance(stereo, FakeStereo)
    assert isinstance(model, FakeModelDescription)
    assert (model.model, model.platform) == ("yolov6-nano", "RVC2")
    assert sdn_build[3] == {"fps": 10.0}
    assert _calls(calls, "input", "setBlocking") == [("input", "setBlocking", (False,), {})]
    sdn_sets = {
        c[1]: c[2] for c in calls if c[0] == "SpatialDetectionNetwork" and c[1][:3] == "set"
    }
    assert sdn_sets == {
        "setConfidenceThreshold": (0.5,),
        "setBoundingBoxScaleFactor": (0.5,),
        "setDepthLowerThreshold": (200,),
        "setDepthUpperThreshold": (6000,),
        "setSpatialCalculationAlgorithm": ("Algo.MEDIAN",),
    }
    assert h.labels == ["person", "bicycle", "car", "cup"]
    assert _calls(calls, "output", "createOutputQueue")[0] == (
        "output", "createOutputQueue", ("sdn.out",), {"maxSize": 4, "blocking": False}
    )  # fmt: skip
    assert h.q_det is h.extras["sdn"].out.queue

    tag = [c for c in _calls(calls, "Camera", "requestOutput") if c[2][1] == (1280, 800)]
    assert len(tag) == 1 and tag[0][2][0] == "CameraBoardSocket.CAM_A"
    assert tag[0][3] == {"type": "Type.NV12", "fps": 5.0}
    assert "enableUndistortion" not in tag[0][3]
    assert h.tag_size == (1280, 800)
    assert _calls(calls, "output", "createOutputQueue")[1][3] == {"maxSize": 2, "blocking": False}
    assert h.q_tag is not None


def test_build_pipeline_optional_flags() -> None:
    calls: list[Call] = []
    cfg = OakConfig(
        lr_check=True,
        extended_disparity=True,
        pp_shaves=3,
        stereo_preset="default",
        spatial_algo="min",
        tag_stream=(640, 400, 10.0),
        tag_undistort=True,
        confidence=0.4,
        bbox_scale=0.7,
        depth_min_mm=300,
        depth_max_mm=5000,
        fps=15.0,
    )
    build_pipeline(make_fake_dai(calls), object(), cfg)
    assert _calls(calls, "StereoDepth", "setLeftRightCheck") == [
        ("StereoDepth", "setLeftRightCheck", (True,), {})
    ]
    assert _calls(calls, "StereoDepth", "setExtendedDisparity") == [
        ("StereoDepth", "setExtendedDisparity", (True,), {})
    ]
    assert _calls(calls, "StereoDepth", "setPostProcessingHardwareResources")[0][2] == (3, 3)
    assert _calls(calls, "StereoDepth", "build")[0][3]["presetMode"] == "PresetMode.DEFAULT"
    assert _calls(calls, "SpatialDetectionNetwork", "setSpatialCalculationAlgorithm")[0][2] == (
        "Algo.MIN",
    )
    assert _calls(calls, "SpatialDetectionNetwork", "setDepthLowerThreshold")[0][2] == (300,)
    assert _calls(calls, "SpatialDetectionNetwork", "setDepthUpperThreshold")[0][2] == (5000,)
    assert _calls(calls, "SpatialDetectionNetwork", "build")[0][3] == {"fps": 15.0}
    assert all(c[3] == {"sensorFps": 15.0} for c in _calls(calls, "Camera", "build"))
    tag = [c for c in _calls(calls, "Camera", "requestOutput") if c[2][1] == (640, 400)]
    tag = [c for c in tag if c[2][0] == "CameraBoardSocket.CAM_A"]
    assert tag[0][3] == {"type": "Type.NV12", "fps": 10.0, "enableUndistortion": True}


def test_build_pipeline_no_tag_stream_and_default_algo() -> None:
    calls: list[Call] = []
    h = build_pipeline(
        make_fake_dai(calls), object(), OakConfig(tag_stream=None, spatial_algo="default")
    )
    assert h.q_tag is None and h.tag_size is None
    assert [
        c for c in _calls(calls, "Camera", "requestOutput") if c[2][0] == "CameraBoardSocket.CAM_A"
    ] == []
    assert _calls(calls, "SpatialDetectionNetwork", "setSpatialCalculationAlgorithm") == []
    assert len(_calls(calls, "output", "createOutputQueue")) == 1


def test_build_pipeline_uses_nn_archive_for_existing_path(tmp_path: Path) -> None:
    archive = tmp_path / "YOLOv6_Nano-R2_COCO_512x384.rvc2.tar.xz"
    archive.write_bytes(b"not really an archive")
    calls: list[Call] = []
    build_pipeline(make_fake_dai(calls), object(), OakConfig(model=str(archive)))
    model = _calls(calls, "SpatialDetectionNetwork", "build")[0][2][2]
    assert isinstance(model, FakeArchive) and model.path == str(archive)
    # a missing path falls back to a zoo description with the same string
    calls.clear()
    build_pipeline(
        make_fake_dai(calls), object(), OakConfig(model=str(tmp_path / "missing.tar.xz"))
    )
    model = _calls(calls, "SpatialDetectionNetwork", "build")[0][2][2]
    assert isinstance(model, FakeModelDescription) and model.platform == "RVC2"


def test_build_pipeline_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="HIGH_DENSITY"):
        build_pipeline(make_fake_dai([]), object(), OakConfig(stereo_preset="HIGH_DENSITY"))


# ---------------------------------------------------------------------------
# spatial_to_detections / message_device_ms / tag_frame_from_img
# ---------------------------------------------------------------------------


def _det(
    label: int, conf: float, box: tuple[float, float, float, float], xyz: tuple[float, float, float]
) -> Any:
    sc = types.SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2])
    xmin, ymin, xmax, ymax = box
    return types.SimpleNamespace(
        label=label,
        confidence=conf,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        spatialCoordinates=sc,
    )


def test_spatial_to_detections_zero_depth_axes_and_labels() -> None:
    labels = ["person", "bicycle", "car", "cup"]
    msg = types.SimpleNamespace(
        detections=[
            _det(3, 0.91, (0.1, 0.2, 0.3, 0.4), (120.0, -50.0, 1840.0)),
            _det(0, 0.70, (0.5, 0.5, 0.9, 1.1), (0.0, 0.0, 0.0)),  # no depth -> zeros
            _det(7, 0.60, (0.6, 0.1, 0.8, 0.3), (10.0, 10.0, 500.0)),  # unknown class index
        ]
    )
    out = spatial_to_detections(msg, labels, OakConfig())
    assert [d.label for d in out] == ["cup", "person", "class_7"]
    assert out[0].xyz_cam_m == pytest.approx((0.12, -0.05, 1.84)) and out[0].label_id == 3
    assert out[1].xyz_cam_m is None and out[1].depth_m is None
    assert (out[1].xmax, out[1].ymax) == (0.9, 1.0), "boxes are clipped to [0,1]"
    assert out[2].label_id == 7 and out[2].xyz_cam_m == pytest.approx((0.01, 0.01, 0.5))
    flipped = spatial_to_detections(msg, labels, OakConfig(axes="ruf"))
    assert flipped[0].xyz_cam_m == pytest.approx((0.12, 0.05, 1.84))
    assert flipped[1].xyz_cam_m is None


def test_message_device_ms() -> None:
    assert (
        message_device_ms(types.SimpleNamespace(getTimestamp=lambda: timedelta(seconds=12.3456)))
        == 12346
    )
    assert message_device_ms(types.SimpleNamespace()) is None
    assert message_device_ms(types.SimpleNamespace(getTimestamp=lambda: 1.5)) == 1500


class _FakeTransformation:
    def __init__(self, w: int, h: int, dist: list[float]) -> None:
        self._w, self._h, self._dist = w, h, dist

    def getIntrinsicMatrix(self) -> list[list[float]]:  # noqa: N802
        return [[918.3, 0.0, 645.8], [0.0, 917.5, 417.4], [0.0, 0.0, 1.0]]

    def getDistortionCoefficients(self) -> list[float]:  # noqa: N802
        return list(self._dist)

    def getSize(self) -> tuple[int, int]:  # noqa: N802
        return (self._w, self._h)


class _FakeImgFrame:
    def __init__(self, raw: np.ndarray, w: int, h: int, dist: list[float] | None = None) -> None:
        self._raw, self._w, self._h = raw, w, h
        self._tf = _FakeTransformation(
            w,
            h,
            dist
            if dist is not None
            else [-0.78, -9.39, 0.0006, 0.0004, 31.35, -0.93, -8.76, 30.43],
        )
        self.cv_calls = 0

    def getTransformation(self) -> _FakeTransformation:  # noqa: N802
        return self._tf

    def getFrame(self) -> np.ndarray:  # noqa: N802
        return self._raw

    def getCvFrame(self) -> np.ndarray:  # noqa: N802
        self.cv_calls += 1
        if self._raw.ndim == 3:
            return self._raw
        return np.dstack([self._raw, self._raw, self._raw])

    def getWidth(self) -> int:  # noqa: N802
        return self._w

    def getHeight(self) -> int:  # noqa: N802
        return self._h


def test_tag_frame_from_img_luma_plane() -> None:
    w, h = 64, 32
    nv12 = np.zeros((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h] = np.arange(h, dtype=np.uint8)[:, None]  # luma rows
    nv12[h:] = 128  # chroma plane must not leak into gray
    frame = _FakeImgFrame(nv12, w, h)
    tf = tag_frame_from_img(frame, OakConfig(), 1234)
    assert isinstance(tf, TagFrame) and tf.timestamp_ms == 1234
    assert (tf.width, tf.height) == (w, h) and tf.tags is None
    assert tf.gray.shape == (h, w) and tf.gray.dtype == np.uint8 and tf.gray[5, 0] == 5
    assert tf.gray.flags["C_CONTIGUOUS"]
    assert tf.K.shape == (3, 3) and tf.K[0, 0] == 918.3
    assert tf.dist is not None and tf.dist.shape == (8,) and frame.cv_calls == 0


def test_tag_frame_from_img_undistorted_has_no_dist() -> None:
    w, h = 16, 8
    frame = _FakeImgFrame(np.zeros((h * 3 // 2, w), dtype=np.uint8), w, h)
    assert tag_frame_from_img(frame, OakConfig(tag_undistort=True), 1).dist is None
    frame_nodist = _FakeImgFrame(np.zeros((h * 3 // 2, w), dtype=np.uint8), w, h, dist=[])
    assert tag_frame_from_img(frame_nodist, OakConfig(), 1).dist is None


def test_tag_frame_from_img_fallback_uses_cv2(monkeypatch: pytest.MonkeyPatch) -> None:
    w, h = 16, 8
    calls: list[tuple] = []

    def cvt(img: np.ndarray, code: int) -> np.ndarray:
        calls.append((img.shape, code))
        return img[:, :, 0].copy()

    fake_cv2 = types.SimpleNamespace(cvtColor=cvt, COLOR_BGR2GRAY=6)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    # a frame whose getFrame() is not the NV12 plane layout (e.g. an interleaved BGR buffer)
    frame = _FakeImgFrame(np.full((h, w, 3), 7, dtype=np.uint8), w, h)
    tf = tag_frame_from_img(frame, OakConfig(), 5)
    assert frame.cv_calls == 1 and calls == [((h, w, 3), 6)]
    assert tf.gray.shape == (h, w)


# ---------------------------------------------------------------------------
# wait_for_unbooted
# ---------------------------------------------------------------------------


def test_wait_for_unbooted_states() -> None:
    unbooted = types.SimpleNamespace(state="XLinkDeviceState.X_LINK_UNBOOTED")
    booted = types.SimpleNamespace(state="XLinkDeviceState.X_LINK_BOOTED")
    assert wait_for_unbooted(make_fake_dai([], [unbooted]), timeout_s=0.1, poll_s=0.01)
    assert not wait_for_unbooted(make_fake_dai([], [booted]), timeout_s=0.05, poll_s=0.01)
    assert not wait_for_unbooted(make_fake_dai([], []), timeout_s=0.05, poll_s=0.01)

    def boom() -> list[Any]:
        raise RuntimeError("X_LINK_ERROR")

    broken = types.SimpleNamespace(Device=types.SimpleNamespace(getAllAvailableDevices=boom))
    assert not wait_for_unbooted(broken, timeout_s=0.05, poll_s=0.01)
    stop = threading.Event()
    stop.set()
    assert not wait_for_unbooted(make_fake_dai([], [booted]), timeout_s=10, poll_s=0.01, stop=stop)


# ---------------------------------------------------------------------------
# Supervisor state machine
# ---------------------------------------------------------------------------


class _FakeDevice:
    closed = 0

    def getUsbSpeed(self) -> Any:  # noqa: N802
        return types.SimpleNamespace(name="HIGH")

    def getDeviceName(self) -> str:  # noqa: N802
        return "OAK-D-LITE"

    def close(self) -> None:
        self.closed += 1


def _msg(ts_s: float, *dets: Any) -> Any:
    return types.SimpleNamespace(
        detections=list(dets), getTimestamp=lambda: timedelta(seconds=ts_s)
    )


def _handles(n_frames: int, with_tag: bool = False) -> PipelineHandles:
    pipeline = FakePipeline(object(), [])
    q_det = FakeQueue(
        [
            _msg(10.0 + i * 0.1, _det(3, 0.9, (0.1, 0.1, 0.3, 0.3), (0.0, 0.0, 1500.0)))
            for i in range(n_frames)
        ]
    )
    q_tag = None
    if with_tag:
        w, h = 16, 8
        img = _FakeImgFrame(np.zeros((h * 3 // 2, w), dtype=np.uint8), w, h)
        img.getTimestamp = lambda: timedelta(seconds=10.05)  # type: ignore[attr-defined]
        q_tag = FakeQueue([img])
    return PipelineHandles(
        pipeline, _FakeDevice(), q_det, q_tag, ["person", "bicycle", "car", "cup"], None
    )


def test_supervisor_reconnects_after_two_failures_then_streams(tmp_path: Path) -> None:
    attempts: list[int] = []
    handles = _handles(3, with_tag=True)

    def opener() -> PipelineHandles:
        attempts.append(1)
        if len(attempts) <= 2:
            raise RuntimeError("X_LINK_DEVICE_NOT_FOUND")
        return handles

    waits: list[bool] = []
    cfg = OakConfig(reconnect_backoff_s=(0.01, 0.02), stall_timeout_s=0.2)
    rec = JsonlRecorder(tmp_path / "oak.jsonl")
    src = OakDetectionSource(
        cfg,
        recorder=rec,
        clock=HostClock(),
        opener=opener,
        wait_fn=lambda stop: waits.append(True) or True,
    )
    src.start()
    got: list[Any] = []

    def consume() -> None:
        for ev in src.events():
            got.append(ev)
            if len([e for e in got if isinstance(e, DetectionFrame)]) == 3:
                src.stop()

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    th.join(timeout=5.0)
    assert not th.is_alive(), "events() must return after stop()"
    rec.close()

    dets = [e for e in got if isinstance(e, DetectionFrame)]
    tags = [e for e in got if isinstance(e, TagFrame)]
    assert len(dets) == 3 and len(tags) == 1
    assert [d.seq for d in dets] == [0, 1, 2]
    assert dets[0].detections[0].label == "cup" and dets[0].detections[
        0
    ].xyz_cam_m == pytest.approx((0.0, 0.0, 1.5))
    assert dets[0].source == "oak" and dets[0].device_ts_ms == 10000
    now_ms = time.time() * 1000
    assert abs(dets[0].timestamp_ms - now_ms) < 5_000, (
        "device steady time mapped onto the host clock"
    )
    assert tags[0].gray is not None and tags[0].tags is None

    st = src.status()
    assert st.reconnects == 2 and st.frames == 3 and st.state == "stopped"
    assert st.usb_speed == "HIGH" and st.device == "OAK-D-LITE"
    assert src.classes == ["person", "bicycle", "car", "cup"]
    assert src.states_seen[:5] == [
        "starting",
        "waiting_for_camera",
        "starting",
        "waiting_for_camera",
        "starting",
    ]
    assert src.states_seen[5:] == ["running", "stopped"]
    assert len(attempts) == 3 and len(waits) == 2
    assert handles.pipeline.started and handles.pipeline.stopped and handles.device.closed == 1
    assert (tmp_path / "oak.jsonl").read_text().count("\n") == 4


def test_supervisor_stall_triggers_reconnect() -> None:
    opened: list[PipelineHandles] = []

    def opener() -> PipelineHandles:
        h = _handles(0)  # pipeline "runs" but never delivers detections
        opened.append(h)
        return h

    cfg = OakConfig(stall_timeout_s=0.05, reconnect_backoff_s=(0.01, 0.01))
    src = OakDetectionSource(cfg, opener=opener)
    src.start()

    def consume() -> None:
        for _ in src.events():  # pragma: no cover - nothing is ever yielded
            pass

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    deadline = time.monotonic() + 3.0
    while src.status().reconnects < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    src.stop()
    th.join(timeout=3.0)
    assert not th.is_alive()
    st = src.status()
    assert st.reconnects >= 2 and st.frames == 0
    assert "StallError" in (st.error or "")
    assert len(opened) >= 2 and all(h.pipeline.stopped and h.device.closed == 1 for h in opened)
    assert src.states_seen.count("waiting_for_camera") >= 2 and src.states_seen[-1] == "stopped"


def test_supervisor_pipeline_not_running_is_a_loss() -> None:
    def opener() -> PipelineHandles:
        h = _handles(0)
        h.pipeline.start = lambda: None  # type: ignore[method-assign]  # never reports running
        return h

    src = OakDetectionSource(OakConfig(reconnect_backoff_s=(0.01, 0.01)), opener=opener)
    src.start()
    th = threading.Thread(target=lambda: list(src.events()), daemon=True)
    th.start()
    deadline = time.monotonic() + 3.0
    while src.status().reconnects < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    src.stop()
    th.join(timeout=3.0)
    assert src.status().reconnects >= 1
    assert "stopped running" in (src.status().error or "")


def test_stop_before_events_returns_immediately() -> None:
    src = OakDetectionSource(OakConfig(), opener=lambda: _handles(5))
    src.stop()
    assert list(src.events()) == []
    assert src.status().state == "stopped"
    assert isinstance(StallError("x"), RuntimeError)


# ---------------------------------------------------------------------------
# Real depthai attribute check (skipped without the wheel)
# ---------------------------------------------------------------------------


def test_every_dai_name_used_resolves_on_installed_depthai() -> None:
    dai = pytest.importorskip("depthai")
    missing: list[str] = []
    for dotted in DAI_NAMES_USED:
        obj: Any = dai
        for part in dotted.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                missing.append(dotted)
                break
    assert missing == [], f"depthai {dai.__version__} lacks: {missing}"
    assert "FAST_DENSITY" in dai.node.StereoDepth.PresetMode.__members__
    # keyword names used by build_pipeline must appear in the binding signatures
    assert "sensorFps" in dai.node.Camera.build.__doc__
    assert "enableUndistortion" in dai.node.Camera.requestOutput.__doc__
    assert "presetMode" in dai.node.StereoDepth.build.__doc__
    assert "fps" in dai.node.SpatialDetectionNetwork.build.__doc__
    assert "maxSize" in dai.Node.Output.createOutputQueue.__doc__
    assert "platform" in dai.NNModelDescription.__init__.__doc__
    assert "maxUsbSpeed" in dai.Device.__init__.__doc__
    assert "defaultDevice" in dai.Pipeline.__init__.__doc__


def test_dai_fake_covers_build_pipeline_against_real_enums() -> None:
    """With real depthai present, the enum lookups used by build_pipeline must work."""
    dai = pytest.importorskip("depthai")
    from app.oak_source import _enum_member

    assert (
        _enum_member(dai.node.StereoDepth.PresetMode, "FAST_DENSITY")
        == dai.node.StereoDepth.PresetMode.FAST_DENSITY
    )
    assert (
        _enum_member(dai.SpatialLocationCalculatorAlgorithm, "MEDIAN")
        == dai.SpatialLocationCalculatorAlgorithm.MEDIAN
    )
    with pytest.raises(ValueError, match="HIGH_DENSITY"):
        _enum_member(dai.node.StereoDepth.PresetMode, "HIGH_DENSITY")


# ---------------------------------------------------------------------------
# Review fixes: vocabulary before the device opens, per-message decode errors
# ---------------------------------------------------------------------------


def test_classes_default_to_coco80_until_the_device_reports_labels() -> None:
    from app.detection_source import COCO80

    handles = _handles(2)
    src = OakDetectionSource(OakConfig(stall_timeout_s=5.0), opener=lambda: handles)
    assert src.classes == list(COCO80) and len(src.classes) == 80
    assert "cup" in src.classes and "bottle" in src.classes
    src.start()
    got: list[Any] = []

    def consume() -> None:
        for ev in src.events():
            got.append(ev)
            src.stop()

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    th.join(timeout=3.0)
    assert not th.is_alive() and len(got) == 1
    assert src.classes == ["person", "bicycle", "car", "cup"]  # the model's own labels


def test_message_conversion_error_drops_the_message_not_the_camera() -> None:
    handles = _handles(3, with_tag=True)
    handles.q_tag = FakeQueue([object()])  # no getTransformation(): conversion fails
    cfg = OakConfig(stall_timeout_s=5.0, reconnect_backoff_s=(0.01, 0.01))
    src = OakDetectionSource(cfg, opener=lambda: handles)
    src.start()
    got: list[Any] = []

    def consume() -> None:
        for ev in src.events():
            got.append(ev)
            if len(got) >= 3:
                src.stop()

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    th.join(timeout=3.0)
    assert not th.is_alive()
    assert len(got) == 3 and all(isinstance(ev, DetectionFrame) for ev in got)
    st = src.status()
    assert src.decode_errors == 1 and st.reconnects == 0 and st.frames == 3
    assert "waiting_for_camera" not in src.states_seen
    assert handles.device.closed == 1  # closed once, on stop()
