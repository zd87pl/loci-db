"""Hardware-neutral detection source contract for the handheld demo (design §4.2).

This module defines the message types exchanged between a detection source (the
OAK-D camera, the scripted simulator or a JSONL replay) and the handheld runtime,
plus the simulator and the record/replay helpers.  It imports neither ``loci`` nor
``cv2`` nor ``depthai``: the only third-party dependency is numpy.

Coordinate conventions (design §4.1, single source of truth):

* DepthAI v3 ``spatialCoordinates`` are millimetres in a right/down/forward (RDF)
  frame with the origin at the RGB camera.  ``x = y = z = 0`` exactly means "no
  depth" and such detections carry ``xyz_cam_m=None``.
* The camera frame is the OpenCV frame (+x right, +y down, +z forward) in metres.
  :func:`depthai_xyz_to_cam` is the only place where the DepthAI axis sign lives.
* The room frame is right-handed, +y up, +z from the anchor wall into the room,
  +x to the viewer's right when facing the anchor tag; origin on the floor below
  the anchor tag.  ``T_a_b`` denotes a 4x4 matrix with ``p_a = T_a_b @ p_b``.
* Timestamps are Unix milliseconds (``int``) everywhere.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class vocabulary of the on-camera detector (COCO-80, model order)
# ---------------------------------------------------------------------------

COCO80: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]  # fmt: skip

# ---------------------------------------------------------------------------
# Axis mapping (the ONLY place the DepthAI y-sign lives)
# ---------------------------------------------------------------------------

AXIS_FIX: dict[str, np.ndarray] = {
    "rdf": np.eye(3),
    "ruf": np.diag([1.0, -1.0, 1.0]),
}


def depthai_xyz_to_cam(
    xyz_mm: Sequence[float], axes: str = "rdf"
) -> tuple[float, float, float] | None:
    """Convert DepthAI spatial coordinates (mm) to the OpenCV camera frame (metres).

    Returns ``None`` when ``z_mm <= 0`` (DepthAI reports exact zeros for boxes
    without valid depth).  ``axes="rdf"`` is the identity; ``"ruf"`` flips y for
    DepthAI v2 or a firmware that reports y up.
    """
    try:
        fix = AXIS_FIX[axes]
    except KeyError as exc:
        raise ValueError(f"unknown axes {axes!r}; expected one of {sorted(AXIS_FIX)}") from exc
    x_mm, y_mm, z_mm = (float(v) for v in xyz_mm)
    if not (z_mm > 0.0) or not math.isfinite(z_mm):
        return None
    v = fix @ np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0
    return (float(v[0]), float(v[1]), float(v[2]))


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


@dataclass(frozen=True)
class SpatialDetection:
    """One detection of the on-camera network with optional metric position."""

    label: str
    label_id: int
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    xyz_cam_m: tuple[float, float, float] | None

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def depth_m(self) -> float | None:
        return None if self.xyz_cam_m is None else self.xyz_cam_m[2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "confidence": self.confidence,
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
            "xyz_cam_m": None if self.xyz_cam_m is None else list(self.xyz_cam_m),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SpatialDetection:
        xyz = d.get("xyz_cam_m")
        return cls(
            label=str(d["label"]),
            label_id=int(d.get("label_id", -1)),
            confidence=float(d.get("confidence", 0.0)),
            xmin=float(d["xmin"]),
            ymin=float(d["ymin"]),
            xmax=float(d["xmax"]),
            ymax=float(d["ymax"]),
            xyz_cam_m=None if xyz is None else (float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )


@dataclass
class TagDetection:
    """An AprilTag seen in a :class:`TagFrame`; corners TL, TR, BR, BL in pixels."""

    tag_id: int
    corners_px: np.ndarray
    decision_margin: float
    hamming: int

    def __post_init__(self) -> None:
        self.corners_px = np.asarray(self.corners_px, dtype=np.float64).reshape(4, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag_id": int(self.tag_id),
            "corners_px": self.corners_px.tolist(),
            "decision_margin": float(self.decision_margin),
            "hamming": int(self.hamming),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TagDetection:
        return cls(
            tag_id=int(d["tag_id"]),
            corners_px=np.asarray(d["corners_px"], dtype=np.float64),
            decision_margin=float(d.get("decision_margin", 0.0)),
            hamming=int(d.get("hamming", 0)),
        )


@dataclass
class TagFrame:
    """A frame of the tag stream: either a luma image to detect on, or pre-detected tags."""

    timestamp_ms: int
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray | None
    gray: np.ndarray | None
    tags: list[TagDetection] | None

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        if self.dist is not None:
            self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form without the image (design §4.14)."""
        return {
            "timestamp_ms": int(self.timestamp_ms),
            "width": int(self.width),
            "height": int(self.height),
            "K": self.K.tolist(),
            "dist": None if self.dist is None else self.dist.tolist(),
            "tags": None if self.tags is None else [t.to_dict() for t in self.tags],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TagFrame:
        tags = d.get("tags")
        return cls(
            timestamp_ms=int(d["timestamp_ms"]),
            width=int(d["width"]),
            height=int(d["height"]),
            K=np.asarray(d["K"], dtype=np.float64),
            dist=None if d.get("dist") is None else np.asarray(d["dist"], dtype=np.float64),
            gray=None,
            tags=None if tags is None else [TagDetection.from_dict(t) for t in tags],
        )


@dataclass
class CameraPose:
    """Camera pose in the room frame as produced by the localizer (or the sim oracle)."""

    room: str | None
    frame: str
    T_room_cam: np.ndarray | None
    anchored: bool
    anchor_confidence: float
    tag_ids: tuple[int, ...]
    age_s: float

    def __post_init__(self) -> None:
        if self.T_room_cam is not None:
            self.T_room_cam = np.asarray(self.T_room_cam, dtype=np.float64).reshape(4, 4)
        self.tag_ids = tuple(int(t) for t in self.tag_ids)

    def to_dict(self) -> dict[str, Any]:
        xyz: list[float] | None = None
        heading: float | None = None
        if self.T_room_cam is not None:
            xyz = [float(v) for v in self.T_room_cam[:3, 3]]
            heading = yaw_deg_of(self.T_room_cam)
        age = None if self.age_s is None or not math.isfinite(self.age_s) else float(self.age_s)
        return {
            "room": self.room,
            "frame": self.frame,
            "anchored": bool(self.anchored),
            "anchor_confidence": float(self.anchor_confidence),
            "camera_xyz_m": xyz,
            "heading_deg": heading,
            "tag_ids": list(self.tag_ids),
            "age_s": age,
        }

    def to_record(self) -> dict[str, Any]:
        """Lossless JSON form (``to_dict`` plus the full matrix); used by the recorder."""
        d = self.to_dict()
        d["T_room_cam"] = None if self.T_room_cam is None else self.T_room_cam.tolist()
        d["age_s"] = None if not math.isfinite(self.age_s) else float(self.age_s)
        return d

    @classmethod
    def from_record(cls, d: Mapping[str, Any]) -> CameraPose:
        T = d.get("T_room_cam")
        age = d.get("age_s")
        return cls(
            room=d.get("room"),
            frame=str(d.get("frame", "camera")),
            T_room_cam=None if T is None else np.asarray(T, dtype=np.float64),
            anchored=bool(d.get("anchored", False)),
            anchor_confidence=float(d.get("anchor_confidence", 0.0)),
            tag_ids=tuple(int(t) for t in d.get("tag_ids", ())),
            age_s=float("inf") if age is None else float(age),
        )


@dataclass
class DetectionFrame:
    """All detections of one camera frame, time-stamped in host Unix milliseconds."""

    timestamp_ms: int
    seq: int
    detections: list[SpatialDetection]
    source: str
    device_ts_ms: int | None = None
    oracle_pose: CameraPose | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "timestamp_ms": int(self.timestamp_ms),
            "seq": int(self.seq),
            "source": self.source,
            "device_ts_ms": None if self.device_ts_ms is None else int(self.device_ts_ms),
            "detections": [x.to_dict() for x in self.detections],
        }
        if self.oracle_pose is not None:
            d["oracle_pose"] = self.oracle_pose.to_record()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], source: str | None = None) -> DetectionFrame:
        pose = d.get("oracle_pose")
        dev = d.get("device_ts_ms")
        return cls(
            timestamp_ms=int(d["timestamp_ms"]),
            seq=int(d.get("seq", 0)),
            detections=[SpatialDetection.from_dict(x) for x in d.get("detections", [])],
            source=source if source is not None else str(d.get("source", "replay")),
            device_ts_ms=None if dev is None else int(dev),
            oracle_pose=None if pose is None else CameraPose.from_record(pose),
        )


SourceEvent = DetectionFrame | TagFrame


@dataclass
class SourceStatus:
    """Snapshot of a source's health, surfaced in ``/api/handheld/status``."""

    state: str
    fps: float
    frames: int
    reconnects: int
    last_frame_age_s: float | None
    usb_speed: str | None
    device: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class DetectionSource(Protocol):
    """Blocking event producer; ``events()`` ends soon after ``stop()``."""

    name: str
    classes: list[str]

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def events(self) -> Iterator[SourceEvent]: ...

    def status(self) -> SourceStatus: ...


# ---------------------------------------------------------------------------
# Host clock (device steady-clock milliseconds -> Unix milliseconds)
# ---------------------------------------------------------------------------


class HostClock:
    """Map device timestamps onto the host Unix clock with a min-latency offset.

    ``offset = min(arrival_ms - device_ms)`` over the first ``warmup`` samples; a
    converted value further than ``max_skew_ms`` from the arrival time (device
    reboot, clock jump) falls back to the arrival time, is counted in
    :attr:`fallbacks` and re-arms the warm-up.
    """

    def __init__(self, max_skew_ms: int = 1000, warmup: int = 20) -> None:
        self.max_skew_ms = int(max_skew_ms)
        self.warmup = max(1, int(warmup))
        self.fallbacks = 0
        self._offset: int | None = None
        self._samples = 0

    @property
    def offset_ms(self) -> int | None:
        return self._offset

    @property
    def synced(self) -> bool:
        return self._offset is not None and self._samples >= self.warmup

    def reset(self) -> None:
        self._offset = None
        self._samples = 0

    def to_unix_ms(self, device_ms: int, arrival_ms: int) -> int:
        device_ms = int(device_ms)
        arrival_ms = int(arrival_ms)
        delta = arrival_ms - device_ms
        if self._offset is None:
            self._offset = delta
            self._samples = 1
            return arrival_ms
        if self._samples < self.warmup:
            self._offset = min(self._offset, delta)
            self._samples += 1
        converted = device_ms + self._offset
        if abs(converted - arrival_ms) > self.max_skew_ms:
            self.fallbacks += 1
            self.reset()
            return arrival_ms
        return converted


# ---------------------------------------------------------------------------
# Minimal SE(3) helpers for the simulator (same conventions as geometry.py §4.4)
# ---------------------------------------------------------------------------

_R_CAM_FACING_WALL = np.diag([1.0, -1.0, -1.0])  # yaw 0: camera +z -> room -z, +y down -> room -y


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_T(T: np.ndarray, p: Sequence[float]) -> np.ndarray:
    q = np.asarray(p, dtype=float)
    return T[:3, :3] @ q.reshape(3) + T[:3, 3] if q.ndim == 1 else (q @ T[:3, :3].T) + T[:3, 3]


def camera_pose_T(xyz_room_m: Sequence[float], yaw_deg: float) -> np.ndarray:
    """``T_room_cam`` for a level camera at ``xyz_room_m`` with the given heading.

    Heading 0 looks toward -z (at the anchor wall); +90 looks toward +x.
    """
    R = _rot_y(-yaw_deg) @ _R_CAM_FACING_WALL
    return make_T(R, xyz_room_m)


def yaw_deg_of(T_room_cam: np.ndarray) -> float:
    """Heading of the camera +z axis in the room x-z plane (see :func:`camera_pose_T`)."""
    d = np.asarray(T_room_cam, dtype=float)[:3, 2]
    return float(math.degrees(math.atan2(d[0], -d[2])))


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole projection of (N,3) camera-frame points, no distortion."""
    P = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = P[:, 2:3]
    uv = (P[:, :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv


def _tag_corners_tag_frame(tag_size_m: float) -> np.ndarray:
    s = tag_size_m / 2.0
    return np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])


# ---------------------------------------------------------------------------
# Scripted scenario
# ---------------------------------------------------------------------------


def _default_K_tag() -> np.ndarray:
    return np.array([[918.3, 0.0, 645.8], [0.0, 917.5, 417.4], [0.0, 0.0, 1.0]])


def _default_K_det() -> np.ndarray:
    return np.array([[459.2, 0.0, 322.9], [0.0, 458.7, 208.7], [0.0, 0.0, 1.0]])


@dataclass
class SimObject:
    label: str
    pos_room_m: tuple[float, float, float]
    size_m: float = 0.15


@dataclass
class Segment:
    duration_s: float
    room: str | None
    camera_xyz_m: tuple[float, float, float] | None
    yaw_deg: float = 0.0
    objects: list[SimObject] = field(default_factory=list)
    tag_ids: tuple[int, ...] = ()


@dataclass
class Scenario:
    name: str
    segments: list[Segment]
    tag_size_m: float = 0.16
    rooms: dict[str, dict] = field(default_factory=dict)
    K_tag: np.ndarray = field(default_factory=_default_K_tag)
    tag_frame_size: tuple[int, int] = (1280, 800)
    K_det: np.ndarray = field(default_factory=_default_K_det)
    det_frame_size: tuple[int, int] = (640, 400)

    @property
    def duration_s(self) -> float:
        return float(sum(s.duration_s for s in self.segments))

    def segment_at(self, t_s: float) -> tuple[int, Segment] | None:
        """Segment index and segment active at scenario time ``t_s`` (None past the end)."""
        acc = 0.0
        for i, seg in enumerate(self.segments):
            acc += seg.duration_s
            if t_s < acc:
                return i, seg
        return None

    def tag_pose_room(self, room: str | None, tag_id: int) -> np.ndarray | None:
        """``T_room_tag`` for a tag of ``room``: the anchor (0, h, 0) or an ``extra_tags`` entry."""
        if room is None:
            return None
        spec = self.rooms.get(room)
        if not spec:
            return None
        if int(spec.get("anchor_tag", -1)) == int(tag_id):
            return make_T(np.eye(3), (0.0, float(spec.get("anchor_height_m", 1.4)), 0.0))
        extra = spec.get("extra_tags") or {}
        xyz = extra.get(str(tag_id), extra.get(tag_id))
        if xyz is None:
            return None
        return make_T(np.eye(3), tuple(float(v) for v in xyz))

    @classmethod
    def mug_two_rooms(cls) -> Scenario:
        """Kitchen (cup, bottle) -> hallway -> living room (laptop, couch) -> kitchen, cup moved."""
        rooms = {
            "kitchen": {"anchor_tag": 0, "anchor_height_m": 1.4},
            "living_room": {"anchor_tag": 1, "anchor_height_m": 1.4},
        }
        kitchen_1 = [SimObject("cup", (1.0, 0.8, 1.0)), SimObject("bottle", (-0.6, 0.9, 1.2))]
        living = [SimObject("laptop", (-0.8, 0.7, 1.0)), SimObject("couch", (1.0, 0.4, 1.5), 1.2)]
        kitchen_2 = [SimObject("cup", (-1.2, 0.8, 1.5)), SimObject("bottle", (-0.6, 0.9, 1.2))]
        segments = [
            Segment(6.0, "kitchen", (0.0, 1.4, 2.5), 0.0, kitchen_1, (0,)),
            Segment(3.0, None, None, 0.0, [], ()),
            Segment(4.0, "living_room", (0.2, 1.3, 3.2), 0.0, living, (1,)),
            Segment(5.0, "kitchen", (-0.4, 1.2, 3.0), 0.0, kitchen_2, (0,)),
        ]
        return cls(name="mug_two_rooms", segments=segments, rooms=rooms)

    def to_json(self) -> str:
        d = {
            "name": self.name,
            "tag_size_m": self.tag_size_m,
            "rooms": self.rooms,
            "K_tag": self.K_tag.tolist(),
            "tag_frame_size": list(self.tag_frame_size),
            "K_det": self.K_det.tolist(),
            "det_frame_size": list(self.det_frame_size),
            "segments": [
                {
                    "duration_s": s.duration_s,
                    "room": s.room,
                    "camera_xyz_m": None if s.camera_xyz_m is None else list(s.camera_xyz_m),
                    "yaw_deg": s.yaw_deg,
                    "objects": [asdict(o) | {"pos_room_m": list(o.pos_room_m)} for o in s.objects],
                    "tag_ids": list(s.tag_ids),
                }
                for s in self.segments
            ],
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> Scenario:
        d = json.loads(s)
        segments = [
            Segment(
                duration_s=float(seg["duration_s"]),
                room=seg.get("room"),
                camera_xyz_m=None
                if seg.get("camera_xyz_m") is None
                else tuple(float(v) for v in seg["camera_xyz_m"]),
                yaw_deg=float(seg.get("yaw_deg", 0.0)),
                objects=[
                    SimObject(
                        label=str(o["label"]),
                        pos_room_m=tuple(float(v) for v in o["pos_room_m"]),
                        size_m=float(o.get("size_m", 0.15)),
                    )
                    for o in seg.get("objects", [])
                ],
                tag_ids=tuple(int(t) for t in seg.get("tag_ids", ())),
            )
            for seg in d["segments"]
        ]
        return cls(
            name=str(d.get("name", "custom")),
            segments=segments,
            tag_size_m=float(d.get("tag_size_m", 0.16)),
            rooms=dict(d.get("rooms", {})),
            K_tag=np.asarray(d["K_tag"], dtype=float) if "K_tag" in d else _default_K_tag(),
            tag_frame_size=tuple(d.get("tag_frame_size", (1280, 800))),
            K_det=np.asarray(d["K_det"], dtype=float) if "K_det" in d else _default_K_det(),
            det_frame_size=tuple(d.get("det_frame_size", (640, 400))),
        )


SCENARIOS: dict[str, Callable[[], Scenario]] = {"mug_two_rooms": Scenario.mug_two_rooms}
DEMO_SCENARIO = Scenario.mug_two_rooms()


def load_scenario(scenario: Scenario | str) -> Scenario:
    """Resolve a scenario object, registry name or path to a ``Scenario.to_json`` file."""
    if isinstance(scenario, Scenario):
        return scenario
    key = str(scenario)
    if key in SCENARIOS:
        return SCENARIOS[key]()
    p = Path(key)
    if p.suffix == ".json" and p.is_file():
        return Scenario.from_json(p.read_text(encoding="utf-8"))
    raise ValueError(f"unknown scenario {scenario!r}; known: {sorted(SCENARIOS)}")


# ---------------------------------------------------------------------------
# Shared helpers for sources
# ---------------------------------------------------------------------------


class _FpsMeter:
    """Exponentially smoothed frame rate from wall-clock intervals."""

    def __init__(self, alpha: float = 0.2) -> None:
        self._alpha = alpha
        self._last: float | None = None
        self.fps = 0.0

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self._last is not None:
            dt = now - self._last
            inst = 1.0 / dt if dt > 1e-6 else 1000.0
            self.fps = (
                inst if self.fps == 0.0 else (1 - self._alpha) * self.fps + self._alpha * inst
            )
        self._last = now

    @property
    def last(self) -> float | None:
        return self._last

    def reset(self) -> None:
        self._last = None
        self.fps = 0.0


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class SimDetectionSource:
    """Scripted twin of the camera: emits ``DetectionFrame``/``TagFrame`` from a scenario.

    Deterministic under ``numpy.random.default_rng(seed)``.  With ``oracle_pose=True``
    no tag frames are emitted and every detection frame carries the true camera
    pose so the chain runs without cv2/PnP.
    """

    name = "sim"

    def __init__(
        self,
        scenario: Scenario | str = "mug_two_rooms",
        fps: float = 10.0,
        realtime: bool = False,
        oracle_pose: bool = False,
        emit_tags: bool = True,
        loop: bool = False,
        seed: int = 0,
        noise_px: float = 0.3,
        noise_xy_m: float = 0.02,
        noise_z_m: float = 0.04,
        tag_fps: float = 5.0,
        start_ms: int | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.scenario = load_scenario(scenario)
        self.classes: list[str] = list(COCO80)
        self.fps = float(fps)
        self.realtime = bool(realtime)
        self.oracle_pose = bool(oracle_pose)
        self.emit_tags = bool(emit_tags)
        self.loop = bool(loop)
        self.seed = int(seed)
        self.noise_px = float(noise_px)
        self.noise_xy_m = float(noise_xy_m)
        self.noise_z_m = float(noise_z_m)
        self.tag_fps = float(tag_fps)
        self.start_ms = start_ms
        self.frames_total = int(round(self.scenario.duration_s * self.fps))
        self._rng = np.random.default_rng(self.seed)
        self._stop = threading.Event()
        self._state = "starting"
        self._frames = 0
        self._meter = _FpsMeter()
        self._last_frame_mono: float | None = None
        self._lock = threading.Lock()

    # -- DetectionSource protocol -------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        with self._lock:
            self._state = "starting"

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> SourceStatus:
        with self._lock:
            age = None
            if self._last_frame_mono is not None:
                age = max(0.0, time.monotonic() - self._last_frame_mono)
            return SourceStatus(
                state=self._state,
                fps=round(self._meter.fps, 2),
                frames=self._frames,
                reconnects=0,
                last_frame_age_s=age,
                usb_speed=None,
                device=f"sim:{self.scenario.name}",
                error=None,
            )

    def events(self) -> Iterator[SourceEvent]:
        self._rng = np.random.default_rng(self.seed)
        start_ms = _now_ms() if self.start_ms is None else int(self.start_ms)
        with self._lock:
            self._state = "running"
        lap = 0
        seq = 0
        wall0 = time.monotonic()
        frame_ms = 1000.0 / self.fps
        duration_ms = self.scenario.duration_s * 1000.0
        completed = False
        try:
            while not self._stop.is_set():
                next_tag_t = 0.0
                for i in range(self.frames_total):
                    if self._stop.is_set():
                        return
                    t_s = i / self.fps
                    ts_ms = start_ms + int(round(lap * duration_ms + i * frame_ms))
                    if self.realtime:
                        target = wall0 + (lap * self.scenario.duration_s) + t_s
                        delay = target - time.monotonic()
                        if delay > 0 and self._stop.wait(delay):
                            return
                    found = self.scenario.segment_at(t_s)
                    if found is None:
                        break
                    _, seg = found
                    T_room_cam = (
                        camera_pose_T(seg.camera_xyz_m, seg.yaw_deg)
                        if seg.camera_xyz_m is not None
                        else None
                    )
                    tags_due = self.tag_fps > 0 and t_s + 1e-9 >= next_tag_t
                    if tags_due:
                        next_tag_t += 1.0 / self.tag_fps
                    if (
                        tags_due
                        and self.emit_tags
                        and not self.oracle_pose
                        and seg.tag_ids
                        and T_room_cam is not None
                    ):
                        yield self._tag_frame(seg, T_room_cam, ts_ms)
                    frame = self._detection_frame(seg, T_room_cam, ts_ms, seq)
                    seq += 1
                    with self._lock:
                        self._frames += 1
                        self._last_frame_mono = time.monotonic()
                        self._meter.tick(self._last_frame_mono)
                    yield frame
                if not self.loop:
                    completed = True
                    break
                lap += 1
        finally:
            with self._lock:
                self._state = "done" if completed else "stopped"

    # -- synthesis ----------------------------------------------------------------

    def _oracle(self, seg: Segment, T_room_cam: np.ndarray | None) -> CameraPose:
        if seg.room is None or T_room_cam is None:
            return CameraPose(None, "camera", None, False, 0.0, (), float("inf"))
        spec = self.scenario.rooms.get(seg.room, {})
        anchor = spec.get("anchor_tag")
        tag_ids = (int(anchor),) if anchor is not None else tuple(seg.tag_ids)
        return CameraPose(seg.room, "room", T_room_cam, True, 1.0, tag_ids, 0.0)

    def _detection_frame(
        self, seg: Segment, T_room_cam: np.ndarray | None, ts_ms: int, seq: int
    ) -> DetectionFrame:
        dets: list[SpatialDetection] = []
        if T_room_cam is not None:
            T_cam_room = invert_T(T_room_cam)
            W, H = self.scenario.det_frame_size
            K = self.scenario.K_det
            for obj in seg.objects:
                p_cam = apply_T(T_cam_room, obj.pos_room_m)
                p_cam = p_cam + self._rng.normal(
                    0.0, (self.noise_xy_m, self.noise_xy_m, self.noise_z_m)
                )
                if p_cam[2] <= 0.3:
                    continue
                h = obj.size_m / 2.0
                offs = np.array(
                    [[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)]
                )
                corners = p_cam + offs
                corners = corners[corners[:, 2] > 0.05]
                if len(corners) == 0:
                    continue
                uv = project_points(corners, K)
                xmin, ymin = _clip01(uv[:, 0].min() / W), _clip01(uv[:, 1].min() / H)
                xmax, ymax = _clip01(uv[:, 0].max() / W), _clip01(uv[:, 1].max() / H)
                if xmax - xmin <= 0.0 or ymax - ymin <= 0.0:
                    continue  # entirely outside the image
                conf = float(np.clip(0.85 + self._rng.normal(0.0, 0.05), 0.5, 0.99))
                label = obj.label.lower()
                label_id = self.classes.index(label) if label in self.classes else -1
                dets.append(
                    SpatialDetection(
                        label=label,
                        label_id=label_id,
                        confidence=conf,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        xyz_cam_m=(float(p_cam[0]), float(p_cam[1]), float(p_cam[2])),
                    )
                )
        return DetectionFrame(
            timestamp_ms=ts_ms,
            seq=seq,
            detections=dets,
            source=self.name,
            device_ts_ms=None,
            oracle_pose=self._oracle(seg, T_room_cam) if self.oracle_pose else None,
        )

    def _tag_frame(self, seg: Segment, T_room_cam: np.ndarray, ts_ms: int) -> TagFrame:
        W, H = self.scenario.tag_frame_size
        K = self.scenario.K_tag
        T_cam_room = invert_T(T_room_cam)
        obj_pts = _tag_corners_tag_frame(self.scenario.tag_size_m)
        tags: list[TagDetection] = []
        for tag_id in seg.tag_ids:
            T_room_tag = self.scenario.tag_pose_room(seg.room, tag_id)
            if T_room_tag is None:
                continue
            T_cam_tag = T_cam_room @ T_room_tag
            pts_cam = apply_T(T_cam_tag, obj_pts)
            if np.any(pts_cam[:, 2] <= 0.1):
                continue
            uv = project_points(pts_cam, K)
            uv = uv + self._rng.normal(0.0, self.noise_px, size=uv.shape)
            if np.any(uv[:, 0] < 0) or np.any(uv[:, 0] >= W) or np.any(uv[:, 1] < 0):
                continue
            if np.any(uv[:, 1] >= H):
                continue
            tags.append(TagDetection(int(tag_id), uv, decision_margin=80.0, hamming=0))
        return TagFrame(
            timestamp_ms=ts_ms, width=W, height=H, K=K.copy(), dist=None, gray=None, tags=tags
        )


# ---------------------------------------------------------------------------
# JSONL record / replay (design §4.14)
# ---------------------------------------------------------------------------


class JsonlRecorder:
    """Append every source event as one JSON line (tag frames without the image)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.lines = 0

    def write(self, ev: SourceEvent) -> None:
        if isinstance(ev, DetectionFrame):
            rec: dict[str, Any] = {"t": "det", **ev.to_dict()}
        elif isinstance(ev, TagFrame):
            rec = {"t": "tag", **ev.to_dict()}
        else:  # pragma: no cover - protocol violation
            raise TypeError(f"unsupported event {type(ev).__name__}")
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(line + "\n")
            self.lines += 1

    def flush(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def __enter__(self) -> JsonlRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def parse_event_line(line: str) -> SourceEvent | None:
    """Parse one recorder line; returns None for blank/unknown/corrupt lines."""
    line = line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    kind = d.get("t")
    try:
        if kind == "det":
            return DetectionFrame.from_dict(d, source="replay")
        if kind == "tag":
            return TagFrame.from_dict(d)
    except (KeyError, TypeError, ValueError):
        return None
    return None


class JsonlReplaySource:
    """Play a :class:`JsonlRecorder` file back as a detection source."""

    name = "replay"

    def __init__(self, path: Path, realtime: bool = False, loop: bool = False) -> None:
        self.path = Path(path)
        self.realtime = bool(realtime)
        self.loop = bool(loop)
        self.classes: list[str] = list(COCO80)
        self._stop = threading.Event()
        self._state = "starting"
        self._frames = 0
        self._bad_lines = 0
        self._meter = _FpsMeter()
        self._last_frame_mono: float | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._stop.clear()
        with self._lock:
            self._state = "starting"

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> SourceStatus:
        with self._lock:
            age = None
            if self._last_frame_mono is not None:
                age = max(0.0, time.monotonic() - self._last_frame_mono)
            return SourceStatus(
                state=self._state,
                fps=round(self._meter.fps, 2),
                frames=self._frames,
                reconnects=0,
                last_frame_age_s=age,
                usb_speed=None,
                device=f"replay:{self.path.name}",
                error=f"{self._bad_lines} unparsable lines" if self._bad_lines else None,
            )

    def _load(self) -> list[SourceEvent]:
        events: list[SourceEvent] = []
        bad = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                ev = parse_event_line(line)
                if ev is None:
                    bad += 1
                    continue
                events.append(ev)
        with self._lock:
            self._bad_lines = bad
        return events

    def events(self) -> Iterator[SourceEvent]:
        events = self._load()
        with self._lock:
            self._state = "running"
        if not events:
            with self._lock:
                self._state = "done"
            return
        first_ts = events[0].timestamp_ms
        last_ts = events[-1].timestamp_ms
        span_ms = max(1, last_ts - first_ts + 100)
        lap = 0
        completed = False
        try:
            while not self._stop.is_set():
                wall0 = time.monotonic()
                shift = lap * span_ms
                for ev in events:
                    if self._stop.is_set():
                        return
                    if self.realtime:
                        target = wall0 + (ev.timestamp_ms - first_ts) / 1000.0
                        delay = target - time.monotonic()
                        if delay > 0 and self._stop.wait(delay):
                            return
                    out = _shift_event(ev, shift) if shift else ev
                    if isinstance(out, DetectionFrame):
                        with self._lock:
                            self._frames += 1
                            self._last_frame_mono = time.monotonic()
                            self._meter.tick(self._last_frame_mono)
                    yield out
                if not self.loop:
                    completed = True
                    break
                lap += 1
        finally:
            with self._lock:
                self._state = "done" if completed else "stopped"


def _shift_event(ev: SourceEvent, shift_ms: int) -> SourceEvent:
    if isinstance(ev, DetectionFrame):
        return DetectionFrame(
            timestamp_ms=ev.timestamp_ms + shift_ms,
            seq=ev.seq,
            detections=ev.detections,
            source=ev.source,
            device_ts_ms=ev.device_ts_ms,
            oracle_pose=ev.oracle_pose,
        )
    return TagFrame(
        timestamp_ms=ev.timestamp_ms + shift_ms,
        width=ev.width,
        height=ev.height,
        K=ev.K,
        dist=ev.dist,
        gray=ev.gray,
        tags=ev.tags,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def env_flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def make_detection_source(
    kind: str | None = None, *, env: Mapping[str, str] | None = None
) -> DetectionSource | None:
    """Build the source named by ``kind`` or ``LOCI_DETECTION_SOURCE`` (``none`` -> None)."""
    env = os.environ if env is None else env
    kind = (kind or env.get("LOCI_DETECTION_SOURCE", "none") or "none").strip().lower()
    if kind in {"none", "off", ""}:
        return None
    if kind == "sim":
        return SimDetectionSource(
            scenario=env.get("LOCI_SIM_SCENARIO", "mug_two_rooms") or "mug_two_rooms",
            fps=float(env.get("LOCI_SIM_FPS", "10") or 10),
            realtime=env_flag(env, "LOCI_SIM_REALTIME", True),
            oracle_pose=env_flag(env, "LOCI_SIM_ORACLE_POSE", False),
            loop=env_flag(env, "LOCI_SIM_LOOP", False),
            seed=int(env.get("LOCI_SIM_SEED", "0") or 0),
        )
    if kind == "replay":
        path = env.get("LOCI_REPLAY_PATH", "").strip()
        if not path:
            raise ValueError("LOCI_DETECTION_SOURCE=replay requires LOCI_REPLAY_PATH")
        return JsonlReplaySource(
            Path(path),
            realtime=env_flag(env, "LOCI_SIM_REALTIME", True),
            loop=env_flag(env, "LOCI_SIM_LOOP", False),
        )
    if kind == "oak":
        from .oak_source import OakConfig, OakDetectionSource

        return OakDetectionSource(OakConfig.from_env(env))
    raise ValueError(f"unknown detection source {kind!r} (none|sim|replay|oak)")


__all__ = [
    "AXIS_FIX",
    "COCO80",
    "DEMO_SCENARIO",
    "SCENARIOS",
    "CameraPose",
    "DetectionFrame",
    "DetectionSource",
    "HostClock",
    "JsonlRecorder",
    "JsonlReplaySource",
    "Scenario",
    "Segment",
    "SimDetectionSource",
    "SimObject",
    "SourceEvent",
    "SourceStatus",
    "SpatialDetection",
    "TagDetection",
    "TagFrame",
    "apply_T",
    "camera_pose_T",
    "depthai_xyz_to_cam",
    "env_flag",
    "invert_T",
    "load_scenario",
    "make_T",
    "make_detection_source",
    "parse_event_line",
    "project_points",
    "yaw_deg_of",
]
