"""AprilTag room anchoring for the handheld demo.

Chain per tag frame (``docs/HANDHELD_DEMO_DESIGN.md`` section 6): ``TagDetector.detect`` (or
the frame's pre-detected ``tags``) -> ``solve_tag_pose`` per tag -> ``TagFilter`` -> group by
room (``RoomMap``) -> joint PnP when two or more known tags of one room are visible, else
``RoomAnchorMap.observe`` (which also self-registers co-visible tags) -> ``PoseHold`` ->
``RoomLocalizer.current()`` yields the ``CameraPose`` used for every detection frame until
the next tag frame.

Only numpy is imported at module level.  ``pyapriltags`` and ``cv2`` are imported lazily
inside the detector constructors; the PnP functions of :mod:`geometry` import ``cv2`` lazily
as well, so this module imports (and the pre-detected-tag / oracle paths work) without
OpenCV.

Environment variables read here: ``LOCI_TAG_DETECTOR`` (``make_tag_detector(kind=None)``),
``LOCI_TAG_SIZE_M`` (``RoomMap.load``; overrides ``rooms.json``) and ``LOCI_TAG_UPRIGHT_FIX``
(``RoomLocalizer``; default ``0``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .detection_source import CameraPose, TagDetection, TagFrame
from .geometry import (
    Array,
    TagPose,
    apply_T,
    camera_pose_in_room,
    log_rot,
    make_T,
    mean_T,
    normalize_upright,
    rodrigues,
    rot_y,
    rotation_angle_deg,
    solve_room_pose_multi,
    solve_tag_pose,
)

logger = logging.getLogger(__name__)

Extents = dict[str, tuple[float, float]]

#: Box used for camera-frame (unanchored) observations, metres -> LOCI [0, 1].
CAMERA_BOX: Extents = {"x": (-3.0, 3.0), "y": (-1.5, 1.5), "z": (0.0, 6.0)}
#: Default per-room box, metres -> LOCI [0, 1] (x across, y up from the floor, z into the room).
DEFAULT_ROOM_EXTENTS: Extents = {"x": (-3.0, 3.0), "y": (0.0, 2.5), "z": (0.0, 6.0)}
DEFAULT_TAG_SIZE_M = 0.16
DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_ANCHOR_HEIGHT_M = 1.4
DEFAULT_ANCHOR_NAME = "door tag"
ROOMS_FILE_FORMAT = 1
TAGS_FILE_FORMAT = 1

#: pyapriltags returns the corners of a ``cv2.aruco``-generated tag as [TR, TL, BL, BR]; this
#: index turns them into the TL, TR, BR, BL order the PnP object points expect (verified on the
#: Mac and the Pi for ids 0-586, see the design research).
PYAPRILTAGS_CORNER_ORDER: tuple[int, int, int, int] = (1, 0, 3, 2)

_ARUCO_DICTS: dict[str, str] = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}


# ------------------------------------------------------------------------------- detectors
@runtime_checkable
class TagDetector(Protocol):
    """Finds AprilTags in a grayscale image; corners come back as TL, TR, BR, BL."""

    name: str

    def detect(self, gray: np.ndarray) -> list[TagDetection]: ...


def _as_gray_u8(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(g)


def remap_pyapriltags_corners(corners: Any) -> np.ndarray:
    """pyapriltags corner order [TR, TL, BL, BR] -> TL, TR, BR, BL (float64, shape (4, 2))."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return c[list(PYAPRILTAGS_CORNER_ORDER)]


class PyAprilTagsDetector:
    """AprilTag 3 via ``pyapriltags`` (recommended: 9 ms per 1280x800 frame at qd=2 on a Pi 5)."""

    name = "pyapriltags"

    def __init__(
        self,
        family: str = DEFAULT_TAG_FAMILY,
        quad_decimate: float = 2.0,
        nthreads: int = 1,
        min_margin: float = 30.0,
    ) -> None:
        import pyapriltags  # lazy: optional dependency

        self.family = family
        self.min_margin = float(min_margin)
        self._det = pyapriltags.Detector(
            families=family,
            nthreads=int(nthreads),
            quad_decimate=float(quad_decimate),
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )

    def detect(self, gray: np.ndarray) -> list[TagDetection]:
        out: list[TagDetection] = []
        for r in self._det.detect(_as_gray_u8(gray)):
            hamming = int(r.hamming)
            margin = float(r.decision_margin)
            if hamming != 0 or margin < self.min_margin:
                continue
            out.append(
                TagDetection(
                    tag_id=int(r.tag_id),
                    corners_px=remap_pyapriltags_corners(r.corners),
                    decision_margin=margin,
                    hamming=hamming,
                )
            )
        return out


class ArucoTagDetector:
    """``cv2.aruco`` fallback (``DICT_APRILTAG_36h11`` + sub-pixel corner refinement)."""

    name = "aruco"

    def __init__(self, family: str = DEFAULT_TAG_FAMILY) -> None:
        import cv2  # lazy: optional dependency

        try:
            dict_id = getattr(cv2.aruco, _ARUCO_DICTS[family])
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"unsupported tag family for cv2.aruco: {family!r}") from exc
        self.family = family
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)

    def detect(self, gray: np.ndarray) -> list[TagDetection]:
        corners, ids, _ = self._det.detectMarkers(_as_gray_u8(gray))
        if ids is None:
            return []
        return [
            TagDetection(
                tag_id=int(tid),
                corners_px=np.asarray(c, dtype=np.float64).reshape(4, 2),
                decision_margin=100.0,
                hamming=0,
            )
            for c, tid in zip(corners, np.asarray(ids).flatten(), strict=True)
        ]


def make_tag_detector(
    kind: str | None = "auto",
    *,
    family: str = DEFAULT_TAG_FAMILY,
    env: Mapping[str, str] | None = None,
) -> TagDetector | None:
    """``auto`` = pyapriltags if importable, else cv2.aruco if cv2 exists, else ``None``.

    ``kind=None`` reads ``LOCI_TAG_DETECTOR`` (auto|pyapriltags|aruco|none).  An explicit
    backend that cannot be imported raises ``ImportError``.
    """
    source = os.environ if env is None else env
    k = (kind if kind is not None else source.get("LOCI_TAG_DETECTOR", "auto")).strip().lower()
    if k in ("none", "off", ""):
        return None
    if k == "pyapriltags":
        return PyAprilTagsDetector(family=family)
    if k == "aruco":
        return ArucoTagDetector(family=family)
    if k != "auto":
        raise ValueError(f"unknown tag detector {kind!r} (auto|pyapriltags|aruco|none)")
    try:
        return PyAprilTagsDetector(family=family)
    except (ImportError, OSError) as exc:
        logger.info("pyapriltags unavailable (%s); trying cv2.aruco", exc)
    try:
        return ArucoTagDetector(family=family)
    except ImportError:
        logger.warning(
            "no AprilTag detector available (pyapriltags and cv2 missing); "
            "only pre-detected tag frames will anchor"
        )
        return None


# ---------------------------------------------------------------------------- sanity filter
@dataclass
class TagFilter:
    """Per-detection gate applied before a tag pose may update the room pose."""

    min_side_px: float = 30.0  # below ~30 px the planar-pose ambiguity is not resolvable
    max_reproj_err_px: float = 1.0
    min_ambiguity_ratio: float = 1.5
    max_view_angle_deg: float = 60.0
    max_range_m: float = 6.0
    border_margin_px: float = 2.0  # corners closer than this to the image edge = partial tag

    def reasons(self, p: TagPose) -> list[str]:
        """Why ``p`` would be rejected (empty when accepted); for diagnostics."""
        out: list[str] = []
        rng = p.range_m
        if p.side_px < self.min_side_px:
            out.append("small")
        if not (p.reproj_err_px <= self.max_reproj_err_px):
            out.append("reproj")
        if not (p.ambiguity_ratio >= self.min_ambiguity_ratio):
            out.append("ambiguous")
        if not (p.view_angle_deg <= self.max_view_angle_deg):
            out.append("oblique")
        if not (0.0 < rng <= self.max_range_m):
            out.append("range")
        return out

    def accept(self, p: TagPose) -> bool:
        return not self.reasons(p)

    def inside_image(self, corners_px: Any, width: int, height: int) -> bool:
        """False when any corner lies within ``border_margin_px`` of the image border."""
        c = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
        m = self.border_margin_px
        return bool(
            np.all(c[:, 0] >= m)
            and np.all(c[:, 1] >= m)
            and np.all(c[:, 0] <= width - 1 - m)
            and np.all(c[:, 1] <= height - 1 - m)
        )


# ------------------------------------------------------------------------- room anchor map
@dataclass
class RoomAnchorMap:
    """Tag poses in one room's frame.  The anchor tag is the origin (floor directly below it).

    Other tags are registered automatically when they are seen together with an already-known
    tag: ``T_room_tagB = T_room_cam @ T_cam_tagB``, averaged over ``register_samples`` frames
    before the tag is trusted (and only from quads of at least ``register_min_side_px``).
    """

    room: str
    anchor_height_m: float = DEFAULT_ANCHOR_HEIGHT_M
    register_samples: int = 10
    tags: dict[int, Array] = field(default_factory=dict)
    anchor_id: int | None = None
    register_min_side_px: float = 40.0
    registrations: int = 0
    _pending: dict[int, list[Array]] = field(default_factory=dict, repr=False)

    def seed_anchor(self, tag_id: int) -> None:
        """Declare ``tag_id`` the anchor: ``T_room_tag = [I | (0, anchor_height_m, 0)]``."""
        self.anchor_id = int(tag_id)
        self.tags[int(tag_id)] = make_T(np.eye(3), (0.0, self.anchor_height_m, 0.0))

    @property
    def pending_ids(self) -> list[int]:
        return sorted(self._pending)

    def observe(self, poses: list[TagPose], T_room_cam: Array | None = None) -> Array | None:
        """Return ``T_room_cam`` for this frame (``None`` if no known tag is visible).

        When the caller already solved the camera pose jointly from several known tags it
        passes it as ``T_room_cam``; the map then only performs the registration step.
        """
        if not poses:
            return None
        if self.anchor_id is None:
            first = max(poses, key=lambda p: p.side_px)
            self.seed_anchor(first.tag_id)
        known = [p for p in poses if p.tag_id in self.tags]
        if not known:
            return None
        if T_room_cam is not None:
            T = np.asarray(T_room_cam, dtype=np.float64)
        elif len(known) == 1:
            T = camera_pose_in_room(self.tags[known[0].tag_id], known[0].T_cam_tag)
        else:  # average the camera pose implied by every known tag
            T = mean_T([camera_pose_in_room(self.tags[p.tag_id], p.T_cam_tag) for p in known])
        for p in poses:
            if p.tag_id in self.tags or p.side_px < self.register_min_side_px:
                continue
            samples = self._pending.setdefault(p.tag_id, [])
            samples.append(T @ p.T_cam_tag)
            if len(samples) >= self.register_samples:
                self.tags[p.tag_id] = mean_T(samples)
                del self._pending[p.tag_id]
                self.registrations += 1
                logger.info(
                    "room %s: registered tag %d at %s",
                    self.room,
                    p.tag_id,
                    np.round(self.tags[p.tag_id][:3, 3], 2).tolist(),
                )
        return T

    def to_dict(self) -> dict[str, Any]:
        return {
            "room": self.room,
            "anchor_id": self.anchor_id,
            "anchor_height_m": self.anchor_height_m,
            "tags": {str(k): v.tolist() for k, v in self.tags.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RoomAnchorMap:
        m = cls(
            room=str(d["room"]),
            anchor_height_m=float(d.get("anchor_height_m", DEFAULT_ANCHOR_HEIGHT_M)),
        )
        anchor = d.get("anchor_id")
        m.anchor_id = None if anchor is None else int(anchor)
        m.tags = {
            int(k): np.asarray(v, dtype=np.float64).reshape(4, 4)
            for k, v in dict(d.get("tags", {})).items()
        }
        return m


# ----------------------------------------------------------------------- pose hold / smoothing
@dataclass
class PoseHold:
    """Smooth ``T_room_cam`` while tags are visible; hold it (with decaying confidence) when not."""

    hold_s: float = 10.0
    alpha_t: float = 0.35
    alpha_r: float = 0.35
    max_jump_m: float = 0.75
    max_jump_deg: float = 25.0
    jump_confirm: int = 3
    T: Array | None = None
    room: str | None = None
    last_seen_s: float | None = None
    _jumps: int = 0

    def update(self, T_room_cam: Array, now_s: float, room: str) -> None:
        if self.T is None or room != self.room:
            self.T, self.room, self._jumps = np.array(T_room_cam, dtype=np.float64), room, 0
        else:
            dt = float(np.linalg.norm(T_room_cam[:3, 3] - self.T[:3, 3]))
            dr = rotation_angle_deg(self.T[:3, :3], T_room_cam[:3, :3])
            if dt > self.max_jump_m or dr > self.max_jump_deg:
                self._jumps += 1
                if self._jumps < self.jump_confirm:
                    return  # ignore an isolated outlier; accept it if it persists
                self.T, self._jumps = np.array(T_room_cam, dtype=np.float64), 0
            else:
                self._jumps = 0
                Rf, R = self.T[:3, :3], T_room_cam[:3, :3]
                Rf = Rf @ rodrigues(self.alpha_r * log_rot(Rf.T @ R))
                tf = (1.0 - self.alpha_t) * self.T[:3, 3] + self.alpha_t * T_room_cam[:3, 3]
                self.T = make_T(Rf, tf)
        self.last_seen_s = now_s

    def reset(
        self, T: Array | None = None, room: str | None = None, now_s: float | None = None
    ) -> None:
        """Replace the held pose without smoothing or jump gating (oracle / test injection)."""
        self.T = None if T is None else np.array(T, dtype=np.float64)
        self.room = room
        self.last_seen_s = None if T is None else now_s
        self._jumps = 0

    def age_s(self, now_s: float) -> float:
        if self.last_seen_s is None:
            return math.inf
        return max(0.0, now_s - self.last_seen_s)

    def current(self, now_s: float) -> tuple[Array | None, float, bool]:
        """``(T_room_cam, anchor_confidence in [0, 1], anchored)``."""
        if self.T is None or self.last_seen_s is None:
            return None, 0.0, False
        age = self.age_s(now_s)
        conf = max(0.0, 1.0 - age / self.hold_s) if self.hold_s > 0 else 0.0
        return self.T, conf, age <= self.hold_s


# ------------------------------------------------------------------------- rooms / extents
def slug(room: str) -> str:
    """Room name -> scene_id prefix: lower case, runs of spaces and ``/`` become ``_``."""
    return re.sub(r"[\s/]+", "_", str(room).strip().lower())


def _parse_extents(d: Mapping[str, Any] | None) -> Extents:
    out: Extents = dict(DEFAULT_ROOM_EXTENTS)
    if not d:
        return out
    for axis in ("x", "y", "z"):
        if axis not in d:
            continue
        lo, hi = (float(v) for v in d[axis])
        if not hi > lo:
            raise ValueError(f"extents_m[{axis!r}] must satisfy lo < hi, got {lo} >= {hi}")
        out[axis] = (lo, hi)
    return out


def _parse_extra_tag(tag_id: int, v: Any) -> Array:
    """``T_room_tag`` from a 4x4 nested list or ``{"pos_m": [x, y, z], "normal_yaw_deg": deg}``.

    ``normal_yaw_deg`` is the heading of the tag's normal (+z, out of the printed face) in the
    room x-z plane: 0 = facing into the room like the anchor (+z), +90 = facing +x.
    """
    if isinstance(v, Mapping):
        pos = np.asarray(v.get("pos_m", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
        yaw = math.radians(float(v.get("normal_yaw_deg", 0.0)))
        return make_T(rot_y(yaw), pos)
    T = np.asarray(v, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(
            f"extra_tags[{tag_id}] must be a 4x4 matrix or a pos_m/normal_yaw_deg object"
        )
    return T


@dataclass
class RoomSpec:
    """One room of ``rooms.json``."""

    name: str
    anchor_tag: int
    anchor_height_m: float = DEFAULT_ANCHOR_HEIGHT_M
    anchor_name: str = DEFAULT_ANCHOR_NAME
    extents_m: Extents = field(default_factory=lambda: dict(DEFAULT_ROOM_EXTENTS))
    landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    #: manually surveyed ``T_room_tag`` for tags never co-visible with the anchor
    extra_tags: dict[int, Array] = field(default_factory=dict)

    @property
    def T_room_anchor(self) -> Array:
        return make_T(np.eye(3), (0.0, self.anchor_height_m, 0.0))

    def tag_ids(self) -> list[int]:
        return [int(self.anchor_tag), *sorted(int(t) for t in self.extra_tags)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "anchor_tag": int(self.anchor_tag),
            "anchor_height_m": float(self.anchor_height_m),
            "anchor_name": self.anchor_name,
            "extents_m": {k: [float(lo), float(hi)] for k, (lo, hi) in self.extents_m.items()},
            "landmarks": {k: [float(c) for c in v] for k, v in self.landmarks.items()},
            "extra_tags": {str(k): np.asarray(v).tolist() for k, v in self.extra_tags.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RoomSpec:
        if "name" not in d or "anchor_tag" not in d:
            raise ValueError("each room needs 'name' and 'anchor_tag'")
        name = str(d["name"]).strip()
        if not name:
            raise ValueError("room name must not be empty")
        landmarks: dict[str, tuple[float, float, float]] = {}
        for k, v in dict(d.get("landmarks") or {}).items():
            xyz = tuple(float(c) for c in v)
            if len(xyz) != 3:
                raise ValueError(f"landmark {k!r} must have 3 coordinates")
            landmarks[str(k)] = (xyz[0], xyz[1], xyz[2])
        extra: dict[int, Array] = {}
        for k, v in dict(d.get("extra_tags") or {}).items():
            extra[int(k)] = _parse_extra_tag(int(k), v)
        return cls(
            name=name,
            anchor_tag=int(d["anchor_tag"]),
            anchor_height_m=float(d.get("anchor_height_m", DEFAULT_ANCHOR_HEIGHT_M)),
            anchor_name=str(d.get("anchor_name", DEFAULT_ANCHOR_NAME)),
            extents_m=_parse_extents(d.get("extents_m")),
            landmarks=landmarks,
            extra_tags=extra,
        )


class RoomMap:
    """All rooms of the installation plus the shared tag size / family."""

    def __init__(
        self,
        rooms: Iterable[RoomSpec] | Mapping[str, RoomSpec] = (),
        tag_size_m: float = DEFAULT_TAG_SIZE_M,
        tag_family: str = DEFAULT_TAG_FAMILY,
    ) -> None:
        specs = list(rooms.values()) if isinstance(rooms, Mapping) else list(rooms)
        self.rooms: dict[str, RoomSpec] = {}
        self._tag_to_room: dict[int, str] = {}
        for spec in specs:
            self._add(spec)
        if not tag_size_m > 0:
            raise ValueError(f"tag_size_m must be positive, got {tag_size_m}")
        self.tag_size_m = float(tag_size_m)
        self.tag_family = str(tag_family)

    def _add(self, spec: RoomSpec) -> None:
        if spec.name in self.rooms:
            raise ValueError(f"duplicate room name {spec.name!r}")
        for tid in spec.tag_ids():
            if tid in self._tag_to_room:
                raise ValueError(
                    f"tag {tid} is used by both {self._tag_to_room[tid]!r} and {spec.name!r}"
                )
        self.rooms[spec.name] = spec
        for tid in spec.tag_ids():
            self._tag_to_room[tid] = spec.name

    # -- construction -------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, tag_size_m: float | None = None) -> RoomMap:
        fmt = int(d.get("format", ROOMS_FILE_FORMAT))
        if fmt != ROOMS_FILE_FORMAT:
            logger.warning("rooms file format %s; expected %s", fmt, ROOMS_FILE_FORMAT)
        rooms_raw = d.get("rooms", [])
        if isinstance(rooms_raw, Mapping):  # tolerate {"name": {...}} layout
            rooms_raw = [{"name": k, **v} for k, v in rooms_raw.items()]
        specs = [RoomSpec.from_dict(r) for r in rooms_raw]
        size = tag_size_m if tag_size_m is not None else d.get("tag_size_m", DEFAULT_TAG_SIZE_M)
        return cls(
            specs, tag_size_m=float(size), tag_family=str(d.get("tag_family", DEFAULT_TAG_FAMILY))
        )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        tag_size_m: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RoomMap:
        """Load ``rooms.json``.

        Tag size precedence: ``tag_size_m`` kwarg > ``LOCI_TAG_SIZE_M`` > the file.
        """
        source = os.environ if env is None else env
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if tag_size_m is None and source.get("LOCI_TAG_SIZE_M"):
            tag_size_m = float(source["LOCI_TAG_SIZE_M"])
        return cls.from_dict(data, tag_size_m=tag_size_m)

    @classmethod
    def default(cls) -> RoomMap:
        """Built-in two-room layout: kitchen (tag 0) and living_room (tag 1)."""
        return cls(
            [
                RoomSpec(name="kitchen", anchor_tag=0),
                RoomSpec(name="living_room", anchor_tag=1),
            ]
        )

    # -- queries ------------------------------------------------------------------------
    def room_for_tag(self, tag_id: int) -> RoomSpec | None:
        name = self._tag_to_room.get(int(tag_id))
        return None if name is None else self.rooms[name]

    def get(self, room: str) -> RoomSpec | None:
        """Look a room up by exact name, else by slug (``"living room"`` == ``"living_room"``)."""
        if room in self.rooms:
            return self.rooms[room]
        wanted = slug(room)
        for spec in self.rooms.values():
            if slug(spec.name) == wanted:
                return spec
        return None

    def extents(self, room: str | None) -> Extents:
        """Metric box of ``room``; unknown or ``None`` room -> ``CAMERA_BOX``."""
        spec = None if room is None else self.get(room)
        return dict(spec.extents_m) if spec is not None else dict(CAMERA_BOX)

    def names(self) -> list[str]:
        return list(self.rooms)

    def tag_ids(self) -> list[int]:
        return sorted(self._tag_to_room)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ROOMS_FILE_FORMAT,
            "tag_size_m": self.tag_size_m,
            "tag_family": self.tag_family,
            "rooms": [spec.to_dict() for spec in self.rooms.values()],
        }

    def __len__(self) -> int:
        return len(self.rooms)

    def __contains__(self, room: object) -> bool:
        return isinstance(room, str) and self.get(room) is not None


def room_to_unit(
    p_m: Sequence[float], extents: Mapping[str, tuple[float, float]]
) -> tuple[tuple[float, float, float], bool]:
    """Metres -> LOCI unit cube: ``u_i = clip((p_i - lo_i) / (hi_i - lo_i), 0, 1)``.

    Returns ``(u, clipped)``; ``clipped`` is True when any axis had to be clamped.
    """
    u: list[float] = []
    clipped = False
    for value, axis in zip(p_m, ("x", "y", "z"), strict=True):
        lo, hi = extents[axis]
        raw = (float(value) - lo) / (hi - lo)
        if raw < 0.0 or raw > 1.0 or not math.isfinite(raw):
            clipped = True
        u.append(min(1.0, max(0.0, raw)) if math.isfinite(raw) else 0.0)
    return (u[0], u[1], u[2]), clipped


def unit_to_room(
    u: Sequence[float], extents: Mapping[str, tuple[float, float]]
) -> tuple[float, float, float]:
    """Inverse of :func:`room_to_unit` (for unclipped points)."""
    out: list[float] = []
    for value, axis in zip(u, ("x", "y", "z"), strict=True):
        lo, hi = extents[axis]
        out.append(lo + float(value) * (hi - lo))
    return (out[0], out[1], out[2])


# ---------------------------------------------------------------------------- localizer
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


class RoomLocalizer:
    """Turns tag frames into a smoothed, held ``CameraPose`` in the current room's frame.

    Thread-safe: ``update`` runs on the runtime thread, ``current``/``status`` are read from
    the API/WebSocket side.
    """

    def __init__(
        self,
        rooms: RoomMap,
        detector: TagDetector | None,
        *,
        tag_filter: TagFilter | None = None,
        hold: PoseHold | None = None,
        stale_s: float = 60.0,
        tags_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        upright_fix: bool | None = None,
        multi_tag_max_reproj_px: float = 3.0,
    ) -> None:
        self._rooms = rooms
        self._detector = detector
        self._filter = tag_filter if tag_filter is not None else TagFilter()
        self._hold = hold if hold is not None else PoseHold()
        self._stale_s = float(stale_s)
        self._tags_path = None if tags_path is None else Path(tags_path)
        self._clock = clock
        self._upright_fix = (
            _env_flag("LOCI_TAG_UPRIGHT_FIX") if upright_fix is None else upright_fix
        )
        self._multi_max_reproj = float(multi_tag_max_reproj_px)
        self._lock = threading.RLock()
        self._maps: dict[str, RoomAnchorMap] = {}
        self._tag_ids: tuple[int, ...] = ()
        self._tags_visible: tuple[int, ...] = ()
        self._last_tag_ms: int | None = None
        self._last_anchored_ms: dict[str, int] = {}
        self._tag_frames = 0
        self._poses_accepted = 0
        self._poses_rejected = 0
        self._multi_solves = 0
        self._detect_errors = 0
        self._unknown_tags: set[int] = set()
        self._pnp_unavailable = False
        if self._tags_path is not None:
            self.load()

    # -- properties -----------------------------------------------------------------
    @property
    def rooms(self) -> RoomMap:
        return self._rooms

    @property
    def detector(self) -> TagDetector | None:
        return self._detector

    @property
    def hold(self) -> PoseHold:
        return self._hold

    @property
    def tag_size_m(self) -> float:
        return self._rooms.tag_size_m

    def anchor_map(self, room: str) -> RoomAnchorMap:
        """The (lazily created) anchor map of ``room``; its anchor tag is seeded from the spec."""
        with self._lock:
            return self._map(room)

    def _map(self, room: str) -> RoomAnchorMap:
        amap = self._maps.get(room)
        if amap is None:
            spec = self._rooms.get(room)
            if spec is None:
                raise KeyError(room)
            amap = RoomAnchorMap(room=spec.name, anchor_height_m=spec.anchor_height_m)
            self._seed_from_spec(amap, spec)
            self._maps[spec.name] = amap
        return amap

    @staticmethod
    def _seed_from_spec(amap: RoomAnchorMap, spec: RoomSpec) -> None:
        amap.anchor_height_m = spec.anchor_height_m
        amap.seed_anchor(spec.anchor_tag)
        for tid, T in spec.extra_tags.items():
            amap.tags[int(tid)] = np.asarray(T, dtype=np.float64)

    def _room_of_tag(self, tag_id: int) -> str | None:
        spec = self._rooms.room_for_tag(tag_id)
        if spec is not None:
            return spec.name
        for name, amap in self._maps.items():
            if tag_id in amap.tags:
                return name
        return None

    # -- update ---------------------------------------------------------------------
    def _detect(self, tf: TagFrame) -> list[TagDetection]:
        if tf.tags is not None:
            return list(tf.tags)
        if self._detector is None or tf.gray is None:
            return []
        try:
            return self._detector.detect(tf.gray)
        except Exception:  # a detector crash must not kill the runtime thread
            self._detect_errors += 1
            if self._detect_errors <= 3:
                logger.exception("tag detector failed")
            return []

    def _solve(self, det: TagDetection, tf: TagFrame) -> TagPose | None:
        corners = np.asarray(det.corners_px, dtype=np.float64).reshape(4, 2)
        if int(det.hamming) != 0 or not self._filter.inside_image(corners, tf.width, tf.height):
            return None
        try:
            pose = solve_tag_pose(det.tag_id, corners, tf.K, tf.dist, self.tag_size_m)
        except ValueError:
            return None
        except ImportError as exc:
            if not self._pnp_unavailable:
                logger.error("tag PnP unavailable: %s", exc)
            self._pnp_unavailable = True
            return None
        if self._upright_fix:
            T, flipped = normalize_upright(pose.T_cam_tag)
            if flipped:
                pose = replace(pose, T_cam_tag=T)
        return pose if self._filter.accept(pose) else None

    def update(self, tf: TagFrame) -> CameraPose:
        """Consume one tag frame; returns the pose that is current afterwards."""
        now = self._clock()
        with self._lock:
            self._tag_frames += 1
            self._last_tag_ms = int(tf.timestamp_ms)
            dets = self._detect(tf)
            if tf.tags is None and self._detector is not None and tf.gray is not None:
                # Keep what was detected on the luma plane with the frame: the JSONL
                # recorder never serialises ``gray``, so a replay of a camera recording
                # can only anchor through these (design §4.14).
                tf.tags = list(dets)
            self._tags_visible = tuple(sorted({int(d.tag_id) for d in dets}))
            accepted: list[tuple[TagPose, Array]] = []
            for det in dets:
                pose = self._solve(det, tf)
                if pose is None:
                    self._poses_rejected += 1
                    continue
                self._poses_accepted += 1
                accepted.append((pose, np.asarray(det.corners_px, dtype=np.float64).reshape(4, 2)))
            if accepted:
                self._apply(accepted, tf, now)
            return self._current_locked(now)

    def _apply(self, accepted: list[tuple[TagPose, Array]], tf: TagFrame, now: float) -> None:
        by_room: dict[str, list[tuple[TagPose, Array]]] = {}
        for pose, corners in accepted:
            room = self._room_of_tag(pose.tag_id)
            if room is None:
                if pose.tag_id not in self._unknown_tags:
                    self._unknown_tags.add(pose.tag_id)
                    logger.warning(
                        "tag %d is not in rooms.json and not co-visible with a known tag; ignoring",
                        pose.tag_id,
                    )
                continue
            by_room.setdefault(room, []).append((pose, corners))
        if not by_room:
            return
        # prefer the room with the most known tags in view, then the largest quad
        room, items = max(
            by_room.items(),
            key=lambda kv: (len(kv[1]), max(p.side_px for p, _ in kv[1])),
        )
        amap = self._map(room)
        T_multi: Array | None = None
        if len(items) >= 2:
            res = solve_room_pose_multi(
                [(p.tag_id, c) for p, c in items], amap.tags, tf.K, tf.dist, self.tag_size_m
            )
            if res is not None and res[1] <= self._multi_max_reproj:
                T_multi = res[0]
                self._multi_solves += 1
        # registration candidates: this room's tags plus tags that belong to no room (yet)
        candidates = [p for p, _ in accepted if self._room_of_tag(p.tag_id) in (None, room)]
        before = amap.registrations
        T_room_cam = amap.observe(candidates, T_room_cam=T_multi)
        if T_room_cam is None:
            return
        if amap.registrations != before:
            self._unknown_tags.difference_update(amap.tags)
            self._save_quiet()
        if self._hold.room is not None and self._hold.room != room:
            logger.info("room switch %s -> %s", self._hold.room, room)
        self._hold.update(T_room_cam, now, room)
        self._tag_ids = tuple(sorted(p.tag_id for p, _ in items))
        self._last_anchored_ms[room] = int(tf.timestamp_ms)

    # -- read -----------------------------------------------------------------------
    def set_oracle(self, pose: CameraPose, now_s: float | None = None) -> None:
        """Inject a pose (simulator / tests) in place of a tag observation."""
        now = self._clock() if now_s is None else now_s
        with self._lock:
            if pose.frame == "room" and pose.T_room_cam is not None and pose.room:
                age = pose.age_s if math.isfinite(pose.age_s) else 0.0
                self._hold.reset(
                    np.asarray(pose.T_room_cam, dtype=np.float64), pose.room, now - age
                )
                self._tag_ids = tuple(int(t) for t in pose.tag_ids)
                if pose.anchored:
                    self._last_anchored_ms[pose.room] = int(time.time() * 1000)
            else:
                self._hold.reset(None, pose.room, None)
                self._tag_ids = ()

    def current(self, now_s: float | None = None) -> CameraPose:
        now = self._clock() if now_s is None else now_s
        with self._lock:
            return self._current_locked(now)

    def _current_locked(self, now: float) -> CameraPose:
        T, conf, anchored = self._hold.current(now)
        age = self._hold.age_s(now)
        if T is not None and age <= self._stale_s:
            return CameraPose(
                room=self._hold.room,
                frame="room",
                T_room_cam=T,
                anchored=anchored,
                anchor_confidence=conf,
                tag_ids=self._tag_ids,
                age_s=age,
            )
        return CameraPose(
            room=self._hold.room,
            frame="camera",
            T_room_cam=None,
            anchored=False,
            anchor_confidence=0.0,
            tag_ids=(),
            age_s=age,
        )

    @staticmethod
    def to_room(pose: CameraPose, xyz_cam_m: Sequence[float]) -> tuple[float, float, float] | None:
        """Camera-frame metres -> room metres; ``None`` when the pose is not a room pose."""
        if pose.frame != "room" or pose.T_room_cam is None:
            return None
        p = apply_T(np.asarray(pose.T_room_cam, dtype=np.float64), xyz_cam_m)
        return (float(p[0]), float(p[1]), float(p[2]))

    def status(self) -> dict[str, Any]:
        with self._lock:
            pose = self._current_locked(self._clock())
            return {
                "room": pose.room,
                "frame": pose.frame,
                "anchored": pose.anchored,
                "anchor_confidence": pose.anchor_confidence,
                "age_s": pose.age_s if math.isfinite(pose.age_s) else None,
                "tags_visible": list(self._tags_visible),
                "tags_registered": {
                    name: sorted(amap.tags) for name, amap in sorted(self._maps.items())
                },
                "tags_pending": {
                    name: amap.pending_ids
                    for name, amap in sorted(self._maps.items())
                    if amap.pending_ids
                },
                "last_anchored_ms": dict(self._last_anchored_ms),
                "detector": None if self._detector is None else self._detector.name,
                "last_tag_ms": self._last_tag_ms,
                "tag_frames": self._tag_frames,
                "poses_accepted": self._poses_accepted,
                "poses_rejected": self._poses_rejected,
                "multi_tag_solves": self._multi_solves,
                "unknown_tags": sorted(self._unknown_tags),
                "pnp_available": not self._pnp_unavailable,
                "tags_path": None if self._tags_path is None else str(self._tags_path),
            }

    # -- persistence (tags.json) ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "format": TAGS_FILE_FORMAT,
                "rooms": {name: amap.to_dict() for name, amap in sorted(self._maps.items())},
            }

    def save(self) -> None:
        """Atomically write ``tags.json`` (no-op without ``tags_path``)."""
        if self._tags_path is None:
            return
        payload = json.dumps(self.to_dict(), indent=1)
        path = self._tags_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _save_quiet(self) -> None:
        try:
            self.save()
        except OSError:
            logger.exception("could not write %s", self._tags_path)

    def load(self) -> None:
        """Merge registered tags from ``tags.json``; the ``rooms.json`` anchor is authoritative."""
        if self._tags_path is None or not self._tags_path.exists():
            return
        try:
            with open(self._tags_path, encoding="utf-8") as fh:
                data = json.load(fh)
            rooms = dict(data.get("rooms", {}))
        except (OSError, ValueError, AttributeError, TypeError):
            logger.exception("ignoring unreadable tags file %s", self._tags_path)
            return
        with self._lock:
            for name, raw in rooms.items():
                spec = self._rooms.get(str(name))
                if spec is None:
                    logger.warning("tags.json room %r is not in rooms.json; skipped", name)
                    continue
                try:
                    amap = RoomAnchorMap.from_dict({**dict(raw), "room": spec.name})
                except (KeyError, ValueError, TypeError):
                    logger.exception("tags.json entry for %r is malformed; skipped", name)
                    continue
                self._seed_from_spec(amap, spec)
                self._maps[spec.name] = amap
