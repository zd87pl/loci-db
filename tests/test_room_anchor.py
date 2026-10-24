"""Tests for demo_spatial/app/room_anchor.py (detectors, TagFilter, RoomAnchorMap, PoseHold,
RoomMap, room_to_unit/unit_to_room, RoomLocalizer, tags.json).

numpy only: the localizer logic is exercised with the PnP solvers replaced by a fake that maps
tag ids to known poses; one test runs the real cv2 solver on projected corners and the
rendered-tag detector tests skip without cv2 / pyapriltags.

``app.detection_source`` (task T2) is used when present; otherwise a minimal fake of its
section 4.2 dataclasses is installed so this module can be tested in isolation.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.geometry import (
    TagPose,
    apply_T,
    camera_pose_T,
    invert_T,
    make_T,
    project_points,
    rot_y,
    rot_z,
    rotation_angle_deg,
    tag_object_points,
)


def _install_fake_detection_source() -> None:
    """Fall back to local stand-ins for T2's dataclasses when app.detection_source is absent."""
    try:
        importlib.import_module("app.detection_source")
        return
    except ImportError:
        pass
    mod = types.ModuleType("app.detection_source")

    @dataclass
    class TagDetection:
        tag_id: int
        corners_px: np.ndarray
        decision_margin: float
        hamming: int

    @dataclass
    class TagFrame:
        timestamp_ms: int
        width: int
        height: int
        K: np.ndarray
        dist: np.ndarray | None
        gray: np.ndarray | None
        tags: list | None

    @dataclass
    class CameraPose:
        room: str | None
        frame: str
        T_room_cam: np.ndarray | None
        anchored: bool
        anchor_confidence: float
        tag_ids: tuple
        age_s: float

    mod.TagDetection = TagDetection  # type: ignore[attr-defined]
    mod.TagFrame = TagFrame  # type: ignore[attr-defined]
    mod.CameraPose = CameraPose  # type: ignore[attr-defined]
    mod.__loci_test_fake__ = True  # type: ignore[attr-defined]
    sys.modules["app.detection_source"] = mod


_install_fake_detection_source()

import app.room_anchor as ra  # noqa: E402
from app.detection_source import CameraPose, TagDetection, TagFrame  # noqa: E402
from app.room_anchor import (  # noqa: E402
    CAMERA_BOX,
    DEFAULT_ROOM_EXTENTS,
    ArucoTagDetector,
    PoseHold,
    PyAprilTagsDetector,
    RoomAnchorMap,
    RoomLocalizer,
    RoomMap,
    RoomSpec,
    TagFilter,
    make_tag_detector,
    remap_pyapriltags_corners,
    room_to_unit,
    slug,
    unit_to_room,
)

W, H = 1280, 800
TAG = 0.16
K_TAG = np.array([[918.3, 0.0, 645.8], [0.0, 917.5, 417.4], [0.0, 0.0, 1.0]])
T_ROOM_TAG0 = make_T(np.eye(3), (0.0, 1.4, 0.0))  # anchor of every default room
EXAMPLE = (
    Path(__file__).resolve().parents[1] / "deploy" / "handheld" / "config" / "rooms.example.json"
)


# ----------------------------------------------------------------------------- helpers
class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _tag_pose(
    tag_id: int,
    T_cam_tag: np.ndarray,
    side: float = 60.0,
    reproj: float = 0.1,
    ratio: float = 5.0,
    view: float = 10.0,
) -> TagPose:
    return TagPose(tag_id, np.asarray(T_cam_tag, dtype=np.float64), reproj, ratio, side, view)


def _square(cx: float = 640.0, cy: float = 400.0, side: float = 60.0) -> np.ndarray:
    s = side / 2.0
    return np.array([[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s], [cx - s, cy + s]])


def _det(
    tag_id: int, corners: np.ndarray | None = None, margin: float = 50.0, hamming: int = 0
) -> TagDetection:
    return TagDetection(
        tag_id=tag_id,
        corners_px=_square() if corners is None else corners,
        decision_margin=margin,
        hamming=hamming,
    )


def _frame(ts: int, dets: list[TagDetection] | None, gray: np.ndarray | None = None) -> TagFrame:
    return TagFrame(timestamp_ms=ts, width=W, height=H, K=K_TAG, dist=None, gray=gray, tags=dets)


def _corners(T_room_cam: np.ndarray, T_room_tag: np.ndarray) -> np.ndarray:
    return project_points(tag_object_points(TAG), invert_T(T_room_cam) @ T_room_tag, K_TAG)


class FakePnP:
    """Stands in for geometry's cv2 solvers inside room_anchor: tag id -> T_cam_tag per frame."""

    def __init__(self) -> None:
        self.truth: dict[int, np.ndarray] = {}
        self.side = 60.0
        self.multi: tuple[np.ndarray, float] | None = None
        self.calls = 0
        self.multi_calls = 0

    def solve(self, tag_id, corners, K, dist, tag_size_m, min_distinct_deg=3.0) -> TagPose:
        self.calls += 1
        assert np.asarray(corners).shape == (4, 2)
        assert tag_size_m == pytest.approx(TAG)
        if tag_id not in self.truth:
            raise ValueError("no valid PnP solution")
        return _tag_pose(tag_id, self.truth[tag_id], side=self.side)

    def solve_multi(self, detections, tag_map, K, dist, tag_size_m):
        self.multi_calls += 1
        assert len(detections) >= 2
        return self.multi


@pytest.fixture
def fake_pnp(monkeypatch: pytest.MonkeyPatch) -> FakePnP:
    f = FakePnP()
    monkeypatch.setattr(ra, "solve_tag_pose", f.solve)
    monkeypatch.setattr(ra, "solve_room_pose_multi", f.solve_multi)
    return f


def _localizer(tmp_path: Path, clock: FakeClock, **kw) -> RoomLocalizer:
    kw.setdefault("hold", PoseHold(hold_s=10.0))
    kw.setdefault("stale_s", 60.0)
    kw.setdefault("tags_path", tmp_path / "tags.json")
    kw.setdefault("upright_fix", False)
    return RoomLocalizer(RoomMap.default(), None, clock=clock, **kw)


# ----------------------------------------------------------------------------- TagFilter
def test_tag_filter_thresholds() -> None:
    f = TagFilter()
    T = make_T(np.eye(3), (0.0, 0.0, 2.0))
    good = _tag_pose(0, T)
    assert f.accept(good)
    assert f.reasons(good) == []
    assert f.reasons(_tag_pose(0, T, side=20.0)) == ["small"]
    assert f.reasons(_tag_pose(0, T, reproj=1.5)) == ["reproj"]
    assert f.reasons(_tag_pose(0, T, reproj=math.nan)) == ["reproj"]
    assert f.reasons(_tag_pose(0, T, ratio=1.2)) == ["ambiguous"]
    assert f.reasons(_tag_pose(0, T, view=70.0)) == ["oblique"]
    assert f.reasons(_tag_pose(0, make_T(np.eye(3), (0.0, 0.0, 7.0)))) == ["range"]
    assert f.reasons(_tag_pose(0, make_T(np.eye(3), (0.0, 0.0, 0.0)))) == ["range"]
    assert not f.accept(_tag_pose(0, T, side=20.0, view=70.0))
    # boundaries are inclusive
    assert f.accept(
        _tag_pose(
            0, make_T(np.eye(3), (0.0, 0.0, 6.0)), side=30.0, reproj=1.0, ratio=1.5, view=60.0
        )
    )
    assert TagFilter(min_side_px=10.0).accept(_tag_pose(0, T, side=20.0))


def test_tag_filter_inside_image() -> None:
    f = TagFilter(border_margin_px=2.0)
    assert f.inside_image(_square(), W, H)
    assert f.inside_image(_square(cx=33.0, cy=400.0), W, H)  # left edge at 3 px
    assert not f.inside_image(_square(cx=31.0, cy=400.0), W, H)  # left edge at 1 px
    assert not f.inside_image(_square(cx=1250.0, cy=400.0), W, H)  # right edge beyond 1277
    assert not f.inside_image(_square(cx=640.0, cy=30.0), W, H)
    assert not f.inside_image(_square(cx=640.0, cy=775.0), W, H)


# ----------------------------------------------------------------------------- RoomAnchorMap
def test_room_anchor_map_registers_after_ten_covisible_frames() -> None:
    T_room_tagB = make_T(rot_y(math.radians(90)), (3.0, 1.3, 2.0))  # right-hand wall
    m = RoomAnchorMap("kitchen", anchor_height_m=1.4)
    assert m.register_samples == 10
    m.seed_anchor(1)
    assert m.anchor_id == 1
    assert np.allclose(m.tags[1], T_ROOM_TAG0)
    for i in range(10):
        T_room_cam = camera_pose_T((0.1 * i, 1.4, 2.5), -3.0 * i)
        out = m.observe(
            [
                _tag_pose(1, invert_T(T_room_cam) @ T_ROOM_TAG0),
                _tag_pose(2, invert_T(T_room_cam) @ T_room_tagB),
            ]
        )
        assert out is not None
        assert np.allclose(out, T_room_cam, atol=1e-9)
        if i < 9:
            assert 2 not in m.tags
            assert m.pending_ids == [2]
    assert 2 in m.tags
    assert np.allclose(m.tags[2], T_room_tagB, atol=1e-9)
    assert m.registrations == 1
    assert m.pending_ids == []
    # dict round trip (through JSON like tags.json)
    m2 = RoomAnchorMap.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.room == "kitchen"
    assert m2.anchor_id == 1
    assert m2.anchor_height_m == 1.4
    assert sorted(m2.tags) == [1, 2]
    assert np.allclose(m2.tags[2], T_room_tagB)
    # tag 2 alone now localises the camera
    T_room_cam = camera_pose_T((1.0, 1.5, 2.0), 30.0)
    out = m2.observe([_tag_pose(2, invert_T(T_room_cam) @ T_room_tagB)])
    assert np.allclose(out, T_room_cam, atol=1e-9)
    # two known tags: the mean of both estimates (identical here)
    out = m2.observe(
        [
            _tag_pose(1, invert_T(T_room_cam) @ T_ROOM_TAG0),
            _tag_pose(2, invert_T(T_room_cam) @ T_room_tagB),
        ]
    )
    assert np.allclose(out, T_room_cam, atol=1e-9)
    assert m2.observe([_tag_pose(9, np.eye(4))]) is None
    assert m2.observe([]) is None


def test_room_anchor_map_unseeded_first_largest_tag_is_anchor() -> None:
    m = RoomAnchorMap("x")
    T = make_T(np.diag([1.0, -1.0, -1.0]), (0.0, 0.0, 2.0))
    m.observe([_tag_pose(4, T, side=50.0), _tag_pose(5, T, side=80.0)])
    assert m.anchor_id == 5
    assert np.allclose(m.tags[5], T_ROOM_TAG0)


def test_room_anchor_map_small_or_provided_pose_registration() -> None:
    T_room_tagB = make_T(rot_y(math.radians(90)), (3.0, 1.3, 2.0))
    T_room_cam = camera_pose_T((0.2, 1.4, 2.5), 0.0)
    pA = _tag_pose(1, invert_T(T_room_cam) @ T_ROOM_TAG0)
    pB_small = _tag_pose(2, invert_T(T_room_cam) @ T_room_tagB, side=35.0)
    m = RoomAnchorMap("k", register_samples=1)
    m.seed_anchor(1)
    assert m.observe([pA, pB_small]) is not None
    assert 2 not in m.tags and m.pending_ids == []  # below register_min_side_px
    pB = _tag_pose(2, invert_T(T_room_cam) @ T_room_tagB, side=45.0)
    T_given = camera_pose_T((0.2, 1.4, 2.6), 1.0)  # a joint solve result
    out = m.observe([pA, pB], T_room_cam=T_given)
    assert np.allclose(out, T_given)
    assert np.allclose(m.tags[2], T_given @ pB.T_cam_tag)


# ----------------------------------------------------------------------------- PoseHold
def test_pose_hold_confidence_and_expiry() -> None:
    T = camera_pose_T((0.0, 1.4, 2.5), 0.0)
    ph = PoseHold(hold_s=10.0)
    assert ph.current(0.0) == (None, 0.0, False)
    assert ph.age_s(0.0) == math.inf
    ph.update(T, 0.0, "kitchen")
    T0, c0, a0 = ph.current(0.0)
    assert a0 and c0 == pytest.approx(1.0) and np.allclose(T0, T)
    _, c5, a5 = ph.current(5.0)
    assert a5 and c5 == pytest.approx(0.5)
    T12, c12, a12 = ph.current(12.0)
    assert not a12 and c12 == 0.0 and T12 is not None
    assert ph.age_s(12.0) == pytest.approx(12.0)
    assert ph.room == "kitchen"


def test_pose_hold_jump_gating_and_ema() -> None:
    T = camera_pose_T((0.0, 1.4, 2.5), 0.0)
    ph = PoseHold(hold_s=10.0)
    ph.update(T, 0.0, "kitchen")
    wild = make_T(T[:3, :3], T[:3, 3] + [5.0, 0.0, 0.0])
    ph.update(wild, 1.0, "kitchen")
    assert np.allclose(ph.T, T)  # isolated outlier ignored
    assert ph.last_seen_s == 0.0  # and it does not refresh the hold
    ph.update(wild, 1.1, "kitchen")
    ph.update(wild, 1.2, "kitchen")
    assert np.allclose(ph.T, wild)  # persistent jump accepted
    assert ph.last_seen_s == 1.2
    ph2 = PoseHold()
    ph2.update(T, 0.0, "k")
    target = make_T(T[:3, :3] @ rot_y(math.radians(10)), T[:3, 3] + [0.3, 0.0, 0.0])
    for i in range(40):
        ph2.update(target, 0.1 * i, "k")
    assert np.allclose(ph2.T, target, atol=1e-3)
    assert rotation_angle_deg(ph2.T[:3, :3], target[:3, :3]) < 0.1


def test_pose_hold_room_switch_and_reset() -> None:
    T_k = camera_pose_T((0.0, 1.4, 2.5), 0.0)
    T_l = camera_pose_T((5.0, 1.4, 1.0), 90.0)
    ph = PoseHold()
    ph.update(T_k, 0.0, "kitchen")
    ph.update(T_l, 0.2, "living_room")  # a 5 m / 90 deg jump, but a new room: accepted at once
    assert ph.room == "living_room"
    assert np.allclose(ph.T, T_l)
    ph.reset(T_k, "kitchen", 3.0)
    assert ph.current(3.0)[2] and np.allclose(ph.T, T_k) and ph.room == "kitchen"
    ph.reset()
    assert ph.current(3.0) == (None, 0.0, False)
    assert ph.room is None


# ----------------------------------------------------------------------------- unit cube
def test_room_to_unit_and_unit_to_room() -> None:
    assert CAMERA_BOX == {"x": (-3.0, 3.0), "y": (-1.5, 1.5), "z": (0.0, 6.0)}
    assert DEFAULT_ROOM_EXTENTS == {"x": (-3.0, 3.0), "y": (0.0, 2.5), "z": (0.0, 6.0)}
    u, clipped = room_to_unit((0.0, 1.25, 3.0), DEFAULT_ROOM_EXTENTS)
    assert u == pytest.approx((0.5, 0.5, 0.5)) and not clipped
    u, clipped = room_to_unit((3.0, 2.5, 6.0), DEFAULT_ROOM_EXTENTS)
    assert u == (1.0, 1.0, 1.0) and not clipped
    u, clipped = room_to_unit((-4.0, 3.0, 7.0), DEFAULT_ROOM_EXTENTS)
    assert u == (0.0, 1.0, 1.0) and clipped
    u, clipped = room_to_unit((0.0, 0.0, 0.0), CAMERA_BOX)
    assert u == pytest.approx((0.5, 0.5, 0.0)) and not clipped
    u, clipped = room_to_unit((0.0, -2.0, 1.0), CAMERA_BOX)
    assert clipped and u[1] == 0.0
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = (rng.uniform(-3, 3), rng.uniform(0, 2.5), rng.uniform(0, 6))
        u, clipped = room_to_unit(p, DEFAULT_ROOM_EXTENTS)
        assert not clipped
        assert all(0.0 <= v <= 1.0 for v in u)
        assert unit_to_room(u, DEFAULT_ROOM_EXTENTS) == pytest.approx(p)
    assert unit_to_room((0.5, 0.5, 0.5), CAMERA_BOX) == pytest.approx((0.0, 0.0, 3.0))


def test_slug() -> None:
    assert slug("Living Room") == "living_room"
    assert slug("living_room") == "living_room"
    assert slug("a/b") == "a_b"
    assert slug("  kitchen ") == "kitchen"
    assert slug("x  y") == "x_y"


# ----------------------------------------------------------------------------- RoomMap
def test_room_map_load_example_file() -> None:
    rm = RoomMap.load(EXAMPLE, env={})
    assert rm.names() == ["kitchen", "living_room"]
    assert rm.tag_size_m == pytest.approx(0.16)
    assert rm.tag_family == "tag36h11"
    k = rm.get("kitchen")
    assert k is not None
    assert k.anchor_tag == 0
    assert k.anchor_height_m == 1.4
    assert k.anchor_name == "door tag"
    assert k.landmarks == {"table": (1.0, 0.75, 1.2)}
    assert k.extents_m == DEFAULT_ROOM_EXTENTS
    assert k.extra_tags == {}
    assert np.allclose(k.T_room_anchor, T_ROOM_TAG0)
    lr = rm.room_for_tag(1)
    assert lr is not None and lr.name == "living_room"
    assert lr.extents_m == DEFAULT_ROOM_EXTENTS and lr.landmarks == {}
    assert rm.room_for_tag(7) is None
    assert rm.extents(None) == CAMERA_BOX
    assert rm.extents("garage") == CAMERA_BOX
    assert rm.extents("living room") == DEFAULT_ROOM_EXTENTS  # slug lookup
    assert rm.get("Living Room") is lr
    assert "living room" in rm and "garage" not in rm
    assert len(rm) == 2
    assert rm.tag_ids() == [0, 1]
    d = rm.to_dict()
    assert d["format"] == 1
    assert d["rooms"][0]["landmarks"] == {"table": [1.0, 0.75, 1.2]}
    rm2 = RoomMap.from_dict(json.loads(json.dumps(d)))
    assert rm2.names() == rm.names() and rm2.tag_size_m == rm.tag_size_m


def test_room_map_tag_size_precedence() -> None:
    assert RoomMap.load(EXAMPLE, env={"LOCI_TAG_SIZE_M": "0.24"}).tag_size_m == pytest.approx(0.24)
    assert RoomMap.load(EXAMPLE, tag_size_m=0.2, env={"LOCI_TAG_SIZE_M": "0.24"}).tag_size_m == 0.2
    assert RoomMap.load(EXAMPLE, env={"LOCI_TAG_SIZE_M": ""}).tag_size_m == pytest.approx(0.16)


def test_room_map_default_and_validation(tmp_path: Path) -> None:
    rm = RoomMap.default()
    assert rm.names() == ["kitchen", "living_room"]
    assert rm.room_for_tag(0).name == "kitchen" and rm.room_for_tag(1).name == "living_room"
    assert rm.tag_size_m == 0.16
    with pytest.raises(ValueError, match="tag 0"):
        RoomMap([RoomSpec("a", 0), RoomSpec("b", 0)])
    with pytest.raises(ValueError, match="duplicate room"):
        RoomMap([RoomSpec("a", 0), RoomSpec("a", 1)])
    with pytest.raises(ValueError, match="tag_size_m"):
        RoomMap([RoomSpec("a", 0)], tag_size_m=0.0)
    with pytest.raises(ValueError, match="anchor_tag"):
        RoomSpec.from_dict({"name": "a"})
    with pytest.raises(ValueError, match="extents_m"):
        RoomSpec.from_dict({"name": "a", "anchor_tag": 0, "extents_m": {"x": [3.0, -3.0]}})
    with pytest.raises(ValueError, match="landmark"):
        RoomSpec.from_dict({"name": "a", "anchor_tag": 0, "landmarks": {"t": [1.0, 2.0]}})
    bad = tmp_path / "rooms.json"
    bad.write_text(
        '{"format": 1, "rooms": [{"name": "a", "anchor_tag": 0}, {"name": "b", "anchor_tag": 0}]}'
    )
    with pytest.raises(ValueError):
        RoomMap.load(bad, env={})
    # the {"name": {...}} layout is tolerated
    rm3 = RoomMap.from_dict({"rooms": {"hall": {"anchor_tag": 4, "extents_m": {"z": [0, 10]}}}})
    assert rm3.get("hall").extents_m["z"] == (0.0, 10.0)
    assert rm3.get("hall").extents_m["x"] == DEFAULT_ROOM_EXTENTS["x"]


def test_room_spec_extra_tags_forms() -> None:
    T5 = make_T(rot_y(math.radians(-90)), (3.0, 1.3, 2.0))
    spec = RoomSpec.from_dict(
        {
            "name": "k",
            "anchor_tag": 0,
            "extra_tags": {
                "5": T5.tolist(),
                "6": {"pos_m": [3.0, 1.3, 2.0], "normal_yaw_deg": 90.0},
            },
        }
    )
    assert spec.tag_ids() == [0, 5, 6]
    assert np.allclose(spec.extra_tags[5], T5)
    T6 = spec.extra_tags[6]
    assert np.allclose(T6[:3, 3], (3.0, 1.3, 2.0))
    assert np.allclose(T6[:3, 2], (1.0, 0.0, 0.0), atol=1e-12)  # normal faces +x
    assert np.allclose(T6[:3, 1], (0.0, 1.0, 0.0))  # tag y stays up
    rm = RoomMap([spec])
    assert rm.room_for_tag(6).name == "k"
    assert json.loads(json.dumps(spec.to_dict()))["extra_tags"]["5"] == T5.tolist()
    with pytest.raises(ValueError, match="extra_tags"):
        RoomSpec.from_dict({"name": "k", "anchor_tag": 0, "extra_tags": {"5": [1, 2, 3]}})


# ----------------------------------------------------------------------------- detectors
def test_remap_pyapriltags_corners() -> None:
    raw = [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # TR, TL, BL, BR
    assert np.array_equal(remap_pyapriltags_corners(raw), [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_make_tag_detector_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_tag_detector("none") is None
    assert make_tag_detector("off") is None
    assert make_tag_detector(None, env={"LOCI_TAG_DETECTOR": "none"}) is None
    with pytest.raises(ValueError, match="unknown tag detector"):
        make_tag_detector("bogus")
    monkeypatch.setitem(sys.modules, "pyapriltags", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError):
        make_tag_detector("pyapriltags")
    with pytest.raises(ImportError):
        make_tag_detector("aruco")
    assert make_tag_detector("auto") is None
    assert make_tag_detector(None, env={}) is None


def test_make_tag_detector_auto_matches_installed() -> None:
    d = make_tag_detector("auto")
    if importlib.util.find_spec("pyapriltags") is not None:
        assert d is not None and d.name == "pyapriltags"
    elif importlib.util.find_spec("cv2") is not None:
        assert d is not None and d.name == "aruco"
    else:
        assert d is None


def _render_tag(cv2, T_cam_tag: np.ndarray, tag_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Render a cv2.aruco 36h11 tag at a known pose; returns (gray, true corners TL,TR,BR,BL)."""
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    cell = 40
    marker = cv2.aruco.generateImageMarker(dic, tag_id, 8 * cell)
    canvas = np.full((10 * cell, 10 * cell), 255, np.uint8)
    canvas[cell : 9 * cell, cell : 9 * cell] = marker
    a, b = cell - 0.5, 9 * cell - 0.5
    src = np.array([[a, a], [b, a], [b, b], [a, b]], dtype=np.float32)
    corners = project_points(tag_object_points(TAG), T_cam_tag, K_TAG)
    M = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    frame = np.full((H, W), 150, np.uint8)
    warped = cv2.warpPerspective(canvas, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full(canvas.shape, 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST
    )
    frame = np.where(mask > 127, warped, frame).astype(np.uint8)
    return cv2.GaussianBlur(frame, (0, 0), 0.8), corners


_T_CAM_TAG_RENDER = make_T(
    np.diag([1.0, -1.0, -1.0]) @ rot_y(math.radians(-20)) @ rot_z(math.radians(5)), (-0.4, 0.2, 2.5)
)


def test_aruco_tag_detector_on_rendered_tag() -> None:
    cv2 = pytest.importorskip("cv2")
    det = ArucoTagDetector()
    assert det.name == "aruco"
    frame, true_corners = _render_tag(cv2, _T_CAM_TAG_RENDER, 7)
    out = det.detect(frame)
    hits = [d for d in out if d.tag_id == 7]
    assert len(hits) == 1
    d = hits[0]
    assert d.corners_px.shape == (4, 2) and d.corners_px.dtype == np.float64
    assert np.abs(d.corners_px - true_corners).max() < 1.5
    assert d.hamming == 0 and d.decision_margin == 100.0
    assert det.detect(np.full((100, 100), 128, np.uint8)) == []
    # three-channel input is accepted too
    assert [d.tag_id for d in det.detect(np.stack([frame] * 3, axis=2))] == [7]
    with pytest.raises(ValueError, match="family"):
        ArucoTagDetector(family="tag99h99")


def test_pyapriltags_tag_detector_on_rendered_tag() -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("pyapriltags")
    det = PyAprilTagsDetector(quad_decimate=1.0)
    assert det.name == "pyapriltags"
    frame, true_corners = _render_tag(cv2, _T_CAM_TAG_RENDER, 7)
    hits = [d for d in det.detect(frame) if d.tag_id == 7]
    assert len(hits) == 1
    d = hits[0]
    assert np.abs(d.corners_px - true_corners).max() < 1.5  # TL, TR, BR, BL after the remap
    assert d.hamming == 0 and d.decision_margin >= 30.0
    assert PyAprilTagsDetector(quad_decimate=1.0, min_margin=1e9).detect(frame) == []


# ----------------------------------------------------------------------------- RoomLocalizer
def test_localizer_anchoring_hold_stale_and_status(tmp_path: Path, fake_pnp: FakePnP) -> None:
    clock = FakeClock(100.0)
    loc = _localizer(tmp_path, clock)
    p = loc.current()
    assert p.frame == "camera" and p.room is None and not p.anchored
    assert p.anchor_confidence == 0.0 and p.tag_ids == () and p.age_s == math.inf
    assert loc.to_room(p, (0.0, 0.0, 1.0)) is None
    T_true = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    fake_pnp.truth = {0: invert_T(T_true) @ T_ROOM_TAG0}
    pose = loc.update(_frame(1_000, [_det(0)]))
    assert pose.room == "kitchen" and pose.frame == "room" and pose.anchored
    assert (
        pose.anchor_confidence == pytest.approx(1.0) and pose.tag_ids == (0,) and pose.age_s == 0.0
    )
    assert np.allclose(pose.T_room_cam, T_true)
    p_room = np.array([1.0, 0.8, 1.0])
    xyz_cam = apply_T(invert_T(T_true), p_room)
    assert xyz_cam[2] > 0
    assert loc.to_room(pose, xyz_cam) == pytest.approx(tuple(p_room))
    assert loc.to_room(pose, tuple(xyz_cam)) == pytest.approx(tuple(p_room))
    clock.t = 105.0
    p = loc.current()
    assert (
        p.anchored and p.anchor_confidence == pytest.approx(0.5) and p.age_s == pytest.approx(5.0)
    )
    clock.t = 111.0
    p = loc.current()
    assert p.frame == "room" and p.room == "kitchen" and not p.anchored
    assert p.anchor_confidence == 0.0 and p.T_room_cam is not None and p.tag_ids == (0,)
    assert loc.to_room(p, xyz_cam) == pytest.approx(tuple(p_room))
    clock.t = 161.0
    p = loc.current()
    assert p.frame == "camera" and p.room == "kitchen" and p.T_room_cam is None
    assert not p.anchored and p.tag_ids == () and p.age_s == pytest.approx(61.0)
    assert loc.to_room(p, xyz_cam) is None
    st = loc.status()
    for key in (
        "room",
        "frame",
        "anchored",
        "anchor_confidence",
        "age_s",
        "tags_visible",
        "tags_registered",
        "detector",
        "last_tag_ms",
        "tag_frames",
        "poses_accepted",
        "poses_rejected",
    ):
        assert key in st
    assert st["room"] == "kitchen" and st["frame"] == "camera" and st["anchored"] is False
    assert st["age_s"] == pytest.approx(61.0)
    assert st["tags_visible"] == [0] and st["tags_registered"] == {"kitchen": [0]}
    assert st["detector"] is None and st["last_tag_ms"] == 1000 and st["tag_frames"] == 1
    assert st["poses_accepted"] == 1 and st["poses_rejected"] == 0
    assert st["last_anchored_ms"] == {"kitchen": 1000}
    json.dumps(st)  # JSON-safe for /api/handheld/status
    # re-anchoring works after the stale period
    clock.t = 162.0
    pose = loc.update(_frame(62_000, [_det(0)]))
    assert pose.anchored and pose.frame == "room" and np.allclose(pose.T_room_cam, T_true)
    assert loc.current(now_s=200.0).frame == "room"
    assert loc.current(now_s=300.0).frame == "camera"


def test_localizer_room_switch_resets_hold(tmp_path: Path, fake_pnp: FakePnP) -> None:
    clock = FakeClock(0.0)
    loc = _localizer(tmp_path, clock)
    T_k = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    T_l = camera_pose_T((-1.0, 1.5, 3.0), 20.0)
    fake_pnp.truth = {0: invert_T(T_k) @ T_ROOM_TAG0}
    assert loc.update(_frame(0, [_det(0)])).room == "kitchen"
    clock.t = 0.2
    fake_pnp.truth = {1: invert_T(T_l) @ T_ROOM_TAG0}
    pose = loc.update(_frame(200, [_det(1)]))
    assert pose.room == "living_room" and pose.anchored and pose.tag_ids == (1,)
    assert np.allclose(pose.T_room_cam, T_l)  # no EMA blend across rooms
    clock.t = 0.4
    fake_pnp.truth = {0: invert_T(T_k) @ T_ROOM_TAG0}
    pose = loc.update(_frame(400, [_det(0)]))
    assert pose.room == "kitchen" and np.allclose(pose.T_room_cam, T_k)
    assert loc.status()["last_anchored_ms"] == {"kitchen": 400, "living_room": 200}
    # both rooms' tags in one frame: the larger quad decides
    clock.t = 0.6
    fake_pnp.truth = {0: invert_T(T_k) @ T_ROOM_TAG0, 1: invert_T(T_l) @ T_ROOM_TAG0}
    pose = loc.update(_frame(600, [_det(0), _det(1)]))
    assert pose.room == "kitchen"  # equal sides: max() keeps the first room encountered


def test_localizer_rejections_and_unknown_tags(tmp_path: Path, fake_pnp: FakePnP) -> None:
    clock = FakeClock(0.0)
    loc = _localizer(tmp_path, clock)
    T_true = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    fake_pnp.truth = {0: invert_T(T_true) @ T_ROOM_TAG0, 9: invert_T(T_true) @ T_ROOM_TAG0}
    assert loc.update(_frame(0, [_det(0, hamming=1)])).frame == "camera"
    assert loc.status()["poses_rejected"] == 1 and fake_pnp.calls == 0
    assert loc.update(_frame(100, [_det(0, corners=_square(cx=30.0))])).frame == "camera"
    assert loc.status()["poses_rejected"] == 2 and fake_pnp.calls == 0
    fake_pnp.side = 20.0  # TagFilter rejects small quads
    assert loc.update(_frame(200, [_det(0)])).frame == "camera"
    assert loc.status()["poses_rejected"] == 3 and fake_pnp.calls == 1
    fake_pnp.side = 60.0
    assert loc.update(_frame(300, [_det(3)])).frame == "camera"  # solver: no solution
    assert loc.status()["poses_rejected"] == 4
    pose = loc.update(_frame(400, [_det(9)]))  # accepted pose, but tag 9 belongs to no room
    assert pose.frame == "camera"
    st = loc.status()
    assert st["poses_accepted"] == 1 and st["unknown_tags"] == [9] and st["tags_visible"] == [9]
    pose = loc.update(_frame(500, [_det(0), _det(9)]))
    assert pose.room == "kitchen" and pose.anchored and pose.tag_ids == (0,)
    assert loc.status()["poses_accepted"] == 3
    assert not (tmp_path / "tags.json").exists()  # nothing registered -> nothing written yet
    # empty / None tag lists are harmless
    assert loc.update(_frame(600, [])).room == "kitchen"
    assert loc.update(_frame(700, None)).room == "kitchen"
    assert loc.status()["tag_frames"] == 8


def test_localizer_covisible_registration_multi_solve_and_tags_json(
    tmp_path: Path, fake_pnp: FakePnP
) -> None:
    clock = FakeClock(0.0)
    loc = _localizer(tmp_path, clock)
    T_true = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    T_room_tag5 = make_T(rot_y(math.radians(-90)), (3.0, 1.3, 2.0))
    fake_pnp.truth = {0: invert_T(T_true) @ T_ROOM_TAG0, 5: invert_T(T_true) @ T_room_tag5}
    for i in range(10):
        clock.t = 0.2 * i
        pose = loc.update(_frame(200 * i, [_det(0), _det(5)]))
        assert pose.room == "kitchen" and np.allclose(pose.T_room_cam, T_true, atol=1e-9)
        if i < 9:
            assert loc.status()["tags_registered"] == {"kitchen": [0]}
            assert loc.status()["tags_pending"] == {"kitchen": [5]}
    st = loc.status()
    assert st["tags_registered"] == {"kitchen": [0, 5]} and st["tags_pending"] == {}
    assert st["unknown_tags"] == []
    assert np.allclose(loc.anchor_map("kitchen").tags[5], T_room_tag5, atol=1e-9)
    tags_file = tmp_path / "tags.json"
    assert tags_file.exists() and not list(tmp_path.glob("*.tmp"))
    data = json.loads(tags_file.read_text())
    assert data["format"] == 1
    assert data["rooms"]["kitchen"]["anchor_id"] == 0
    assert sorted(data["rooms"]["kitchen"]["tags"]) == ["0", "5"]
    assert np.allclose(np.array(data["rooms"]["kitchen"]["tags"]["5"]), T_room_tag5)
    assert fake_pnp.multi_calls == 0  # tag 5 was unknown during registration
    # a fresh localizer loads tags.json and localises from tag 5 alone
    loc2 = _localizer(tmp_path, clock)
    assert loc2.status()["tags_registered"] == {"kitchen": [0, 5]}
    T2 = camera_pose_T((1.5, 1.4, 1.0), 60.0)
    fake_pnp.truth = {5: invert_T(T2) @ T_room_tag5}
    pose = loc2.update(_frame(5_000, [_det(5)]))
    assert pose.room == "kitchen" and pose.tag_ids == (5,)
    assert np.allclose(pose.T_room_cam, T2, atol=1e-9)
    # both known tags visible: the joint solve is used when its residual is good ...
    fake_pnp.truth[0] = invert_T(T2) @ T_ROOM_TAG0
    T_joint = camera_pose_T((1.5, 1.4, 1.02), 60.0)
    fake_pnp.multi = (T_joint, 0.3)
    pose = loc2.update(_frame(5_200, [_det(0), _det(5)]))
    assert fake_pnp.multi_calls == 1 and loc2.status()["multi_tag_solves"] == 1
    assert pose.tag_ids == (0, 5)
    assert np.allclose(pose.T_room_cam[:3, 3], (1.5, 1.4, 1.007), atol=1e-6)  # EMA toward T_joint
    # ... and ignored when the residual is poor (falls back to the per-tag mean)
    fake_pnp.multi = (np.eye(4), 9.0)
    pose = loc2.update(_frame(5_400, [_det(0), _det(5)]))
    assert fake_pnp.multi_calls == 2 and loc2.status()["multi_tag_solves"] == 1
    assert np.linalg.norm(pose.T_room_cam[:3, 3] - T2[:3, 3]) < 0.01
    fake_pnp.multi = None
    pose = loc2.update(_frame(5_600, [_det(0), _det(5)]))
    assert loc2.status()["multi_tag_solves"] == 1 and pose.room == "kitchen"
    # to_dict mirrors the file
    assert loc2.to_dict()["rooms"]["kitchen"]["anchor_id"] == 0


def test_localizer_tags_json_robustness(tmp_path: Path) -> None:
    tags_file = tmp_path / "tags.json"
    tags_file.write_text("{not json")
    loc = _localizer(tmp_path, FakeClock())
    assert loc.status()["tags_registered"] == {}
    bogus_anchor = make_T(np.eye(3), (9.0, 9.0, 9.0))
    tags_file.write_text(
        json.dumps(
            {
                "format": 1,
                "rooms": {
                    "kitchen": {
                        "room": "kitchen",
                        "anchor_id": 0,
                        "anchor_height_m": 1.4,
                        "tags": {"0": bogus_anchor.tolist(), "7": np.eye(4).tolist()},
                    },
                    "garage": {"room": "garage", "anchor_id": 3, "tags": {}},
                    "living_room": {"room": "living_room", "tags": "not-a-dict"},
                },
            }
        )
    )
    loc = _localizer(tmp_path, FakeClock())
    st = loc.status()
    assert st["tags_registered"] == {"kitchen": [0, 7]}
    assert "garage" not in st["tags_registered"]
    amap = loc.anchor_map("kitchen")
    assert np.allclose(amap.tags[0], T_ROOM_TAG0)  # rooms.json anchor wins over the file
    assert np.allclose(amap.tags[7], np.eye(4))
    loc.save()
    saved = json.loads(tags_file.read_text())
    assert np.allclose(np.array(saved["rooms"]["kitchen"]["tags"]["0"]), T_ROOM_TAG0)
    # extra_tags from rooms.json are pre-seeded as known tags
    spec = RoomSpec.from_dict(
        {
            "name": "k",
            "anchor_tag": 0,
            "extra_tags": {"8": {"pos_m": [2.0, 1.3, 1.0], "normal_yaw_deg": -90}},
        }
    )
    loc3 = RoomLocalizer(RoomMap([spec]), None, clock=FakeClock())
    assert sorted(loc3.anchor_map("k").tags) == [0, 8]
    assert loc3.status()["tags_path"] is None
    loc3.save()  # no path: no-op
    loc3.load()
    with pytest.raises(KeyError):
        loc3.anchor_map("nowhere")


def test_localizer_set_oracle(tmp_path: Path) -> None:
    clock = FakeClock(10.0)
    loc = RoomLocalizer(
        RoomMap.default(), None, hold=PoseHold(hold_s=10.0), stale_s=60.0, clock=clock
    )
    T = camera_pose_T((0.0, 1.4, 2.5), 0.0)
    loc.set_oracle(
        CameraPose(
            room="kitchen",
            frame="room",
            T_room_cam=T,
            anchored=True,
            anchor_confidence=1.0,
            tag_ids=(0,),
            age_s=0.0,
        )
    )
    p = loc.current()
    assert p.room == "kitchen" and p.frame == "room" and p.anchored
    assert p.anchor_confidence == pytest.approx(1.0) and p.tag_ids == (0,)
    assert loc.to_room(p, (0.5, 0.0, 1.0)) == pytest.approx((0.5, 1.4, 1.5))
    clock.t = 21.0
    p = loc.current()
    assert p.frame == "room" and not p.anchored and p.anchor_confidence == 0.0
    clock.t = 71.0
    assert loc.current().frame == "camera"
    loc.set_oracle(
        CameraPose(
            room=None,
            frame="camera",
            T_room_cam=None,
            anchored=False,
            anchor_confidence=0.0,
            tag_ids=(),
            age_s=math.inf,
        )
    )
    p = loc.current()
    assert p.frame == "camera" and p.room is None and p.T_room_cam is None
    assert loc.to_room(p, (0.0, 0.0, 1.0)) is None
    # an explicit age is honoured
    loc.set_oracle(
        CameraPose(
            room="living_room",
            frame="room",
            T_room_cam=T,
            anchored=False,
            anchor_confidence=0.0,
            tag_ids=(1,),
            age_s=30.0,
        ),
        now_s=100.0,
    )
    p = loc.current(now_s=100.0)
    assert p.room == "living_room" and p.frame == "room" and not p.anchored
    assert p.age_s == pytest.approx(30.0)
    assert loc.current(now_s=135.0).frame == "camera"
    assert loc.status()["room"] == "living_room"


def test_localizer_with_detector_on_gray(tmp_path: Path, fake_pnp: FakePnP) -> None:
    class FakeDetector:
        name = "fake"

        def __init__(self, dets: list[TagDetection]) -> None:
            self.dets = dets
            self.calls = 0
            self.fail = False

        def detect(self, gray: np.ndarray) -> list[TagDetection]:
            self.calls += 1
            if self.fail:
                raise RuntimeError("boom")
            assert gray.shape == (H, W)
            return self.dets

    det = FakeDetector([_det(0)])
    loc = RoomLocalizer(RoomMap.default(), det, clock=FakeClock(), upright_fix=False)
    assert loc.detector is det and loc.rooms.names() == ["kitchen", "living_room"]
    assert loc.tag_size_m == pytest.approx(TAG)
    T_true = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    fake_pnp.truth = {0: invert_T(T_true) @ T_ROOM_TAG0}
    gray = np.zeros((H, W), np.uint8)
    pose = loc.update(_frame(0, None, gray=gray))
    assert pose.anchored and det.calls == 1
    assert loc.status()["detector"] == "fake"
    # no gray and no pre-detected tags: nothing to do
    loc.update(_frame(100, None))
    assert det.calls == 1
    # pre-detected tags win over the detector
    loc.update(_frame(200, [_det(0)], gray=gray))
    assert det.calls == 1
    # a crashing detector is contained
    det.fail = True
    pose = loc.update(_frame(300, None, gray=gray))
    assert pose.room == "kitchen" and det.calls == 2
    assert loc.status()["tag_frames"] == 4


def test_localizer_upright_fix(
    tmp_path: Path, fake_pnp: FakePnP, monkeypatch: pytest.MonkeyPatch
) -> None:
    T_true = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    T_cam_tag = invert_T(T_true) @ T_ROOM_TAG0
    fake_pnp.truth = {
        0: T_cam_tag @ make_T(rot_z(math.pi), (0.0, 0.0, 0.0))
    }  # tag frame upside down
    monkeypatch.delenv("LOCI_TAG_UPRIGHT_FIX", raising=False)
    loc_off = RoomLocalizer(RoomMap.default(), None, clock=FakeClock())
    pose = loc_off.update(_frame(0, [_det(0)]))
    assert rotation_angle_deg(pose.T_room_cam[:3, :3], T_true[:3, :3]) > 170.0
    monkeypatch.setenv("LOCI_TAG_UPRIGHT_FIX", "1")
    loc_on = RoomLocalizer(RoomMap.default(), None, clock=FakeClock())
    pose = loc_on.update(_frame(0, [_det(0)]))
    assert np.allclose(pose.T_room_cam, T_true, atol=1e-9)
    loc_kw = RoomLocalizer(RoomMap.default(), None, clock=FakeClock(), upright_fix=False)
    pose = loc_kw.update(_frame(0, [_det(0)]))
    assert rotation_angle_deg(pose.T_room_cam[:3, :3], T_true[:3, :3]) > 170.0


def test_localizer_threads_smoke(tmp_path: Path, fake_pnp: FakePnP) -> None:
    clock = FakeClock(0.0)
    loc = _localizer(tmp_path, clock)
    T_true = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    fake_pnp.truth = {0: invert_T(T_true) @ T_ROOM_TAG0}
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(300):
                loc.update(_frame(i, [_det(0)]))
        except BaseException as exc:  # noqa: BLE001 - surfaced through `errors`
            errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    for _ in range(300):
        p = loc.current()
        assert p.frame in ("room", "camera")
        json.dumps(loc.status())
    t.join(5.0)
    assert not t.is_alive() and errors == []
    assert loc.status()["tag_frames"] == 300


def test_localizer_real_pnp_with_projected_corners(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    clock = FakeClock(0.0)
    loc = _localizer(tmp_path, clock)
    T_true = camera_pose_T((0.3, 1.4, 2.5), -10.0)
    corners = _corners(T_true, T_ROOM_TAG0)
    assert TagFilter().inside_image(corners, W, H)
    pose = loc.update(_frame(0, [_det(0, corners=corners)]))
    assert pose.room == "kitchen" and pose.anchored and pose.tag_ids == (0,)
    assert np.linalg.norm(pose.T_room_cam[:3, 3] - T_true[:3, 3]) < 0.01
    assert rotation_angle_deg(pose.T_room_cam[:3, :3], T_true[:3, :3]) < 0.5
    p_room = np.array([1.0, 0.8, 1.0])
    xyz_cam = apply_T(invert_T(T_true), p_room)
    est = loc.to_room(pose, xyz_cam)
    assert np.linalg.norm(np.array(est) - p_room) < 0.05
    rng = np.random.default_rng(0)
    clock.t = 0.2
    pose = loc.update(_frame(200, [_det(0, corners=corners + rng.normal(scale=0.3, size=(4, 2)))]))
    est = loc.to_room(pose, xyz_cam)
    assert np.linalg.norm(np.array(est) - p_room) < 0.05
    # a second tag on the anchor wall registers over 10 frames with the real solver ...
    T_room_tag5 = make_T(np.eye(3), (1.5, 1.4, 0.0))
    T_cam = camera_pose_T((0.75, 1.4, 2.5), 0.0)
    c0, c5 = _corners(T_cam, T_ROOM_TAG0), _corners(T_cam, T_room_tag5)
    for i in range(10):
        clock.t = 1.0 + 0.2 * i
        loc.update(_frame(1_000 + 200 * i, [_det(0, corners=c0), _det(5, corners=c5)]))
    assert loc.status()["tags_registered"] == {"kitchen": [0, 5]}
    T5 = loc.anchor_map("kitchen").tags[5]
    assert np.linalg.norm(T5[:3, 3] - T_room_tag5[:3, 3]) < 0.02
    assert rotation_angle_deg(T5[:3, :3], T_room_tag5[:3, :3]) < 1.0
    # ... after which both tags feed the joint solve
    clock.t = 4.0
    pose = loc.update(_frame(4_000, [_det(0, corners=c0), _det(5, corners=c5)]))
    assert loc.status()["multi_tag_solves"] >= 1 and pose.tag_ids == (0, 5)
    assert np.linalg.norm(pose.T_room_cam[:3, 3] - T_cam[:3, 3]) < 0.02
