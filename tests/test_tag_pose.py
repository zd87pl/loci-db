"""cv2-dependent tests for the robust tag PnP in demo_spatial/app/geometry.py.

Skipped entirely when OpenCV is not importable (the Mac test runner, CI); run them with the
3.11 scratch venv that has opencv-python-headless (and optionally pyapriltags).
"""

from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

import app.geometry as geometry
from app.geometry import (
    FRONTAL_R_CAM_TAG,
    TagPose,
    camera_pose_in_room,
    camera_pose_T,
    invert_T,
    make_T,
    normalize_upright,
    rot_x,
    rot_y,
    rot_z,
    rotation_angle_deg,
    solve_room_pose_multi,
    solve_tag_pose,
    tag_object_points,
)

cv2 = pytest.importorskip("cv2")

TAG = 0.16
# OAK-D Lite CAM_A intrinsics at 1280x800 and 640x400 (device calibration, see hardware facts)
K_1280 = np.array([[918.3, 0.0, 645.8], [0.0, 917.5, 417.4], [0.0, 0.0, 1.0]])
K_640 = np.array([[459.2, 0.0, 322.9], [0.0, 458.7, 208.7], [0.0, 0.0, 1.0]])
DIST_MILD = np.array([-0.05, 0.01, 0.0005, -0.0003, 0.0])
# CAM_A rational model (k1 k2 p1 p2 k3 k4 k5 k6) from the device calibration
DIST_OAK = np.array([-0.7791, -9.3857, 0.0006, 0.0004, 31.35, -0.9325, -8.7559, 30.4295])


def _project(T_cam_tag: np.ndarray, K: np.ndarray, dist: np.ndarray | None) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(T_cam_tag[:3, :3])
    img, _ = cv2.projectPoints(tag_object_points(TAG), rvec, T_cam_tag[:3, 3], K, dist)
    return img.reshape(4, 2)


def _inside(c: np.ndarray, K: np.ndarray) -> bool:
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    return bool(c.min() >= 0 and c[:, 0].max() < w and c[:, 1].max() < h)


def _pose_error(tp: TagPose, T_true: np.ndarray) -> tuple[float, float]:
    dt = float(np.linalg.norm(tp.T_cam_tag[:3, 3] - T_true[:3, 3]))
    dr = rotation_angle_deg(tp.T_cam_tag[:3, :3], T_true[:3, :3])
    return dt, dr


# --------------------------------------------------------------------------- tag rendering
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _marker_canvas(tag_id: int, cell_px: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """10x10-cell canvas: quiet zone + 8x8-cell 36h11 marker -> (canvas, square TL,TR,BR,BL)."""
    marker = cv2.aruco.generateImageMarker(_ARUCO_DICT, tag_id, 8 * cell_px)
    canvas = np.full((10 * cell_px, 10 * cell_px), 255, np.uint8)
    canvas[cell_px : 9 * cell_px, cell_px : 9 * cell_px] = marker
    a, b = cell_px - 0.5, 9 * cell_px - 0.5
    src = np.array([[a, a], [b, a], [b, b], [a, b]], dtype=np.float32)
    return canvas, src


def _render_tag(
    width: int, height: int, K: np.ndarray, T_cam_tag: np.ndarray, tag_id: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Render one tag at a known pose (no lens distortion). Returns (gray frame, true corners)."""
    frame = np.full((height, width), 150, np.uint8)
    corners = _project(T_cam_tag, K, None)
    canvas, src = _marker_canvas(tag_id)
    M = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    warped = cv2.warpPerspective(canvas, M, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(
        np.full(canvas.shape, 255, np.uint8), M, (width, height), flags=cv2.INTER_NEAREST
    )
    frame = np.where(mask > 127, warped, frame).astype(np.uint8)
    return cv2.GaussianBlur(frame, (0, 0), 0.8), corners


# ---------------------------------------------------------------------------
# solve_tag_pose
# ---------------------------------------------------------------------------


def test_frontal_level_tag_is_exact() -> None:
    """A level camera looking straight at a wall tag: the case where bare IPPE_SQUARE fails."""
    T_true = make_T(FRONTAL_R_CAM_TAG, (0.0, 0.0, 2.0))
    tp = solve_tag_pose(3, _project(T_true, K_1280, None), K_1280, None, TAG)
    assert tp.tag_id == 3
    assert np.allclose(tp.T_cam_tag[:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-6)
    assert np.allclose(tp.T_cam_tag[:3, 3], (0.0, 0.0, 2.0), atol=1e-6)
    assert tp.reproj_err_px < 1e-4
    assert tp.view_angle_deg < 1e-3
    # mean of the two horizontal (fx) and two vertical (fy) edges
    assert abs(tp.side_px - (K_1280[0, 0] + K_1280[1, 1]) / 2.0 * TAG / 2.0) < 0.01
    assert abs(tp.range_m - 2.0) < 1e-6
    assert np.allclose(tp.T_tag_cam @ tp.T_cam_tag, np.eye(4), atol=1e-9)


@pytest.mark.parametrize("K", [K_1280, K_640], ids=["1280x800", "640x400"])
def test_exact_solve_with_distortion(K: np.ndarray) -> None:
    R = FRONTAL_R_CAM_TAG @ rot_y(math.radians(-20)) @ rot_x(math.radians(8))
    T_true = make_T(R, (-0.4, 0.2, 2.5))
    tp = solve_tag_pose(7, _project(T_true, K, DIST_MILD), K, DIST_MILD, TAG)
    dt, dr = _pose_error(tp, T_true)
    assert dt < 1e-6 and dr < 1e-5
    assert tp.reproj_err_px < 1e-6
    assert tp.ambiguity_ratio > 1.5
    # the view angle combines the 20 deg yaw / 8 deg pitch with the 0.4 m off-axis offset
    assert 20.0 < tp.view_angle_deg < 45.0


def test_exact_solve_with_oak_rational_distortion() -> None:
    R = FRONTAL_R_CAM_TAG @ rot_y(math.radians(15)) @ rot_x(math.radians(-6))
    T_true = make_T(R, (0.3, -0.1, 1.8))
    c = _project(T_true, K_1280, DIST_OAK)
    assert _inside(c, K_1280)
    tp = solve_tag_pose(0, c, K_1280, DIST_OAK, TAG)
    dt, dr = _pose_error(tp, T_true)
    assert dt < 5e-3 and dr < 0.3


def test_exact_solve_random_poses() -> None:
    rng = np.random.default_rng(7)
    solved = 0
    for _ in range(80):
        R = (
            FRONTAL_R_CAM_TAG
            @ rot_y(math.radians(rng.uniform(-50, 50)))
            @ rot_x(math.radians(rng.uniform(-40, 40)))
            @ rot_z(math.radians(rng.uniform(-30, 30)))
        )
        t = (rng.uniform(-1.0, 1.0), rng.uniform(-0.7, 0.7), rng.uniform(0.6, 4.0))
        T_true = make_T(R, t)
        c = _project(T_true, K_1280, DIST_MILD)
        if not _inside(c, K_1280):
            continue
        tp = solve_tag_pose(1, c, K_1280, DIST_MILD, TAG)
        dt, dr = _pose_error(tp, T_true)
        assert dt < 1e-5, (t, dt)
        assert dr < 1e-3, (t, dr)
        solved += 1
    assert solved >= 40


@pytest.mark.parametrize("yaw_deg", [0.0, -10.0, 10.0, -30.0, 30.0])
@pytest.mark.parametrize("x", [-0.8, -0.3, 0.0, 0.3, 0.8])
@pytest.mark.parametrize("z", [1.0, 2.0, 3.0])
def test_pitch_zero_wall_tag_cases(yaw_deg: float, x: float, z: float) -> None:
    """The degenerate manifold (camera pitch exactly 0) must solve exactly, with/without dist."""
    T_true = make_T(FRONTAL_R_CAM_TAG @ rot_y(math.radians(yaw_deg)), (x, 0.0, z))
    for dist in (None, DIST_MILD):
        c = _project(T_true, K_1280, dist)
        if not _inside(c, K_1280):
            pytest.skip("tag outside the image for this pose")
        tp = solve_tag_pose(2, c, K_1280, dist, TAG)
        dt, dr = _pose_error(tp, T_true)
        assert dt < 1e-5 and dr < 1e-3, (yaw_deg, x, z, dist is None, dt, dr)
        assert tp.T_cam_tag[2, 3] > 0


def test_noisy_corners_keep_pose_close_and_ratio_informative() -> None:
    rng = np.random.default_rng(3)
    T_true = make_T(FRONTAL_R_CAM_TAG @ rot_y(math.radians(25)), (0.4, 0.1, 2.0))
    c = _project(T_true, K_1280, None) + rng.normal(scale=0.3, size=(4, 2))
    tp = solve_tag_pose(4, c, K_1280, None, TAG)
    dt, dr = _pose_error(tp, T_true)
    assert dt < 0.05 and dr < 3.0
    assert 0.0 < tp.reproj_err_px < 1.0
    assert tp.ambiguity_ratio >= 1.0


def test_missing_cv2_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match="cv2"):
        solve_tag_pose(0, np.zeros((4, 2)), K_1280, None, TAG)
    with pytest.raises(ImportError, match="cv2"):
        solve_room_pose_multi(
            [(0, np.zeros((4, 2))), (1, np.zeros((4, 2)))],
            {0: np.eye(4), 1: np.eye(4)},
            K_1280,
            None,
            TAG,
        )


# ---------------------------------------------------------------------------
# pyapriltags corner remap
# ---------------------------------------------------------------------------


def test_pyapriltags_corner_remap_matches_ippe_order() -> None:
    pyapriltags = pytest.importorskip("pyapriltags")
    R = FRONTAL_R_CAM_TAG @ rot_y(math.radians(-20)) @ rot_x(math.radians(8))
    T_true = make_T(R, (-0.4, 0.2, 2.5))
    det = pyapriltags.Detector(families="tag36h11", nthreads=1, quad_decimate=1.0)
    checked = 0
    for tag_id in (0, 1, 7, 42):
        frame, true_corners = _render_tag(1280, 800, K_1280, T_true, tag_id=tag_id)
        found = [d for d in det.detect(frame) if d.tag_id == tag_id]
        if not found:
            continue
        raw = np.asarray(found[0].corners, dtype=np.float64)
        # raw order is TR, TL, BL, BR of the cv2.aruco-generated tag;
        # [1, 0, 3, 2] gives TL, TR, BR, BL
        remapped = raw[[1, 0, 3, 2]]
        assert np.abs(remapped - true_corners).max() < 1.5, tag_id
        assert np.abs(raw - true_corners).max() > 10.0  # the raw order really is different
        tp = solve_tag_pose(tag_id, remapped, K_1280, None, TAG)
        dt, dr = _pose_error(tp, T_true)
        assert dt < 0.04 and dr < 2.5, (tag_id, dt, dr)
        checked += 1
    assert checked >= 2


def test_aruco_detector_returns_ippe_order() -> None:
    R = FRONTAL_R_CAM_TAG @ rot_y(math.radians(-20)) @ rot_x(math.radians(8))
    T_true = make_T(R, (-0.4, 0.2, 2.5))
    frame, true_corners = _render_tag(1280, 800, K_1280, T_true, tag_id=7)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(_ARUCO_DICT, params)
    corners, ids, _ = det.detectMarkers(frame)
    assert ids is not None and 7 in ids.flatten()
    c = corners[list(ids.flatten()).index(7)].reshape(4, 2)
    assert np.abs(c - true_corners).max() < 1.5
    tp = solve_tag_pose(7, c, K_1280, None, TAG)
    dt, dr = _pose_error(tp, T_true)
    assert dt < 0.04 and dr < 2.5


# ---------------------------------------------------------------------------
# multi-tag solve
# ---------------------------------------------------------------------------


def test_solve_room_pose_multi_needs_two_known_tags() -> None:
    tag_map = {1: make_T(np.eye(3), (0, 1.4, 0))}
    c = _project(make_T(FRONTAL_R_CAM_TAG, (0, 0, 2.0)), K_1280, None)
    assert solve_room_pose_multi([(1, c), (9, c)], tag_map, K_1280, None, TAG) is None
    assert solve_room_pose_multi([], tag_map, K_1280, None, TAG) is None


@pytest.mark.parametrize(
    "layout,T2",
    [
        ("same_wall", make_T(np.eye(3), (1.5, 1.4, 0.0))),
        ("adjacent_wall", make_T(rot_y(math.radians(-90)), (3.0, 1.3, 0.6))),
    ],
)
def test_multi_tag_pnp_exact_and_beats_single_tag(layout: str, T2: np.ndarray) -> None:
    rng = np.random.default_rng(11)
    T_room_tag1 = make_T(np.eye(3), (0.0, 1.4, 0.0))
    tag_map = {1: T_room_tag1, 2: T2}
    single_err: list[float] = []
    multi_err: list[float] = []
    exact_checked = False
    for _ in range(200):
        cam_pos = np.array([rng.uniform(0.6, 1.4), 1.3, rng.uniform(2.2, 3.2)])
        mid = 0.5 * (T_room_tag1[:3, 3] + T2[:3, 3])
        fwd = mid - cam_pos
        fwd /= np.linalg.norm(fwd)
        # heading toward the mid point (+-4 deg) in the CameraPose convention
        yaw = math.degrees(math.atan2(fwd[0], -fwd[2])) + rng.uniform(-4, 4)
        Tc = camera_pose_T(cam_pos, yaw)
        exact: list[tuple[int, np.ndarray]] = []
        noisy: list[tuple[int, np.ndarray]] = []
        inside = True
        for tid, Trt in tag_map.items():
            Tct = invert_T(Tc) @ Trt
            img = _project(Tct, K_1280, None)
            if Tct[2, 3] < 0.3 or not _inside(img, K_1280):
                inside = False
            exact.append((tid, img))
            noisy.append((tid, img + rng.normal(scale=0.5, size=(4, 2))))
        if not inside:
            continue
        if not exact_checked:
            res = solve_room_pose_multi(exact, tag_map, K_1280, None, TAG)
            assert res is not None
            T_est, err = res
            assert np.allclose(T_est, Tc, atol=1e-5), layout
            assert err < 1e-4
            exact_checked = True
        tp = solve_tag_pose(1, noisy[0][1], K_1280, None, TAG)
        single_err.append(
            float(np.linalg.norm(camera_pose_in_room(T_room_tag1, tp.T_cam_tag)[:3, 3] - Tc[:3, 3]))
        )
        res = solve_room_pose_multi(noisy, tag_map, K_1280, None, TAG)
        assert res is not None
        multi_err.append(float(np.linalg.norm(res[0][:3, 3] - Tc[:3, 3])))
    assert exact_checked
    assert len(multi_err) > 30
    assert float(np.median(multi_err)) < float(np.median(single_err)), layout
    assert float(np.median(multi_err)) < 0.10


# ---------------------------------------------------------------------------
# normalize_upright and the IPPE guard
# ---------------------------------------------------------------------------


def test_normalize_upright_on_solved_poses() -> None:
    R = FRONTAL_R_CAM_TAG @ rot_y(math.radians(-20)) @ rot_x(math.radians(8))
    T_true = make_T(R, (-0.4, 0.2, 2.5))
    tp = solve_tag_pose(7, _project(T_true, K_1280, None), K_1280, None, TAG)
    _, flipped = normalize_upright(tp.T_cam_tag)
    assert not flipped
    # corners of a tag printed with the other canonical orientation arrive rotated by 180 deg
    rotated = np.roll(_project(T_true, K_1280, None), 2, axis=0)
    tp_rot = solve_tag_pose(7, rotated, K_1280, None, TAG)
    assert tp_rot.T_cam_tag[1, 1] > 0  # tag y now points down in the image
    fixed, flipped = normalize_upright(tp_rot.T_cam_tag)
    assert flipped
    assert np.allclose(fixed, T_true, atol=1e-5)


def test_ippe_square_is_used_only_inside_solve_tag_pose() -> None:
    """Source-level guard: bare SOLVEPNP_IPPE_SQUARE is unsafe on level wall tags (design D6)."""
    src = Path(geometry.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solve_tag_pose"
    )
    uses = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "SOLVEPNP_IPPE_SQUARE"
    ]
    assert uses, "solve_tag_pose must still try the IPPE_SQUARE candidates"
    for n in uses:
        assert func.lineno <= n.lineno <= func.end_lineno, (
            f"IPPE_SQUARE used outside solve_tag_pose (line {n.lineno})"
        )
    # and it is always accompanied by the ITERATIVE candidate inside that function
    flags = {
        n.attr
        for n in ast.walk(func)
        if isinstance(n, ast.Attribute) and n.attr.startswith("SOLVEPNP_")
    }
    assert "SOLVEPNP_ITERATIVE" in flags
    # the only other PnP flag in the module is SQPNP (multi-tag solve)
    all_flags = {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr.startswith("SOLVEPNP_")
    }
    assert all_flags == {"SOLVEPNP_IPPE_SQUARE", "SOLVEPNP_ITERATIVE", "SOLVEPNP_SQPNP"}
    # and solvePnP is never called with IPPE (non-square) either
    assert "SOLVEPNP_IPPE\n" not in src and "SOLVEPNP_IPPE)" not in src
