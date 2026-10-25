"""numpy-only tests for demo_spatial/app/geometry.py (SE(3) helpers, room chain, projection).

The cv2-dependent PnP functions are covered by tests/test_tag_pose.py.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.geometry import (
    FRONTAL_R_CAM_TAG,
    apply_T,
    camera_pose_in_room,
    camera_pose_T,
    invert_T,
    log_rot,
    make_T,
    mean_T,
    normalize_upright,
    project_points,
    rodrigues,
    rot_x,
    rot_y,
    rot_z,
    rotation_angle_deg,
    so3_mean,
    tag_object_points,
    yaw_deg_of,
)

K_1280 = np.array([[918.3, 0.0, 645.8], [0.0, 917.5, 417.4], [0.0, 0.0, 1.0]])


def _random_T(rng: np.random.Generator) -> np.ndarray:
    return make_T(rodrigues(rng.normal(size=3)), rng.normal(size=3) * 2.0)


# ---------------------------------------------------------------------------
# SE(3) basics
# ---------------------------------------------------------------------------


def test_invert_and_apply_identities() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = _random_T(rng)
        assert np.allclose(invert_T(T) @ T, np.eye(4), atol=1e-12)
        assert np.allclose(T @ invert_T(T), np.eye(4), atol=1e-12)
        p = rng.normal(size=3)
        assert np.allclose(apply_T(invert_T(T), apply_T(T, p)), p, atol=1e-12)
        pts = rng.normal(size=(7, 3))
        many = apply_T(T, pts)
        assert many.shape == (7, 3)
        assert np.allclose(many[3], apply_T(T, pts[3]))


def test_make_T_accepts_lists_and_rejects_nothing_silently() -> None:
    T = make_T(np.eye(3), [1, 2, 3])
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_rodrigues_log_rot_round_trip() -> None:
    rng = np.random.default_rng(2)
    for _ in range(50):
        r = rng.normal(size=3)
        r = r / np.linalg.norm(r) * rng.uniform(0.01, 3.1)  # angle < pi: the log is unique
        assert np.allclose(log_rot(rodrigues(r)), r, atol=1e-9)
    assert np.allclose(rodrigues([0, 0, 0]), np.eye(3))
    assert np.allclose(log_rot(np.eye(3)), 0.0)


def test_log_rot_near_pi() -> None:
    r = log_rot(rot_x(math.pi))
    assert abs(np.linalg.norm(r) - math.pi) < 1e-6
    assert abs(abs(r[0]) - math.pi) < 1e-6
    r = log_rot(rot_y(math.pi - 1e-7))
    assert abs(np.linalg.norm(r) - math.pi) < 1e-5


def test_rot_axes_match_cv_conventions() -> None:
    assert np.allclose(rot_z(math.pi / 2) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rot_x(math.pi / 2) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(rot_y(math.pi / 2) @ [0, 0, 1], [1, 0, 0])


def test_rotation_angle_deg() -> None:
    assert abs(rotation_angle_deg(np.eye(3), rot_z(math.radians(37))) - 37.0) < 1e-9
    assert abs(rotation_angle_deg(rot_y(0.3), rot_y(0.3))) < 1e-9
    assert abs(rotation_angle_deg(np.eye(3), rot_x(math.pi)) - 180.0) < 1e-6


def test_so3_mean_and_mean_T() -> None:
    assert np.allclose(so3_mean([rot_y(0.3)] * 5), rot_y(0.3))
    # symmetric pair averages to the identity
    assert np.allclose(so3_mean([rot_y(0.2), rot_y(-0.2)]), np.eye(3), atol=1e-12)
    Ta = make_T(np.eye(3), (0, 0, 0))
    Tb = make_T(np.eye(3), (2, 4, 6))
    Tm = mean_T([Ta, Tb])
    assert np.allclose(Tm[:3, 3], (1, 2, 3))
    assert np.allclose(Tm[:3, :3], np.eye(3))
    assert abs(np.linalg.det(so3_mean([rot_z(0.1), rot_x(0.2)])) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Heading / camera pose helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaw", [-170.0, -90.0, -45.0, 0.0, 12.5, 90.0, 179.0])
def test_camera_pose_T_yaw_round_trip(yaw: float) -> None:
    T = camera_pose_T((1.0, 1.4, 2.0), yaw)
    assert np.allclose(T[:3, 3], (1.0, 1.4, 2.0))
    assert abs(yaw_deg_of(T) - yaw) < 1e-9
    # rigid
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-12


def test_heading_definition() -> None:
    # yaw 0: the optical axis (camera +z) looks toward -z (at the anchor wall); camera y points down
    T0 = camera_pose_T((0, 1.4, 2.5), 0.0)
    assert np.allclose(T0[:3, 2], (0, 0, -1))
    assert np.allclose(T0[:3, 1], (0, -1, 0))
    assert np.allclose(T0[:3, 0], (1, 0, 0))
    assert np.allclose(T0[:3, :3], FRONTAL_R_CAM_TAG)
    # +90: looking toward +x
    T90 = camera_pose_T((0, 1.4, 2.5), 90.0)
    assert np.allclose(T90[:3, 2], (1, 0, 0), atol=1e-12)
    assert abs(yaw_deg_of(T90) - 90.0) < 1e-9
    # -90: looking toward -x
    Tm90 = camera_pose_T((0, 1.4, 2.5), -90.0)
    assert np.allclose(Tm90[:3, 2], (-1, 0, 0), atol=1e-12)
    # a pitched camera keeps its heading
    Tp = T0.copy()
    Tp[:3, :3] = T0[:3, :3] @ rot_x(math.radians(15))
    assert abs(yaw_deg_of(Tp)) < 1e-9


def test_room_chain_with_known_camera_pose() -> None:
    T = camera_pose_T((0.0, 1.4, 2.5), 0.0)
    # a point 1 m in front of the camera lands 1 m closer to the anchor wall
    assert np.allclose(apply_T(T, (0.0, 0.0, 1.0)), (0.0, 1.4, 1.5))
    # camera +x is room +x: the viewer's right when facing the tag
    assert np.allclose(apply_T(T, (0.5, 0.0, 1.0)), (0.5, 1.4, 1.5))
    # camera -y (above the lens) is higher in the room
    assert np.allclose(apply_T(T, (0.0, -0.5, 1.0)), (0.0, 1.9, 1.5))
    # and the floor directly below the camera is at y = 0
    assert np.allclose(apply_T(T, (0.0, 1.4, 0.0)), (0.0, 0.0, 2.5))


def test_tag_chain_pure_numpy() -> None:
    """Design self-check 2: tag pose -> camera pose in room -> camera point to room metres."""
    R_cam_tag = FRONTAL_R_CAM_TAG @ rot_y(math.radians(-20)) @ rot_x(math.radians(8))
    t = np.array([-0.4, 0.2, 2.5])
    T_cam_tag = make_T(R_cam_tag, t)
    h = 1.4
    T_room_tag = make_T(np.eye(3), (0, h, 0))
    T_room_cam = camera_pose_in_room(T_room_tag, T_cam_tag)
    cam_pos = T_room_cam[:3, 3]
    assert abs(np.linalg.norm(t) - np.linalg.norm(cam_pos - [0, h, 0])) < 1e-9
    assert cam_pos[2] > 2.0  # the camera stands inside the room, in front of the wall
    # an object 0.5 m right of the tag, 0.8 m below its centre, 1.2 m into the room
    p_tag = np.array([0.5, -0.8, 1.2])
    p_cam = apply_T(T_cam_tag, p_tag)
    assert p_cam[2] > 0
    p_room = apply_T(T_room_cam, p_cam)
    assert np.allclose(p_room, p_tag + [0, h, 0], atol=1e-9)
    # yaw_deg_of agrees with the construction: the camera was yawed 20 deg about the tag's y
    # (the 8 deg pitch applied afterwards in the camera frame shifts the projected heading
    # by ~0.2 deg, hence the tolerance)
    assert abs(abs(yaw_deg_of(T_room_cam)) - 20.0) < 0.5


# ---------------------------------------------------------------------------
# Tag geometry / projection
# ---------------------------------------------------------------------------


def test_tag_object_points_order_and_frontal_projection() -> None:
    pts = tag_object_points(0.16)
    assert pts.shape == (4, 3)
    assert np.allclose(
        pts, [[-0.08, 0.08, 0], [0.08, 0.08, 0], [0.08, -0.08, 0], [-0.08, -0.08, 0]]
    )
    T_cam_tag = make_T(FRONTAL_R_CAM_TAG, (0.0, 0.0, 2.0))
    uv = project_points(pts, T_cam_tag, K_1280)
    cx, cy = K_1280[0, 2], K_1280[1, 2]
    # TL, TR, BR, BL in image coordinates (x right, y down)
    assert uv[0, 0] < cx and uv[0, 1] < cy
    assert uv[1, 0] > cx and uv[1, 1] < cy
    assert uv[2, 0] > cx and uv[2, 1] > cy
    assert uv[3, 0] < cx and uv[3, 1] > cy
    side = np.linalg.norm(uv[0] - uv[1])
    assert abs(side - K_1280[0, 0] * 0.16 / 2.0) < 1e-6


def test_project_points_pinhole_values() -> None:
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    uv = project_points(np.array([[0.5, 0.25, 2.0]]), np.eye(4), K)
    assert np.allclose(uv, [[890.0, 525.0]])
    # frame change is applied before projection
    T_cam_x = make_T(np.eye(3), (0.0, 0.0, 2.0))
    uv2 = project_points(np.array([[0.5, 0.25, 0.0]]), T_cam_x, K)
    assert np.allclose(uv2, [[890.0, 525.0]])
    # a point on the camera plane has no projection
    uv3 = project_points(np.array([[0.1, 0.1, 0.0]]), np.eye(4), K)
    assert np.all(np.isnan(uv3))


def test_normalize_upright_numpy() -> None:
    T = make_T(
        FRONTAL_R_CAM_TAG @ rot_y(math.radians(-20)) @ rot_x(math.radians(8)), (-0.4, 0.2, 2.5)
    )
    same, flipped = normalize_upright(T)
    assert not flipped
    assert same is T
    upside_down = T @ make_T(rot_z(math.pi), (0, 0, 0))
    fixed, flipped = normalize_upright(upside_down)
    assert flipped
    assert np.allclose(fixed, T, atol=1e-12)
