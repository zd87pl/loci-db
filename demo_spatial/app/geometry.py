"""SE(3) helpers and the robust AprilTag PnP used by the handheld room anchoring.

Only numpy is imported at module level; ``cv2`` is imported lazily inside the two PnP entry
points (:func:`solve_tag_pose`, :func:`solve_room_pose_multi`) so that this module imports on
hosts without OpenCV (CI, the Mac test runner).

The maths is a faithful port of the validated design recipe (``apriltag_pose.py``, 81
self-checks on the Mac and the Pi).  Frame conventions (``docs/HANDHELD_DEMO_DESIGN.md``
section 4.1):

* ``cam``  : OpenCV camera frame, metres: +x right, +y down, +z forward (optical axis).
* ``tag``  : +x right, +y up, +z out of the printed face toward the viewer, origin at the tag
  centre.  The corners TL, TR, BR, BL correspond to the ``SOLVEPNP_IPPE_SQUARE`` object order
  ``(-s,+s), (+s,+s), (+s,-s), (-s,-s)``; ``cv2.aruco`` returns that order, pyapriltags
  corners must be remapped ``[1, 0, 3, 2]`` first.  A frontal tag has
  ``R_cam_tag = diag(1, -1, -1)``.
* ``room`` : the anchor tag frame shifted down by the mounting height so that y = 0 is the
  floor; y up, z from the anchor wall into the room, x to the viewer's RIGHT when facing the
  tag (the walker's left when entering with their back to the tag).

``T_a_b`` is a 4x4 homogeneous matrix such that ``p_a = T_a_b @ p_b``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

Array = np.ndarray

#: Rotation of a tag seen head-on by an upright camera (tag y up / z toward the viewer vs
#: camera y down / z forward).
FRONTAL_R_CAM_TAG = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------- basic SE(3)
def tag_object_points(tag_size_m: float) -> Array:
    """Tag-frame corner coordinates (4, 3) in the ``SOLVEPNP_IPPE_SQUARE`` order TL, TR, BR, BL."""
    s = float(tag_size_m) / 2.0
    return np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)


def make_T(R: Array, t: Any) -> Array:
    """Build a 4x4 rigid transform from a 3x3 rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: Array) -> Array:
    """Inverse of a rigid transform: ``[R t]^-1 = [R^T  -R^T t]``."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_T(T: Array, p: Any) -> Array:
    """Apply ``T`` to one point ``(3,)`` or many points ``(N, 3)``."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return p @ T[:3, :3].T + T[:3, 3]


def rot_x(a: float) -> Array:
    """Rotation about +x by ``a`` radians."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> Array:
    """Rotation about +y by ``a`` radians."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> Array:
    """Rotation about +z by ``a`` radians."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rodrigues(rvec: Any) -> Array:
    """Rotation vector -> rotation matrix (same as ``cv2.Rodrigues``, numpy only)."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def log_rot(R: Array) -> Array:
    """Rotation matrix -> rotation vector (inverse of :func:`rodrigues`)."""
    cos_th = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(cos_th)
    if th < 1e-9:
        return np.zeros(3)
    if abs(math.pi - th) < 1e-6:  # near pi: take the axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        # fix signs using the off-diagonal terms
        if A[0, 1] < 0:
            axis[1] = -axis[1]
        if A[0, 2] < 0:
            axis[2] = -axis[2]
        return axis / np.linalg.norm(axis) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * math.sin(th)) * th


def rotation_angle_deg(R1: Array, R2: Array) -> float:
    """Geodesic angle between two rotations in degrees."""
    return math.degrees(float(np.linalg.norm(log_rot(R1.T @ R2))))


def so3_mean(Rs: list[Array]) -> Array:
    """Chordal (projection) mean of rotation matrices."""
    M = np.sum(np.stack(Rs), axis=0)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def mean_T(Ts: list[Array]) -> Array:
    """Mean rigid transform: chordal rotation mean + arithmetic translation mean."""
    return make_T(so3_mean([T[:3, :3] for T in Ts]), np.mean([T[:3, 3] for T in Ts], axis=0))


# ------------------------------------------------------------------- room / camera helpers
def camera_pose_in_room(T_room_tag: Array, T_cam_tag: Array) -> Array:
    """``T_room_cam = T_room_tag @ T_tag_cam``."""
    return T_room_tag @ invert_T(T_cam_tag)


def yaw_deg_of(T_room_cam: Array) -> float:
    """Heading of the camera's optical axis in the room x-z plane, in degrees.

    0 = looking toward -z (at the anchor wall), +90 = looking toward +x, -90 toward -x and
    +-180 = looking away from the anchor wall.  This is the ``heading_deg`` of
    ``CameraPose.to_dict()``.
    """
    f = np.asarray(T_room_cam, dtype=np.float64)[:3, 2]
    return math.degrees(math.atan2(float(f[0]), -float(f[2])))


def camera_pose_T(xyz_room_m: Sequence[float], yaw_deg: float) -> Array:
    """``T_room_cam`` of an upright camera at ``xyz_room_m`` with heading ``yaw_deg``.

    Inverse of :func:`yaw_deg_of` (used by the simulator and the tests).
    """
    return make_T(rot_y(-math.radians(yaw_deg)) @ FRONTAL_R_CAM_TAG, xyz_room_m)


def project_points(points_3d: Array, T_cam_x: Array, K: Array) -> Array:
    """Pinhole projection (no distortion) of points given in frame ``x`` into pixels.

    Returns an ``(N, 2)`` array; points at z == 0 project to NaN (points behind the camera
    are returned as-is, callers drop them by checking the depth).
    """
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    p = apply_T(T_cam_x, pts)
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_safe = np.where(np.abs(z) < 1e-12, np.nan, z)
        u = (K[0, 0] * p[:, 0] + K[0, 1] * p[:, 1]) / z_safe + K[0, 2]
        v = K[1, 1] * p[:, 1] / z_safe + K[1, 2]
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------- tag pose (cv2)
@dataclass
class TagPose:
    """One tag's pose in the camera frame plus the quality numbers ``TagFilter`` looks at."""

    tag_id: int
    T_cam_tag: Array
    reproj_err_px: float  # RMS reprojection error of the chosen solution
    ambiguity_ratio: float  # err(best distinct alternative) / err(best); small => ambiguous
    side_px: float  # mean edge length of the quad in pixels
    view_angle_deg: float  # angle between the tag normal and the ray to the camera (0 = frontal)

    @property
    def T_tag_cam(self) -> Array:
        return invert_T(self.T_cam_tag)

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.T_cam_tag[:3, 3]))


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - only without OpenCV
        raise ImportError("opencv-python-headless (cv2) is required for AprilTag PnP") from exc
    return cv2


def _reproj_rms(
    cv2: Any, obj: Array, img: Array, rvec: Array, tvec: Array, K: Array, dist: Array | None
) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - img.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1))))


def _refine(
    cv2: Any, obj: Array, img: Array, K: Array, dist: Array | None, rvec: Array, tvec: Array
) -> tuple[Array, Array, float]:
    r, t = cv2.solvePnPRefineLM(
        obj, img, K, dist, np.array(rvec, dtype=np.float64), np.array(tvec, dtype=np.float64)
    )
    return r, t, _reproj_rms(cv2, obj, img, r, t, K, dist)


def solve_tag_pose(
    tag_id: int,
    corners_px: Any,
    K: Array,
    dist: Array | None,
    tag_size_m: float,
    min_distinct_deg: float = 3.0,
) -> TagPose:
    """Robust single-tag PnP.  ``corners_px``: (4, 2) in TL, TR, BR, BL order.

    Candidates = both ``SOLVEPNP_IPPE_SQUARE`` solutions (``solvePnPGeneric``) plus
    ``SOLVEPNP_ITERATIVE``, each ``solvePnPRefineLM``-refined; solutions with the tag behind
    the camera are dropped and the lowest RMS reprojection error wins.  Plain IPPE_SQUARE
    alone is NOT safe: OpenCV 4.11-5.0 return a flipped/garbage pose for axis-aligned views
    such as a level camera looking straight at a wall tag; ITERATIVE does not have that
    failure and the refine step makes the ambiguity ratio meaningful.

    Raises ``ValueError`` when no valid solution exists.
    """
    cv2 = _cv2()
    K = np.asarray(K, dtype=np.float64)
    obj = tag_object_points(tag_size_m).reshape(4, 1, 3)
    img = np.ascontiguousarray(np.asarray(corners_px, dtype=np.float64).reshape(4, 1, 2))
    dist_arr = None if dist is None else np.asarray(dist, dtype=np.float64)
    cands: list[tuple[Array, Array]] = []
    try:
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj, img, K, dist_arr, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        cands += [(rvecs[i], tvecs[i]) for i in range(n)]
    except cv2.error as exc:
        logger.debug("IPPE_SQUARE produced no candidates for tag %d: %s", tag_id, exc)
    try:
        ok, r0, t0 = cv2.solvePnP(obj, img, K, dist_arr, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error as exc:
        logger.debug("ITERATIVE PnP failed for tag %d: %s", tag_id, exc)
        ok = False
    if ok:
        cands.append((r0, t0))
    refined: list[tuple[Array, Array, float]] = []
    for r, t in cands:
        try:
            rr, tt, e = _refine(cv2, obj, img, K, dist_arr, r, t)
        except cv2.error as exc:
            logger.debug("PnP refinement failed for tag %d: %s", tag_id, exc)
            continue
        if math.isfinite(e) and tt.reshape(3)[2] > 0:
            refined.append((rr, tt, e))
    if not refined:
        raise ValueError(f"no valid PnP solution for tag {tag_id}")
    refined.sort(key=lambda x: x[2])
    r_best, t_best, err_best = refined[0]
    R = rodrigues(r_best)
    t = np.asarray(t_best, dtype=np.float64).reshape(3)
    err_other = math.inf
    for rr, _tt, e in refined[1:]:
        if rotation_angle_deg(rodrigues(rr), R) > min_distinct_deg:
            err_other = e
            break
    ratio = err_other / max(err_best, 1e-9)
    c = img.reshape(4, 2)
    side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    normal = R[:, 2]  # tag z axis (out of the face) in camera coords
    to_cam = -t / max(float(np.linalg.norm(t)), 1e-9)  # unit vector from tag centre to camera
    view = math.degrees(math.acos(max(-1.0, min(1.0, float(normal @ to_cam)))))
    return TagPose(tag_id, make_T(R, t), err_best, ratio, side, view)


def solve_room_pose_multi(
    detections: list[tuple[int, Any]],
    tag_map: dict[int, Array],
    K: Array,
    dist: Array | None,
    tag_size_m: float,
) -> tuple[Array, float] | None:
    """Camera pose in the room from ALL visible known tags in one PnP (8+ points).

    ``detections``: ``[(tag_id, corners (4,2) TL,TR,BR,BL), ...]``; ``tag_map``:
    ``tag_id -> T_room_tag``.  Tags on different walls make the point set non-coplanar, which
    removes the single-square flip ambiguity and roughly halves the pose error.

    Returns ``(T_room_cam, reprojection RMS px)`` or ``None`` when fewer than two known tags
    are visible or the solver fails.
    """
    known = [(tid, c) for tid, c in detections if tid in tag_map]
    if len(known) < 2:
        return None
    cv2 = _cv2()
    K = np.asarray(K, dtype=np.float64)
    obj = np.concatenate(
        [apply_T(tag_map[tid], tag_object_points(tag_size_m)) for tid, _ in known]
    ).reshape(-1, 1, 3)
    img = np.concatenate([np.asarray(c, dtype=np.float64).reshape(4, 2) for _, c in known]).reshape(
        -1, 1, 2
    )
    dist_arr = None if dist is None else np.asarray(dist, dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist_arr, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist_arr, rvec, tvec)
        err = _reproj_rms(cv2, obj, img, rvec, tvec, K, dist_arr)
    except cv2.error as exc:
        logger.debug("multi-tag PnP failed: %s", exc)
        return None
    T_cam_room = make_T(rodrigues(rvec), tvec.reshape(3))
    return invert_T(T_cam_room), err


def normalize_upright(T_cam_tag: Array) -> tuple[Array, bool]:
    """Rotate the tag frame by 180 deg about its normal when its y axis points DOWN in the image.

    Needed only for tags printed with the AprilRobotics canonical orientation instead of
    ``cv2.aruco.generateImageMarker`` (or mounted upside down).  Valid while the camera is
    held roughly upright.  Returns ``(T, was_flipped)``.
    """
    if T_cam_tag[1, 1] > 0:  # tag +y has a +y_cam (downward) component
        return T_cam_tag @ make_T(rot_z(math.pi), (0.0, 0.0, 0.0)), True
    return T_cam_tag, False
