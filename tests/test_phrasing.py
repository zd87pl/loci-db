"""Tests for demo_spatial/app/phrasing.py (design D23, section 4.9).

Pure wording tests: no LOCI client, no camera, fakes for observations, changes,
room specs and poses.  The room-frame conventions come from
``detection_source.camera_pose_T`` so the perspective maths is checked against
the same transform the simulator and the localizer use.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo_spatial"))

from app import phrasing as P  # noqa: E402
from app.detection_source import CameraPose, apply_T, camera_pose_T, make_T  # noqa: E402

NOW_DT = datetime(2026, 10, 24, 14, 18, 0)
NOW_MS = int(NOW_DT.timestamp() * 1000)
TS_3MIN = NOW_MS - 180_000  # 2:15 PM local
CLOCK_3MIN = datetime.fromtimestamp(TS_3MIN / 1000).strftime("%I:%M %p").lstrip("0")

# Camera 2.5 m into the kitchen at 1.4 m, looking at the anchor wall (yaw 0).
T_KITCHEN = camera_pose_T((0.0, 1.4, 2.5), 0.0)


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@dataclass
class FakeObs:
    """Shape of the post-T4 ObjectObservation (section 4.8), fields optional."""

    label: str = "cup"
    cx: float = 0.5
    cy: float = 0.5
    confidence: float = 0.9
    timestamp_ms: int = TS_3MIN
    state_id: str = "s1"
    depth_m: float | None = None
    room: str | None = None
    frame: str = "image"
    pos_m: tuple[float, float, float] | None = None
    xyz_cam_m: tuple[float, float, float] | None = None
    anchored: bool = False
    anchor_confidence: float = 0.0
    source: str = "camera"
    first_seen_ms: int | None = None
    merge_count: int = 0
    restored: bool = False
    scene_id: str = ""

    @property
    def age_seconds(self) -> float:
        return (NOW_MS - self.timestamp_ms) / 1000.0


@dataclass
class FakeChange:
    kind: str
    label: str
    room: str | None = "kitchen"
    last_seen_ms: int = TS_3MIN
    first_seen_ms: int | None = None
    pos_m: tuple[float, float, float] | None = None
    pos_before_m: tuple[float, float, float] | None = None
    delta_m: float | None = None


def room_obs(pos, *, room="kitchen", anchored=True, **kw) -> FakeObs:
    return FakeObs(frame="room", room=room, pos_m=pos, anchored=anchored, **kw)


def pose(room="kitchen", T=T_KITCHEN, anchored=True, frame="room") -> CameraPose:
    return CameraPose(
        room=room,
        frame=frame,
        T_room_cam=T if frame == "room" else None,
        anchored=anchored,
        anchor_confidence=1.0 if anchored else 0.0,
        tag_ids=(0,),
        age_s=0.5,
    )


KITCHEN_SPEC = SimpleNamespace(
    name="kitchen",
    anchor_tag=0,
    anchor_height_m=1.4,
    anchor_name="door tag",
    landmarks={"table": (1.0, 0.75, 1.2)},
)


# ---------------------------------------------------------------------------
# Perspective table, bands, formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "perspective", "expected"),
    [
        (1.0, "walk_in", "to your left"),
        (1.0, "facing_tag", "to your right"),
        (-1.0, "walk_in", "to your right"),
        (-1.0, "facing_tag", "to your left"),
        (0.3, "walk_in", "straight ahead"),
        (-0.49, "facing_tag", "straight ahead"),
        (0.5, "walk_in", "to your left"),
    ],
)
def test_lateral_perspective_table(x, perspective, expected):
    assert P.lateral_phrase(x, perspective) == expected


def test_lateral_rejects_unknown_perspective():
    with pytest.raises(ValueError):
        P.lateral_phrase(1.0, "sideways")


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        (0.0, "on the floor"),
        (0.29, "on the floor"),
        (0.3, "low down"),
        (0.59, "low down"),
        (0.6, "at table height"),
        (1.1, "at table height"),
        (1.11, "at chest height"),
        (1.5, "at chest height"),
        (1.51, "up high"),
        (2.2, "up high"),
    ],
)
def test_height_bands(y, expected):
    assert P.height_phrase(y) == expected


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.0),
        (0.24, 0.0),
        (0.25, 0.5),
        (0.74, 0.5),
        (0.76, 1.0),
        (1.25, 1.5),
        (2.3, 2.5),
        (2.24, 2.0),
    ],
)
def test_round_half_m(x, expected):
    assert P.round_half_m(x) == expected


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.1, "less than half a metre"),
        (0.5, "half a metre"),
        (1.0, "1 metre"),
        (-1.0, "1 metre"),
        (2.5, "2.5 metres"),
        (3.0, "3 metres"),
        (3.2, "3 metres"),
    ],
)
def test_format_distance(x, expected):
    assert P.format_distance(x) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "just now"),
        (9.9, "just now"),
        (42, "42 seconds ago"),
        (60, "1 minute ago"),
        (125, "2 minutes ago"),
        (3600, "1 hour ago"),
        (7300, "2 hours ago"),
        (86400, "1 day ago"),
        (200_000, "2 days ago"),
    ],
)
def test_format_age(seconds, expected):
    assert P.format_age(seconds) == expected


def test_format_clock_local_time():
    ts = int(datetime(2026, 10, 24, 14, 15).timestamp() * 1000)
    assert P.format_clock(ts) == "2:15 PM"
    ts = int(datetime(2026, 10, 24, 9, 5).timestamp() * 1000)
    assert P.format_clock(ts) == "9:05 AM"


def test_window_phrase():
    assert P.window_phrase(600) == "in the last 10 minutes"
    assert P.window_phrase(60) == "in the last minute"
    assert P.window_phrase(45) == "in the last 45 seconds"
    assert P.window_phrase(3600) == "in the last hour"
    assert P.window_phrase(7200) == "in the last 2 hours"


def test_room_display():
    assert P.room_display("living_room") == "living room"
    assert P.room_display("kitchen") == "kitchen"
    assert P.room_display(None) == "room"


# ---------------------------------------------------------------------------
# describe_position per frame
# ---------------------------------------------------------------------------


def test_room_position_walk_in_wording():
    text = P.room_position_phrase((1.0, 0.8, 1.0), room="kitchen")
    assert text == (
        "about 1 metre into the kitchen and 1 metre to your left as you walk in from the "
        "door tag, at table height"
    )
    assert P.room_position_phrase((-2.0, 0.1, 3.0)) == (
        "about 3 metres into the room and 2 metres to your right as you walk in from the "
        "door tag, on the floor"
    )


def test_room_position_straight_ahead_and_custom_anchor():
    text = P.room_position_phrase((0.2, 1.3, 2.5), anchor_name="hallway tag")
    assert text == (
        "about 2.5 metres into the room, straight ahead as you walk in from the hallway tag, "
        "at chest height"
    )


def test_room_position_at_the_wall():
    assert P.room_position_phrase((0.0, 0.9, 0.1)).startswith("just inside the room")


def test_describe_position_dispatches_on_frame():
    assert P.describe_position("room", pos_m=(1.0, 0.8, 1.0), room="kitchen") == (
        P.room_position_phrase((1.0, 0.8, 1.0), room="kitchen")
    )
    cam = P.describe_position("camera", xyz_cam_m=(0.0, 0.0, 2.0))
    assert cam == "about 2 metres in front of the camera, about level with the camera"
    # room frame without metres falls back to the camera wording
    assert "in front of the camera" in P.describe_position("room", xyz_cam_m=(0.0, 0.0, 2.0))
    # legacy image wording is byte-identical to the old text
    assert P.describe_position("image", cx=0.1, cy=0.1) == "on the left, toward the back"
    assert P.describe_position("image", cx=0.5, cy=0.5) == "in the center, in the middle area"
    assert P.describe_position("image", cx=0.9, cy=0.9) == "on the right, near the front"
    assert P.describe_position("camera") == "in the center, in the middle area"


def test_camera_position_sides_and_eye_line():
    assert P.camera_position_phrase((0.0, -0.5, 2.0)).endswith("above eye line")
    assert P.camera_position_phrase((0.0, 0.5, 2.0)).endswith("below eye line")
    assert ", slightly to the left, " in P.camera_position_phrase((-0.5, 0.0, 2.0))
    assert ", 1.5 metres to the right, " in P.camera_position_phrase((1.5, 0.0, 2.0))
    assert P.camera_position_phrase((0.0, 0.0, 0.1)).startswith("right in front of the camera")


# ---------------------------------------------------------------------------
# Egocentric wording
# ---------------------------------------------------------------------------


def test_egocentric_known_pose_ahead_and_left():
    # Camera at (0, 1.4, 2.5) facing the wall; object 3 m ahead, 1 m to the camera's left.
    text = P.egocentric_phrase((-1.0, 0.8, -0.5), T_KITCHEN)
    assert text == "about 3 metres ahead of you and 1 metre to your left, at table height"


def test_egocentric_uses_the_camera_frame_conventions():
    # Points expressed in the camera frame (+x right, +z forward) via T2's transform.
    right = apply_T(T_KITCHEN, (1.0, 0.0, 3.0))
    left = apply_T(T_KITCHEN, (-1.0, 0.0, 3.0))
    assert P.egocentric_phrase(right, T_KITCHEN) == (
        "about 3 metres ahead of you and 1 metre to your right, at chest height"
    )
    assert P.egocentric_phrase(left, T_KITCHEN).startswith(
        "about 3 metres ahead of you and 1 metre to your left"
    )


def test_egocentric_straight_ahead_behind_and_next_to_you():
    assert P.egocentric_phrase((0.1, 0.8, 0.5), T_KITCHEN) == (
        "about 2 metres ahead of you, at table height"
    )
    assert (
        P.egocentric_phrase((0.0, 0.2, 4.5), T_KITCHEN) == "about 2 metres behind you, on the floor"
    )
    assert P.egocentric_phrase((0.1, 0.9, 2.4), T_KITCHEN).startswith("right next to you")


def test_egocentric_follows_heading():
    T = camera_pose_T((0.0, 1.4, 2.5), 90.0)  # looking toward +x
    assert P.egocentric_phrase((3.0, 0.5, 2.5), T) == "about 3 metres ahead of you, low down"


def test_egocentric_ignores_camera_pitch():
    # Tilt the handheld 45 deg down: "ahead" is still measured on the floor plane.
    T_pitched = T_KITCHEN @ make_T(rot_x(45.0), (0.0, 0.0, 0.0))
    assert P.egocentric_phrase((0.0, 0.8, -0.5), T_pitched) == (
        "about 3 metres ahead of you, at table height"
    )


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------


def test_landmark_within_and_outside_radius():
    landmarks = {"table": (1.0, 0.75, 1.2)}
    assert P.landmark_phrase((1.2, 0.8, 1.0), landmarks) == "near the table"
    assert P.landmark_phrase((1.0, 0.75, 1.2 + 1.19), landmarks) == "near the table"
    assert P.landmark_phrase((1.0, 0.75, 1.2 + 1.21), landmarks) is None
    assert P.landmark_phrase((3.0, 0.8, 1.0), landmarks) is None


def test_landmark_nearest_wins_and_names_are_spoken():
    landmarks = {"coffee_table": (0.0, 0.5, 2.0), "couch": (0.5, 0.5, 2.5)}
    assert P.landmark_phrase((0.1, 0.5, 2.0), landmarks) == "near the coffee table"
    assert P.landmark_phrase((0.5, 0.5, 2.4), landmarks) == "near the couch"
    assert P.landmark_phrase((0.1, 0.5, 2.0), {}) is None
    assert P.landmark_phrase((0.1, 0.5, 2.0), None) is None
    assert P.landmark_phrase((0.1, 0.5, 2.0), {"bad": (1.0,)}) is None


# ---------------------------------------------------------------------------
# where_is_answer and suffixes
# ---------------------------------------------------------------------------


def test_where_is_answer_room_frame_alias_disclosure():
    obs = room_obs((1.0, 0.8, 1.0))
    text = P.where_is_answer("mug", obs, now_ms=NOW_MS, resolved_label="cup")
    assert text == (
        "Your mug (I track it as a cup) was last seen in the kitchen, about 1 metre into the "
        "room and 1 metre to your left as you walk in from the door tag, at table height. "
        f"That was 3 minutes ago, at {CLOCK_3MIN}."
    )
    assert P.UNANCHORED_SUFFIX not in text
    assert P.RESTORED_SUFFIX not in text


def test_where_is_answer_no_alias_clause_when_names_match():
    obs = room_obs((1.0, 0.8, 1.0))
    assert "(I track" not in P.where_is_answer("cup", obs, now_ms=NOW_MS, resolved_label="cup")
    assert "(I track" not in P.where_is_answer("cups", obs, now_ms=NOW_MS, resolved_label="cup")
    assert "(I track" not in P.where_is_answer("my cup", obs, now_ms=NOW_MS, resolved_label="cup")
    assert P.where_is_answer("my cup", obs, now_ms=NOW_MS).startswith("Your cup was last seen")


def test_where_is_answer_hedges_low_confidence_embedding_match():
    obs = room_obs((1.0, 0.8, 1.0), label="bird")
    text = P.where_is_answer("beaker", obs, now_ms=NOW_MS, resolved_label="bird", hedge=True)
    assert text.startswith("Your beaker (I think you mean a bird) was last seen")


def test_where_is_answer_landmark_wins_over_walk_in():
    obs = room_obs((1.1, 0.8, 1.1))
    text = P.where_is_answer("mug", obs, room_spec=KITCHEN_SPEC, now_ms=NOW_MS)
    assert "in the kitchen, near the table." in text
    assert "walk in" not in text


def test_where_is_answer_egocentric_when_anchored_in_the_same_room():
    obs = room_obs((-1.0, 0.8, -0.5))
    text = P.where_is_answer("mug", obs, current_pose=pose("kitchen"), now_ms=NOW_MS)
    assert "about 3 metres ahead of you and 1 metre to your left, at table height" in text
    assert "walk in" not in text
    # landmark + egocentric when both apply
    obs2 = room_obs((1.1, 0.8, 1.1))
    text2 = P.where_is_answer(
        "mug", obs2, room_spec=KITCHEN_SPEC, current_pose=pose("kitchen"), now_ms=NOW_MS
    )
    assert "near the table, about" in text2 and "ahead of you" in text2


@pytest.mark.parametrize(
    "current",
    [
        None,
        pose("living_room"),  # anchored elsewhere
        pose("kitchen", anchored=False),  # held pose only
        pose("kitchen", frame="camera"),  # no room pose at all
    ],
)
def test_where_is_answer_walk_in_when_not_anchored_here(current):
    obs = room_obs((-1.0, 0.8, -0.5))
    text = P.where_is_answer("mug", obs, current_pose=current, now_ms=NOW_MS)
    assert "as you walk in from the door tag" in text
    assert "ahead of you" not in text


def test_where_is_answer_unanchored_suffix():
    held = room_obs((1.0, 0.8, 1.0), anchored=False)
    text = P.where_is_answer("mug", held, now_ms=NOW_MS)
    assert "at table height, roughly — I wasn't sure of my position. That was" in text

    cam = FakeObs(frame="camera", xyz_cam_m=(0.3, 0.4, 1.6), anchored=False)
    text = P.where_is_answer("mug", cam, now_ms=NOW_MS)
    assert text.startswith(
        "Your mug was last seen about 1.5 metres in front of the camera, slightly to the right, "
        "below eye line, roughly — I wasn't sure of my position. That was 3 minutes ago"
    )


def test_where_is_answer_restored_suffix():
    obs = room_obs((1.0, 0.8, 1.0), restored=True)
    assert P.where_is_answer("mug", obs, now_ms=NOW_MS).endswith(" That was before I restarted.")
    fresh = room_obs((1.0, 0.8, 1.0))
    assert P.where_is_answer("mug", fresh, now_ms=NOW_MS, started_at_ms=TS_3MIN + 1).endswith(
        P.RESTORED_SUFFIX
    )
    assert not P.where_is_answer("mug", fresh, now_ms=NOW_MS, started_at_ms=TS_3MIN - 1).endswith(
        P.RESTORED_SUFFIX
    )


def test_where_is_answer_image_frame_keeps_legacy_wording():
    obs = FakeObs(label="keys", cx=0.1, cy=0.1, timestamp_ms=NOW_MS - 5000)
    text = P.where_is_answer("keys", obs, now_ms=NOW_MS)
    clock = P.format_clock(NOW_MS - 5000)
    assert (
        text
        == f"Your keys was last seen on the left, toward the back. That was just now, at {clock}."
    )


def test_history_answer_lists_previous_sightings():
    newest = room_obs((1.0, 0.8, 1.0))
    older = room_obs((-0.8, 0.7, 1.0), room="living_room", timestamp_ms=NOW_MS - 3600_000)
    oldest = FakeObs(label="cup", cx=0.8, cy=0.8, timestamp_ms=NOW_MS - 7200_000)
    text = P.history_answer("mug", [newest, older, oldest], now_ms=NOW_MS, resolved_label="cup")
    assert text.startswith("Your mug (I track it as a cup) was last seen in the kitchen")
    assert "Previous sightings:" in text
    assert f"At {P.format_clock(NOW_MS - 3600_000)} (1 hour ago), in the living room" in text
    assert "(2 hours ago), on the right, near the front." in text
    assert P.history_answer("mug", [], resolved_label="cup").startswith("I haven't seen your mug")


def test_sighting_phrase_short_form():
    obs = room_obs((1.1, 0.8, 1.1), anchored=False)
    text = P.sighting_phrase(obs, room_spec=KITCHEN_SPEC, now_ms=NOW_MS)
    assert text == (
        "in the kitchen, near the table, roughly — I wasn't sure of my position, seen 3 minutes ago"
    )


def test_anchored_in_room():
    assert P.anchored_in_room(pose("kitchen"), "kitchen")
    assert P.anchored_in_room(pose("living_room"), "living room")
    assert not P.anchored_in_room(pose("kitchen"), "living_room")
    assert not P.anchored_in_room(pose("kitchen", anchored=False), "kitchen")
    assert not P.anchored_in_room(pose("kitchen", frame="camera"), "kitchen")
    assert not P.anchored_in_room(None, "kitchen")
    assert not P.anchored_in_room(pose("kitchen"), None)


# ---------------------------------------------------------------------------
# changes_answer
# ---------------------------------------------------------------------------


def test_changes_answer_each_kind():
    changes = [
        FakeChange("moved", "cup", delta_m=2.4),
        FakeChange("new", "bottle"),
        FakeChange("gone", "laptop"),
        FakeChange("seen", "couch"),
    ]
    assert P.changes_answer(changes, "kitchen", 600) == (
        "In the kitchen in the last 10 minutes: the cup moved about 2.5 metres; "
        "a bottle is new; I no longer see the laptop."
    )
    assert P.changes_answer([FakeChange("moved", "cup")], None, 60) == (
        "In the last minute: the cup moved."
    )
    assert P.changes_answer([FakeChange("new", "apple")], "living_room", 3600) == (
        "In the living room in the last hour: an apple is new."
    )


def test_changes_answer_empty_and_seen_only():
    assert (
        P.changes_answer([], "kitchen", 600)
        == "Nothing changed in the kitchen in the last 10 minutes."
    )
    assert P.changes_answer([], None, 600) == "Nothing changed in the last 10 minutes."
    seen = [FakeChange("seen", "cup"), FakeChange("seen", "bottle")]
    assert P.changes_answer(seen, "kitchen", 600) == (
        "Nothing changed in the kitchen in the last 10 minutes. I still see the cup and the bottle."
    )
    assert P.changes_answer(None, None, 600) == "Nothing changed in the last 10 minutes."


# ---------------------------------------------------------------------------
# unknown / not seen / where am I
# ---------------------------------------------------------------------------


def test_unknown_label_answer_with_and_without_suggestion():
    res = SimpleNamespace(label=None, how="none", score=0.2, suggestions=["keyboard"])
    examples = ["cup", "bottle", "cell phone", "remote", "book"]
    assert P.unknown_label_answer("keys", res, examples) == (
        "I can't recognise 'keys' with this camera. Did you mean keyboard? "
        "I can look for cup, bottle, cell phone, remote, book."
    )
    none = SimpleNamespace(label=None, how="none", score=0.0, suggestions=[])
    assert P.unknown_label_answer("my wallet", none, examples) == (
        "I can't recognise 'wallet' with this camera. I can look for cup, bottle, cell phone, "
        "remote, book."
    )
    assert (
        P.unknown_label_answer("wallet", None, []) == "I can't recognise 'wallet' with this camera."
    )
    two = SimpleNamespace(
        label=None, how="none", score=0.3, suggestions=["keyboard", "mouse", "tv"]
    )
    assert "Did you mean keyboard or mouse?" in P.unknown_label_answer("keys", two, [])
    # a suggestion equal to the spoken word is not offered
    same = SimpleNamespace(label=None, how="none", score=0.3, suggestions=["keys"])
    assert "Did you mean" not in P.unknown_label_answer("keys", same, [])


def test_not_seen_answer():
    assert P.not_seen_answer("mug", None) == "I haven't seen your mug yet. Try scanning the area."
    assert (
        P.not_seen_answer("my mug", "living_room")
        == "I haven't seen your mug in the living room yet."
    )
    assert P.not_seen_answer("mug", None, resolved_label="cup") == (
        "I haven't seen your mug yet (I look for it as a cup). Try scanning the area."
    )
    assert P.not_seen_answer("cup", "kitchen", resolved_label="cup") == (
        "I haven't seen your cup in the kitchen yet."
    )


def test_where_am_i_answer():
    assert P.where_am_i_answer(None).startswith("I'm not sure which room you're in")
    assert P.where_am_i_answer(pose("kitchen", frame="camera")).startswith("I'm not sure")
    assert P.where_am_i_answer(pose("kitchen")) == (
        "You're in the kitchen, about 2.5 metres in from the door tag, facing the door tag."
    )
    assert P.where_am_i_answer(pose("kitchen", anchored=False)).endswith(", roughly.")
    away = pose("kitchen", T=camera_pose_T((0.0, 1.4, 2.5), 180.0))
    assert "with the door tag behind you" in P.where_am_i_answer(away)
    side = pose("living_room", T=camera_pose_T((1.0, 1.4, 2.0), 90.0))
    text = P.where_am_i_answer(side, anchor_name="hall tag")
    assert text == (
        "You're in the living room, about 2 metres in from the hall tag and 1 metre to the "
        "right of it, with the hall tag to your left."
    )
