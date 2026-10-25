"""Answer wording for the handheld spatial-memory demo (design D23, section 4.9).

This module is the single place that turns metric observations into spoken
English.  Everything else (``voice_pipeline``, ``spatial_memory``'s
``position_description``, the phone page) calls these helpers instead of
formatting positions itself, so the perspective conventions live in one file:

* **Room frame** (``frame == "room"``): right-handed, ``+y`` up, ``+z`` from the
  anchor wall into the room, ``+x`` to the viewer's right when *facing* the tag,
  which is the walker's **left** when entering with their back to the tag.
  Answers use the walk-in perspective and say so ("as you walk in from the door
  tag").  When the camera is currently anchored in the same room the answer is
  egocentric instead ("about 3 metres ahead of you and 1 metre to your left").
* **Camera frame** (``frame == "camera"``): OpenCV axes, ``+x`` right, ``+y``
  down, ``+z`` forward.  The device had no usable room pose, so the wording is
  relative to the camera and carries the "roughly" suffix.
* **Image frame** (``frame == "image"``): the legacy browser/YOLO wording built
  from normalised box centres ("on the left, toward the back"), kept identical
  to the pre-handheld text.

Distances are rounded to the nearest half metre and spelled out ("2.5 metres",
"half a metre") so Piper and screen readers do not read a bare "m".
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .detection_source import CameraPose
    from .label_resolver import Resolution

DEFAULT_ANCHOR_NAME = "door tag"
#: |x| below this is "straight ahead" (walk-in and facing-tag perspectives).
STRAIGHT_AHEAD_M = 0.5
#: Nearest landmark within this distance names the spot ("near the table").
LANDMARK_MAX_DIST_M = 1.2
UNANCHORED_SUFFIX = ", roughly — I wasn't sure of my position"
RESTORED_SUFFIX = " That was before I restarted."
PERSPECTIVES = ("walk_in", "facing_tag")

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def format_age(seconds: float) -> str:
    """Relative time: "just now", "42 seconds ago", "3 minutes ago", "2 hours ago"."""
    age = float(seconds)
    if age < 10:
        return "just now"
    if age < 60:
        return f"{int(age)} seconds ago"
    if age < 3600:
        mins = int(age / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if age < 86400:
        hrs = int(age / 3600)
        return f"{hrs} hour{'s' if hrs != 1 else ''} ago"
    days = int(age / 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def format_clock(timestamp_ms: int) -> str:
    """Local clock time of a Unix-ms timestamp, e.g. "2:15 PM"."""
    dt = datetime.fromtimestamp(int(timestamp_ms) / 1000)
    return dt.strftime("%I:%M %p").lstrip("0")


def round_half_m(x: float) -> float:
    """Round to the nearest 0.5 m (ties go up): 0.74 -> 0.5, 0.76 -> 1.0, 2.3 -> 2.5."""
    return math.floor(float(x) * 2.0 + 0.5) / 2.0


def format_distance(x: float) -> str:
    """Spoken distance for an already-positive length, rounded to half metres."""
    r = round_half_m(abs(float(x)))
    if r == 0.0:
        return "less than half a metre"
    if r == 0.5:
        return "half a metre"
    if r == 1.0:
        return "1 metre"
    return f"{r:g} metres"


def room_display(room: str | None) -> str:
    """ "living_room" -> "living room" for speech; None -> "room"."""
    if not room:
        return "room"
    return str(room).replace("_", " ").strip() or "room"


def _article(word: str) -> str:
    w = word.strip().lower()
    return "an" if w[:1] in ("a", "e", "i", "o", "u") else "a"


def _norm_name(text: str) -> str:
    t = " ".join(str(text).strip().lower().split())
    for prefix in ("my ", "the ", "a ", "an "):
        if t.startswith(prefix):
            t = t[len(prefix) :]
    return t


def _same_label(spoken: str, label: str) -> bool:
    a, b = _norm_name(spoken), _norm_name(label)
    return a == b or a.rstrip("s") == b.rstrip("s")


def _join_names(names: Sequence[str]) -> str:
    names = [str(n) for n in names]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# ---------------------------------------------------------------------------
# Position bands and perspectives
# ---------------------------------------------------------------------------


def height_phrase(y_m: float) -> str:
    """Height above the floor (room frame, y up) as a coarse band."""
    y = float(y_m)
    if y < 0.3:
        return "on the floor"
    if y < 0.6:
        return "low down"
    if y <= 1.1:
        return "at table height"
    if y <= 1.5:
        return "at chest height"
    return "up high"


def lateral_phrase(x_m: float, perspective: str = "walk_in") -> str:
    """Left/right of a room-frame ``x`` for the given perspective.

    ``"walk_in"``: the walker enters with their back to the anchor tag, so room
    ``+x`` is on their **left**.  ``"facing_tag"``: mirrored (``+x`` is right).
    """
    if perspective not in PERSPECTIVES:
        raise ValueError(f"perspective must be one of {PERSPECTIVES}, got {perspective!r}")
    x = float(x_m)
    if abs(x) < STRAIGHT_AHEAD_M:
        return "straight ahead"
    left = x > 0 if perspective == "walk_in" else x < 0
    return "to your left" if left else "to your right"


def image_position_phrase(cx: float, cy: float) -> str:
    """Legacy wording for normalised image coordinates (identical to the old text)."""
    if cx < 0.33:
        h_pos = "on the left"
    elif cx > 0.67:
        h_pos = "on the right"
    else:
        h_pos = "in the center"
    if cy < 0.35:
        v_pos = "toward the back"
    elif cy > 0.70:
        v_pos = "near the front"
    else:
        v_pos = "in the middle area"
    return f"{h_pos}, {v_pos}"


def room_position_phrase(
    pos_m: Sequence[float],
    *,
    anchor_name: str = DEFAULT_ANCHOR_NAME,
    room: str | None = None,
) -> str:
    """Walk-in description of a room-frame point.

    "about 2 metres into the kitchen and 1 metre to your left as you walk in
    from the door tag, at table height".  ``room=None`` says "into the room".
    """
    x, y, z = (float(v) for v in pos_m)
    place = f"the {room_display(room)}" if room else "the room"
    depth = round_half_m(max(z, 0.0))
    head = f"about {format_distance(depth)} into {place}" if depth > 0 else f"just inside {place}"
    lateral = lateral_phrase(x, "walk_in")
    if lateral == "straight ahead":
        middle = f"{head}, straight ahead as you walk in from the {anchor_name}"
    else:
        middle = f"{head} and {format_distance(x)} {lateral} as you walk in from the {anchor_name}"
    return f"{middle}, {height_phrase(y)}"


def camera_position_phrase(xyz_cam_m: Sequence[float]) -> str:
    """Description of a camera-frame point (OpenCV axes: +x right, +y down, +z forward)."""
    x, y, z = (float(v) for v in xyz_cam_m)
    dist = round_half_m(max(z, 0.0))
    head = (
        f"about {format_distance(dist)} in front of the camera"
        if dist > 0
        else ("right in front of the camera")
    )
    side = ""
    if abs(x) >= 0.3:
        direction = "left" if x < 0 else "right"
        side = (
            f", slightly to the {direction}"
            if abs(x) < 1.0
            else f", {format_distance(x)} to the {direction}"
        )
    if y < -0.3:
        vert = "above eye line"
    elif y > 0.3:
        vert = "below eye line"
    else:
        vert = "about level with the camera"
    return f"{head}{side}, {vert}"


def describe_position(
    frame: str,
    pos_m: Sequence[float] | None = None,
    xyz_cam_m: Sequence[float] | None = None,
    cx: float = 0.5,
    cy: float = 0.5,
    *,
    anchor_name: str = DEFAULT_ANCHOR_NAME,
    room: str | None = None,
) -> str:
    """Frame-aware position wording (used by ``ObjectObservation.position_description``).

    ``"room"`` with ``pos_m`` -> walk-in wording; ``"camera"`` (or a room frame
    without metres) with ``xyz_cam_m`` -> camera wording; anything else -> the
    legacy image wording from ``cx``/``cy``.
    """
    if frame == "room" and pos_m is not None:
        return room_position_phrase(pos_m, anchor_name=anchor_name, room=room)
    if frame in ("room", "camera") and xyz_cam_m is not None:
        return camera_position_phrase(xyz_cam_m)
    return image_position_phrase(float(cx), float(cy))


def _ground_axes(T_room_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Forward and right unit vectors of the camera projected onto the floor plane."""
    R = np.asarray(T_room_cam, dtype=float)[:3, :3]
    forward = R[:, 2].copy()
    right = R[:, 0].copy()
    forward[1] = 0.0
    right[1] = 0.0
    nf, nr = float(np.linalg.norm(forward)), float(np.linalg.norm(right))
    if nf < 1e-6 or nr < 1e-6:
        return None
    return forward / nf, right / nr


def egocentric_phrase(pos_room_m: Sequence[float], T_room_cam: np.ndarray) -> str:
    """Position relative to the camera's current heading, on the floor plane.

    "about 3 metres ahead of you and 1 metre to your left, at table height".
    Pitch and roll of the handheld are ignored (ahead/left are measured on the
    floor plane) so a tilted camera does not turn "ahead" into "below".
    """
    T = np.asarray(T_room_cam, dtype=float).reshape(4, 4)
    p = np.asarray(pos_room_m, dtype=float).reshape(3)
    d = p - T[:3, 3]
    axes = _ground_axes(T)
    if axes is None:  # camera pointing straight up/down: fall back to the raw frame
        p_cam = T[:3, :3].T @ d
        ahead_m, right_m = float(p_cam[2]), float(p_cam[0])
    else:
        forward, right = axes
        ahead_m, right_m = float(d @ forward), float(d @ right)

    ahead = round_half_m(abs(ahead_m))
    if ahead == 0.0:
        head = "right next to you"
    elif ahead_m > 0:
        head = f"about {format_distance(ahead)} ahead of you"
    else:
        head = f"about {format_distance(ahead)} behind you"
    side = round_half_m(abs(right_m))
    if side > 0.0:
        head += f" and {format_distance(side)} to your {'right' if right_m > 0 else 'left'}"
    return f"{head}, {height_phrase(float(p[1]))}"


def landmark_phrase(
    pos_room_m: Sequence[float],
    landmarks: Mapping[str, Sequence[float]] | None,
    max_dist_m: float = LANDMARK_MAX_DIST_M,
) -> str | None:
    """ "near the table" for the nearest landmark within ``max_dist_m``; else None."""
    if not landmarks or not isinstance(landmarks, Mapping):
        return None
    p = np.asarray(pos_room_m, dtype=float).reshape(3)
    best: tuple[float, str] | None = None
    for name, xyz in landmarks.items():
        try:
            q = np.asarray(xyz, dtype=float).reshape(3)
        except (TypeError, ValueError):
            continue
        dist = float(np.linalg.norm(p - q))
        if dist <= max_dist_m and (best is None or dist < best[0]):
            best = (dist, str(name))
    if best is None:
        return None
    return f"near the {room_display(best[1])}"


# ---------------------------------------------------------------------------
# Observation-level phrases
# ---------------------------------------------------------------------------


def _obs_get(obs: Any, name: str, default: Any = None) -> Any:
    return getattr(obs, name, default)


def _landmarks_of(room_spec: Any) -> Mapping[str, Sequence[float]] | None:
    lm = getattr(room_spec, "landmarks", None) if room_spec is not None else None
    return lm if isinstance(lm, Mapping) else None


def _anchor_name_of(room_spec: Any) -> str:
    name = getattr(room_spec, "anchor_name", None) if room_spec is not None else None
    return name if isinstance(name, str) and name.strip() else DEFAULT_ANCHOR_NAME


def anchored_in_room(pose: CameraPose | None, room: str | None) -> bool:
    """True when ``pose`` is a live, anchored room pose for ``room``."""
    if pose is None or not room:
        return False
    if getattr(pose, "frame", None) != "room" or not getattr(pose, "anchored", False):
        return False
    if getattr(pose, "T_room_cam", None) is None:
        return False
    return room_display(getattr(pose, "room", None)) == room_display(room)


def location_phrase(
    obs: Any,
    *,
    room_spec: Any = None,
    current_pose: CameraPose | None = None,
    egocentric: bool = True,
    name_room: bool = True,
) -> str:
    """Where an observation is, in the right perspective for the listener.

    Room frame: ``"in the kitchen, near the table"`` / ``"in the kitchen, about 2
    metres into the room and 1 metre to your left as you walk in from the door
    tag, at table height"``; egocentric when the camera is anchored in that room.
    Camera frame: camera wording.  Image frame: legacy wording.  The unanchored
    suffix is appended for camera-frame and held (``anchored=False``) room-frame
    observations.
    """
    frame = _obs_get(obs, "frame", "image") or "image"
    room = _obs_get(obs, "room")
    pos = _obs_get(obs, "pos_m")
    xyz = _obs_get(obs, "xyz_cam_m")
    anchored = bool(_obs_get(obs, "anchored", False))

    if frame == "room" and pos is not None:
        parts: list[str] = []
        near = landmark_phrase(pos, _landmarks_of(room_spec))
        if near:
            parts.append(near)
        if egocentric and anchored_in_room(current_pose, room):
            parts.append(egocentric_phrase(pos, current_pose.T_room_cam))
        elif not near:
            parts.append(room_position_phrase(pos, anchor_name=_anchor_name_of(room_spec)))
        text = ", ".join(parts)
        if name_room:
            text = f"in the {room_display(room)}, {text}" if room else f"in the room, {text}"
        return text + ("" if anchored else UNANCHORED_SUFFIX)

    if frame == "camera" or (xyz is not None and pos is None and frame != "image"):
        if xyz is not None:
            return camera_position_phrase(xyz) + UNANCHORED_SUFFIX
        return (
            image_position_phrase(float(_obs_get(obs, "cx", 0.5)), float(_obs_get(obs, "cy", 0.5)))
            + UNANCHORED_SUFFIX
        )

    return image_position_phrase(float(_obs_get(obs, "cx", 0.5)), float(_obs_get(obs, "cy", 0.5)))


def sighting_phrase(obs: Any, *, room_spec: Any = None, now_ms: int | None = None) -> str:
    """Short "where and when" for history lines and object lists."""
    where = location_phrase(obs, room_spec=room_spec, current_pose=None, egocentric=False)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    ts = int(_obs_get(obs, "timestamp_ms", now))
    return f"{where}, seen {format_age((now - ts) / 1000.0)}"


def where_is_answer(
    spoken_name: str,
    obs: Any,
    *,
    room_spec: Any = None,
    current_pose: CameraPose | None = None,
    now_ms: int | None = None,
    resolved_label: str | None = None,
    started_at_ms: int | None = None,
    hedge: bool = False,
) -> str:
    """Full spoken answer for "where is my X".

    "Your mug (I track it as a cup) was last seen in the kitchen, near the table.
    That was 3 minutes ago, at 2:15 PM." plus the unanchored suffix when the
    position was not anchored and " That was before I restarted." when the
    sighting predates this process (``obs.restored`` or ``timestamp_ms <
    started_at_ms``).  ``hedge=True`` says "(I think you mean a cup)" instead,
    for low-confidence embedding matches.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    ts = int(_obs_get(obs, "timestamp_ms", now))
    spoken = _norm_name(spoken_name) or str(_obs_get(obs, "label", "object"))

    subject = f"Your {spoken}"
    if resolved_label and not _same_label(spoken, resolved_label):
        verb = "I think you mean" if hedge else "I track it as"
        subject += f" ({verb} {_article(resolved_label)} {resolved_label})"

    where = location_phrase(obs, room_spec=room_spec, current_pose=current_pose)
    when = f"That was {format_age((now - ts) / 1000.0)}, at {format_clock(ts)}."
    answer = f"{subject} was last seen {where}. {when}"

    restored = bool(_obs_get(obs, "restored", False)) or (
        started_at_ms is not None and ts < int(started_at_ms)
    )
    if restored:
        answer += RESTORED_SUFFIX
    return answer


def history_answer(
    spoken_name: str,
    observations: Sequence[Any],
    *,
    room_spec_for: Mapping[str, Any] | None = None,
    current_pose: CameraPose | None = None,
    now_ms: int | None = None,
    resolved_label: str | None = None,
    started_at_ms: int | None = None,
    hedge: bool = False,
    max_previous: int = 3,
) -> str:
    """``where_is_answer`` for the newest sighting plus up to ``max_previous`` older ones."""
    if not observations:
        return not_seen_answer(spoken_name, None, resolved_label=resolved_label)
    newest = observations[0]
    specs = room_spec_for or {}
    answer = where_is_answer(
        spoken_name,
        newest,
        room_spec=specs.get(_obs_get(newest, "room") or ""),
        current_pose=current_pose,
        now_ms=now_ms,
        resolved_label=resolved_label,
        started_at_ms=started_at_ms,
        hedge=hedge,
    )
    previous = list(observations[1 : 1 + max_previous])
    if previous:
        answer += " Previous sightings:"
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        for prev in previous:
            ts = int(_obs_get(prev, "timestamp_ms", now))
            where = location_phrase(
                prev, room_spec=specs.get(_obs_get(prev, "room") or ""), egocentric=False
            )
            answer += f" At {format_clock(ts)} ({format_age((now - ts) / 1000.0)}), {where}."
    return answer


# ---------------------------------------------------------------------------
# Changes, unknown labels, not seen, where am I
# ---------------------------------------------------------------------------


def window_phrase(window_s: float) -> str:
    """ "in the last 10 minutes" / "in the last minute" / "in the last hour"."""
    s = float(window_s)
    if s < 60:
        n = max(1, int(round(s)))
        return f"in the last {n} second{'s' if n != 1 else ''}"
    if s < 3600:
        m = max(1, int(round(s / 60)))
        return "in the last minute" if m == 1 else f"in the last {m} minutes"
    h = s / 3600
    if abs(h - 1.0) < 1e-9:
        return "in the last hour"
    return f"in the last {h:g} hours"


def _change_item(change: Any) -> str | None:
    kind = str(_obs_get(change, "kind", "") or "")
    label = str(_obs_get(change, "label", "object") or "object")
    if kind == "moved":
        delta = _obs_get(change, "delta_m")
        if delta is None:
            return f"the {label} moved"
        return f"the {label} moved about {format_distance(float(delta))}"
    if kind == "new":
        return f"{_article(label)} {label} is new"
    if kind == "gone":
        return f"I no longer see the {label}"
    return None


def changes_answer(changes: Iterable[Any], room: str | None, window_s: float) -> str:
    """ "In the kitchen in the last 10 minutes: the cup moved about 2.5 metres; a bottle
    is new; I no longer see the laptop." or "Nothing changed in the kitchen in the last
    10 minutes." (plus "I still see the cup and the bottle." when only unchanged
    sightings fall inside the window)."""
    changes = list(changes or [])
    window = window_phrase(window_s)
    place = f"in the {room_display(room)}" if room else ""
    items = [item for item in (_change_item(c) for c in changes) if item]
    if items:
        head = f"In the {room_display(room)} {window}" if room else window.capitalize()
        return f"{head}: {'; '.join(items)}."
    still = [
        str(_obs_get(c, "label", "") or "")
        for c in changes
        if str(_obs_get(c, "kind", "") or "") == "seen"
    ]
    still = [s for s in still if s]
    text = f"Nothing changed {place} {window}.".replace("  ", " ")
    if still:
        text += f" I still see {_join_names([f'the {s}' for s in still[:4]])}."
    return text


def unknown_label_answer(name: str, resolution: Resolution | None, examples: Sequence[str]) -> str:
    """Honest refusal for a noun the detector cannot see (D13).

    "I can't recognise 'keys' with this camera. Did you mean keyboard? I can look
    for cup, bottle, cell phone, remote, book."
    """
    spoken = _norm_name(name) or "that"
    text = f"I can't recognise '{spoken}' with this camera."
    suggestions = list(getattr(resolution, "suggestions", None) or [])
    suggestions = [str(s) for s in suggestions if s and not _same_label(spoken, str(s))][:2]
    if suggestions:
        text += f" Did you mean {' or '.join(suggestions)}?"
    shown = [str(e) for e in list(examples or []) if e][:5]
    if shown:
        text += f" I can look for {', '.join(shown)}."
    return text


def not_seen_answer(name: str, room: str | None, *, resolved_label: str | None = None) -> str:
    """ "I haven't seen your mug yet. Try scanning the area." / "... in the kitchen yet."."""
    spoken = _norm_name(name) or "that"
    tracked = ""
    if resolved_label and not _same_label(spoken, resolved_label):
        tracked = f" (I look for it as {_article(resolved_label)} {resolved_label})"
    if room:
        return f"I haven't seen your {spoken} in the {room_display(room)} yet{tracked}."
    return f"I haven't seen your {spoken} yet{tracked}. Try scanning the area."


def where_am_i_answer(pose: CameraPose | None, *, anchor_name: str = DEFAULT_ANCHOR_NAME) -> str:
    """ "You're in the kitchen, about 2 metres in from the door tag, facing the door tag."."""
    if (
        pose is None
        or getattr(pose, "frame", None) != "room"
        or not getattr(pose, "room", None)
        or getattr(pose, "T_room_cam", None) is None
    ):
        return (
            "I'm not sure which room you're in right now. "
            f"Look toward a {anchor_name} so I can anchor."
        )
    T = np.asarray(pose.T_room_cam, dtype=float).reshape(4, 4)
    x, _, z = (float(v) for v in T[:3, 3])
    depth = round_half_m(max(z, 0.0))
    text = f"You're in the {room_display(pose.room)}"
    text += (
        f", about {format_distance(depth)} in from the {anchor_name}"
        if depth > 0
        else f", right by the {anchor_name}"
    )
    lateral = lateral_phrase(x, "facing_tag")
    if lateral != "straight ahead":
        side = "left" if lateral == "to your left" else "right"
        text += f" and {format_distance(x)} to the {side} of it"
    axes = _ground_axes(T)
    if axes is not None:
        forward, right = axes
        to_tag = np.array([0.0, 0.0, -1.0])
        if float(forward @ to_tag) > math.cos(math.radians(45)):
            text += f", facing the {anchor_name}"
        elif float(forward @ to_tag) < -math.cos(math.radians(45)):
            text += f", with the {anchor_name} behind you"
        else:
            side = "right" if float(right @ to_tag) > 0 else "left"
            text += f", with the {anchor_name} to your {side}"
    if not getattr(pose, "anchored", False):
        text += ", roughly"
    return text + "."
