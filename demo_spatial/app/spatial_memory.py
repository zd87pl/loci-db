"""LOCI-DB spatial memory wrapper for the assistive AI demo.

Provides a high-level API for tracking object positions over time and
answering natural-language location queries ("where did I leave my keys?").

Handheld extension (docs/HANDHELD_DEMO_DESIGN.md, section 4.8 / 7):

* three observation frames share one LOCI client: ``image`` (legacy browser
  mode, ``scene_id == label``), ``room`` (metric room frame, ``"<room>/<label>"``)
  and ``camera`` (metric but unanchored, ``"unanchored/<label>"``);
* a keyed side index per ``scene_id`` drives the cross-frame merge (D10:
  delete + re-insert through ``client.insert`` so Hilbert ids and causal links
  are recomputed) and the exact-recency time window of ``where_is`` (D11);
* the label embedder is pluggable (D12; hash by default, potion-base-8M on the Pi);
* optional crash-safe persistence (D14) through :mod:`app.persistence`;
* every public method takes one re-entrant lock (D17).
"""

from __future__ import annotations

import bisect
import logging
import math
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loci.schema import WorldState

from . import loci_bridge
from .label_embedding import HASH_DIM, LabelEmbedder, hash_label_embedding, make_embedder
from .persistence import (
    MemoryPersistence,
    ObservationRecord,
    RestoreReport,
    copy_record,
)
from .room_anchor import CAMERA_BOX, DEFAULT_ROOM_EXTENTS, RoomMap, room_to_unit, slug

if TYPE_CHECKING:
    from .label_resolver import Resolution

logger = logging.getLogger(__name__)

EMBED_DIM = HASH_DIM
"""Dimension of the legacy hash embedding (kept for compatibility)."""

Vec3 = tuple[float, float, float]

UNANCHORED = "unanchored"
_FRAMES = ("image", "camera", "room")
_MERGE_SCAN = 20
_PRESENCE_GAP_MS = 5000
_GONE_MIN_PRESENCE_MS = 5000
_CHANGE_PRIORITY = {"moved": 0, "new": 1, "gone": 2, "seen": 3}


def _label_embedding(label: str) -> list[float]:
    """Legacy 128-d hash embedding (byte-identical to the pre-T6 function).

    Still exported because ``tests/test_spatial_deduplication.py`` queries the raw
    client with it; it is only a valid query vector while the hash embedder is active.
    """
    return hash_label_embedding(label, EMBED_DIM)


def _normalize(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def _vec3(v: Any) -> Vec3 | None:
    if v is None:
        return None
    try:
        x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return (x, y, z)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(p) - float(q)) ** 2 for p, q in zip(a, b, strict=True)))


def _weighted(a: Vec3 | None, b: Vec3 | None, w0: float, w1: float) -> Vec3 | None:
    if a is None:
        return b
    if b is None:
        return a
    tot = w0 + w1
    if tot <= 0.0:
        w0 = w1 = 0.5
        tot = 1.0
    return tuple((w0 * p + w1 * q) / tot for p, q in zip(a, b, strict=True))  # type: ignore[return-value]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def scene_key(label: str, frame: str, room: str | None) -> str:
    """``scene_id`` layout (D9): image -> label; camera -> ``unanchored/<label>``;
    room -> ``<slug(room)>/<label>``.  ``label`` is already normalised."""
    if frame == "image":
        return label
    if frame == "camera":
        return f"{UNANCHORED}/{label}"
    if frame == "room":
        if not room:
            raise ValueError("room frame requires a room name")
        return f"{slug(room)}/{label}"
    raise ValueError(f"unknown frame {frame!r}")


def split_scene_key(scene_id: str) -> tuple[str | None, str]:
    """Inverse of :func:`scene_key`: ``(room_slug | "unanchored" | None, label)``."""
    prefix, sep, label = scene_id.partition("/")
    if not sep:
        return None, scene_id
    return prefix, label


def _iou_proxy(cx1: float, cy1: float, cx2: float, cy2: float, box_size: float = 0.1) -> float:
    """Compute IoU between two proxy bounding boxes of fixed size centered at (cx, cy)."""
    half = box_size / 2
    ax1, ay1, ax2, ay2 = cx1 - half, cy1 - half, cx1 + half, cy1 + half
    bx1, by1, bx2, by2 = cx2 - half, cy2 - half, cx2 + half, cy2 + half
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    box_area = box_size * box_size
    union_area = 2 * box_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _height_phrase(y_m: float) -> str:
    if y_m < 0.3:
        return "on the floor"
    if y_m < 0.6:
        return "low down"
    if y_m <= 1.1:
        return "at table height"
    if y_m <= 1.5:
        return "at chest height"
    return "up high"


def _fallback_describe_position(
    frame: str, pos_m: Vec3 | None, xyz_cam_m: Vec3 | None, room: str | None
) -> str:
    """Minimal metric wording used only when :mod:`app.phrasing` (T10) is unavailable."""
    if frame == "room" and pos_m is not None:
        x, y, z = pos_m
        where = f"about {z:.1f} m into the {room or 'room'}"
        if abs(x) >= 0.5:
            side = "left" if x > 0 else "right"
            where += f" and {abs(x):.1f} m to your {side}"
        return f"{where} as you walk in from the door tag, {_height_phrase(y)}"
    if xyz_cam_m is not None:
        x, y, z = xyz_cam_m
        where = f"about {z:.1f} m in front of the camera"
        if abs(x) >= 0.3:
            where += ", slightly right" if x > 0 else ", slightly left"
        if y < -0.3:
            where += ", above eye line"
        elif y > 0.3:
            where += ", below eye line"
        return where
    return "somewhere nearby"


@dataclass
class ObjectObservation:
    """A single tracked observation of a physical object."""

    label: str
    cx: float  # normalized [0, 1] — horizontal position in camera frame
    cy: float  # normalized [0, 1] — vertical position in camera frame
    confidence: float
    timestamp_ms: int
    state_id: str
    depth_m: float | None = None  # depth in meters (None if 2D only)
    room: str | None = None
    frame: str = "image"  # "image" | "camera" | "room"
    pos_m: Vec3 | None = None  # room frame, metres
    xyz_cam_m: Vec3 | None = None  # camera frame, metres
    anchored: bool = False
    anchor_confidence: float = 0.0
    source: str = "camera"
    first_seen_ms: int | None = None
    merge_count: int = 0
    restored: bool = False  # inserted by restore() from disk (before this process started)
    scene_id: str = ""

    @property
    def age_seconds(self) -> float:
        return (int(time.time() * 1000) - self.timestamp_ms) / 1000.0

    @property
    def is_metric(self) -> bool:
        return self.frame != "image"

    def to_dict(self) -> dict:
        from datetime import datetime

        dt = datetime.fromtimestamp(self.timestamp_ms / 1000)
        pos = self.pos_m
        result = {
            "label": self.label,
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "confidence": round(self.confidence, 3),
            "timestamp_ms": self.timestamp_ms,
            "age_seconds": round(self.age_seconds, 1),
            "last_seen_at": dt.strftime("%H:%M:%S"),
            "last_seen_time": dt.strftime("%I:%M %p").lstrip("0"),
            "position_description": self.position_description,
            "state_id": self.state_id,
        }
        if self.depth_m is not None:
            result["depth_m"] = round(self.depth_m, 3)
        result.update(
            {
                "room": self.room,
                "frame": self.frame,
                "x_m": round(pos[0], 3) if pos is not None else None,
                "y_m": round(pos[1], 3) if pos is not None else None,
                "z_m": round(pos[2], 3) if pos is not None else None,
                "xyz_cam_m": [round(v, 3) for v in self.xyz_cam_m] if self.xyz_cam_m else None,
                "anchored": bool(self.anchored),
                "anchor_confidence": round(float(self.anchor_confidence), 3),
                "source": self.source,
                "first_seen_ms": self.first_seen_ms,
                "merge_count": int(self.merge_count),
                "restored": bool(self.restored),
                "scene_id": self.scene_id,
            }
        )
        return result

    @property
    def position_description(self) -> str:
        """Human-readable spatial description (legacy wording for image-frame sightings)."""
        if self.frame == "image":
            if self.cx < 0.33:
                h = "on the left"
            elif self.cx > 0.67:
                h = "on the right"
            else:
                h = "in the center"
            if self.cy < 0.35:
                v = "toward the back"
            elif self.cy > 0.70:
                v = "near the front"
            else:
                v = "in the middle area"
            return f"{h}, {v}"
        try:
            from .phrasing import describe_position  # T10, optional at runtime
        except ImportError:
            return _fallback_describe_position(self.frame, self.pos_m, self.xyz_cam_m, self.room)
        try:
            return str(
                describe_position(
                    self.frame,
                    self.pos_m,
                    self.xyz_cam_m,
                    self.cx,
                    self.cy,
                    room=self.room,
                )
            )
        except Exception as exc:  # phrasing must never break a query
            logger.debug("phrasing.describe_position failed: %s", exc)
            return _fallback_describe_position(self.frame, self.pos_m, self.xyz_cam_m, self.room)


@dataclass
class Change:
    """One entry of :meth:`SpatialMemory.changes` (section 7.5)."""

    kind: str  # "new" | "moved" | "gone" | "seen"
    label: str
    room: str | None
    last_seen_ms: int
    first_seen_ms: int | None
    pos_m: Vec3 | None
    pos_before_m: Vec3 | None
    delta_m: float | None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "room": self.room,
            "last_seen_ms": self.last_seen_ms,
            "first_seen_ms": self.first_seen_ms,
            "pos_m": [round(v, 3) for v in self.pos_m] if self.pos_m else None,
            "pos_before_m": [round(v, 3) for v in self.pos_before_m] if self.pos_before_m else None,
            "delta_m": round(self.delta_m, 3) if self.delta_m is not None else None,
        }


@dataclass
class _Slot:
    """Side-index entry mirroring one live point (section 7.2)."""

    state_id: str
    timestamp_ms: int
    confidence: float
    x: float
    y: float
    z: float
    pos_m: Vec3 | None
    xyz_cam_m: Vec3 | None
    cx: float
    cy: float
    anchored: bool
    first_seen_ms: int
    merge_count: int = 0


@dataclass
class _SceneMeta:
    label: str
    frame: str
    room: str | None
    room_slug: str | None = None


@dataclass
class _Pending:
    """Everything observe() needs to insert one point (new or merged)."""

    label: str
    scene_id: str
    frame: str
    room: str | None
    last_room: str | None
    cx: float
    cy: float
    confidence: float
    timestamp_ms: int
    depth_m: float | None
    pos_m: Vec3 | None
    xyz_cam_m: Vec3 | None
    anchored: bool
    anchor_confidence: float
    source: str
    first_seen_ms: int
    merge_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class SpatialMemory:
    """Thread-safe wrapper around LocalLociClient for object tracking.

    Uses LOCI-DB's spatial + temporal indexing to answer questions like:
      - "Where is my water bottle?" (latest known position)
      - "Where was my phone last Tuesday?" (temporal history)
      - "What objects are in the left side of the room?" (spatial range query)
      - "What changed in the kitchen?" (new / moved / gone per room)
    """

    def __init__(
        self,
        epoch_size_ms: int = 5000,
        label_confidence_overrides: dict[str, float] | None = None,
        dedup_iou_threshold: float = 0.5,
        dedup_window_ms: int = 5000,
        *,
        embedder: LabelEmbedder | None = None,
        rooms: RoomMap | None = None,
        dedup_radius_m: float = 0.35,
        moved_threshold_m: float = 0.5,
        persistence: MemoryPersistence | None = None,
    ) -> None:
        self._embedder: LabelEmbedder = embedder if embedder is not None else make_embedder()
        self._rooms: RoomMap = rooms if rooms is not None else RoomMap.default()
        self._epoch_size_ms = int(epoch_size_ms)
        self._client = loci_bridge.make_client(self._epoch_size_ms, int(self._embedder.dim))
        self._persistence = persistence
        self._lock = threading.RLock()

        self.started_at_ms = int(time.time() * 1000)
        self.version = 0

        # Per-label confidence threshold overrides (label -> min_confidence)
        self._label_confidence_overrides: dict[str, float] = {
            _normalize(k): float(v) for k, v in (label_confidence_overrides or {}).items()
        }
        self.dedup_iou_threshold = float(dedup_iou_threshold)
        self.dedup_window_ms = int(dedup_window_ms)
        self.dedup_radius_m = float(dedup_radius_m)
        self.moved_threshold_m = float(moved_threshold_m)

        self._observation_count = 0
        self._session_started = False
        self._restore_report: RestoreReport | None = None
        self._vocab: list[str] = []
        self._resolver: Any = None
        self._reset_indices()

    # ------------------------------------------------------------------
    # Index bookkeeping
    # ------------------------------------------------------------------

    def _reset_indices(self) -> None:
        self._scene_index: dict[str, list[_Slot]] = {}
        self._scene_meta: dict[str, _SceneMeta] = {}
        self._records: dict[str, ObservationRecord] = {}
        self._latest: dict[str, ObjectObservation] = {}
        self._latest_by_scene: dict[str, ObjectObservation] = {}
        self._room_names: dict[str, str] = {}  # slug -> display name
        self._presence: dict[str, list[list[int]]] = {}
        self._restored_ids: set[str] = set()

    def _slot_from_record(self, rec: ObservationRecord) -> _Slot:
        meta = rec.get("metadata") or {}
        return _Slot(
            state_id=rec["id"],
            timestamp_ms=int(rec["timestamp_ms"]),
            confidence=float(rec["confidence"]),
            x=float(rec["x"]),
            y=float(rec["y"]),
            z=float(rec["z"]),
            pos_m=_vec3(meta.get("pos_m")),
            xyz_cam_m=_vec3(meta.get("xyz_cam_m")),
            cx=float(meta.get("cx", rec["x"])),
            cy=float(meta.get("cy", rec["y"])),
            anchored=bool(meta.get("anchored", False)),
            first_seen_ms=int(meta.get("first_seen_ms") or rec["timestamp_ms"]),
            merge_count=int(meta.get("merge_count", 0) or 0),
        )

    def _obs_from_record(self, rec: ObservationRecord) -> ObjectObservation:
        meta = rec.get("metadata") or {}
        depth = meta.get("depth_m")
        first_seen = meta.get("first_seen_ms")
        return ObjectObservation(
            label=rec["label"],
            cx=float(meta.get("cx", rec["x"])),
            cy=float(meta.get("cy", rec["y"])),
            confidence=float(rec["confidence"]),
            timestamp_ms=int(rec["timestamp_ms"]),
            state_id=rec["id"],
            depth_m=float(depth) if depth is not None else None,
            room=rec.get("room"),
            frame=rec.get("frame", "image"),
            pos_m=_vec3(meta.get("pos_m")),
            xyz_cam_m=_vec3(meta.get("xyz_cam_m")),
            anchored=bool(meta.get("anchored", False)),
            anchor_confidence=float(meta.get("anchor_confidence", 0.0) or 0.0),
            source=str(meta.get("source", "camera")),
            first_seen_ms=int(first_seen) if first_seen is not None else None,
            merge_count=int(meta.get("merge_count", 0) or 0),
            restored=rec["id"] in self._restored_ids,
            scene_id=rec["scene_id"],
        )

    def _obs_from_state(self, state: WorldState) -> ObjectObservation:
        rec = self._records.get(state.id)
        if rec is not None:
            return self._obs_from_record(rec)
        meta = state.metadata or {}
        prefix, parsed_label = split_scene_key(state.scene_id)
        frame = str(meta.get("frame") or ("image" if prefix is None else "camera"))
        room = meta.get("room")
        if room is None and prefix not in (None, UNANCHORED):
            room = self._room_names.get(prefix, prefix)
        synthetic: ObservationRecord = {
            "id": state.id,
            "scene_id": state.scene_id,
            "label": str(meta.get("label") or parsed_label or "unknown"),
            "room": room if frame == "room" else None,
            "frame": frame if frame in _FRAMES else "image",
            "x": state.x,
            "y": state.y,
            "z": state.z,
            "timestamp_ms": state.timestamp_ms,
            "confidence": state.confidence,
            "prev_state_id": state.prev_state_id,
            "metadata": dict(meta),
        }
        return self._obs_from_record(synthetic)

    def _add_record(self, rec: ObservationRecord) -> ObjectObservation:
        scene_id = rec["scene_id"]
        slot = self._slot_from_record(rec)
        slots = self._scene_index.setdefault(scene_id, [])
        bisect.insort(slots, slot, key=lambda s: s.timestamp_ms)
        self._records[rec["id"]] = rec
        room = rec.get("room")
        room_slug = slug(room) if room else None
        if scene_id not in self._scene_meta:
            self._scene_meta[scene_id] = _SceneMeta(rec["label"], rec["frame"], room, room_slug)
        if room and room_slug:
            self._room_names.setdefault(room_slug, room)
        obs = self._obs_from_record(rec)
        prev_scene = self._latest_by_scene.get(scene_id)
        if prev_scene is None or obs.timestamp_ms >= prev_scene.timestamp_ms:
            self._latest_by_scene[scene_id] = obs
        prev_label = self._latest.get(rec["label"])
        if prev_label is None or obs.timestamp_ms >= prev_label.timestamp_ms:
            self._latest[rec["label"]] = obs
        return obs

    def _remove_record(self, state_id: str) -> ObservationRecord | None:
        rec = self._records.pop(state_id, None)
        if rec is None:
            return None
        scene_id = rec["scene_id"]
        slots = self._scene_index.get(scene_id)
        if slots:
            for i in range(len(slots) - 1, -1, -1):
                if slots[i].state_id == state_id:
                    del slots[i]
                    break
            if not slots:
                del self._scene_index[scene_id]
        self._restored_ids.discard(state_id)
        latest_scene = self._latest_by_scene.get(scene_id)
        if latest_scene is not None and latest_scene.state_id == state_id:
            if slots:
                self._latest_by_scene[scene_id] = self._obs_from_record(
                    self._records[slots[-1].state_id]
                )
            else:
                del self._latest_by_scene[scene_id]
        label = rec["label"]
        latest_label = self._latest.get(label)
        if latest_label is not None and latest_label.state_id == state_id:
            candidates = [
                o
                for sid, o in self._latest_by_scene.items()
                if self._scene_meta.get(sid) and self._scene_meta[sid].label == label
            ]
            if candidates:
                self._latest[label] = max(candidates, key=lambda o: o.timestamp_ms)
            else:
                del self._latest[label]
        return rec

    def _extents(self, room: str | None) -> dict[str, tuple[float, float]]:
        spec = self._rooms.get(room) if room else None
        return dict(spec.extents_m) if spec is not None else dict(DEFAULT_ROOM_EXTENTS)

    def _unit_coords(
        self,
        frame: str,
        room: str | None,
        pos_m: Vec3 | None,
        xyz_cam_m: Vec3 | None,
        cx: float,
        cy: float,
        depth_m: float | None,
    ) -> tuple[float, float, float, bool]:
        if frame == "room" and pos_m is not None:
            (x, y, z), clipped = room_to_unit(pos_m, self._extents(room))
            return x, y, z, clipped
        if frame == "camera" and xyz_cam_m is not None:
            (x, y, z), clipped = room_to_unit(xyz_cam_m, CAMERA_BOX)
            return x, y, z, clipped
        z = _clamp01(depth_m / 5.0) if depth_m is not None else 0.0
        return _clamp01(cx), _clamp01(cy), z, False

    def _embed(self, label: str) -> list[float]:
        return list(self._embedder.embed(label))

    def _prev_state_id(self, scene_id: str, before_ms: int) -> str | None:
        slots = self._scene_index.get(scene_id)
        if not slots:
            return None
        for s in reversed(slots):
            if s.timestamp_ms < before_ms:
                return s.state_id
        return None

    def _insert_pending(self, p: _Pending) -> str:
        x, y, z, clipped = self._unit_coords(
            p.frame, p.room, p.pos_m, p.xyz_cam_m, p.cx, p.cy, p.depth_m
        )
        metadata: dict[str, Any] = {
            "label": p.label,
            "room": p.room,
            "frame": p.frame,
            "pos_m": list(p.pos_m) if p.pos_m is not None else None,
            "xyz_cam_m": list(p.xyz_cam_m) if p.xyz_cam_m is not None else None,
            "cx": p.cx,
            "cy": p.cy,
            "depth_m": p.depth_m,
            "anchored": bool(p.anchored),
            "anchor_confidence": float(p.anchor_confidence),
            "source": p.source,
            "first_seen_ms": int(p.first_seen_ms),
            "merge_count": int(p.merge_count),
            "clipped": bool(clipped),
            "embedder": str(self._embedder.name),
        }
        if p.last_room:
            metadata["last_room"] = p.last_room
        metadata.update(p.extra)
        prev_id = self._prev_state_id(p.scene_id, p.timestamp_ms)
        state = WorldState(
            x=x,
            y=y,
            z=z,
            timestamp_ms=p.timestamp_ms,
            vector=self._embed(p.label),
            scene_id=p.scene_id,
            confidence=p.confidence,
            metadata=metadata,
        )
        state_id = self._client.insert(state)
        # The sighting keeps its first id for life: merges and restores re-id the record,
        # but carry this, so mirrors such as cloud_sync see one stable object.
        metadata.setdefault("sighting_id", state_id)
        rec: ObservationRecord = {
            "id": state_id,
            "scene_id": p.scene_id,
            "label": p.label,
            "room": p.room,
            "frame": p.frame,
            "x": x,
            "y": y,
            "z": z,
            "timestamp_ms": p.timestamp_ms,
            "confidence": p.confidence,
            "prev_state_id": prev_id,
            "metadata": metadata,
        }
        self._add_record(rec)
        if self._persistence is not None:
            self._persistence.append_upsert(rec)
        self.version += 1
        return state_id

    def _ensure_session(self) -> None:
        """Restore from disk before the first write when nobody called ``restore()``."""
        if self._persistence is not None and not self._session_started:
            self.restore()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def observe(
        self,
        label: str,
        cx: float,
        cy: float,
        confidence: float = 1.0,
        timestamp_ms: int | None = None,
        depth_m: float | None = None,
        min_confidence: float = 0.55,
        *,
        xyz_cam_m: Vec3 | None = None,
        pos_room_m: Vec3 | None = None,
        room: str | None = None,
        anchored: bool = False,
        anchor_confidence: float = 0.0,
        source: str = "camera",
    ) -> str | None:
        """Record an object sighting in spatial memory. Returns the state_id.

        Returns ``None`` without storing if ``confidence`` is below the effective
        threshold for this label (``label_confidence_overrides`` takes precedence
        over ``min_confidence``).

        Frame selection: ``room`` when ``pos_room_m`` and ``room`` are given,
        ``camera`` when only ``xyz_cam_m`` is known, else the legacy ``image`` frame.
        """
        norm = _normalize(label)
        if not norm:
            return None
        effective_min = self._label_confidence_overrides.get(norm, min_confidence)
        if confidence < effective_min:
            return None

        ts = int(timestamp_ms) if timestamp_ms is not None else int(time.time() * 1000)
        ts = max(0, ts)
        cx = _clamp01(cx)
        cy = _clamp01(cy)
        confidence = _clamp01(confidence)
        pos = _vec3(pos_room_m)
        cam = _vec3(xyz_cam_m)
        room_name = str(room).strip() if room else None

        if pos is not None and room_name:
            frame = "room"
        elif cam is not None:
            frame = "camera"
        else:
            frame = "image"
        if frame != "room":
            pos = None
        if depth_m is None and cam is not None:
            depth_m = cam[2]
        depth = float(depth_m) if depth_m is not None else None
        sid = scene_key(norm, frame, room_name)

        pending = _Pending(
            label=norm,
            scene_id=sid,
            frame=frame,
            room=room_name if frame == "room" else None,
            last_room=room_name if frame != "room" else None,
            cx=cx,
            cy=cy,
            confidence=confidence,
            timestamp_ms=ts,
            depth_m=depth,
            pos_m=pos,
            xyz_cam_m=cam,
            anchored=bool(anchored) if frame != "image" else False,
            anchor_confidence=_clamp01(anchor_confidence) if frame != "image" else 0.0,
            source=str(source or "camera"),
            first_seen_ms=ts,
        )

        with self._lock:
            self._ensure_session()
            merged_id = self._try_merge(pending)
            if merged_id is not None:
                self._observation_count += 1
                return merged_id
            state_id = self._insert_pending(pending)
            self._observation_count += 1
            return state_id

    def _try_merge(self, p: _Pending) -> str | None:
        """Keyed cross-frame merge (section 7.3).

        Looks only at the side index of ``p.scene_id`` (never a vector query):
        same ``anchored`` flag, inside ``dedup_window_ms``, metric distance
        <= ``dedup_radius_m`` (room / camera frames) or proxy-IoU >
        ``dedup_iou_threshold`` (image frame).  The matched point is deleted and
        the merged observation re-inserted through ``client.insert`` so Hilbert
        ids, epoch routing and causal links are recomputed; returns the NEW id.
        """
        slots = self._scene_index.get(p.scene_id)
        if not slots:
            return None
        cutoff = p.timestamp_ms - self.dedup_window_ms
        scanned = 0
        for s in reversed(slots):
            if s.timestamp_ms < cutoff:
                break
            scanned += 1
            if scanned > _MERGE_SCAN:
                break
            if s.anchored != p.anchored:
                continue
            if p.frame == "image":
                if _iou_proxy(p.cx, p.cy, s.cx, s.cy) <= self.dedup_iou_threshold:
                    continue
            else:
                new_p = p.pos_m if p.frame == "room" else p.xyz_cam_m
                old_p = s.pos_m if p.frame == "room" else s.xyz_cam_m
                if new_p is None or old_p is None or _dist(new_p, old_p) > self.dedup_radius_m:
                    continue

            w0, w1 = s.confidence, p.confidence
            tot = w0 + w1
            if tot <= 0.0:
                w0 = w1 = 0.5
                tot = 1.0
            old_rec = self._records.get(s.state_id)
            old_meta = (old_rec or {}).get("metadata") or {}
            if not loci_bridge.delete_point(self._client, s.state_id):
                logger.warning("merge candidate %s vanished from the store", s.state_id)
                self._remove_record(s.state_id)
                continue
            self._remove_record(s.state_id)
            if self._persistence is not None:
                self._persistence.append_delete(s.state_id)

            merged = _Pending(
                label=p.label,
                scene_id=p.scene_id,
                frame=p.frame,
                room=p.room,
                last_room=p.last_room or old_meta.get("last_room"),
                cx=(w0 * s.cx + w1 * p.cx) / tot,
                cy=(w0 * s.cy + w1 * p.cy) / tot,
                confidence=max(w0, w1),
                timestamp_ms=max(p.timestamp_ms, s.timestamp_ms),
                depth_m=p.depth_m if p.depth_m is not None else old_meta.get("depth_m"),
                pos_m=_weighted(s.pos_m, p.pos_m, w0, w1),
                xyz_cam_m=_weighted(s.xyz_cam_m, p.xyz_cam_m, w0, w1),
                anchored=p.anchored,
                anchor_confidence=max(
                    p.anchor_confidence, float(old_meta.get("anchor_confidence", 0.0) or 0.0)
                ),
                source=p.source,
                first_seen_ms=s.first_seen_ms,
                merge_count=s.merge_count + 1,
                # a record without the field is known to mirrors by its own id
                extra={"sighting_id": old_meta.get("sighting_id") or s.state_id},
            )
            return self._insert_pending(merged)
        return None

    def note_presence(self, room: str, timestamp_ms: int) -> None:
        """Runtime hook (<= 1 Hz): the camera is anchored in ``room`` right now.

        Feeds the in-memory presence log that gates ``gone`` in :meth:`changes` (D24).
        """
        if not room:
            return
        key = slug(room)
        ts = int(timestamp_ms)
        with self._lock:
            self._room_names.setdefault(key, str(room).strip())
            intervals = self._presence.setdefault(key, [])
            if intervals and ts - intervals[-1][1] < _PRESENCE_GAP_MS and ts >= intervals[-1][0]:
                intervals[-1][1] = max(intervals[-1][1], ts)
            else:
                intervals.append([ts, ts])
                if len(intervals) > 2000:
                    del intervals[: len(intervals) - 2000]

    def _presence_ms(self, room_slug: str, lo: int, hi: int) -> int:
        if hi <= lo:
            return 0
        total = 0
        for start, end in self._presence.get(room_slug, ()):
            a, b = max(start, lo), min(end, hi)
            if b > a:
                total += b - a
        return total

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _scene_ids_for(self, label: str, room: str | None) -> list[str]:
        if room:
            ids = [f"{slug(room)}/{label}"]
        else:
            ids = [label]
            ids.extend(f"{rs}/{label}" for rs in sorted(self._room_names))
            ids.append(f"{UNANCHORED}/{label}")
        return [sid for sid in ids if self._scene_index.get(sid)]

    def where_is(
        self,
        label: str,
        limit: int = 5,
        min_confidence: float | None = None,
        *,
        room: str | None = None,
    ) -> list[ObjectObservation]:
        """Recent observations of an exact label, newest first (never resolves labels).

        The side index picks the time window of the ``limit`` newest sightings across
        the label's scene ids, so the LOCI query (exact ``scene_id`` MatchAny + window)
        returns exactly those points however old they are and however many points of
        the label exist (D11; MemoryStore truncates ties before decay re-ranking).

        Args:
            min_confidence: If provided, only return observations with
                confidence >= this value.
            room: restrict to room-frame sightings in that room.
        """
        norm = _normalize(label)
        limit = max(1, int(limit))
        with self._lock:
            scene_ids = self._scene_ids_for(norm, room)
            if not scene_ids:
                return []
            slots = [
                s
                for sid in scene_ids
                for s in self._scene_index.get(sid, ())
                if min_confidence is None or s.confidence >= min_confidence
            ]
            if not slots:
                return []
            slots.sort(key=lambda s: s.timestamp_ms, reverse=True)
            top = slots[:limit]
            window = (top[-1].timestamp_ms, top[0].timestamp_ms)
            results = self._client.query(
                vector=self._embed(norm),
                time_window_ms=window,
                limit=limit * 4,
                _extra_payload_filter={"scene_id": scene_ids},
                min_confidence=min_confidence,
            )
            observations = [self._obs_from_state(r) for r in results]
        observations.sort(key=lambda o: o.timestamp_ms, reverse=True)
        return observations[:limit]

    def history(
        self, label: str, limit: int = 20, *, room: str | None = None
    ) -> list[ObjectObservation]:
        """Return full temporal history of an object's positions."""
        return self.where_is(label, limit=limit, room=room)

    def objects_in_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        limit: int = 20,
        *,
        room: str | None = None,
    ) -> list[ObjectObservation]:
        """Find all objects observed in a spatial region (normalized [0,1] coords).

        With ``room`` the box is interpreted inside that room's scene ids only.
        One result per label, newest first.
        """
        query_vec = [0.0] * int(self._embedder.dim)  # neutral vector — spatial filter dominates
        spatial_bounds = {
            "x_min": max(0.0, x_min),
            "x_max": min(1.0, x_max),
            "y_min": max(0.0, y_min),
            "y_max": min(1.0, y_max),
            "z_min": 0.0,
            "z_max": 1.0,
        }
        with self._lock:
            extra: dict[str, Any] | None = None
            if room:
                prefix = f"{slug(room)}/"
                scene_ids = [
                    sid
                    for sid, slots in self._scene_index.items()
                    if slots and sid.startswith(prefix)
                ]
                if not scene_ids:
                    return []
                extra = {"scene_id": scene_ids}
            results = self._client.query(
                vector=query_vec,
                spatial_bounds=spatial_bounds,
                limit=max(200, int(limit)),
                _extra_payload_filter=extra,
            )
            observations = [self._obs_from_state(r) for r in results]
        observations.sort(key=lambda o: o.timestamp_ms, reverse=True)
        seen_labels: set[str] = set()
        out: list[ObjectObservation] = []
        for o in observations:
            if o.label in seen_labels:
                continue
            seen_labels.add(o.label)
            out.append(o)
            if len(out) >= limit:
                break
        return out

    def current_objects(self, room: str | None = None) -> list[ObjectObservation]:
        """Most recently observed position of every tracked object (optionally per room)."""
        with self._lock:
            if not room:
                values = list(self._latest.values())
            else:
                prefix = f"{slug(room)}/"
                per_label: dict[str, ObjectObservation] = {}
                for sid, obs in self._latest_by_scene.items():
                    if not sid.startswith(prefix):
                        continue
                    cur = per_label.get(obs.label)
                    if cur is None or obs.timestamp_ms > cur.timestamp_ms:
                        per_label[obs.label] = obs
                values = list(per_label.values())
        return sorted(values, key=lambda o: o.timestamp_ms, reverse=True)

    def recent_changes(
        self, window_seconds: float = 30.0, room: str | None = None
    ) -> list[ObjectObservation]:
        """Objects observed in the last N seconds."""
        cutoff_ms = int(time.time() * 1000) - int(window_seconds * 1000)
        return [o for o in self.current_objects(room) if o.timestamp_ms >= cutoff_ms]

    def changes(
        self,
        room: str | None,
        window_seconds: float = 600.0,
        *,
        now_ms: int | None = None,
    ) -> list[Change]:
        """New / moved / gone / seen objects inside the window (section 7.5, D24)."""
        now = int(now_ms) if now_ms is not None else int(time.time() * 1000)
        since = now - int(window_seconds * 1000)
        out: list[Change] = []
        with self._lock:
            if room:
                prefix = f"{slug(room)}/"
                scenes = [sid for sid in self._scene_index if sid.startswith(prefix)]
            else:
                scenes = list(self._scene_index)
            for sid in scenes:
                slots = self._scene_index.get(sid)
                meta = self._scene_meta.get(sid)
                if not slots or meta is None:
                    continue
                latest = slots[-1]
                first_seen = min(s.first_seen_ms for s in slots)
                pos_before: Vec3 | None = None
                delta: float | None = None
                if latest.timestamp_ms < since:
                    if meta.room_slug is None:
                        continue
                    lo = max(since, latest.timestamp_ms)
                    if self._presence_ms(meta.room_slug, lo, now) < _GONE_MIN_PRESENCE_MS:
                        continue
                    kind = "gone"
                else:
                    # "moved" is checked before "new": a scene whose first sighting
                    # is itself inside the window (every fresh demo run) still has to
                    # report a displacement, as the spec's acceptance test and demo
                    # story (sections 13 and 14) expect; section 7.5's pseudo-code
                    # ordering would call such a cup "new" for its first 10 minutes.
                    older = [s for s in slots[:-1] if s.timestamp_ms < latest.timestamp_ms]
                    kind = "new" if first_seen >= since else "seen"
                    if older and latest.first_seen_ms >= since and latest.pos_m is not None:
                        # Compare against the newest earlier slot that lies farther than
                        # the threshold, not only the immediately previous one: the first
                        # consensus fire after re-entering a room comes from a single,
                        # not-yet-smoothed tag pose and lands between the old and the new
                        # place, which would otherwise mask a metre-scale move.
                        for prev in reversed(older):
                            if prev.pos_m is None:
                                continue
                            d = _dist(prev.pos_m, latest.pos_m)
                            if d > self.moved_threshold_m:
                                kind = "moved"
                                pos_before = prev.pos_m
                                delta = d
                                break
                out.append(
                    Change(
                        kind=kind,
                        label=meta.label,
                        room=meta.room,
                        last_seen_ms=latest.timestamp_ms,
                        first_seen_ms=first_seen,
                        pos_m=latest.pos_m,
                        pos_before_m=pos_before,
                        delta_m=delta,
                    )
                )
        out.sort(key=lambda c: (_CHANGE_PRIORITY.get(c.kind, 9), -c.last_seen_ms))
        return out

    # ------------------------------------------------------------------
    # Label resolution (voice / API layer helper, D13)
    # ------------------------------------------------------------------

    def set_vocabulary(self, labels: Iterable[str]) -> None:
        """Detector classes the resolver may answer with (union with tracked labels)."""
        with self._lock:
            self._vocab = list(dict.fromkeys(_normalize(x) for x in labels if _normalize(x)))

    def resolve_label(self, text: str) -> Resolution:
        """Resolve a spoken noun onto detector classes ∪ tracked labels."""
        from .label_resolver import LabelResolver

        with self._lock:
            vocab = list(dict.fromkeys([*self._vocab, *self._latest.keys()]))
            if self._resolver is None:
                self._resolver = LabelResolver(vocab, self._embedder)
            else:
                self._resolver.set_vocabulary(vocab)
            return self._resolver.resolve(text)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def tracked_labels(self) -> list[str]:
        with self._lock:
            return list(self._latest.keys())

    @property
    def rooms_seen(self) -> list[str]:
        with self._lock:
            live = {
                m.room_slug for sid, m in self._scene_meta.items() if self._scene_index.get(sid)
            }
            return [self._room_names[s] for s in sorted(self._room_names) if s in live]

    @property
    def rooms(self) -> RoomMap:
        return self._rooms

    @property
    def embedder(self) -> LabelEmbedder:
        return self._embedder

    @property
    def persistence(self) -> MemoryPersistence | None:
        return self._persistence

    @property
    def restore_report(self) -> RestoreReport | None:
        return self._restore_report

    def stats(self) -> dict:
        with self._lock:
            qs = self._client.last_query_stats
            frames = {"image": 0, "camera": 0, "room": 0}
            for rec in self._records.values():
                frames[rec.get("frame", "image")] = frames.get(rec.get("frame", "image"), 0) + 1
            return {
                "observation_count": self._observation_count,
                "tracked_objects": len(self._latest),
                "labels": list(self._latest.keys()),
                "last_query": {
                    "shards_searched": qs.shards_searched if qs else 0,
                    "elapsed_ms": round(qs.elapsed_ms, 2) if qs else 0,
                },
                "rooms_seen": self.rooms_seen,
                "frames": frames,
                "records": len(self._records),
                "points": loci_bridge.point_count(self._client),
                "persistence": self._persistence.status() if self._persistence else None,
                "embedder": str(self._embedder.name),
                "embedding_dim": int(self._embedder.dim),
                "version": self.version,
                "started_at_ms": self.started_at_ms,
            }

    # ------------------------------------------------------------------
    # Lifecycle / persistence
    # ------------------------------------------------------------------

    def _snapshot_header(self) -> dict:
        return {
            "format": 2,
            "embedder": {"name": str(self._embedder.name), "dim": int(self._embedder.dim)},
            "epoch_size_ms": self._epoch_size_ms,
            "observation_count": self._observation_count,
            "started_at_ms": self.started_at_ms,
            "version": self.version,
        }

    def _snapshot_payload(self) -> tuple[dict, list[ObservationRecord]]:
        with self._lock:
            return self._snapshot_header(), self.export_records()

    def export_records(self) -> list[ObservationRecord]:
        """Copy of every live record (source of truth for snapshots)."""
        with self._lock:
            return [copy_record(r) for r in self._records.values()]

    def restore(self) -> RestoreReport:
        """Load the persisted records, insert them through ``insert_batch`` and start
        the persistence session (synchronous snapshot + journal truncation).

        Idempotent: a second call returns the first report.  Without persistence a
        ``disabled`` report is returned and nothing happens.
        """
        if self._persistence is None:
            report = RestoreReport(0, [], None, 0, "disabled")
            self._restore_report = report
            return report
        with self._lock:
            if self._session_started and self._restore_report is not None:
                return self._restore_report
            self._session_started = True
            try:
                records, header, report = self._persistence.load()
                if records:
                    self._insert_restored(records)
                    prev_count = (
                        header.get("observation_count") if isinstance(header, dict) else None
                    )
                    if isinstance(prev_count, int) and prev_count > 0:
                        self._observation_count += prev_count
                    else:
                        self._observation_count += len(records)
            except Exception as exc:
                # A failed restore must not take persistence down for the session: drop
                # any half-inserted points, report it, and still start the writer below.
                logger.exception("restore failed; starting with an empty memory: %s", exc)
                self._client = loci_bridge.make_client(self._epoch_size_ms, int(self._embedder.dim))
                self._reset_indices()
                detail = f"{type(exc).__name__}: {exc}"
                # the empty session below must not snapshot over the files it could not load
                try:
                    kept = self._persistence.set_aside(detail)
                except Exception as aside_exc:
                    kept = []
                    logger.exception("could not set the memory files aside: %s", aside_exc)
                if kept:
                    detail = f"{detail}; kept {', '.join(kept)}"
                report = RestoreReport(0, [], None, 0, "error", detail=detail)
            self._restore_report = report
            self.version += 1
            try:
                self._persistence.begin_session(
                    self._snapshot_header(), self.export_records(), provider=self._snapshot_payload
                )
            except Exception as exc:
                logger.exception("persistence session did not start: %s", exc)
                sep = "; " if report.detail else ""
                report.detail = f"{report.detail}{sep}persistence: {type(exc).__name__}: {exc}"
        if report.records:
            logger.info(
                "restored %d sightings in %d room(s) from %s",
                report.records,
                len(report.rooms),
                report.source,
            )
        return report

    def _insert_restored(self, records: list[ObservationRecord]) -> None:
        ordered = sorted(records, key=lambda r: (r["scene_id"], int(r["timestamp_ms"]), r["id"]))
        labels = list(dict.fromkeys(r["label"] for r in ordered))
        vectors = dict(zip(labels, self._embedder.embed_many(labels), strict=True))
        states = [
            WorldState(
                x=float(r["x"]),
                y=float(r["y"]),
                z=float(r["z"]),
                timestamp_ms=int(r["timestamp_ms"]),
                vector=list(vectors[r["label"]]),
                scene_id=r["scene_id"],
                confidence=float(r["confidence"]),
                metadata=dict(r.get("metadata") or {}),
            )
            for r in ordered
        ]
        new_ids = self._client.insert_batch(states)
        chain: dict[str, str] = {}
        for old, new_id in zip(ordered, new_ids, strict=True):
            rec = copy_record(old)
            rec["id"] = new_id
            rec["prev_state_id"] = chain.get(rec["scene_id"])
            chain[rec["scene_id"]] = new_id
            rec["metadata"]["embedder"] = str(self._embedder.name)
            # records saved before sighting ids existed keep the id mirrors already know
            rec["metadata"].setdefault("sighting_id", old["id"])
            self._restored_ids.add(new_id)
            self._add_record(rec)

    def flush(self, timeout_s: float = 10.0) -> None:
        """Synchronous snapshot (no-op without persistence)."""
        if self._persistence is None:
            return
        with self._lock:
            self._ensure_session()
        self._persistence.request_snapshot(wait=True, timeout_s=timeout_s)

    def clear(self, timeout_s: float = 10.0) -> None:
        """Forget everything: fresh client, empty indices, persisted files removed."""
        with self._lock:
            self._client = loci_bridge.make_client(self._epoch_size_ms, int(self._embedder.dim))
            self._reset_indices()
            self._observation_count = 0
            self.version += 1
            done = self._persistence.clear(wait=False) if self._persistence is not None else None
        if done is not None:
            done.wait(timeout_s)

    def close(self, timeout_s: float = 10.0) -> None:
        """Final snapshot and writer shutdown (no-op without persistence)."""
        if self._persistence is None:
            return
        if self._session_started:
            self._persistence.close(timeout_s)
