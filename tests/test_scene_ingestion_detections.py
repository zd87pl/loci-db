"""Tests for the handheld entry point of SceneIngestion (design task T5).

Covers:
- ``process_detections`` with room-frame ``Detection``s calls ``observe`` once per
  ``consensus_min_count`` frames with averaged metres, ``anchored = all(...)``, the mean
  anchor confidence and ``room`` forwarded
- metric matching within ``consensus_radius_m`` while the image boxes drift (and no
  match beyond the radius)
- image-frame legacy path and camera-frame path
- buffer keys isolate frames and rooms
- matched entries are consumed on fire; ``consensus_refire_ms`` hold-off and the
  ``consensus_stats`` counters
- eviction by window
- ``Detection.to_dict`` adds the new keys only when set
- the legacy ``_update_consensus`` keyword call shape is still accepted
- the memory's ``observe`` is always called with keyword arguments (legacy six + new)

Everything runs against a ``MagicMock`` memory: no LOCI client, no hardware.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import threading
import time
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.scene_ingestion import Detection, SceneIngestion  # noqa: E402

LEGACY_OBSERVE_KEYS = {"label", "cx", "cy", "confidence", "timestamp_ms", "depth_m"}
METRIC_OBSERVE_KEYS = {"xyz_cam_m", "pos_room_m", "room", "anchored", "anchor_confidence", "source"}


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _memory() -> MagicMock:
    memory = MagicMock()
    memory.observe = MagicMock(return_value="fake-state-id")
    return memory


def _ingestion(**kwargs) -> SceneIngestion:
    return SceneIngestion(memory=_memory(), vlm_client=None, use_vlm_fallback=False, **kwargs)


def _room_det(
    label: str = "cup",
    pos: tuple[float, float, float] = (1.0, 0.8, 1.0),
    xyz: tuple[float, float, float] | None = (0.2, -0.3, 1.5),
    room: str = "kitchen",
    cx: float = 0.5,
    cy: float = 0.5,
    confidence: float = 0.8,
    anchored: bool = True,
    anchor_confidence: float = 1.0,
    source: str = "oak",
) -> Detection:
    return Detection(
        label=label,
        cx=cx,
        cy=cy,
        width=0.1,
        height=0.1,
        confidence=confidence,
        source=source,
        xyz_cam_m=xyz,
        pos_room_m=pos,
        room=room,
        frame="room",
        anchored=anchored,
        anchor_confidence=anchor_confidence,
    )


def _cam_det(
    label: str = "cup",
    xyz: tuple[float, float, float] = (0.2, -0.3, 1.5),
    cx: float = 0.5,
    cy: float = 0.5,
    confidence: float = 0.8,
) -> Detection:
    return Detection(
        label=label,
        cx=cx,
        cy=cy,
        width=0.1,
        height=0.1,
        confidence=confidence,
        source="oak",
        xyz_cam_m=xyz,
        frame="camera",
    )


def _image_det(
    label: str = "cup", cx: float = 0.5, cy: float = 0.5, conf: float = 0.8
) -> Detection:
    return Detection(label=label, cx=cx, cy=cy, width=0.1, height=0.1, confidence=conf)


def _observe_kwargs(ing: SceneIngestion) -> dict:
    call = ing._memory.observe.call_args
    assert call is not None
    assert call.args == (), "observe must be called with keyword arguments only"
    return call.kwargs


def _close(a, b, tol: float = 1e-6) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Room frame: averaged metres, anchored = all, mean anchor confidence
# ---------------------------------------------------------------------------


def test_room_frame_fires_once_with_averaged_metres() -> None:
    ing = _ingestion(consensus_min_count=2)

    out1 = ing.process_detections(
        [_room_det(pos=(1.0, 0.8, 1.0), xyz=(0.2, -0.3, 1.5), cx=0.50, confidence=0.7)],
        timestamp_ms=1000,
    )
    ing._memory.observe.assert_not_called()
    assert len(out1) == 1 and ing.frame_count == 1 and ing.last_detections == out1

    ing.process_detections(
        [
            _room_det(
                pos=(1.1, 0.8, 1.0),
                xyz=(0.3, -0.3, 1.7),
                cx=0.52,
                confidence=0.9,
                anchor_confidence=0.5,
            )
        ],
        timestamp_ms=1100,
    )
    ing._memory.observe.assert_called_once()
    kw = _observe_kwargs(ing)

    assert set(kw) == LEGACY_OBSERVE_KEYS | METRIC_OBSERVE_KEYS
    assert kw["label"] == "cup"
    assert abs(kw["cx"] - 0.51) < 1e-6 and abs(kw["cy"] - 0.5) < 1e-6
    assert abs(kw["confidence"] - 0.9) < 1e-6
    assert kw["timestamp_ms"] == 1100
    assert _close(kw["pos_room_m"], (1.05, 0.8, 1.0))
    assert _close(kw["xyz_cam_m"], (0.25, -0.3, 1.6))
    assert kw["room"] == "kitchen"
    assert kw["anchored"] is True
    assert abs(kw["anchor_confidence"] - 0.75) < 1e-6
    assert kw["source"] == "oak"
    # depth_m is the newest sample's camera z (derived from xyz_cam_m)
    assert abs(kw["depth_m"] - 1.7) < 1e-6
    assert ing.frame_count == 2


def test_anchored_is_all_of_the_agreeing_samples() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_room_det(anchored=True, anchor_confidence=1.0)], 1000)
    ing.process_detections([_room_det(anchored=False, anchor_confidence=0.0)], 1100)
    kw = _observe_kwargs(ing)
    assert kw["anchored"] is False
    assert abs(kw["anchor_confidence"] - 0.5) < 1e-6


def test_min_count_three_needs_three_metric_frames() -> None:
    ing = _ingestion(consensus_min_count=3)
    for i in range(2):
        ing.process_detections([_room_det()], 1000 + 100 * i)
    ing._memory.observe.assert_not_called()
    ing.process_detections([_room_det()], 1200)
    ing._memory.observe.assert_called_once()


# ---------------------------------------------------------------------------
# Metric matching vs the image-box proxy
# ---------------------------------------------------------------------------


def test_metric_match_within_radius_while_boxes_drift() -> None:
    """Boxes at opposite image corners (IoU 0) still agree when the room metres do."""
    ing = _ingestion(consensus_min_count=2, consensus_radius_m=0.3)
    ing.process_detections([_room_det(pos=(1.0, 0.8, 1.0), cx=0.1, cy=0.1)], 1000)
    ing.process_detections([_room_det(pos=(1.2, 0.8, 1.1), cx=0.9, cy=0.9)], 1100)
    ing._memory.observe.assert_called_once()
    kw = _observe_kwargs(ing)
    assert _close(kw["pos_room_m"], (1.1, 0.8, 1.05))


def test_no_metric_match_beyond_radius_even_with_identical_boxes() -> None:
    ing = _ingestion(consensus_min_count=2, consensus_radius_m=0.3)
    ing.process_detections([_room_det(pos=(1.0, 0.8, 1.0), cx=0.5, cy=0.5)], 1000)
    ing.process_detections([_room_det(pos=(1.5, 0.8, 1.0), cx=0.5, cy=0.5)], 1100)
    ing._memory.observe.assert_not_called()
    assert ing.consensus_stats["buffer_entries"] == 2


def test_radius_is_inclusive_and_configurable() -> None:
    ing = _ingestion(consensus_min_count=2, consensus_radius_m=0.5)
    ing.process_detections([_room_det(pos=(0.0, 0.0, 0.0))], 1000)
    ing.process_detections([_room_det(pos=(0.3, 0.4, 0.0))], 1100)  # exactly 0.5 m
    ing._memory.observe.assert_called_once()


def test_camera_metres_used_when_no_room_metres() -> None:
    ing = _ingestion(consensus_min_count=2, consensus_radius_m=0.3)
    ing.process_detections([_cam_det(xyz=(0.0, 0.0, 2.0), cx=0.2, cy=0.2)], 1000)
    ing.process_detections([_cam_det(xyz=(0.1, 0.0, 2.1), cx=0.8, cy=0.8)], 1100)
    ing._memory.observe.assert_called_once()
    kw = _observe_kwargs(ing)
    assert _close(kw["xyz_cam_m"], (0.05, 0.0, 2.05))
    assert kw["pos_room_m"] is None
    assert kw["room"] is None
    assert kw["anchored"] is False
    assert kw["source"] == "oak"


def test_mixed_depth_falls_back_to_iou_proxy() -> None:
    """A depthless sample and a metric one in the same key agree through the boxes."""
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_room_det(xyz=None, pos=None, cx=0.5, cy=0.5)], 1000)
    ing.process_detections([_room_det(xyz=(0.1, 0.0, 1.0), pos=(1.0, 0.8, 1.0), cx=0.52)], 1100)
    ing._memory.observe.assert_called_once()
    kw = _observe_kwargs(ing)
    # Means are taken over the samples that have metres
    assert _close(kw["pos_room_m"], (1.0, 0.8, 1.0))
    assert _close(kw["xyz_cam_m"], (0.1, 0.0, 1.0))
    assert kw["room"] == "kitchen"


def test_mixed_depth_no_iou_means_no_match() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_room_det(xyz=None, pos=None, cx=0.1, cy=0.1)], 1000)
    ing.process_detections([_room_det(cx=0.9, cy=0.9)], 1100)
    ing._memory.observe.assert_not_called()


# ---------------------------------------------------------------------------
# Image frame (legacy) path through process_detections
# ---------------------------------------------------------------------------


def test_image_frame_legacy_path_unchanged() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_image_det(cx=0.50, cy=0.50, conf=0.7)], 1000)
    ing._memory.observe.assert_not_called()
    ing.process_detections([_image_det(cx=0.52, cy=0.52, conf=0.9)], 1100)
    ing._memory.observe.assert_called_once()
    kw = _observe_kwargs(ing)
    assert abs(kw["cx"] - 0.51) < 1e-6 and abs(kw["cy"] - 0.51) < 1e-6
    assert abs(kw["confidence"] - 0.9) < 1e-6
    assert kw["timestamp_ms"] == 1100
    assert kw["depth_m"] is None
    assert kw["xyz_cam_m"] is None and kw["pos_room_m"] is None and kw["room"] is None
    assert kw["anchored"] is False and kw["anchor_confidence"] == 0.0
    assert kw["source"] == "yolo"


def test_image_frame_low_iou_is_a_different_object() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_image_det(cx=0.1, cy=0.1)], 1000)
    ing.process_detections([_image_det(cx=0.9, cy=0.9)], 1100)
    ing._memory.observe.assert_not_called()


def test_process_detections_defaults_timestamp_to_now() -> None:
    ing = _ingestion(consensus_min_count=1)
    before = int(time.time() * 1000)
    ing.process_detections([_image_det()])
    after = int(time.time() * 1000)
    kw = _observe_kwargs(ing)
    assert before <= kw["timestamp_ms"] <= after


def test_process_detections_handles_several_labels_per_frame() -> None:
    ing = _ingestion(consensus_min_count=2)
    frame = [_room_det("cup", pos=(1.0, 0.8, 1.0)), _room_det("bottle", pos=(-0.6, 0.9, 1.2))]
    ing.process_detections(frame, 1000)
    ing.process_detections(frame, 1100)
    labels = sorted(c.kwargs["label"] for c in ing._memory.observe.call_args_list)
    assert labels == ["bottle", "cup"]
    assert ing.consensus_stats == {"buffer_entries": 0, "fires": 2, "suppressed_by_holdoff": 0}


# ---------------------------------------------------------------------------
# Buffer keys isolate frames and rooms
# ---------------------------------------------------------------------------


def test_buffer_keys_isolate_rooms() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_room_det(room="kitchen")], 1000)
    ing.process_detections([_room_det(room="living_room")], 1100)
    ing._memory.observe.assert_not_called()
    assert ing.consensus_stats["buffer_entries"] == 2


def test_buffer_keys_isolate_frames() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_image_det(cx=0.5, cy=0.5)], 1000)
    ing.process_detections([_cam_det(cx=0.5, cy=0.5)], 1050)
    ing.process_detections([_room_det(cx=0.5, cy=0.5)], 1100)
    ing._memory.observe.assert_not_called()
    assert ing.consensus_stats["buffer_entries"] == 3
    # Keys are (label, frame, room or "")
    assert set(ing._consensus_buffer) == {
        ("cup", "image", ""),
        ("cup", "camera", ""),
        ("cup", "room", "kitchen"),
    }


def test_buffer_keys_isolate_labels() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_room_det("cup")], 1000)
    ing.process_detections([_room_det("bottle")], 1100)
    ing._memory.observe.assert_not_called()


# ---------------------------------------------------------------------------
# Consumption on fire and the refire hold-off
# ---------------------------------------------------------------------------


def test_matched_entries_are_consumed_on_fire() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing.process_detections([_room_det()], 1000)
    ing.process_detections([_room_det()], 1100)
    assert ing._memory.observe.call_count == 1
    assert ing.consensus_stats["buffer_entries"] == 0
    # A third frame alone starts a new buffer; the fourth fires again
    ing.process_detections([_room_det()], 1200)
    assert ing._memory.observe.call_count == 1
    ing.process_detections([_room_det()], 1300)
    assert ing._memory.observe.call_count == 2
    assert ing.consensus_stats["fires"] == 2


def test_consumption_leaves_unmatched_entries_in_place() -> None:
    """Two cups in one room: firing for one must not forget the other."""
    ing = _ingestion(consensus_min_count=2, consensus_radius_m=0.3)
    ing.process_detections([_room_det(pos=(1.0, 0.8, 1.0)), _room_det(pos=(-2.0, 0.8, 3.0))], 1000)
    ing.process_detections([_room_det(pos=(1.0, 0.8, 1.0))], 1100)
    assert ing._memory.observe.call_count == 1
    assert ing.consensus_stats["buffer_entries"] == 1
    ing.process_detections([_room_det(pos=(-2.0, 0.8, 3.0))], 1200)
    assert ing._memory.observe.call_count == 2
    assert ing.consensus_stats["buffer_entries"] == 0


def test_refire_holdoff_suppresses_and_averages_more_samples() -> None:
    ing = _ingestion(consensus_min_count=2, consensus_window_ms=1000, consensus_refire_ms=2000)
    # 10 fps stream of one cup from t=1000 to t=3100 inclusive, slowly drifting in x
    for i in range(22):
        ts = 1000 + 100 * i
        ing.process_detections([_room_det(pos=(1.0 + 0.001 * i, 0.8, 1.0))], ts)
        if ts < 1100:
            assert ing._memory.observe.call_count == 0
        elif ts < 3100:
            # fired once at 1100; everything after that is held off
            assert ing._memory.observe.call_count == 1
    assert ing._memory.observe.call_count == 2
    first, second = ing._memory.observe.call_args_list
    assert first.kwargs["timestamp_ms"] == 1100
    assert second.kwargs["timestamp_ms"] == 3100
    # The second fire averages every sample still inside the 1 s window (2100..3100)
    expected_x = sum(1.0 + 0.001 * i for i in range(11, 22)) / 11
    assert abs(second.kwargs["pos_room_m"][0] - expected_x) < 1e-9
    stats = ing.consensus_stats
    assert stats["fires"] == 2
    # 1200 only re-seeds the emptied buffer; 1300..3000 (18 frames) are suppressed
    assert stats["suppressed_by_holdoff"] == 18
    assert stats["buffer_entries"] == 0


def test_refire_holdoff_is_per_key() -> None:
    ing = _ingestion(consensus_min_count=2, consensus_refire_ms=5000)
    ing.process_detections([_room_det("cup")], 1000)
    ing.process_detections([_room_det("cup")], 1100)
    ing.process_detections([_room_det("bottle", pos=(-0.6, 0.9, 1.2))], 1200)
    ing.process_detections([_room_det("bottle", pos=(-0.6, 0.9, 1.2))], 1300)
    assert ing._memory.observe.call_count == 2
    assert ing.consensus_stats["suppressed_by_holdoff"] == 0


def test_refire_zero_is_legacy_behaviour() -> None:
    ing = _ingestion(consensus_min_count=2)  # consensus_refire_ms defaults to 0
    for i in range(6):
        ing.process_detections([_room_det()], 1000 + 100 * i)
    assert ing._memory.observe.call_count == 3
    assert ing.consensus_stats["suppressed_by_holdoff"] == 0


# ---------------------------------------------------------------------------
# Eviction by window
# ---------------------------------------------------------------------------


def test_eviction_by_window() -> None:
    ing = _ingestion(consensus_window_ms=500, consensus_min_count=2)
    ing.process_detections([_room_det()], 0)
    ing.process_detections([_room_det()], 600)
    ing._memory.observe.assert_not_called()
    assert ing.consensus_stats["buffer_entries"] == 1


def test_consensus_stats_shape() -> None:
    ing = _ingestion()
    assert ing.consensus_stats == {"buffer_entries": 0, "fires": 0, "suppressed_by_holdoff": 0}
    status = ing.status()
    assert status["consensus"]["fires"] == 0
    assert status["consensus"]["radius_m"] == 0.3
    assert status["consensus"]["refire_ms"] == 0


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------


def test_detection_to_dict_legacy_keys_only() -> None:
    d = _image_det()
    assert set(d.to_dict()) == {"label", "cx", "cy", "width", "height", "confidence", "source"}
    d.depth_m = 1.234
    assert d.to_dict()["depth_m"] == 1.234
    assert d.frame == "image" and d.is_metric is False and d.distance_m is None


def test_detection_to_dict_metric_keys_when_set() -> None:
    d = _room_det(pos=(1.0, 0.8, 1.0), xyz=(0.0, 0.0, 2.0), anchor_confidence=0.6667)
    out = d.to_dict()
    assert out["xyz_cam_m"] == [0.0, 0.0, 2.0]
    assert out["pos_room_m"] == [1.0, 0.8, 1.0]
    assert out["room"] == "kitchen"
    assert out["frame"] == "room"
    assert out["anchored"] is True
    assert out["anchor_confidence"] == 0.667
    assert out["distance_m"] == 2.0
    assert out["depth_m"] == 2.0  # derived from xyz_cam_m when not given
    json.dumps(out)  # JSON-safe


def test_detection_distance_is_euclidean() -> None:
    d = _cam_det(xyz=(3.0, 0.0, 4.0))
    assert d.distance_m == 5.0
    assert d.depth_m == 4.0
    assert d.to_dict()["distance_m"] == 5.0


def test_detection_camera_frame_dict_has_no_room_keys() -> None:
    out = _cam_det().to_dict()
    assert "room" not in out and "pos_room_m" not in out
    assert out["frame"] == "camera" and out["anchored"] is False


def test_detection_explicit_depth_wins_over_xyz() -> None:
    d = Detection("cup", 0.5, 0.5, 0.1, 0.1, 0.9, depth_m=1.0, xyz_cam_m=(0.0, 0.0, 2.0))
    assert d.depth_m == 1.0


def test_detection_coerces_numpy_points_to_float_tuples() -> None:
    d = Detection(
        "cup",
        0.5,
        0.5,
        0.1,
        0.1,
        0.9,
        xyz_cam_m=np.array([0.1, -0.2, 1.5], dtype=np.float32),
        pos_room_m=[1.0, 0.8, 1.0],
        room="kitchen",
        frame="room",
    )
    assert isinstance(d.xyz_cam_m, tuple) and all(type(v) is float for v in d.xyz_cam_m)
    assert isinstance(d.pos_room_m, tuple) and d.pos_room_m == (1.0, 0.8, 1.0)
    assert abs(d.xyz_cam_m[2] - 1.5) < 1e-6
    json.dumps(d.to_dict())


# ---------------------------------------------------------------------------
# _update_consensus call shapes and observe keyword contract
# ---------------------------------------------------------------------------


def test_legacy_update_consensus_keyword_shape_still_accepted() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing._update_consensus("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=1000, depth_m=None)
    ing._update_consensus("cup", cx=0.52, cy=0.52, confidence=0.9, timestamp_ms=1100, depth_m=1.5)
    ing._memory.observe.assert_called_once()
    kw = _observe_kwargs(ing)
    assert {k: kw[k] for k in ("label", "timestamp_ms", "depth_m")} == {
        "label": "cup",
        "timestamp_ms": 1100,
        "depth_m": 1.5,
    }
    assert abs(kw["cx"] - 0.51) < 1e-6 and abs(kw["confidence"] - 0.9) < 1e-6
    assert kw["xyz_cam_m"] is None and kw["pos_room_m"] is None and kw["room"] is None
    assert kw["anchored"] is False and kw["anchor_confidence"] == 0.0 and kw["source"] == "yolo"


def test_update_consensus_full_keyword_shape() -> None:
    ing = _ingestion(consensus_min_count=2)
    for ts in (1000, 1100):
        ing._update_consensus(
            label="laptop",
            cx=0.4,
            cy=0.6,
            confidence=0.8,
            timestamp_ms=ts,
            depth_m=None,
            xyz_cam_m=(0.0, 0.2, 1.0),
            pos_room_m=(-0.8, 0.7, 1.0),
            room="living_room",
            frame="room",
            anchored=True,
            anchor_confidence=0.9,
            source="sim",
        )
    kw = _observe_kwargs(ing)
    assert kw["room"] == "living_room" and kw["source"] == "sim"
    assert _close(kw["pos_room_m"], (-0.8, 0.7, 1.0))
    assert kw["depth_m"] is None  # not derived inside _update_consensus; observe decides


def test_depth_uses_newest_non_none_sample() -> None:
    ing = _ingestion(consensus_min_count=3)
    ing._update_consensus("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=1000, depth_m=1.0)
    ing._update_consensus("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=1100, depth_m=2.0)
    ing._update_consensus("cup", cx=0.5, cy=0.5, confidence=0.8, timestamp_ms=1200, depth_m=None)
    assert _observe_kwargs(ing)["depth_m"] == 2.0


def test_observe_errors_are_logged_not_raised() -> None:
    ing = _ingestion(consensus_min_count=2)
    ing._memory.observe.side_effect = RuntimeError("store down")
    ing.process_detections([_room_det()], 1000)
    ing.process_detections([_room_det()], 1100)  # must not raise
    assert ing.consensus_stats["fires"] == 1


def test_legacy_observe_signature_drops_metric_keywords() -> None:
    """A memory with the pre-handheld observe() signature still receives fires."""

    class LegacyMemory:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def observe(
            self,
            label: str,
            cx: float,
            cy: float,
            confidence: float = 1.0,
            timestamp_ms: int | None = None,
            depth_m: float | None = None,
            min_confidence: float = 0.55,
        ) -> str:
            self.calls.append(
                {
                    "label": label,
                    "cx": cx,
                    "cy": cy,
                    "confidence": confidence,
                    "timestamp_ms": timestamp_ms,
                    "depth_m": depth_m,
                }
            )
            return "id"

    mem = LegacyMemory()
    ing = SceneIngestion(memory=mem, vlm_client=None, use_vlm_fallback=False)
    ing.process_detections([_room_det()], 1000)
    ing.process_detections([_room_det()], 1100)
    assert len(mem.calls) == 1
    assert mem.calls[0]["label"] == "cup" and mem.calls[0]["timestamp_ms"] == 1100


def test_contract_observe_signature_receives_metric_keywords() -> None:
    """A memory with the section-4.8 observe() signature receives every keyword."""

    class ContractMemory:
        def __init__(self) -> None:
            self.calls: list[dict] = []

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
            xyz_cam_m: tuple[float, float, float] | None = None,
            pos_room_m: tuple[float, float, float] | None = None,
            room: str | None = None,
            anchored: bool = False,
            anchor_confidence: float = 0.0,
            source: str = "camera",
        ) -> str | None:
            self.calls.append(
                {
                    "label": label,
                    "pos_room_m": pos_room_m,
                    "xyz_cam_m": xyz_cam_m,
                    "room": room,
                    "anchored": anchored,
                    "anchor_confidence": anchor_confidence,
                    "source": source,
                    "depth_m": depth_m,
                }
            )
            return "id"

    mem = ContractMemory()
    ing = SceneIngestion(memory=mem, vlm_client=None, use_vlm_fallback=False)
    ing.process_detections([_room_det()], 1000)
    ing.process_detections([_room_det()], 1100)
    assert len(mem.calls) == 1
    call = mem.calls[0]
    assert call["room"] == "kitchen" and call["anchored"] is True and call["source"] == "oak"
    assert _close(call["pos_room_m"], (1.0, 0.8, 1.0))
    assert _close(call["xyz_cam_m"], (0.2, -0.3, 1.5))
    assert abs(call["depth_m"] - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# process_frame still uses the same buffer (keyword call unchanged)
# ---------------------------------------------------------------------------


def test_process_frame_path_still_reaches_observe_through_the_buffer() -> None:
    """The browser path (no YOLO model) runs with an empty detection list and the
    legacy keyword shape; mixing it with process_detections keeps one buffer."""
    ing = _ingestion(consensus_min_count=2)
    ing._load_yolo = lambda: None  # type: ignore[method-assign]
    out = asyncio.run(ing.process_frame(b"not-an-image", timestamp_ms=1000))
    assert out == [] and ing.frame_count == 1
    ing.process_detections([_image_det()], 1100)
    ing.process_detections([_image_det()], 1200)
    ing._memory.observe.assert_called_once()
    assert ing.frame_count == 3


# ---------------------------------------------------------------------------
# Thread safety smoke test (runtime thread + another feeder)
# ---------------------------------------------------------------------------


def test_concurrent_feeders_do_not_corrupt_the_buffer() -> None:
    ing = _ingestion(consensus_min_count=2, consensus_window_ms=10_000)
    errors: list[BaseException] = []

    def feed(label: str, base_ts: int) -> None:
        try:
            for i in range(200):
                ing.process_detections([_room_det(label, pos=(1.0, 0.8, 1.0))], base_ts + i)
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [
        threading.Thread(target=feed, args=("cup", 1000)),
        threading.Thread(target=feed, args=("bottle", 1000)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    # Every fire consumes its matches: exactly 100 fires per label with min_count=2
    assert ing._memory.observe.call_count == 200
    assert ing.consensus_stats["buffer_entries"] == 0
    assert math.isfinite(ing.consensus_stats["fires"])
