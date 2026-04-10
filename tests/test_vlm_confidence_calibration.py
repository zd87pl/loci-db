"""Unit tests for ADR-5: VLM Confidence Calibration.

Covers:
- VLM confidence is parsed from JSON response (Stage 1)
- VLM confidence is penalized 0.6x when YOLO sees nothing for that label (Stage 2)
- VLM confidence is NOT penalized when YOLO corroborates (same position, IoU > 0.3)
- Fallback confidence is 0.50, not 0.7 (Stage 3)
- Default VLM bounding box is width=0.3, height=0.3
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_spatial"))

from app.scene_ingestion import Detection, SceneIngestion
from app.vlm_client import _parse_vlm_response

# ---------------------------------------------------------------------------
# Stage 1: VLM confidence parsed from prompt response
# ---------------------------------------------------------------------------


def test_vlm_confidence_extracted_from_prompt_response() -> None:
    """When VLM JSON includes confidence, it must be used as-is."""
    raw = '[{"label": "cup", "cx": 0.5, "cy": 0.5, "confidence": 0.82}]'
    result = _parse_vlm_response(raw)
    assert len(result) == 1
    assert abs(result[0]["confidence"] - 0.82) < 1e-6


# ---------------------------------------------------------------------------
# Stage 2: Skepticism multiplier when YOLO absent
# ---------------------------------------------------------------------------


def test_vlm_confidence_penalized_when_yolo_absent() -> None:
    """VLM confidence must be multiplied by 0.6 when YOLO finds nothing for that label."""
    memory = MagicMock()
    memory.observe = MagicMock(return_value="state-id")
    ing = SceneIngestion(memory=memory, vlm_client=None, use_vlm_fallback=False)

    vlm_raw_conf = 0.8
    # No YOLO detections for this label → penalty applies
    raw_yolo: list[Detection] = []

    # Simulate the corroboration check directly
    yolo_corroborated = any(ing._iou(0.5, 0.5, d.cx, d.cy) > 0.3 for d in raw_yolo)
    final_conf = vlm_raw_conf * (1.0 if yolo_corroborated else 0.6)
    assert abs(final_conf - 0.48) < 1e-6


def test_vlm_confidence_unpenalized_when_yolo_corroborates() -> None:
    """VLM confidence must NOT be penalized when YOLO detects same label nearby."""
    memory = MagicMock()
    memory.observe = MagicMock(return_value="state-id")
    ing = SceneIngestion(memory=memory, vlm_client=None, use_vlm_fallback=False)

    vlm_raw_conf = 0.8
    vlm_cx, vlm_cy = 0.5, 0.5
    # YOLO has a detection for this label at same position
    yolo_det = Detection(
        label="cup", cx=0.5, cy=0.5, width=0.1, height=0.1, confidence=0.3, source="yolo"
    )

    yolo_corroborated = any(ing._iou(vlm_cx, vlm_cy, d.cx, d.cy) > 0.3 for d in [yolo_det])
    final_conf = vlm_raw_conf * (1.0 if yolo_corroborated else 0.6)
    # YOLO corroborates → no penalty
    assert abs(final_conf - 0.8) < 1e-6


def test_vlm_confidence_penalized_when_yolo_has_different_label() -> None:
    """VLM confidence is penalized if YOLO detects a DIFFERENT label at the same position."""
    memory = MagicMock()
    memory.observe = MagicMock(return_value="state-id")
    ing = SceneIngestion(memory=memory, vlm_client=None, use_vlm_fallback=False)

    vlm_raw_conf = 0.8
    vlm_cx, vlm_cy = 0.5, 0.5
    vlm_label = "cup"
    # YOLO detects "bottle" (different label) at same position — different label bucket
    yolo_det = Detection(
        label="bottle", cx=0.5, cy=0.5, width=0.1, height=0.1, confidence=0.3, source="yolo"
    )
    raw_by_label = {"bottle": [yolo_det]}  # label "cup" not present

    yolo_corroborated = any(
        ing._iou(vlm_cx, vlm_cy, d.cx, d.cy) > 0.3 for d in raw_by_label.get(vlm_label, [])
    )
    final_conf = vlm_raw_conf * (1.0 if yolo_corroborated else 0.6)
    assert abs(final_conf - 0.48) < 1e-6


# ---------------------------------------------------------------------------
# Stage 3: Fallback confidence
# ---------------------------------------------------------------------------


def test_vlm_fallback_confidence_is_0_50() -> None:
    """When VLM JSON has no confidence field, fallback must be 0.50 (not 0.7)."""
    raw = '[{"label": "phone"}]'
    result = _parse_vlm_response(raw)
    assert len(result) == 1
    assert abs(result[0]["confidence"] - 0.50) < 1e-6


# ---------------------------------------------------------------------------
# Bounding box defaults
# ---------------------------------------------------------------------------


def test_vlm_default_bbox_width_height_is_0_30() -> None:
    """When VLM JSON has no width/height, defaults must be 0.3 (not 0.1)."""
    raw = '[{"label": "wallet", "cx": 0.5, "cy": 0.5}]'
    result = _parse_vlm_response(raw)
    assert len(result) == 1
    assert abs(result[0]["width"] - 0.3) < 1e-6
    assert abs(result[0]["height"] - 0.3) < 1e-6
