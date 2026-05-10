"""Tests for the LOCI world-model proof harness."""

from __future__ import annotations

import json

from benchmarks.world_model_harness import format_markdown, run_harness


def test_world_model_harness_proves_prediction_and_novelty(tmp_path) -> None:
    output_path = tmp_path / "world_model_latest.json"

    result = run_harness(trace_points=80, horizon_steps=4, output_path=output_path)

    assert output_path.exists()
    persisted = json.loads(output_path.read_text())
    assert persisted["config"]["trace_points"] == 80

    analog = result["historical_analog"]
    assert analog["top_step_error"] <= 1
    assert analog["results"] == 5

    prediction = result["prediction_grounded_retrieval"]
    assert prediction["predicted_step_error"] <= 1
    assert prediction["baseline_step_error"] > prediction["predicted_step_error"]

    novelty = result["novelty"]
    assert novelty["novel_corner"] >= 0.9
    assert novelty["novelty_gap"] >= 0.5
    assert novelty["novel_results"] == 0

    trajectory = result["trajectory"]
    assert trajectory["states"] == 21


def test_world_model_harness_markdown_contains_headline() -> None:
    result = run_harness(trace_points=80, horizon_steps=4)

    markdown = format_markdown(result)

    assert "LOCI World-Model Harness" in markdown
    assert "prediction-grounded retrieval reduced future-step error" in markdown
