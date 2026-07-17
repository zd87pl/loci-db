#!/usr/bin/env python3
"""Deterministic world-model proof harness for LOCI.

This is not a synthetic speed test. It creates one repeatable patrol episode
and checks the product claim directly:

1. Similar current states retrieve historical analogs.
2. Predicted future states retrieve the future phase of an episode better than
   querying with the current state.
3. Out-of-distribution states produce a novelty spike.
4. A retrieved state can reconstruct its surrounding trajectory.

Run:
    python benchmarks/world_model_harness.py
    python benchmarks/world_model_harness.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loci.local_client import LocalLociClient
from loci.retrieval.novelty import NoveltyCalibrator
from loci.schema import WorldState

VECTOR_DIM = 16
BASE_MS = 1_700_000_000_000
STEP_MS = 100
DEFAULT_TRACE_POINTS = 240
DEFAULT_HORIZON_STEPS = 12


def _position_for_angle(angle: float) -> tuple[float, float, float]:
    x = 0.5 + 0.32 * math.cos(angle)
    y = 0.5 + 0.08 * math.sin(2.0 * angle)
    z = 0.5 + 0.24 * math.sin(angle)
    return x, y, z


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def _encode_vector(angle: float) -> list[float]:
    x, y, z = _position_for_angle(angle)
    values = [
        math.sin(angle),
        math.cos(angle),
        x,
        y,
        z,
        -math.sin(angle),
        math.cos(angle),
        math.sin(2.0 * angle),
        math.cos(2.0 * angle),
        1.0,
    ]
    values.extend([0.0] * (VECTOR_DIM - len(values)))
    return _normalise(values)


def _novel_vector() -> list[float]:
    values = [-1.0, -1.0, 0.0, 1.0, 0.0, 0.75, -0.75, 0.0, 0.0, -1.0]
    values.extend([0.0] * (VECTOR_DIM - len(values)))
    return _normalise(values)


def _angle_for_step(step: int, trace_points: int) -> float:
    return 2.0 * math.pi * (step % trace_points) / trace_points


def _step_for_state(state: WorldState, trace_points: int) -> int:
    return round((state.timestamp_ms - BASE_MS) / STEP_MS) % trace_points


def _step_error(observed: int, expected: int, trace_points: int) -> int:
    raw = abs(observed - expected)
    return min(raw, trace_points - raw)


def make_patrol_trace(trace_points: int = DEFAULT_TRACE_POINTS) -> list[WorldState]:
    """Create one deterministic closed-loop patrol episode."""
    states: list[WorldState] = []
    for step in range(trace_points):
        angle = _angle_for_step(step, trace_points)
        x, y, z = _position_for_angle(angle)
        states.append(
            WorldState(
                x=x,
                y=y,
                z=z,
                timestamp_ms=BASE_MS + step * STEP_MS,
                vector=_encode_vector(angle),
                scene_id="patrol_loop",
                scale_level="patch",
                confidence=1.0,
            )
        )
    return states


def make_phase_predictor(
    trace_points: int = DEFAULT_TRACE_POINTS,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
) -> Callable[[list[float]], list[float]]:
    """Return a tiny deterministic predictor over the harness embedding."""

    def predictor(context_vector: list[float]) -> list[float]:
        angle = math.atan2(context_vector[0], context_vector[1])
        predicted_angle = angle + _angle_for_step(horizon_steps, trace_points)
        return _encode_vector(predicted_angle)

    return predictor


def _query_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


def run_harness(
    *,
    trace_points: int = DEFAULT_TRACE_POINTS,
    horizon_steps: int = DEFAULT_HORIZON_STEPS,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the proof harness and optionally write a JSON artifact."""
    client = LocalLociClient(vector_size=VECTOR_DIM, epoch_size_ms=1_000, decay_lambda=0)
    states = make_patrol_trace(trace_points)

    insert_start = time.perf_counter()
    ids = client.insert_batch(states)
    insert_ms = (time.perf_counter() - insert_start) * 1000

    predictor = make_phase_predictor(trace_points, horizon_steps)
    current_step = trace_points // 5
    current = states[current_step]
    expected_future_step = (current_step + horizon_steps) % trace_points

    analogs, analog_ms = _query_ms(lambda: client.query(current.vector, limit=5))
    analog_top_step = _step_for_state(analogs[0], trace_points)
    analog_stats = client.last_query_stats

    baseline, baseline_ms = _query_ms(lambda: client.query(current.vector, limit=1))
    baseline_step = _step_for_state(baseline[0], trace_points)

    predicted_result, predicted_ms = _query_ms(
        lambda: client.predict_and_retrieve(
            context_vector=current.vector,
            predictor_fn=predictor,
            future_horizon_ms=horizon_steps * STEP_MS,
            current_timestamp_ms=current.timestamp_ms,
            limit=5,
            return_prediction=True,
        )
    )
    predicted_top_step = _step_for_state(predicted_result.results[0], trace_points)

    calibrator = NoveltyCalibrator(window_size=32, min_samples=6)
    familiar_scores: list[float] = []
    for probe_step in range(0, min(trace_points, 12 * horizon_steps), horizon_steps):
        probe = states[probe_step % trace_points]
        result = client.predict_and_retrieve(
            context_vector=probe.vector,
            predictor_fn=predictor,
            future_horizon_ms=horizon_steps * STEP_MS,
            current_position=(probe.x, probe.y, probe.z),
            current_timestamp_ms=probe.timestamp_ms,
            spatial_search_radius=0.2,
            limit=5,
            calibrator=calibrator,
            return_prediction=True,
        )
        familiar_scores.append(result.prediction_novelty)

    novel_state = WorldState(
        x=0.04,
        y=0.96,
        z=0.04,
        timestamp_ms=BASE_MS + trace_points * STEP_MS,
        vector=_novel_vector(),
        scene_id="novel_corner",
    )
    novel_result = client.predict_and_retrieve(
        context_vector=novel_state.vector,
        predictor_fn=predictor,
        future_horizon_ms=horizon_steps * STEP_MS,
        current_position=(novel_state.x, novel_state.y, novel_state.z),
        current_timestamp_ms=novel_state.timestamp_ms,
        spatial_search_radius=0.04,
        limit=5,
        calibrator=calibrator,
        return_prediction=True,
    )

    anchor_step = trace_points // 2
    trajectory, trajectory_ms = _query_ms(
        lambda: client.get_trajectory(ids[anchor_step], steps_back=10, steps_forward=10)
    )

    result: dict[str, Any] = {
        "config": {
            "trace_points": trace_points,
            "horizon_steps": horizon_steps,
            "horizon_ms": horizon_steps * STEP_MS,
            "vector_dim": VECTOR_DIM,
            "platform": platform.machine(),
            "python": platform.python_version(),
        },
        "insert": {
            "states": len(states),
            "insert_ms": round(insert_ms, 3),
            "insert_rate": round(len(states) / (insert_ms / 1000), 1),
        },
        "historical_analog": {
            "query_ms": round(analog_ms, 3),
            "expected_step": current_step,
            "top_step": analog_top_step,
            "top_step_error": _step_error(analog_top_step, current_step, trace_points),
            "results": len(analogs),
            "shards_searched": analog_stats.shards_searched if analog_stats else 0,
            "candidates": analog_stats.total_candidates if analog_stats else 0,
        },
        "prediction_grounded_retrieval": {
            "query_ms": round(predicted_ms, 3),
            "current_step": current_step,
            "expected_future_step": expected_future_step,
            "baseline_top_step": baseline_step,
            "baseline_query_ms": round(baseline_ms, 3),
            "baseline_step_error": _step_error(baseline_step, expected_future_step, trace_points),
            "predicted_top_step": predicted_top_step,
            "predicted_step_error": _step_error(
                predicted_top_step,
                expected_future_step,
                trace_points,
            ),
            "prediction_novelty": round(predicted_result.prediction_novelty, 3),
            "results": len(predicted_result.results),
        },
        "novelty": {
            "familiar_median": round(statistics.median(familiar_scores), 3),
            "familiar_min": round(min(familiar_scores), 3),
            "familiar_max": round(max(familiar_scores), 3),
            "novel_corner": round(novel_result.prediction_novelty, 3),
            "novel_results": len(novel_result.results),
            "novelty_gap": round(
                novel_result.prediction_novelty - statistics.median(familiar_scores),
                3,
            ),
            "calibrator": calibrator.stats(),
        },
        "trajectory": {
            "query_ms": round(trajectory_ms, 3),
            "anchor_step": anchor_step,
            "states": len(trajectory),
            "first_step": _step_for_state(trajectory[0], trace_points),
            "last_step": _step_for_state(trajectory[-1], trace_points),
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        result["output_path"] = str(output_path)

    return result


def format_markdown(result: dict[str, Any]) -> str:
    """Format harness results as a compact markdown report."""
    analog = result["historical_analog"]
    prediction = result["prediction_grounded_retrieval"]
    novelty = result["novelty"]
    trajectory = result["trajectory"]

    baseline_error = prediction["baseline_step_error"]
    predicted_error = prediction["predicted_step_error"]
    if predicted_error < baseline_error:
        error_clause = f"reduced future-step error from {baseline_error} to {predicted_error}"
    elif predicted_error > baseline_error:
        error_clause = f"increased future-step error from {baseline_error} to {predicted_error}"
    else:
        error_clause = f"left future-step error unchanged at {baseline_error}"

    lines = [
        "",
        "## LOCI World-Model Harness",
        "",
        (
            f"Configuration: n={result['config']['trace_points']}, "
            f"horizon={result['config']['horizon_steps']} steps "
            f"({result['config']['horizon_ms']}ms), dim={result['config']['vector_dim']}"
        ),
        "",
        "| Proof | Metric | Value |",
        "|:--|:--|--:|",
        (f"| Historical analog | top-step error | {analog['top_step_error']} step(s) |"),
        (
            "| Prediction-grounded retrieval | baseline future-step error "
            f"| {prediction['baseline_step_error']} step(s) |"
        ),
        (
            "| Prediction-grounded retrieval | predicted future-step error "
            f"| {prediction['predicted_step_error']} step(s) |"
        ),
        (f"| Novelty detection | familiar median novelty | {novelty['familiar_median']:.3f} |"),
        f"| Novelty detection | novel-corner novelty | {novelty['novel_corner']:.3f} |",
        f"| Novelty detection | novelty gap | {novelty['novelty_gap']:.3f} |",
        f"| Trajectory | reconstructed states | {trajectory['states']} |",
        "",
        (
            f"**Headline:** prediction-grounded retrieval {error_clause}, "
            f"while the novel probe scored {novelty['novel_corner']:.3f} novelty."
        ),
        "",
    ]
    return "\n".join(lines)


def _default_output_path() -> Path:
    return Path(__file__).resolve().parent / "results" / "world_model_latest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LOCI world-model proof harness")
    parser.add_argument("--quick", action="store_true", help="Use a smaller trace for fast CI")
    parser.add_argument("--trace-points", type=int, default=DEFAULT_TRACE_POINTS)
    parser.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS)
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--no-write", action="store_true", help="Do not write JSON results")
    args = parser.parse_args()

    trace_points = min(args.trace_points, 80) if args.quick else args.trace_points
    horizon_steps = min(args.horizon_steps, 4) if args.quick else args.horizon_steps
    output_path = None if args.no_write else args.output

    result = run_harness(
        trace_points=trace_points,
        horizon_steps=horizon_steps,
        output_path=output_path,
    )
    print(format_markdown(result))
    if output_path is not None:
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
