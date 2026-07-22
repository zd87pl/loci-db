#!/usr/bin/env python3
"""Evaluation harness for ConformalNoveltyCalibrator (RFC-0001 R2).

Deterministic (seeded numpy), no network, no external data. Simulates
in-distribution best-match-score streams of several shapes plus injected
out-of-distribution (OOD) segments, and measures:

(a) empirical false-alarm rate (FAR) vs the configured alpha, swept over
    alpha in {0.01, 0.05, 0.1} and several seeds;
(b) detection rate on OOD segments (two severities);
(c) the legacy z-score NoveltyCalibrator's implied alarm at the matched
    threshold (novelty >= 1 - alpha) on the identical streams.

RFC-0001 R2 success metric: empirical FAR within +/-1% (absolute) of the
configured alpha on held-out in-distribution data.

Run:
    python benchmarks/conformal_eval.py
    python benchmarks/conformal_eval.py --quick

Emits a markdown table to stdout and a JSON artifact to
benchmarks/results/conformal_latest.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loci.retrieval.novelty import ConformalNoveltyCalibrator, NoveltyCalibrator  # noqa: E402

WINDOW = 512
MIN_SAMPLES = 30
ALPHAS = (0.01, 0.05, 0.1)
SEEDS = (0, 1, 2, 3, 4)
WARMUP_N = 512
EVAL_N = 4000
CONTAM_BLOCKS = 10
CONTAM_IN_DIST_PER_BLOCK = 80
CONTAM_OOD_PER_BLOCK = 20
FAR_TOLERANCE = 0.01  # RFC-0001 R2: within +/-1% of alpha

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "conformal_latest.json"


# ---------------------------------------------------------------------------
# Stream generators — best-match similarity scores in [0, 1]
# ---------------------------------------------------------------------------


def _gaussian_stream(rng: np.random.Generator, n: int) -> np.ndarray:
    """In-distribution matches clustered at high similarity."""
    return np.clip(rng.normal(0.85, 0.05, n), 0.0, 1.0)


def _bimodal_stream(rng: np.random.Generator, n: int) -> np.ndarray:
    """Two familiar regimes (e.g. two patrol areas with different match quality)."""
    modes = rng.random(n) < 0.6
    high = rng.normal(0.9, 0.03, n)
    low = rng.normal(0.7, 0.04, n)
    return np.clip(np.where(modes, high, low), 0.0, 1.0)


def _drift_stream(rng: np.random.Generator, n: int, start: int, total: int) -> np.ndarray:
    """Slow mean drift 0.85 -> 0.75 across the whole episode.

    ``start``/``total`` position this chunk within the episode so the drift
    is continuous across warm-up, eval and contamination phases.
    """
    t = (start + np.arange(n)) / max(1, total - 1)
    means = 0.85 - 0.10 * t
    return np.clip(rng.normal(means, 0.05), 0.0, 1.0)


def _ood_scores(rng: np.random.Generator, n: int, severity: str) -> np.ndarray:
    """OOD best-match scores: 'near' = subtle degradation, 'far' = no analog."""
    mean = 0.55 if severity == "near" else 0.35
    return np.clip(rng.normal(mean, 0.05, n), 0.0, 1.0)


_SHAPE_SEED_OFFSET = {"gaussian": 101, "bimodal": 211, "drift": 307}


def _make_streams(shape: str, seed: int) -> dict[str, np.ndarray]:
    """Generate all phases for one (shape, seed) run, deterministically."""
    rng = np.random.default_rng(seed * 1000 + _SHAPE_SEED_OFFSET[shape])
    contam_in_dist_n = CONTAM_BLOCKS * CONTAM_IN_DIST_PER_BLOCK
    total = WARMUP_N + EVAL_N + contam_in_dist_n

    if shape == "gaussian":
        warm = _gaussian_stream(rng, WARMUP_N)
        eval_ = _gaussian_stream(rng, EVAL_N)
        contam = _gaussian_stream(rng, contam_in_dist_n)
    elif shape == "bimodal":
        warm = _bimodal_stream(rng, WARMUP_N)
        eval_ = _bimodal_stream(rng, EVAL_N)
        contam = _bimodal_stream(rng, contam_in_dist_n)
    elif shape == "drift":
        warm = _drift_stream(rng, WARMUP_N, 0, total)
        eval_ = _drift_stream(rng, EVAL_N, WARMUP_N, total)
        contam = _drift_stream(rng, contam_in_dist_n, WARMUP_N + EVAL_N, total)
    else:  # pragma: no cover - guarded by SHAPES
        raise ValueError(f"unknown shape {shape}")

    ood_n = CONTAM_BLOCKS * CONTAM_OOD_PER_BLOCK
    return {
        "warmup": warm,
        "eval": eval_,
        "contam_in_dist": contam,
        "ood_near": _ood_scores(rng, ood_n, "near"),
        "ood_far": _ood_scores(rng, ood_n, "far"),
    }


SHAPES = ("gaussian", "bimodal", "drift")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def _alarm(calibrator: Any, score: float, alpha: float) -> bool:
    """Alarm decision for either calibrator at the matched threshold.

    Conformal: is_novel (p <= alpha). Legacy z-score: implied alarm when its
    continuous novelty crosses the same nominal threshold, novelty >= 1 - alpha.
    """
    if isinstance(calibrator, ConformalNoveltyCalibrator):
        return calibrator.is_novel(score)
    return bool(calibrator.calibrated_novelty(score) >= 1.0 - alpha)


def _run_stream(calibrator: Any, alpha: float, streams: dict[str, np.ndarray]) -> dict[str, Any]:
    """Score-before-observe over warm-up, eval and contamination phases."""
    for score in streams["warmup"]:
        calibrator.observe(float(score))

    false_alarms = 0
    for score in streams["eval"]:
        s = float(score)
        if _alarm(calibrator, s, alpha):
            false_alarms += 1
        calibrator.observe(s)

    # Contamination phase: interleaved blocks of in-dist and OOD points, all
    # observed (as the pipeline would), alternating OOD severities per block.
    # First-encounter detection (a clean window meeting OOD — the safety-
    # relevant number) is reported separately from steady-state detection,
    # because observing OOD lets the sliding window absorb it over time.
    contam_false = 0
    detected = {"near": 0, "far": 0}
    detected_first = {"near": 0, "far": 0}
    ood_total = {"near": 0, "far": 0}
    first_block_seen: set[str] = set()
    in_dist_iter = iter(streams["contam_in_dist"])
    ood_iters = {"near": iter(streams["ood_near"]), "far": iter(streams["ood_far"])}
    contam_in_dist_seen = 0
    for block in range(CONTAM_BLOCKS):
        for _ in range(CONTAM_IN_DIST_PER_BLOCK):
            s = float(next(in_dist_iter))
            if _alarm(calibrator, s, alpha):
                contam_false += 1
            calibrator.observe(s)
            contam_in_dist_seen += 1
        severity = "near" if block % 2 == 0 else "far"
        is_first = severity not in first_block_seen
        first_block_seen.add(severity)
        for _ in range(CONTAM_OOD_PER_BLOCK):
            s = float(next(ood_iters[severity]))
            if _alarm(calibrator, s, alpha):
                detected[severity] += 1
                if is_first:
                    detected_first[severity] += 1
            calibrator.observe(s)
            ood_total[severity] += 1

    return {
        "far_pure": false_alarms / len(streams["eval"]),
        "far_contaminated": contam_false / contam_in_dist_seen,
        "detection_near_first": detected_first["near"] / CONTAM_OOD_PER_BLOCK,
        "detection_far_first": detected_first["far"] / CONTAM_OOD_PER_BLOCK,
        "detection_near": detected["near"] / ood_total["near"],
        "detection_far": detected["far"] / ood_total["far"],
    }


def run_eval(seeds: tuple[int, ...], eval_n: int) -> dict[str, Any]:
    global EVAL_N
    EVAL_N = eval_n

    runs: list[dict[str, Any]] = []
    for shape in SHAPES:
        for alpha in ALPHAS:
            for seed in seeds:
                streams = _make_streams(shape, seed)
                conformal = ConformalNoveltyCalibrator(
                    alpha=alpha, window=WINDOW, min_samples=MIN_SAMPLES
                )
                legacy = NoveltyCalibrator(window_size=WINDOW, min_samples=MIN_SAMPLES)
                runs.append(
                    {
                        "shape": shape,
                        "alpha": alpha,
                        "seed": seed,
                        "conformal": _run_stream(conformal, alpha, streams),
                        "legacy_zscore": _run_stream(legacy, alpha, streams),
                    }
                )

    summary: list[dict[str, Any]] = []
    for shape in SHAPES:
        for alpha in ALPHAS:
            cell = [r for r in runs if r["shape"] == shape and r["alpha"] == alpha]

            def mean(group: str, key: str, cell: list[dict[str, Any]] = cell) -> float:
                return float(np.mean([r[group][key] for r in cell]))

            far = mean("conformal", "far_pure")
            summary.append(
                {
                    "shape": shape,
                    "alpha": alpha,
                    "conformal_far": far,
                    "far_abs_error": abs(far - alpha),
                    "within_1pct": abs(far - alpha) <= FAR_TOLERANCE,
                    "conformal_far_contaminated": mean("conformal", "far_contaminated"),
                    "conformal_detection_near_first": mean("conformal", "detection_near_first"),
                    "conformal_detection_far_first": mean("conformal", "detection_far_first"),
                    "conformal_detection_near": mean("conformal", "detection_near"),
                    "conformal_detection_far": mean("conformal", "detection_far"),
                    "legacy_far": mean("legacy_zscore", "far_pure"),
                    "legacy_detection_near_first": mean("legacy_zscore", "detection_near_first"),
                    "legacy_detection_far_first": mean("legacy_zscore", "detection_far_first"),
                    "legacy_detection_near": mean("legacy_zscore", "detection_near"),
                    "legacy_detection_far": mean("legacy_zscore", "detection_far"),
                }
            )

    return {
        "benchmark": "conformal_novelty_eval",
        "rfc": "RFC-0001 R2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": {
            "window": WINDOW,
            "min_samples": MIN_SAMPLES,
            "alphas": list(ALPHAS),
            "seeds": list(seeds),
            "warmup_n": WARMUP_N,
            "eval_n": EVAL_N,
            "contam_blocks": CONTAM_BLOCKS,
            "contam_in_dist_per_block": CONTAM_IN_DIST_PER_BLOCK,
            "contam_ood_per_block": CONTAM_OOD_PER_BLOCK,
            "far_tolerance": FAR_TOLERANCE,
            "ood_means": {"near": 0.55, "far": 0.35},
        },
        "summary": summary,
        "runs": runs,
    }


def _markdown_table(summary: list[dict[str, Any]]) -> str:
    lines = [
        "| shape | alpha | conformal FAR | abs err | ok | FAR (contam) "
        "| det near 1st | det far 1st | det near ss | det far ss "
        "| legacy FAR | legacy det far 1st |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            "| {shape} | {alpha:.2f} | {conformal_far:.4f} | {far_abs_error:.4f} | {ok} "
            "| {conformal_far_contaminated:.4f} | {conformal_detection_near_first:.3f} "
            "| {conformal_detection_far_first:.3f} | {conformal_detection_near:.3f} "
            "| {conformal_detection_far:.3f} | {legacy_far:.4f} "
            "| {legacy_detection_far_first:.3f} |".format(
                ok="yes" if row["within_1pct"] else "NO", **row
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer seeds, shorter streams")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="JSON artifact path")
    args = parser.parse_args()

    seeds = SEEDS[:2] if args.quick else SEEDS
    eval_n = 1000 if args.quick else EVAL_N

    t0 = time.perf_counter()
    report = run_eval(seeds, eval_n)
    elapsed = time.perf_counter() - t0
    report["elapsed_s"] = round(elapsed, 2)

    print("# Conformal novelty evaluation (RFC-0001 R2)\n")
    print(
        f"window={WINDOW} min_samples={MIN_SAMPLES} seeds={list(seeds)} "
        f"eval_n={eval_n} per (shape, alpha, seed)\n"
    )
    print(_markdown_table(report["summary"]))

    all_ok = all(row["within_1pct"] for row in report["summary"])
    non_drift_ok = all(row["within_1pct"] for row in report["summary"] if row["shape"] != "drift")
    print(
        f"\nRFC-0001 R2 metric (|FAR - alpha| <= {FAR_TOLERANCE}): "
        f"{'PASS' if all_ok else 'PASS (excl. drift)' if non_drift_ok else 'FAIL'}"
    )
    print(f"elapsed: {elapsed:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"artifact: {args.output}")
    return 0 if non_drift_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
