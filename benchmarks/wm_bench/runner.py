"""wm_bench runner: tasks x systems matrix, JSON artifact, markdown report.

Run from the repo root:

    python -m benchmarks.wm_bench                # full suite (~2-4 min)
    python -m benchmarks.wm_bench --quick        # reduced sizes for CI
    python -m benchmarks.wm_bench --tasks novelty_auc,recall_vs_age
    python -m benchmarks.wm_bench --systems loci_local,naive_qdrant

Honesty rules baked in: every dataset and every output is labelled
SYNTHETIC (v1 has no real embodied datasets — see datasets.py); the
brute-force oracle rows are always reportable; losing numbers are printed
exactly like winning ones.  All metrics except latency are deterministic
for a given seed.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np

from .report import format_markdown
from .systems import DEFAULT_EPOCH_SIZE_MS, SYSTEM_NAMES, build_system
from .tasks import (
    TASK_NAMES,
    future_analog_recall,
    novelty_auc,
    recall_vs_age,
    sustained_load,
    trajectory_fidelity,
)

SCHEMA_VERSION = 1

SYNTHETIC_NOTICE = (
    "All v1 wm_bench results are computed on SYNTHETIC trajectories "
    "(see benchmarks/wm_bench/datasets.py for generator definitions and "
    "documented realism limitations). No real embodied dataset is involved; "
    "treat every number as a property of the retrieval systems under a "
    "synthetic workload, not a field result."
)

_TASK_FUNCS = {
    "future_analog_recall": future_analog_recall,
    "novelty_auc": novelty_auc,
    "trajectory_fidelity": trajectory_fidelity,
    "recall_vs_age": recall_vs_age,
    "sustained_load": sustained_load,
}


def task_params(quick: bool) -> dict[str, dict]:
    """Per-task size parameters (full vs --quick)."""
    if quick:
        return {
            "future_analog_recall": {"n_points": 600, "n_queries": 24, "k": 10},
            "novelty_auc": {"n_points": 400, "n_ood_segments": 2, "ood_segment_len": 30},
            "trajectory_fidelity": {"n_scenes": 3, "n_visits": 2, "points_per_visit": 30},
            "recall_vs_age": {"n_points": 1200, "n_age_buckets": 6, "queries_per_bucket": 5},
            "sustained_load": {"n_points": 250, "query_every": 5},
        }
    return {
        "future_analog_recall": {"n_points": 1500, "n_queries": 60, "k": 10},
        "novelty_auc": {"n_points": 800, "n_ood_segments": 3, "ood_segment_len": 40},
        "trajectory_fidelity": {"n_scenes": 4, "n_visits": 3, "points_per_visit": 40},
        "recall_vs_age": {"n_points": 3000, "n_age_buckets": 6, "queries_per_bucket": 8},
        "sustained_load": {"n_points": 600, "query_every": 5},
    }


def _versions() -> dict[str, str]:
    def pkg(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "unknown"

    return {
        "python": platform.python_version(),
        "platform": platform.machine(),
        "numpy": np.__version__,
        "qdrant-client": pkg("qdrant-client"),
        "loci-stdb": pkg("loci-stdb"),
    }


def run_suite(
    *,
    tasks: list[str] | None = None,
    systems: list[str] | None = None,
    quick: bool = False,
    seed: int | None = None,
    epoch_size_ms: int = DEFAULT_EPOCH_SIZE_MS,
) -> dict:
    """Run the tasks x systems matrix and return the result document."""
    task_names = tasks or list(TASK_NAMES)
    system_names = systems or list(SYSTEM_NAMES)
    for t in task_names:
        if t not in _TASK_FUNCS:
            raise ValueError(f"unknown task {t!r}; known: {TASK_NAMES}")
    params = task_params(quick)

    results: dict[str, dict[str, dict]] = {}
    for task_name in task_names:
        task_fn = _TASK_FUNCS[task_name]
        kwargs = dict(params[task_name])
        if seed is not None:
            kwargs["seed"] = seed
        if task_name == "recall_vs_age":
            kwargs["epoch_size_ms"] = epoch_size_ms
        results[task_name] = {}
        for system_name in system_names:
            system = build_system(system_name, epoch_size_ms=epoch_size_ms)
            results[task_name][system_name] = task_fn(system, **kwargs)

    return {
        "benchmark": "wm_bench",
        "schema_version": SCHEMA_VERSION,
        "data": "synthetic",
        "synthetic_notice": SYNTHETIC_NOTICE,
        "quick": quick,
        "seed": seed,  # null = each task's documented default seed
        "config": {
            "epoch_size_ms": epoch_size_ms,
            "tasks": {t: params[t] for t in task_names},
            "systems": system_names,
        },
        "versions": _versions(),
        "results": results,
    }


def _default_output_path() -> Path:
    return Path(__file__).resolve().parent.parent / "results" / "wm_bench_latest.json"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.wm_bench",
        description="LOCI world-model memory benchmark (v1: synthetic data only)",
    )
    parser.add_argument(
        "--tasks", type=_csv, default=None, help=f"comma-separated subset of {TASK_NAMES}"
    )
    parser.add_argument(
        "--systems", type=_csv, default=None, help=f"comma-separated subset of {SYSTEM_NAMES}"
    )
    parser.add_argument("--quick", action="store_true", help="reduced sizes for fast runs/CI")
    parser.add_argument(
        "--seed", type=int, default=None, help="override every task's default seed"
    )
    parser.add_argument("--epoch-size-ms", type=int, default=DEFAULT_EPOCH_SIZE_MS)
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--no-write", action="store_true", help="do not write the JSON artifact")
    args = parser.parse_args(argv)

    result = run_suite(
        tasks=args.tasks,
        systems=args.systems,
        quick=args.quick,
        seed=args.seed,
        epoch_size_ms=args.epoch_size_ms,
    )

    print(format_markdown(result))
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"Results saved to {args.output}")
    return result


if __name__ == "__main__":
    main()
