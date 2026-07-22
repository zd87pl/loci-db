"""wm_bench — the RFC-0001 R3 world-model memory benchmark (v1).

A runnable, honest benchmark suite for "memory for world models": five tasks
(future-state analog recall@k, novelty AUC over OOD segments, trajectory
reconstruction fidelity, recall-vs-age under consolidation, sustained-load
latency) run against a matrix of systems (LOCI local, LOCI local +
consolidation, LOCI over Qdrant ':memory:', naive Qdrant, brute-force
oracle).

v1 constraint: ALL DATA IS SYNTHETIC — this environment cannot download
Habitat/TartanAir/nuScenes, so v1 ships seeded synthetic trajectory
generators with documented realism limitations plus a dataset-adapter
protocol (:class:`~benchmarks.wm_bench.datasets.TrajectoryDataset`) so real
datasets plug in later.  Every artifact and report is labelled synthetic.

Usage:
    python -m benchmarks.wm_bench [--quick] [--tasks ...] [--systems ...]

See docs/BENCHMARK_METHODOLOGY.md ("World-model memory benchmark") for the
methodology and benchmarks/results/wm_bench_latest.json for the latest
checked-in run.
"""

from benchmarks.wm_bench.datasets import (
    OodPatrolDataset,
    SmoothPatrolDataset,
    TrajectoryDataset,
    TrajectoryPoint,
    WarehouseDataset,
)
from benchmarks.wm_bench.systems import SYSTEM_NAMES, QueryHit, SystemUnderTest, build_system
from benchmarks.wm_bench.tasks import TASK_NAMES

__all__ = [
    "SYSTEM_NAMES",
    "TASK_NAMES",
    "OodPatrolDataset",
    "QueryHit",
    "SmoothPatrolDataset",
    "SystemUnderTest",
    "TrajectoryDataset",
    "TrajectoryPoint",
    "WarehouseDataset",
    "build_system",
]
