#!/usr/bin/env python3
"""LOCI retrieval benchmark — measures raw data-retrieval latency.

Benchmarks the query path in isolation (no voice, no embedding, no camera)
across dataset sizes and query types that match real usage.

Outputs a markdown summary and writes JSON to benchmarks/results/.

Run:  python benchmarks/benchmark_retrieval.py [--quick]
"""

from __future__ import annotations

import inspect as _inspect
import json
import os
import platform
import random
import statistics
import sys
import time

from loci.local_client import LocalLociClient
from loci.schema import WorldState

VECTOR_DIM = 128
SEED = 42
WARMUP = 20
QUERIES_PER_TEST = 300
LIMIT = 10


# ── Data generation ──────────────────────────────────────────────────────


def _make_states(
    n: int,
    base_ms: int,
    *,
    n_labels: int = 10,
    dim: int = VECTOR_DIM,
) -> list[WorldState]:
    labels = [f"object_{i}" for i in range(n_labels)]
    return [
        WorldState(
            x=random.random(),
            y=random.random(),
            z=random.random(),
            timestamp_ms=base_ms + i * 50,
            vector=[random.gauss(0, 1) for _ in range(dim)],
            scene_id=labels[i % n_labels],
            scale_level="patch",
        )
        for i in range(n)
    ]


def _random_query_vector(dim: int = VECTOR_DIM) -> list[float]:
    return [random.gauss(0, 1) for _ in range(dim)]


# ── Benchmark routines ───────────────────────────────────────────────────


def _bench_latencies(fn, n_warmup: int = WARMUP, n_queries: int = QUERIES_PER_TEST) -> list[float]:
    for _ in range(n_warmup):
        fn()
    latencies: list[float] = []
    for _ in range(n_queries):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1_000_000)  # microseconds
    return latencies


def _summarise(latencies: list[float]) -> dict:
    s = sorted(latencies)
    return {
        "p50_us": round(statistics.median(s), 1),
        "p95_us": round(s[int(len(s) * 0.95)], 1),
        "p99_us": round(s[int(len(s) * 0.99)], 1),
        "mean_us": round(statistics.mean(s), 1),
        "min_us": round(s[0], 1),
        "max_us": round(s[-1], 1),
    }


def bench_vector_only(client: LocalLociClient, dim: int) -> dict:
    """ANN search across all shards, no spatial/temporal filter."""
    lats = _bench_latencies(lambda: client.query(vector=_random_query_vector(dim), limit=LIMIT))
    return _summarise(lats)


def bench_temporal(client: LocalLociClient, dim: int, base_ms: int, n: int) -> dict:
    """ANN search with time-window shard pruning."""
    span = n * 50
    window_start = base_ms + span // 4
    window_end = base_ms + span * 3 // 4
    lats = _bench_latencies(
        lambda: client.query(
            vector=_random_query_vector(dim),
            time_window_ms=(window_start, window_end),
            limit=LIMIT,
        )
    )
    return _summarise(lats)


def bench_label_filter(client: LocalLociClient, dim: int) -> dict:
    """Payload-filtered search (demo 'where is X?' path)."""
    lats = _bench_latencies(
        lambda: client.query(
            vector=_random_query_vector(dim),
            limit=LIMIT,
            _extra_payload_filter={"scene_id": "object_0"},
        )
    )
    return _summarise(lats)


def bench_spatial(client: LocalLociClient, dim: int) -> dict:
    """Spatial bounding-box search with Hilbert pre-filter."""
    bounds = {
        "x_min": 0.3,
        "x_max": 0.7,
        "y_min": 0.3,
        "y_max": 0.7,
        "z_min": 0.0,
        "z_max": 1.0,
    }
    lats = _bench_latencies(
        lambda: client.query(vector=_random_query_vector(dim), spatial_bounds=bounds, limit=LIMIT)
    )
    return _summarise(lats)


def bench_spatial_temporal(client: LocalLociClient, dim: int, base_ms: int, n: int) -> dict:
    """Combined spatial + temporal query."""
    span = n * 50
    window_start = base_ms + span // 4
    window_end = base_ms + span * 3 // 4
    bounds = {
        "x_min": 0.3,
        "x_max": 0.7,
        "y_min": 0.3,
        "y_max": 0.7,
        "z_min": 0.0,
        "z_max": 1.0,
    }
    lats = _bench_latencies(
        lambda: client.query(
            vector=_random_query_vector(dim),
            spatial_bounds=bounds,
            time_window_ms=(window_start, window_end),
            limit=LIMIT,
        )
    )
    return _summarise(lats)


# ── Scenario runner ──────────────────────────────────────────────────────


BENCHMARKS = [
    ("vector_only", "Vector-only ANN", bench_vector_only),
    ("temporal", "Temporal shard pruning", bench_temporal),
    ("label_filter", "Label filter (demo path)", bench_label_filter),
    ("spatial", "Spatial bbox", bench_spatial),
    ("spatial_temporal", "Spatial + temporal", bench_spatial_temporal),
]


def run_dataset_size(n: int, base_ms: int) -> dict:
    random.seed(SEED)
    client = LocalLociClient(vector_size=VECTOR_DIM, epoch_size_ms=5000, decay_lambda=0)
    states = _make_states(n, base_ms)

    t0 = time.perf_counter()
    client.insert_batch(states)
    insert_s = time.perf_counter() - t0

    results: dict = {
        "n": n,
        "insert_s": round(insert_s, 4),
        "insert_rate": round(n / insert_s, 0),
    }

    for key, label, fn in BENCHMARKS:
        sys.stdout.write(f"  {label}...")
        sys.stdout.flush()
        sig = _inspect.signature(fn)
        if len(sig.parameters) == 2:
            result = fn(client, VECTOR_DIM)
        else:
            result = fn(client, VECTOR_DIM, base_ms, n)
        results[key] = result
        p50 = result["p50_us"]
        fmt = f"{p50:.0f}µs" if p50 < 1000 else f"{p50 / 1000:.2f}ms"
        print(f" p50={fmt}")

    return results


# ── Output ───────────────────────────────────────────────────────────────


def _us_to_display(us: float) -> str:
    if us < 1000:
        return f"{us:.0f}µs"
    return f"{us / 1000:.2f}ms"


def print_results(all_results: dict) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("## LOCI Retrieval Benchmark")
    lines.append("")
    lines.append(
        f"Configuration: {platform.machine()}, Python {platform.python_version()}, "
        f"dim={VECTOR_DIM}, {QUERIES_PER_TEST} queries/test, limit={LIMIT}"
    )
    lines.append("")

    for entry in all_results["datasets"]:
        n = entry["n"]
        lines.append(f"### N = {n:,} ({entry['insert_rate']:,.0f} inserts/s)")
        lines.append("")
        lines.append("| Query Type | P50 | P99 | Mean |")
        lines.append("|:--|--:|--:|--:|")

        for key, label, _ in BENCHMARKS:
            m = entry[key]
            lines.append(
                f"| {label} "
                f"| {_us_to_display(m['p50_us'])} "
                f"| {_us_to_display(m['p99_us'])} "
                f"| {_us_to_display(m['mean_us'])} |"
            )
        lines.append("")

    # Headline
    demo_entry = all_results["datasets"][0]
    label_p50 = demo_entry["label_filter"]["p50_us"]
    vec_p50 = demo_entry["vector_only"]["p50_us"]
    lines.append(
        f"**Headline:** On {demo_entry['n']:,} objects, "
        f"label-filtered query (demo path) = **{_us_to_display(label_p50)}** p50, "
        f"vector-only ANN = **{_us_to_display(vec_p50)}** p50."
    )
    lines.append("")

    table = "\n".join(lines)
    print(table)
    return table


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    dataset_sizes = [100, 500, 1_000, 5_000, 10_000]

    if "--quick" in sys.argv:
        dataset_sizes = [100, 500, 1_000]

    print("=" * 60)
    print("  LOCI Retrieval Benchmark")
    print(f"  dim={VECTOR_DIM}, {QUERIES_PER_TEST} queries/test")
    print("=" * 60)

    base_ms = int(time.time() * 1000)

    all_results: dict = {
        "config": {
            "vector_dim": VECTOR_DIM,
            "queries_per_test": QUERIES_PER_TEST,
            "warmup": WARMUP,
            "limit": LIMIT,
            "seed": SEED,
            "dataset_sizes": dataset_sizes,
            "platform": platform.machine(),
            "python": platform.python_version(),
        },
        "datasets": [],
    }

    for n in dataset_sizes:
        print(f"\n--- N = {n:,} ---")
        result = run_dataset_size(n, base_ms)
        all_results["datasets"].append(result)

    print_results(all_results)

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "retrieval_latest.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
