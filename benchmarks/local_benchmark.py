#!/usr/bin/env python3
"""Benchmark Engram's indexing overhead — no Docker required.

Uses LocalEngramClient (in-memory backend) to measure:
1. Batch insert throughput
2. Query latency (brute-force with Hilbert pre-filtering)
3. Spatial query selectivity
4. Temporal shard fan-out overhead
5. Adaptive resolution density tracking overhead

Run:  python benchmarks/local_benchmark.py
"""

from __future__ import annotations

import random
import statistics
import time

from engram.local_client import LocalEngramClient
from engram.schema import WorldState
from engram.spatial.adaptive import AdaptiveResolution

VECTOR_DIM = 128
SEED = 42


def _random_states(n: int, base_ms: int, dim: int = VECTOR_DIM) -> list[WorldState]:
    return [
        WorldState(
            x=random.random(),
            y=random.random(),
            z=random.random(),
            timestamp_ms=base_ms + i * 10,
            vector=[random.gauss(0, 1) for _ in range(dim)],
            scene_id=f"scene_{i % 10}",
            scale_level="patch",
        )
        for i in range(n)
    ]


def bench_insert(n: int = 5000) -> None:
    random.seed(SEED)
    client = LocalEngramClient(vector_size=VECTOR_DIM, decay_lambda=0)
    states = _random_states(n, int(time.time() * 1000))

    t0 = time.perf_counter()
    client.insert_batch(states)
    elapsed = time.perf_counter() - t0

    rate = n / elapsed
    print(f"INSERT  {n:,} states:  {elapsed:.3f}s  ({rate:,.0f} states/s)")


def bench_query(n_data: int = 5000, n_queries: int = 200) -> None:
    random.seed(SEED)
    now_ms = int(time.time() * 1000)
    client = LocalEngramClient(vector_size=VECTOR_DIM, decay_lambda=0)
    client.insert_batch(_random_states(n_data, now_ms))

    latencies: list[float] = []
    for _ in range(n_queries):
        qv = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        t0 = time.perf_counter()
        client.query(vector=qv, limit=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    print(
        f"QUERY   (no filter) {n_queries} queries on {n_data:,} points:  "
        f"p50={p50:.2f}ms  p99={p99:.2f}ms"
    )


def bench_spatial_query(n_data: int = 5000, n_queries: int = 200) -> None:
    random.seed(SEED)
    now_ms = int(time.time() * 1000)
    client = LocalEngramClient(vector_size=VECTOR_DIM, decay_lambda=0)
    client.insert_batch(_random_states(n_data, now_ms))

    latencies: list[float] = []
    total_candidates = 0
    for _ in range(n_queries):
        qv = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        t0 = time.perf_counter()
        client.query(
            vector=qv,
            spatial_bounds={
                "x_min": 0.2,
                "x_max": 0.6,
                "y_min": 0.3,
                "y_max": 0.7,
                "z_min": 0.0,
                "z_max": 1.0,
            },
            limit=10,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        total_candidates += client.last_query_stats.total_candidates

    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    avg_candidates = total_candidates / n_queries
    print(
        f"QUERY   (spatial)   {n_queries} queries on {n_data:,} points:  "
        f"p50={p50:.2f}ms  p99={p99:.2f}ms  avg_candidates={avg_candidates:.0f}"
    )


def bench_temporal_query(n_data: int = 5000, n_queries: int = 200) -> None:
    random.seed(SEED)
    now_ms = int(time.time() * 1000)
    client = LocalEngramClient(
        vector_size=VECTOR_DIM,
        epoch_size_ms=1000,
        decay_lambda=0,
    )
    client.insert_batch(_random_states(n_data, now_ms))

    latencies: list[float] = []
    for _ in range(n_queries):
        qv = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
        # Query a narrow 2-second window (2 shards)
        t0 = time.perf_counter()
        client.query(
            vector=qv,
            time_window_ms=(now_ms + 1000, now_ms + 3000),
            limit=10,
        )
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = statistics.median(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    shards = client.last_query_stats.shards_searched
    print(
        f"QUERY   (temporal)  {n_queries} queries on {n_data:,} points:  "
        f"p50={p50:.2f}ms  p99={p99:.2f}ms  shards={shards}"
    )


def bench_adaptive_resolution(n: int = 10000) -> None:
    random.seed(SEED)
    ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=50)

    # Mix of dense and sparse regions
    t0 = time.perf_counter()
    for i in range(n):
        if i % 3 == 0:
            # Dense cluster at (0.5, 0.5, 0.5)
            x = 0.5 + random.gauss(0, 0.02)
            y = 0.5 + random.gauss(0, 0.02)
            z = 0.5 + random.gauss(0, 0.02)
        else:
            x, y, z = random.random(), random.random(), random.random()
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        z = max(0, min(1, z))
        ar.record(x, y, z, random.random())
    elapsed = (time.perf_counter() - t0) * 1000

    stats = ar.stats()
    print(
        f"ADAPTIVE  {n:,} records:  {elapsed:.1f}ms  "
        f"cells={stats.num_cells}  hot_cells={stats.hot_cells}  "
        f"max_density={stats.max_density}"
    )


def bench_trajectory(n_data: int = 1000) -> None:
    random.seed(SEED)
    now_ms = int(time.time() * 1000)
    client = LocalEngramClient(vector_size=VECTOR_DIM, decay_lambda=0)

    states = [
        WorldState(
            x=0.5,
            y=0.5,
            z=0.5,
            timestamp_ms=now_ms + i * 10,
            vector=[random.gauss(0, 1) for _ in range(VECTOR_DIM)],
            scene_id="trajectory_scene",
        )
        for i in range(n_data)
    ]
    ids = client.insert_batch(states)

    t0 = time.perf_counter()
    traj = client.get_trajectory(ids[n_data // 2], steps_back=50, steps_forward=50)
    elapsed = (time.perf_counter() - t0) * 1000

    print(
        f"TRAJECTORY  {n_data:,} states, walk 100 steps:  {elapsed:.2f}ms  found={len(traj)} states"
    )


def main() -> None:
    print("=" * 70)
    print("Engram Local Benchmark (in-memory backend)")
    print("=" * 70)
    print()

    bench_insert()
    bench_query()
    bench_spatial_query()
    bench_temporal_query()
    bench_adaptive_resolution()
    bench_trajectory()

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
