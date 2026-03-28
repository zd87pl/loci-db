#!/usr/bin/env python3
"""Publication-quality benchmark: LOCI vs naive Qdrant.

Compares four query methods against brute-force ground truth:
1. Naive Qdrant — single collection, 3 float-range filters (x, y, z) + timestamp
2. LOCI r4 — historical fixed-r4 baseline without query-time 4D routing
3. LOCI r4 + overlap — same with overlap_factor=1.2 (20% expanded search)
4. LOCI current — mirrors the shipped query path: epoch-local 4D bounds,
   per-shard overfetch, exact post-filtering, and overlap-based Hilbert routing

Four benchmark scenarios:
A. Tight spatial query  (radius 0.05)
B. Wide spatial query   (radius 0.5)
C. Combined spatial + temporal  (radius 0.05, tight time window)
D. Retrieval stress test        (radius 0.3, short time window)

Dataset sizes: N = [1_000, 10_000, 100_000]  (configurable)

Uses qdrant-client's in-memory mode by default. Set QDRANT_URL for a live server.
Outputs a markdown table and writes results to benchmarks/results/latest.json.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from loci.spatial.filtering import exact_payload_match
from loci.spatial.hilbert import HilbertIndex
from loci.spatial.query_plan import bounds_for_epoch, choose_query_resolution
from loci.temporal.decay import apply_decay
from loci.temporal.sharding import epoch_id, epochs_in_range

# ── Configuration ──────────────────────────────────────────────────────────
VECTOR_DIM = 512
DATASET_SIZES = [1_000, 10_000]  # add 100_000 for full run (slow)
NUM_QUERIES = 50
WARMUP_QUERIES = 5
N_RUNS = 3  # increase to 10 for publication
LIMIT = 10
EPOCH_SIZE_MS = 5_000
SEED = 42
DECAY_LAMBDA = 1e-4

# Scenario definitions: (spatial_radius, time_window_fraction, name, description)
SCENARIOS = {
    "A": {"name": "Tight spatial", "spatial_radius": 0.05, "time_frac": 1.0},
    "B": {"name": "Wide spatial", "spatial_radius": 0.5, "time_frac": 1.0},
    "C": {"name": "Spatial+temporal", "spatial_radius": 0.05, "time_frac": 0.1},
    "D": {"name": "Predict-retrieve", "spatial_radius": 0.3, "time_frac": 0.1},
}

_hilbert = HilbertIndex(resolutions=[4, 8, 12])


# ── Data generation ────────────────────────────────────────────────────────


def generate_dataset(n: int, dim: int, rng: np.random.Generator) -> tuple[list[dict], int, int]:
    """Generate N random world-state points with (x,y,z,t) and a normalized vector."""
    now_ms = int(time.time() * 1000)
    xs = rng.random(n).astype(np.float64)
    ys = rng.random(n).astype(np.float64)
    zs = rng.random(n).astype(np.float64)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    data = []
    for i in range(n):
        ts = now_ms + i * 10
        data.append(
            {
                "x": float(xs[i]),
                "y": float(ys[i]),
                "z": float(zs[i]),
                "timestamp_ms": ts,
                "vector": vecs[i].tolist(),
                "index": i,
            }
        )
    start_ms = data[0]["timestamp_ms"]
    end_ms = data[-1]["timestamp_ms"]
    return data, start_ms, end_ms


def generate_queries(
    n: int,
    dim: int,
    rng: np.random.Generator,
    start_ms: int,
    end_ms: int,
    spatial_radius: float,
    time_frac: float,
) -> list[dict]:
    """Generate N queries with configurable spatial radius and temporal window."""
    queries = []
    for _ in range(n):
        cx, cy, cz = float(rng.random()), float(rng.random()), float(rng.random())
        x_min = max(0.0, cx - spatial_radius)
        x_max = min(1.0, cx + spatial_radius)
        y_min = max(0.0, cy - spatial_radius)
        y_max = min(1.0, cy + spatial_radius)
        z_min = max(0.0, cz - spatial_radius)
        z_max = min(1.0, cz + spatial_radius)

        span = end_ms - start_ms
        w = time_frac * span
        t_start = start_ms + float(rng.random()) * max(1, span - w)
        t_end = t_start + w

        vec = rng.standard_normal(dim).astype(np.float32)
        vec = vec / max(np.linalg.norm(vec), 1e-9)

        queries.append(
            {
                "vector": vec.tolist(),
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
                "t_start": int(t_start),
                "t_end": int(t_end),
                "center": (cx, cy, cz),
            }
        )
    return queries


# ── Ground truth ───────────────────────────────────────────────────────────


def compute_ground_truth(
    data: list[dict],
    q: dict,
    limit: int,
    *,
    now_ms: int | None = None,
    decay_lambda: float = 0.0,
) -> list[int]:
    """Brute-force ground truth: filter by bounds/time, rank by shipped semantics."""
    qv = np.array(q["vector"], dtype=np.float32)
    qv_norm = np.linalg.norm(qv)
    if qv_norm == 0:
        return []

    candidates = []
    for d in data:
        if (
            q["x_min"] <= d["x"] <= q["x_max"]
            and q["y_min"] <= d["y"] <= q["y_max"]
            and q["z_min"] <= d["z"] <= q["z_max"]
            and q["t_start"] <= d["timestamp_ms"] <= q["t_end"]
        ):
            dv = np.array(d["vector"], dtype=np.float32)
            dv_norm = np.linalg.norm(dv)
            if dv_norm == 0:
                continue
            sim = float(np.dot(qv, dv) / (qv_norm * dv_norm))
            candidates.append(
                {
                    "id": d["index"],
                    "score": sim,
                    "timestamp_ms": d["timestamp_ms"],
                }
            )

    if decay_lambda > 0:
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        apply_decay(candidates, now_ms, decay_lambda)
        return [c["id"] for c in candidates[:limit]]

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return [c["id"] for c in candidates[:limit]]


def compute_recall(
    result_ids: list[str], truth_indices: list[int], id_map: dict[str, int]
) -> float:
    if not truth_indices:
        return 1.0
    truth_set = set(truth_indices)
    found = sum(1 for rid in result_ids if id_map.get(rid, -1) in truth_set)
    return found / len(truth_indices)


# ── Naive Qdrant ───────────────────────────────────────────────────────────


def setup_naive(
    qdrant: QdrantClient, data: list[dict], collection: str = "benchmark_naive"
) -> dict[str, int]:
    """Insert all points into a single flat collection with float payloads."""
    if collection in {c.name for c in qdrant.get_collections().collections}:
        qdrant.delete_collection(collection)
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    qdrant.create_payload_index(collection, "x", PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(collection, "y", PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(collection, "z", PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(collection, "timestamp_ms", PayloadSchemaType.INTEGER)

    id_map: dict[str, int] = {}
    batch_size = 500
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        points = []
        for d in batch:
            pid = uuid.uuid4().hex
            id_map[pid] = d["index"]
            points.append(
                PointStruct(
                    id=pid,
                    vector=d["vector"],
                    payload={
                        "x": d["x"],
                        "y": d["y"],
                        "z": d["z"],
                        "timestamp_ms": d["timestamp_ms"],
                    },
                )
            )
        qdrant.upsert(collection_name=collection, points=points)
    return id_map


def query_naive(qdrant: QdrantClient, q: dict, collection: str = "benchmark_naive") -> list[str]:
    resp = qdrant.query_points(
        collection_name=collection,
        query=q["vector"],
        query_filter=Filter(
            must=[
                FieldCondition(key="x", range=Range(gte=q["x_min"], lte=q["x_max"])),
                FieldCondition(key="y", range=Range(gte=q["y_min"], lte=q["y_max"])),
                FieldCondition(key="z", range=Range(gte=q["z_min"], lte=q["z_max"])),
                FieldCondition(
                    key="timestamp_ms",
                    range=Range(gte=q["t_start"], lte=q["t_end"]),
                ),
            ]
        ),
        limit=LIMIT,
        with_vectors=False,
    )
    return [str(h.id) for h in resp.points]


# ── LOCI (Hilbert bucketing + temporal sharding) ──────────────────────────


def _normalise_time(timestamp_ms: int, ep: int) -> float:
    epoch_start = ep * EPOCH_SIZE_MS
    offset = timestamp_ms - epoch_start
    return min(1.0, max(0.0, offset / EPOCH_SIZE_MS))


def setup_loci(
    qdrant: QdrantClient, data: list[dict], prefix: str = "loci"
) -> tuple[dict[str, int], set[str]]:
    """Insert all points into epoch-sharded collections with multi-res Hilbert payloads."""
    groups: dict[str, list[PointStruct]] = {}
    id_map: dict[str, int] = {}

    for d in data:
        ep = epoch_id(d["timestamp_ms"], EPOCH_SIZE_MS)
        col = f"{prefix}_{ep}"
        t_norm = _normalise_time(d["timestamp_ms"], ep)
        hilbert_ids = _hilbert.encode(d["x"], d["y"], d["z"], t_norm)
        pid = uuid.uuid4().hex
        id_map[pid] = d["index"]
        payload = {
            "x": d["x"],
            "y": d["y"],
            "z": d["z"],
            "timestamp_ms": d["timestamp_ms"],
        }
        payload.update(hilbert_ids)
        point = PointStruct(id=pid, vector=d["vector"], payload=payload)
        groups.setdefault(col, []).append(point)

    # Clean up any existing collections with this prefix
    existing = {c.name for c in qdrant.get_collections().collections}
    for col in list(groups.keys()):
        if col in existing:
            qdrant.delete_collection(col)

    for col, points in groups.items():
        qdrant.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        for resolution in _hilbert.resolutions:
            qdrant.create_payload_index(
                col,
                f"hilbert_r{resolution}",
                PayloadSchemaType.INTEGER,
            )
        qdrant.create_payload_index(col, "timestamp_ms", PayloadSchemaType.INTEGER)
        batch_size = 500
        for i in range(0, len(points), batch_size):
            qdrant.upsert(collection_name=col, points=points[i : i + batch_size])

    known = set(groups.keys())
    return id_map, known


def query_loci(
    qdrant: QdrantClient,
    q: dict,
    known_collections: set[str],
    overlap_factor: float = 1.0,
    prefix: str = "loci",
) -> list[str]:
    """Run a LOCI query: Hilbert MatchAny + exact post-filter + shard routing."""
    epochs = epochs_in_range(q["t_start"], q["t_end"], EPOCH_SIZE_MS)
    collections = [f"{prefix}_{e}" for e in epochs]

    bounds = {
        "x_min": q["x_min"],
        "x_max": q["x_max"],
        "y_min": q["y_min"],
        "y_max": q["y_max"],
        "z_min": q["z_min"],
        "z_max": q["z_max"],
    }
    hids = _hilbert.query_buckets(bounds, resolution=4, overlap_factor=overlap_factor)

    must_conditions = [
        FieldCondition(
            key="timestamp_ms",
            range=Range(gte=q["t_start"], lte=q["t_end"]),
        ),
    ]
    if hids:
        must_conditions.append(FieldCondition(key="hilbert_r4", match=MatchAny(any=hids)))
    query_filter = Filter(must=must_conditions)

    per_shard_limit = LIMIT * 3

    all_hits: list[tuple[str, float]] = []
    for col in collections:
        if col not in known_collections:
            continue
        try:
            resp = qdrant.query_points(
                collection_name=col,
                query=q["vector"],
                query_filter=query_filter,
                limit=per_shard_limit,
                with_vectors=False,
                with_payload=True,
            )
            for h in resp.points:
                if exact_payload_match(
                    h.payload or {},
                    spatial_bounds=bounds,
                    time_window_ms=(q["t_start"], q["t_end"]),
                ):
                    all_hits.append((str(h.id), h.score))
        except Exception:
            continue

    all_hits.sort(key=lambda x: x[1], reverse=True)
    return [h[0] for h in all_hits[:LIMIT]]


def query_loci_current(
    qdrant: QdrantClient,
    q: dict,
    known_collections: set[str],
    prefix: str = "loci",
    overlap_factor: float = 1.2,
    now_ms: int | None = None,
    decay_lambda: float = DECAY_LAMBDA,
) -> list[str]:
    """Run the shipped LOCI query path with epoch-local 4D Hilbert routing."""
    epochs = epochs_in_range(q["t_start"], q["t_end"], EPOCH_SIZE_MS)
    time_window = (q["t_start"], q["t_end"])
    spatial_bounds = {
        "x_min": q["x_min"],
        "x_max": q["x_max"],
        "y_min": q["y_min"],
        "y_max": q["y_max"],
        "z_min": q["z_min"],
        "z_max": q["z_max"],
    }
    per_shard_limit = LIMIT * 3

    all_hits: list[dict] = []
    for ep in epochs:
        col = f"{prefix}_{ep}"
        if col not in known_collections:
            continue

        query_resolution = choose_query_resolution(
            _hilbert,
            adaptive=None,
            spatial_bounds=spatial_bounds,
            time_window_ms=time_window,
            ep=ep,
            epoch_size_ms=EPOCH_SIZE_MS,
            overlap_factor=overlap_factor,
        )
        epoch_bounds = bounds_for_epoch(
            spatial_bounds,
            time_window,
            ep,
            EPOCH_SIZE_MS,
        )
        hids = _hilbert.query_buckets(
            epoch_bounds,
            resolution=query_resolution,
            overlap_factor=overlap_factor,
        )
        if not hids:
            continue

        query_filter = Filter(
            must=[
                FieldCondition(
                    key=_hilbert.payload_field(query_resolution),
                    match=MatchAny(any=hids),
                ),
                FieldCondition(
                    key="timestamp_ms",
                    range=Range(gte=q["t_start"], lte=q["t_end"]),
                ),
            ]
        )

        try:
            resp = qdrant.query_points(
                collection_name=col,
                query=q["vector"],
                query_filter=query_filter,
                limit=per_shard_limit,
                with_vectors=False,
                with_payload=True,
            )
        except Exception:
            continue

        for hit in resp.points:
            if exact_payload_match(
                hit.payload or {},
                spatial_bounds=spatial_bounds,
                time_window_ms=time_window,
            ):
                all_hits.append(
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "timestamp_ms": (hit.payload or {}).get("timestamp_ms", 0),
                    }
                )

    if not all_hits:
        return []

    if now_ms is None:
        now_ms = int(time.time() * 1000)
    if decay_lambda > 0:
        apply_decay(all_hits, now_ms, decay_lambda)
    else:
        all_hits.sort(key=lambda x: x["score"], reverse=True)

    return [hit["id"] for hit in all_hits[:LIMIT]]


# ── Benchmark runner ──────────────────────────────────────────────────────


def percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100)."""
    sorted_data = sorted(data)
    idx = int(p / 100 * len(sorted_data))
    return sorted_data[min(idx, len(sorted_data) - 1)]


def run_scenario(
    scenario_id: str,
    n_vectors: int,
    rng: np.random.Generator,
) -> dict:
    """Run a single scenario at a given dataset size and return metrics."""
    scenario = SCENARIOS[scenario_id]
    spatial_radius = scenario["spatial_radius"]
    time_frac = scenario["time_frac"]

    print(
        f"  Scenario {scenario_id} ({scenario['name']}): "
        f"radius={spatial_radius}, time_frac={time_frac}"
    )

    # Generate data and queries
    data, start_ms, end_ms = generate_dataset(n_vectors, VECTOR_DIM, rng)
    total_queries = WARMUP_QUERIES + NUM_QUERIES
    queries = generate_queries(
        total_queries, VECTOR_DIM, rng, start_ms, end_ms, spatial_radius, time_frac
    )
    benchmark_now_ms = int(time.time() * 1000)

    # Setup: use separate in-memory clients for isolation
    qdrant_url = os.environ.get("QDRANT_URL", "")
    if qdrant_url:
        naive_client = QdrantClient(url=qdrant_url, timeout=120)
        loci_client = QdrantClient(url=qdrant_url, timeout=120)
    else:
        naive_client = QdrantClient(location=":memory:")
        loci_client = QdrantClient(location=":memory:")

    # Insert
    t0 = time.perf_counter()
    naive_id_map = setup_naive(naive_client, data, f"naive_{scenario_id}_{n_vectors}")
    naive_insert_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    loci_prefix = f"loci_{scenario_id}_{n_vectors}"
    loci_id_map, known_collections = setup_loci(loci_client, data, loci_prefix)
    loci_insert_s = time.perf_counter() - t0

    # Ground truth
    ground_truths = [compute_ground_truth(data, q, LIMIT) for q in queries]
    current_ground_truths = [
        compute_ground_truth(
            data,
            q,
            LIMIT,
            now_ms=benchmark_now_ms,
            decay_lambda=DECAY_LAMBDA,
        )
        for q in queries
    ]

    results = {}
    methods = [
        (
            "naive_qdrant",
            lambda q: query_naive(naive_client, q, f"naive_{scenario_id}_{n_vectors}"),
            naive_id_map,
            None,
        ),
        (
            "loci_r4",
            lambda q: query_loci(loci_client, q, known_collections, 1.0, loci_prefix),
            loci_id_map,
            None,
        ),
        (
            "loci_r4_overlap",
            lambda q: query_loci(loci_client, q, known_collections, 1.2, loci_prefix),
            loci_id_map,
            None,
        ),
        (
            "loci_current",
            lambda q: query_loci_current(
                loci_client,
                q,
                known_collections,
                loci_prefix,
                1.2,
                benchmark_now_ms,
                DECAY_LAMBDA,
            ),
            loci_id_map,
            None,
        ),
    ]
    for method_name, query_fn, id_map, overlap in methods:
        all_latencies = []
        all_recalls = []

        for run in range(N_RUNS):
            latencies = []
            recalls = []
            for i, q in enumerate(queries):
                t0 = time.perf_counter()
                result_ids = query_fn(q)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if i >= WARMUP_QUERIES:
                    latencies.append(elapsed_ms)
                    truth = (
                        current_ground_truths[i]
                        if method_name == "loci_current"
                        else ground_truths[i]
                    )
                    recalls.append(compute_recall(result_ids, truth, id_map))

            all_latencies.extend(latencies)
            all_recalls.extend(recalls)

        avg_lat = statistics.mean(all_latencies)
        results[method_name] = {
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(percentile(all_latencies, 50), 2),
            "p95_latency_ms": round(percentile(all_latencies, 95), 2),
            "p99_latency_ms": round(percentile(all_latencies, 99), 2),
            "recall_at_10": round(statistics.mean(all_recalls), 4),
            "qps": round(1000.0 / avg_lat, 1) if avg_lat > 0 else 0,
        }

    results["_meta"] = {
        "naive_insert_s": round(naive_insert_s, 2),
        "loci_insert_s": round(loci_insert_s, 2),
        "num_shards": len(known_collections),
    }

    return results


# ── Output ────────────────────────────────────────────────────────────────


def print_markdown_table(all_results: dict) -> str:
    """Print a markdown-formatted summary table."""
    lines = []
    lines.append("")
    lines.append("## LOCI Benchmark Results")
    lines.append("")
    lines.append(
        f"Configuration: dim={VECTOR_DIM}, queries={NUM_QUERIES}, "
        f"runs={N_RUNS}, warmup={WARMUP_QUERIES}, limit={LIMIT}"
    )
    lines.append("")

    for n_vectors in DATASET_SIZES:
        lines.append(f"### N = {n_vectors:,}")
        lines.append("")
        lines.append(
            "| Scenario | Method | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Recall@10 | QPS |"
        )
        lines.append("|:--|:--|--:|--:|--:|--:|--:|--:|")

        for sid in ["A", "B", "C", "D"]:
            key = f"N{n_vectors}_{sid}"
            if key not in all_results:
                continue
            r = all_results[key]
            scenario_name = SCENARIOS[sid]["name"]
            for method in ["naive_qdrant", "loci_r4", "loci_r4_overlap", "loci_current"]:
                if method not in r:
                    continue
                m = r[method]
                method_label = {
                    "naive_qdrant": "Naive Qdrant",
                    "loci_r4": "LOCI r4",
                    "loci_r4_overlap": "LOCI r4+overlap",
                    "loci_current": "LOCI current",
                }[method]
                lines.append(
                    f"| {scenario_name} | {method_label} "
                    f"| {m['avg_latency_ms']:.2f} "
                    f"| {m['p50_latency_ms']:.2f} "
                    f"| {m['p95_latency_ms']:.2f} "
                    f"| {m['p99_latency_ms']:.2f} "
                    f"| {m['recall_at_10']:.4f} "
                    f"| {m['qps']:.1f} |"
                )
        lines.append("")

    table = "\n".join(lines)
    print(table)
    return table


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  LOCI Publication Benchmark")
    print(f"  dim={VECTOR_DIM}, queries={NUM_QUERIES}, runs={N_RUNS}")
    print("=" * 70)

    qdrant_url = os.environ.get("QDRANT_URL", "")
    if qdrant_url:
        print(f"  Using Qdrant at {qdrant_url}")
    else:
        print("  Using in-memory Qdrant (set QDRANT_URL for remote)")

    all_results: dict = {
        "config": {
            "vector_dim": VECTOR_DIM,
            "num_queries": NUM_QUERIES,
            "warmup_queries": WARMUP_QUERIES,
            "n_runs": N_RUNS,
            "limit": LIMIT,
            "epoch_size_ms": EPOCH_SIZE_MS,
            "seed": SEED,
            "dataset_sizes": DATASET_SIZES,
        }
    }

    for n_vectors in DATASET_SIZES:
        print(f"\n--- N = {n_vectors:,} ---")
        for sid in ["A", "B", "C", "D"]:
            rng = np.random.default_rng(SEED)
            key = f"N{n_vectors}_{sid}"
            all_results[key] = run_scenario(sid, n_vectors, rng)

    # Print markdown table
    print_markdown_table(all_results)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "latest.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    import sys

    if "--quick" in sys.argv:
        DATASET_SIZES = [1_000]
        NUM_QUERIES = 10
        WARMUP_QUERIES = 2
        N_RUNS = 1
        print("  (quick mode: N=1000, 10 queries, 1 run)\n")
    main()
