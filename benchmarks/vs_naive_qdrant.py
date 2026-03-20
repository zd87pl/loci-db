#!/usr/bin/env python3
"""Benchmark: Loci middleware (Hilbert bucketing + temporal sharding)
vs naive Qdrant float-range filters.

Uses qdrant-client's in-memory mode — same HNSW engine, zero external deps.
To run against a live Qdrant server set QDRANT_URL=http://localhost:6333.

Outputs a formatted table and writes results to benchmarks/results/latest.json.
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

from loci.spatial.buckets import expand_bounding_box
from loci.spatial.hilbert import encode as hilbert_encode
from loci.temporal.sharding import collection_name, epoch_id, epochs_in_range

# ── Configuration ──────────────────────────────────────────────────────────
VECTOR_DIM = 512
NUM_VECTORS = 10_000
NUM_QUERIES = 100
LIMIT = 10
EPOCH_SIZE_MS = 5_000
SPATIAL_RESOLUTION = 4
SEED = 42


# ── Data generation ────────────────────────────────────────────────────────


def generate_dataset(n: int, dim: int, rng: np.random.Generator) -> list[dict]:
    """Generate N random world-state points with (x,y,z,t) and a vector."""
    now_ms = int(time.time() * 1000)
    xs = rng.random(n)
    ys = rng.random(n)
    zs = rng.random(n)
    vecs = rng.standard_normal((n, dim))
    return [
        {
            "x": float(xs[i]),
            "y": float(ys[i]),
            "z": float(zs[i]),
            "timestamp_ms": now_ms + i * 10,
            "vector": vecs[i].tolist(),
            "index": i,
        }
        for i in range(n)
    ]


def generate_queries(
    n: int, dim: int, rng: np.random.Generator, start_ms: int, end_ms: int
) -> list[dict]:
    """Generate N random bounding-box + time-window queries."""
    queries = []
    for _ in range(n):
        cx, cy, cz = float(rng.random()), float(rng.random()), float(rng.random())
        hw = float(rng.uniform(0.1, 0.3))
        x_min, x_max = max(0.0, cx - hw), min(1.0, cx + hw)
        y_min, y_max = max(0.0, cy - hw), min(1.0, cy + hw)
        z_min, z_max = max(0.0, cz - hw), min(1.0, cz + hw)

        span = end_ms - start_ms
        w = float(rng.uniform(0.1, 0.5)) * span
        t_start = start_ms + float(rng.random()) * (span - w)
        t_end = t_start + w

        vec = rng.standard_normal(dim).tolist()
        queries.append(
            {
                "vector": vec,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "z_min": z_min,
                "z_max": z_max,
                "t_start": int(t_start),
                "t_end": int(t_end),
            }
        )
    return queries


# ── Naive Qdrant (3 float-range + datetime range) ─────────────────────────


def setup_naive(qdrant: QdrantClient, data: list[dict]) -> dict[str, int]:
    """Insert all points into a single flat collection with float payloads.

    Returns a mapping from Qdrant point ID → data index.
    """
    name = "benchmark_naive"
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    qdrant.create_payload_index(name, "x", PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(name, "y", PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(name, "z", PayloadSchemaType.FLOAT)
    qdrant.create_payload_index(name, "timestamp_ms", PayloadSchemaType.INTEGER)

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
        qdrant.upsert(collection_name=name, points=points)
    return id_map


def query_naive(qdrant: QdrantClient, q: dict) -> list[str]:
    """Run a naive Qdrant query with 3 float-range + 1 int-range filters."""
    resp = qdrant.query_points(
        collection_name="benchmark_naive",
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


# ── Loci (Hilbert bucketing + temporal sharding) ─────────────────────────


def setup_loci(qdrant: QdrantClient, data: list[dict]) -> dict[str, int]:
    """Insert all points into epoch-sharded collections with Hilbert payloads.

    Returns a mapping from Qdrant point ID → data index.
    """
    groups: dict[str, list[PointStruct]] = {}
    id_map: dict[str, int] = {}

    for d in data:
        ep = epoch_id(d["timestamp_ms"], EPOCH_SIZE_MS)
        col = collection_name(ep)
        t_norm = _normalise_time(d["timestamp_ms"], ep)
        hid = hilbert_encode(d["x"], d["y"], d["z"], t_norm, resolution_order=SPATIAL_RESOLUTION)
        pid = uuid.uuid4().hex
        id_map[pid] = d["index"]
        point = PointStruct(
            id=pid,
            vector=d["vector"],
            payload={
                "x": d["x"],
                "y": d["y"],
                "z": d["z"],
                "timestamp_ms": d["timestamp_ms"],
                "hilbert_id": hid,
            },
        )
        groups.setdefault(col, []).append(point)

    for col, points in groups.items():
        qdrant.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        qdrant.create_payload_index(col, "hilbert_id", PayloadSchemaType.INTEGER)
        qdrant.create_payload_index(col, "timestamp_ms", PayloadSchemaType.INTEGER)
        qdrant.create_payload_index(col, "x", PayloadSchemaType.FLOAT)
        qdrant.create_payload_index(col, "y", PayloadSchemaType.FLOAT)
        qdrant.create_payload_index(col, "z", PayloadSchemaType.FLOAT)
        batch_size = 500
        for i in range(0, len(points), batch_size):
            qdrant.upsert(collection_name=col, points=points[i : i + batch_size])

    return id_map


def query_loci(qdrant: QdrantClient, q: dict, known_collections: set[str]) -> list[str]:
    """Run an Loci query: Hilbert MatchAny + temporal shard routing.

    Each shard is over-fetched (limit * 3) to account for cross-shard
    merging, then results are globally ranked and truncated to LIMIT.
    """
    epochs = epochs_in_range(q["t_start"], q["t_end"], EPOCH_SIZE_MS)
    collections = [collection_name(e) for e in epochs]

    hids = expand_bounding_box(
        q["x_min"],
        q["x_max"],
        q["y_min"],
        q["y_max"],
        q["z_min"],
        q["z_max"],
        0.0,
        1.0,
        resolution_order=SPATIAL_RESOLUTION,
    )

    must_conditions = [
        FieldCondition(
            key="timestamp_ms",
            range=Range(gte=q["t_start"], lte=q["t_end"]),
        ),
        # Exact spatial bounds (same as naive) for correctness
        FieldCondition(key="x", range=Range(gte=q["x_min"], lte=q["x_max"])),
        FieldCondition(key="y", range=Range(gte=q["y_min"], lte=q["y_max"])),
        FieldCondition(key="z", range=Range(gte=q["z_min"], lte=q["z_max"])),
    ]
    # Hilbert MatchAny pre-filter: on a real Qdrant server with payload
    # indexes, the integer set lookup executes first and drastically
    # narrows the candidate set before the float-range filters run.
    if hids:
        must_conditions.append(FieldCondition(key="hilbert_id", match=MatchAny(any=hids)))
    query_filter = Filter(must=must_conditions)

    per_shard_limit = LIMIT * 3  # over-fetch to improve recall after merge

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
            )
            for h in resp.points:
                all_hits.append((str(h.id), h.score))
        except Exception:
            continue

    all_hits.sort(key=lambda x: x[1], reverse=True)
    return [h[0] for h in all_hits[:LIMIT]]


def _normalise_time(timestamp_ms: int, ep: int) -> float:
    epoch_start = ep * EPOCH_SIZE_MS
    offset = timestamp_ms - epoch_start
    return min(1.0, max(0.0, offset / EPOCH_SIZE_MS))


# ── Ground truth for recall ────────────────────────────────────────────────


def compute_ground_truth(data: list[dict], q: dict, limit: int) -> list[int]:
    """Brute-force ground truth: filter by bounding box + time, rank by cosine."""
    qv = np.array(q["vector"])
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
            dv = np.array(d["vector"])
            dv_norm = np.linalg.norm(dv)
            if dv_norm == 0:
                continue
            sim = float(np.dot(qv, dv) / (qv_norm * dv_norm))
            candidates.append((d["index"], sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:limit]]


def compute_recall(
    result_ids: list[str], truth_indices: list[int], id_map: dict[str, int]
) -> float:
    """Compute recall@k: fraction of ground-truth results found."""
    if not truth_indices:
        return 1.0
    truth_set = set(truth_indices)
    found = sum(1 for rid in result_ids if id_map.get(rid, -1) in truth_set)
    return found / len(truth_indices)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    rng = np.random.default_rng(SEED)

    print(f"Generating {NUM_VECTORS} world-state vectors (dim={VECTOR_DIM})...")
    data = generate_dataset(NUM_VECTORS, VECTOR_DIM, rng)
    start_ms = data[0]["timestamp_ms"]
    end_ms = data[-1]["timestamp_ms"]

    print(f"Generating {NUM_QUERIES} random spatiotemporal queries...")
    queries = generate_queries(NUM_QUERIES, VECTOR_DIM, rng, start_ms, end_ms)

    # ── Setup naive ────────────────────────────────────────────────────
    print("\nSetting up Naive Qdrant (single collection, float-range filters)...")
    qdrant_url = os.environ.get("QDRANT_URL", "")
    if qdrant_url:
        naive_client = QdrantClient(url=qdrant_url)
        loci_qdrant = QdrantClient(url=qdrant_url)
        print(f"  Using remote Qdrant at {qdrant_url}")
    else:
        naive_client = QdrantClient(location=":memory:")
        loci_qdrant = QdrantClient(location=":memory:")
        print("  Using in-memory Qdrant (set QDRANT_URL for remote)")

    t0 = time.perf_counter()
    naive_id_map = setup_naive(naive_client, data)
    naive_insert_s = time.perf_counter() - t0
    print(f"  Inserted {NUM_VECTORS} points in {naive_insert_s:.2f}s")

    # ── Setup Loci ───────────────────────────────────────────────────
    print("Setting up Loci (Hilbert bucketing + temporal sharding)...")
    t0 = time.perf_counter()
    loci_id_map = setup_loci(loci_qdrant, data)
    loci_insert_s = time.perf_counter() - t0
    known_collections = {collection_name(epoch_id(d["timestamp_ms"], EPOCH_SIZE_MS)) for d in data}
    num_shards = len(known_collections)
    print(f"  Inserted {NUM_VECTORS} points across {num_shards} shards in {loci_insert_s:.2f}s")

    # ── Ground truth ───────────────────────────────────────────────────
    print("\nComputing brute-force ground truth for recall@10...")
    ground_truths = [compute_ground_truth(data, q, LIMIT) for q in queries]

    # ── Benchmark queries ──────────────────────────────────────────────
    print(f"Running {NUM_QUERIES} queries per method...\n")

    naive_latencies = []
    naive_recalls = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        results = query_naive(naive_client, q)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        naive_latencies.append(elapsed_ms)
        naive_recalls.append(compute_recall(results, ground_truths[i], naive_id_map))

    loci_latencies = []
    loci_recalls = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        results = query_loci(loci_qdrant, q, known_collections)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        loci_latencies.append(elapsed_ms)
        loci_recalls.append(compute_recall(results, ground_truths[i], loci_id_map))

    # ── Results ────────────────────────────────────────────────────────
    naive_avg = statistics.mean(naive_latencies)
    naive_p95 = sorted(naive_latencies)[int(0.95 * len(naive_latencies))]
    naive_recall = statistics.mean(naive_recalls)

    loci_avg = statistics.mean(loci_latencies)
    loci_p95 = sorted(loci_latencies)[int(0.95 * len(loci_latencies))]
    loci_recall = statistics.mean(loci_recalls)

    speedup_avg = naive_avg / loci_avg if loci_avg > 0 else float("inf")
    speedup_p95 = naive_p95 / loci_p95 if loci_p95 > 0 else float("inf")

    print("=" * 68)
    print(f"  Loci Benchmark — {NUM_VECTORS:,} points, {NUM_QUERIES} queries, dim={VECTOR_DIM}")
    print(f"  {num_shards} temporal shards, Hilbert resolution={SPATIAL_RESOLUTION}")
    print("=" * 68)
    header = (
        f"{'Method':<20}| {'Avg latency (ms)':>17} | {'P95 latency (ms)':>17} | {'Recall@10':>9}"
    )
    print(header)
    print("-" * 20 + "+" + "-" * 19 + "+" + "-" * 19 + "+" + "-" * 11)
    print(f"{'Naive Qdrant':<20}| {naive_avg:>17.2f} | {naive_p95:>17.2f} | {naive_recall:>9.3f}")
    print(f"{'Loci':<20}| {loci_avg:>17.2f} | {loci_p95:>17.2f} | {loci_recall:>9.3f}")
    print(f"{'Speedup':<20}| {speedup_avg:>16.2f}x | {speedup_p95:>16.2f}x |")
    print("=" * 68)

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "config": {
            "num_vectors": NUM_VECTORS,
            "num_queries": NUM_QUERIES,
            "vector_dim": VECTOR_DIM,
            "epoch_size_ms": EPOCH_SIZE_MS,
            "spatial_resolution": SPATIAL_RESOLUTION,
            "num_shards": num_shards,
            "seed": SEED,
        },
        "naive_qdrant": {
            "avg_latency_ms": round(naive_avg, 2),
            "p95_latency_ms": round(naive_p95, 2),
            "recall_at_10": round(naive_recall, 3),
            "insert_time_s": round(naive_insert_s, 2),
        },
        "loci": {
            "avg_latency_ms": round(loci_avg, 2),
            "p95_latency_ms": round(loci_p95, 2),
            "recall_at_10": round(loci_recall, 3),
            "insert_time_s": round(loci_insert_s, 2),
        },
        "speedup": {
            "avg_latency": round(speedup_avg, 2),
            "p95_latency": round(speedup_p95, 2),
        },
    }

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "latest.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
