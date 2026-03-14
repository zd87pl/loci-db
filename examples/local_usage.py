#!/usr/bin/env python3
"""Engram local usage — zero dependencies, no Docker required.

Demonstrates the full Engram API using LocalEngramClient backed by
the pure-Python in-memory store.
"""

from __future__ import annotations

import random
import time

from engram import LocalEngramClient, WorldState

VECTOR_DIM = 128
NUM_STATES = 100


def main() -> None:
    client = LocalEngramClient(
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VECTOR_DIM,
    )

    # --- Insert 100 random states ---
    now_ms = int(time.time() * 1000)
    states: list[WorldState] = []
    for i in range(NUM_STATES):
        state = WorldState(
            x=random.random(),
            y=random.random(),
            z=random.random(),
            timestamp_ms=now_ms + i * 50,
            vector=[random.gauss(0, 1) for _ in range(VECTOR_DIM)],
            scene_id="demo_scene",
            scale_level="patch",
            confidence=random.random(),
        )
        states.append(state)

    t0 = time.perf_counter()
    ids = client.insert_batch(states)
    insert_ms = (time.perf_counter() - t0) * 1000
    print(f"Inserted {len(ids)} states in {insert_ms:.1f}ms")

    # --- Query: nearest neighbours in a spatial region ---
    query_vec = [random.gauss(0, 1) for _ in range(VECTOR_DIM)]
    results = client.query(
        vector=query_vec,
        spatial_bounds={
            "x_min": 0.2,
            "x_max": 0.8,
            "y_min": 0.2,
            "y_max": 0.8,
            "z_min": 0.0,
            "z_max": 1.0,
        },
        time_window_ms=(now_ms, now_ms + NUM_STATES * 50),
        limit=5,
    )

    stats = client.last_query_stats
    print(f"\nQuery stats:")
    print(f"  Shards searched:    {stats.shards_searched}")
    print(f"  Candidates:         {stats.total_candidates}")
    print(f"  Hilbert filter IDs: {stats.hilbert_ids_in_filter}")
    print(f"  Decay applied:      {stats.decay_applied}")
    print(f"  Elapsed:            {stats.elapsed_ms:.2f}ms")

    print(f"\nTop {len(results)} results:")
    for r in results:
        print(f"  id={r.id[:8]}..  pos=({r.x:.2f}, {r.y:.2f}, {r.z:.2f})  t={r.timestamp_ms}")

    # --- Trajectory walk ---
    if ids:
        mid = len(ids) // 2
        traj = client.get_trajectory(ids[mid], steps_back=3, steps_forward=3)
        print(f"\nTrajectory through state {ids[mid][:8]}.. ({len(traj)} states):")
        for s in traj:
            print(f"  t={s.timestamp_ms}  pos=({s.x:.2f}, {s.y:.2f}, {s.z:.2f})")

    # --- Predict and retrieve ---
    future_results = client.predict_and_retrieve(
        context_vector=query_vec,
        predictor_fn=lambda v: [x + random.gauss(0, 0.1) for x in v],
        future_horizon_ms=NUM_STATES * 50,
        limit=3,
    )
    print(f"\nPredict-and-retrieve: {len(future_results)} future matches")


if __name__ == "__main__":
    main()
