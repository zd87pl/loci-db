#!/usr/bin/env python3
"""Basic Loci usage — insert 50 robot states and run a spatiotemporal query.

This script simulates a robot moving linearly through a room over 10 seconds,
inserts the trajectory, then queries for states near a target position in the
last 5 seconds.

Requires a local Qdrant instance running on http://localhost:6333, OR
falls back to the zero-dependency LocalLociClient automatically.

Start Qdrant with:  docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import math
import random
import time

from loci.schema import WorldState

VECTOR_DIM = 128
NUM_STATES = 50
DURATION_S = 10.0


def make_robot_states(now_ms: int) -> list[WorldState]:
    """Simulate a robot moving linearly from (0.1, 0.1, 0.1) to (0.9, 0.9, 0.5)."""
    states: list[WorldState] = []
    for i in range(NUM_STATES):
        t = i / (NUM_STATES - 1)
        x = 0.1 + 0.8 * t
        y = 0.1 + 0.8 * t
        z = 0.1 + 0.4 * math.sin(math.pi * t)  # gentle arc in z
        ts = now_ms + int(t * DURATION_S * 1000)

        # Embedding loosely encodes position (first 3 dims) + noise
        vec = [0.0] * VECTOR_DIM
        vec[0] = x
        vec[1] = y
        vec[2] = z
        for j in range(3, VECTOR_DIM):
            vec[j] = random.gauss(0, 0.1)

        states.append(
            WorldState(
                x=x,
                y=y,
                z=z,
                timestamp_ms=ts,
                vector=vec,
                scene_id="robot_room",
                scale_level="patch",
                confidence=1.0,
            )
        )
    return states


def main() -> None:
    # Try Qdrant, fall back to local
    try:
        from loci import LociClient

        client = LociClient(
            qdrant_url="http://localhost:6333",
            epoch_size_ms=5000,
            spatial_resolution=4,
            vector_size=VECTOR_DIM,
        )
        # Quick connection check
        client._qdrant.get_collections()
        print("Connected to Qdrant at localhost:6333")
    except Exception:
        from loci import LocalLociClient

        client = LocalLociClient(
            epoch_size_ms=5000,
            spatial_resolution=4,
            vector_size=VECTOR_DIM,
        )
        print("Qdrant not available — using LocalLociClient (in-memory)")

    now_ms = int(time.time() * 1000)
    states = make_robot_states(now_ms)

    # --- Insert ---
    print(f"\nInserting {NUM_STATES} robot states over {DURATION_S}s trajectory...")
    t0 = time.perf_counter()
    ids = client.insert_batch(states)
    insert_ms = (time.perf_counter() - t0) * 1000
    print(f"  Inserted {len(ids)} states in {insert_ms:.1f} ms")

    # --- Query: find states near (0.5, 0.5, 0.5) in the last 5 seconds ---
    target_x, target_y, target_z = 0.5, 0.5, 0.5
    query_vec = [0.0] * VECTOR_DIM
    query_vec[0] = target_x
    query_vec[1] = target_y
    query_vec[2] = target_z

    half_ms = int(DURATION_S * 500)
    time_start = now_ms + half_ms
    time_end = now_ms + int(DURATION_S * 1000)

    print(f"\nQuerying: nearest to ({target_x}, {target_y}, {target_z})")
    print(f"  Spatial window:  [{target_x - 0.3:.1f}, {target_x + 0.3:.1f}] per axis")
    print(f"  Time window:     last {DURATION_S / 2:.0f}s of trajectory")

    t0 = time.perf_counter()
    results = client.query(
        vector=query_vec,
        spatial_bounds={
            "x_min": target_x - 0.3,
            "x_max": target_x + 0.3,
            "y_min": target_y - 0.3,
            "y_max": target_y + 0.3,
            "z_min": 0.0,
            "z_max": 1.0,
        },
        time_window_ms=(time_start, time_end),
        limit=5,
    )
    query_ms = (time.perf_counter() - t0) * 1000

    print(f"\nTop {len(results)} results (query took {query_ms:.1f} ms):")
    print(f"  {'ID':<36} {'Position':>24}  {'Time offset':>12}")
    print(f"  {'-' * 36} {'-' * 24}  {'-' * 12}")
    for r in results:
        offset_s = (r.timestamp_ms - now_ms) / 1000
        print(f"  {r.id:<36} ({r.x:.3f}, {r.y:.3f}, {r.z:.3f})  {offset_s:>10.2f}s")

    print(f"\n  Total insert time: {insert_ms:.1f} ms")
    print(f"  Total query time:  {query_ms:.1f} ms")


if __name__ == "__main__":
    main()
