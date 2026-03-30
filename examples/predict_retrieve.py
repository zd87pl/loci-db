#!/usr/bin/env python3
"""Predict-then-retrieve demo — anticipatory retrieval with novelty scoring.

Simulates a robot on a patrol loop (200 states), defines a simple linear
predictor, and demonstrates predict_and_retrieve: finding stored states
that are similar to where the robot is *predicted* to be in the future.

Uses the production PredictRetrieveResult API with novelty scoring.

Requires a local Qdrant instance running on http://localhost:6333, OR
falls back to the zero-dependency LocalLociClient automatically.
"""

from __future__ import annotations

import math
import random
import time

from loci.schema import WorldState

VECTOR_DIM = 128
NUM_STATES = 200
LOOP_DURATION_S = 20.0


def make_patrol_states(now_ms: int) -> list[WorldState]:
    """Simulate a robot patrolling an oval loop in the xz-plane."""
    states: list[WorldState] = []
    for i in range(NUM_STATES):
        t = i / NUM_STATES  # fraction of full loop
        angle = 2 * math.pi * t
        x = 0.5 + 0.35 * math.cos(angle)
        y = 0.5  # constant height
        z = 0.5 + 0.25 * math.sin(angle)
        ts = now_ms + int(t * LOOP_DURATION_S * 1000)

        # Embedding encodes position + velocity direction
        vx = -math.sin(angle)
        vz = math.cos(angle)
        vec = [0.0] * VECTOR_DIM
        vec[0] = x
        vec[1] = y
        vec[2] = z
        vec[3] = vx * 0.5
        vec[4] = vz * 0.5
        for j in range(5, VECTOR_DIM):
            vec[j] = random.gauss(0, 0.05)

        states.append(
            WorldState(
                x=x,
                y=y,
                z=z,
                timestamp_ms=ts,
                vector=vec,
                scene_id="patrol_loop",
                scale_level="patch",
                confidence=1.0,
            )
        )
    return states


def linear_predictor(context_vector: list[float]) -> list[float]:
    """A simple linear 'world model' that extrapolates position along velocity.

    Reads position from dims [0..2] and velocity from dims [3..4],
    then shifts the position forward by the velocity + small noise.
    """
    predicted = list(context_vector)
    # Extrapolate: pos += velocity * scale
    predicted[0] = context_vector[0] + context_vector[3] * 0.5
    predicted[2] = context_vector[2] + context_vector[4] * 0.5
    # Add small perturbation
    for j in range(5, len(predicted)):
        predicted[j] += random.gauss(0, 0.02)
    return predicted


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
    states = make_patrol_states(now_ms)

    # --- Insert patrol trajectory ---
    print(f"\nInserting {NUM_STATES} patrol states ({LOOP_DURATION_S:.0f}s loop)...")
    ids = client.insert_batch(states)
    print(f"  Inserted {len(ids)} states across the patrol loop.")

    # --- Pick a point at 25% through the patrol as "current" ---
    current_idx = NUM_STATES // 4
    current_state = states[current_idx]
    context_vector = current_state.vector

    print(
        "\nCurrent robot position: "
        f"({current_state.x:.3f}, {current_state.y:.3f}, {current_state.z:.3f})"
    )
    print(f"  (patrol step {current_idx}/{NUM_STATES})")

    # Show what the predictor thinks
    predicted = linear_predictor(context_vector)
    print(
        "\nPredictor says the robot will be near: "
        f"({predicted[0]:.3f}, {predicted[1]:.3f}, {predicted[2]:.3f})"
    )

    # --- Run predict-then-retrieve with novelty scoring ---
    print("\nRunning predict_and_retrieve (future_horizon=2000ms, with novelty)...")
    result = client.predict_and_retrieve(
        context_vector=context_vector,
        predictor_fn=linear_predictor,
        future_horizon_ms=2000,
        current_position=(current_state.x, current_state.y, current_state.z),
        spatial_search_radius=0.4,
        limit=5,
        alpha=0.7,
        return_prediction=True,
    )

    # --- Display results with novelty scoring ---
    print("\n--- Predict-Then-Retrieve Results ---")
    print(f"  Prediction novelty:   {result.prediction_novelty:.3f}")
    print(f"  Predictor call time:  {result.predictor_call_ms:.2f} ms")
    print(f"  Retrieval time:       {result.retrieval_latency_ms:.2f} ms")
    print(f"  Results found:        {len(result.results)}")

    if result.prediction_novelty < 0.3:
        print("  Interpretation: LOW novelty — robot has seen this situation before")
    elif result.prediction_novelty < 0.7:
        print("  Interpretation: MODERATE novelty — partially familiar territory")
    else:
        print("  Interpretation: HIGH novelty — new territory, proceed cautiously")

    if result.results:
        print(f"\n  {'#':<4} {'Position':>24}  {'Time offset':>12}  {'Desc'}")
        print(f"  {'-' * 4} {'-' * 24}  {'-' * 12}  {'-' * 30}")
        for i, r in enumerate(result.results, 1):
            offset_s = (r.timestamp_ms - now_ms) / 1000
            frac = (r.timestamp_ms - now_ms) / (LOOP_DURATION_S * 1000)
            angle_deg = frac * 360
            print(
                f"  {i:<4} ({r.x:.3f}, {r.y:.3f}, {r.z:.3f})  "
                f"{offset_s:>10.2f}s  patrol angle ~{angle_deg:.0f}°"
            )

    if result.predicted_vector:
        print(f"\n  Predicted vector (first 5 dims): "
              f"[{', '.join(f'{v:.3f}' for v in result.predicted_vector[:5])}]")

    # --- Also demonstrate legacy API (backward compat) ---
    print("\n--- Legacy API (backward compat, no novelty) ---")
    legacy_results = client.predict_and_retrieve(
        context_vector=context_vector,
        predictor_fn=linear_predictor,
        future_horizon_ms=2000,
        limit=5,
    )
    print(f"  Found {len(legacy_results)} results (returns plain list[WorldState])")

    print("\n--- Anticipatory Retrieval Concept ---")
    print(f"The robot is at step {current_idx}/{NUM_STATES} of its patrol loop.")
    print("The predictor extrapolated its trajectory forward by 2 seconds.")
    print("LOCI found stored states matching where the robot is PREDICTED to be,")
    print("and computed a novelty score indicating how familiar this situation is.")


if __name__ == "__main__":
    main()
