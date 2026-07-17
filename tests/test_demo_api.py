"""Smoke tests for the guided LOCI demo API."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from demo.app.main import app, sim
from demo.app.simulation import EPOCH_SIZE_MS, TICK_INTERVAL_MS, Simulation

client = TestClient(app)


def _advance_ticks(ticks: int) -> None:
    sim.running = True
    for _ in range(ticks):
        asyncio.run(sim.tick())
    sim.running = False


def setup_function() -> None:
    sim.reset()


def teardown_function() -> None:
    sim.reset()


def test_demo_status_exposes_guided_story_fields() -> None:
    response = client.get("/api/demo/status")
    response.raise_for_status()
    data = response.json()

    assert data["phase"] == "idle"
    assert data["next_action"] == "build_memory"
    assert data["predict"]["minimum_memories"] == 12
    assert data["anomaly"]["minimum_memories"] == 8
    assert len(data["route_preview"]) >= 2
    assert "summary" in data["current_view"]


def test_predict_requires_warmup_before_ready() -> None:
    response = client.post("/api/query/predict", json={"steps_ahead": 10})
    response.raise_for_status()
    data = response.json()

    assert data["novelty"] is None
    assert "Wait for" in data["message"]
    assert data["guide"]["predict"]["ready"] is False


def test_demo_status_turns_ready_after_enough_memories() -> None:
    _advance_ticks(12)

    status_response = client.get("/api/demo/status")
    status_response.raise_for_status()
    status_data = status_response.json()

    assert status_data["phase"] == "paused"
    assert status_data["predict"]["ready"] is True

    predict_response = client.post("/api/query/predict", json={"steps_ahead": 10})
    predict_response.raise_for_status()
    predict_data = predict_response.json()

    assert predict_data["guide"]["predict"]["ready"] is True
    assert predict_data["novelty"] is not None


def test_simulation_memory_stays_bounded_with_retention_cap() -> None:
    """The tick loop must not grow the in-memory store without bound.

    Uses a tiny retention cap (3 epochs) so eviction kicks in quickly: at
    500ms ticks each 5s epoch holds 10 points, so 100 ticks span ~11 epochs
    but at most 3 epochs (30 points) may be retained.
    """
    local_sim = Simulation(memory_max_epochs=3)
    local_sim.running = True

    async def run_ticks(n: int) -> None:
        for _ in range(n):
            await local_sim.tick()

    asyncio.run(run_ticks(100))
    local_sim.running = False

    points_per_epoch = EPOCH_SIZE_MS // TICK_INTERVAL_MS
    cap = 3 * points_per_epoch
    assert local_sim.tick_count == 100
    assert local_sim.memory_count <= cap
    # Eviction actually happened (without it there would be 100 points)
    assert local_sim.memory_count < 100
