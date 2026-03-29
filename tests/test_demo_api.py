"""Smoke tests for the guided LOCI demo API."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from demo.app.main import app, sim

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
