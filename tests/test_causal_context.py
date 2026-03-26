"""Tests for get_causal_context using LocalLociClient."""

from __future__ import annotations

import random
import time

from loci.local_client import LocalLociClient
from loci.schema import WorldState

VECTOR_DIM = 32


def _make_client() -> LocalLociClient:
    return LocalLociClient(
        vector_size=VECTOR_DIM,
        epoch_size_ms=5000,
        decay_lambda=0,
    )


def _vec() -> list[float]:
    return [random.gauss(0, 1) for _ in range(VECTOR_DIM)]


def test_causal_context_returns_states_in_window() -> None:
    """Should return states within ±window_ms of the anchor."""
    random.seed(42)
    client = _make_client()
    now = int(time.time() * 1000)

    states = [
        WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=now + i * 100, vector=_vec(), scene_id="s1")
        for i in range(20)
    ]
    ids = client.insert_batch(states)

    # Anchor is the 10th state (now + 1000ms)
    context = client.get_causal_context(ids[10], window_ms=300)

    # Should include states within ±300ms of anchor (now+1000ms)
    # That's states 7..13 (now+700 to now+1300)
    assert len(context) > 0
    for ws in context:
        assert abs(ws.timestamp_ms - states[10].timestamp_ms) <= 300


def test_causal_context_filters_by_scene_id() -> None:
    """Should only return states from the same scene."""
    random.seed(42)
    client = _make_client()
    now = int(time.time() * 1000)

    states_a = [
        WorldState(
            x=0.5, y=0.5, z=0.5, timestamp_ms=now + i * 100, vector=_vec(), scene_id="scene_a"
        )
        for i in range(10)
    ]
    states_b = [
        WorldState(
            x=0.5, y=0.5, z=0.5, timestamp_ms=now + i * 100, vector=_vec(), scene_id="scene_b"
        )
        for i in range(10)
    ]
    ids_a = client.insert_batch(states_a)
    client.insert_batch(states_b)

    context = client.get_causal_context(ids_a[5], window_ms=5000)
    assert all(ws.scene_id == "scene_a" for ws in context)


def test_causal_context_empty_for_no_scene_id() -> None:
    """Should return empty if anchor has no scene_id."""
    random.seed(42)
    client = _make_client()
    now = int(time.time() * 1000)

    ws = WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=now, vector=_vec(), scene_id="")
    state_id = client.insert(ws)

    context = client.get_causal_context(state_id, window_ms=5000)
    assert context == []


def test_causal_context_sorted_by_timestamp() -> None:
    """Results should be sorted by timestamp ascending."""
    random.seed(42)
    client = _make_client()
    now = int(time.time() * 1000)

    states = [
        WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=now + i * 50, vector=_vec(), scene_id="s1")
        for i in range(30)
    ]
    ids = client.insert_batch(states)

    context = client.get_causal_context(ids[15], window_ms=500)
    timestamps = [ws.timestamp_ms for ws in context]
    assert timestamps == sorted(timestamps)


def test_causal_context_missing_state_returns_empty() -> None:
    """Should return empty if state_id is not found."""
    client = _make_client()
    context = client.get_causal_context("nonexistent_id", window_ms=5000)
    assert context == []
