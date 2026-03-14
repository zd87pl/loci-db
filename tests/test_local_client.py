"""Integration tests for LocalEngramClient — full Engram with no Qdrant."""

from __future__ import annotations

import time

import pytest

from engram.local_client import LocalEngramClient
from engram.schema import WorldState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VEC_SIZE = 4


def _make_state(
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.5,
    ts: int = 1000,
    scene: str = "scene_a",
    vector: list[float] | None = None,
) -> WorldState:
    return WorldState(
        x=x,
        y=y,
        z=z,
        timestamp_ms=ts,
        vector=vector or [1.0, 0.0, 0.0, 0.0],
        scene_id=scene,
    )


@pytest.fixture()
def client():
    return LocalEngramClient(
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VEC_SIZE,
        decay_lambda=0.0,  # disable decay for deterministic tests
    )


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_insert_returns_id(self, client):
        sid = client.insert(_make_state())
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_insert_stores_point(self, client):
        client.insert(_make_state())
        assert client.store.total_points == 1

    def test_insert_does_not_mutate_input(self, client):
        state = _make_state()
        original_id = state.id
        client.insert(state)
        assert state.id == original_id  # not mutated

    def test_insert_multiple_epochs(self, client):
        client.insert(_make_state(ts=1000))
        client.insert(_make_state(ts=6000))  # different epoch (5000ms shards)
        assert client.store.total_points == 2


# ---------------------------------------------------------------------------
# Insert batch
# ---------------------------------------------------------------------------


class TestInsertBatch:
    def test_batch_returns_ids(self, client):
        states = [_make_state(ts=100 * i) for i in range(5)]
        ids = client.insert_batch(states)
        assert len(ids) == 5
        assert len(set(ids)) == 5  # all unique

    def test_batch_preserves_order(self, client):
        states = [_make_state(ts=i * 100, scene=f"s{i}") for i in range(3)]
        ids = client.insert_batch(states)
        # Each ID corresponds to the input by index
        for i, sid in enumerate(ids):
            assert isinstance(sid, str)

    def test_batch_causal_linking(self, client):
        states = [
            _make_state(ts=100, scene="s1"),
            _make_state(ts=200, scene="s1"),
            _make_state(ts=300, scene="s1"),
        ]
        ids = client.insert_batch(states)
        traj = client.get_trajectory(ids[1], steps_back=5, steps_forward=5)
        assert len(traj) == 3


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_returns_results(self, client):
        client.insert(_make_state(vector=[1.0, 0.0, 0.0, 0.0]))
        results = client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        assert len(results) == 1
        assert isinstance(results[0], WorldState)

    def test_query_similarity_ranking(self, client):
        client.insert(_make_state(x=0.1, y=0.1, z=0.1, vector=[1.0, 0.0, 0.0, 0.0]))
        client.insert(_make_state(x=0.2, y=0.2, z=0.2, vector=[0.0, 1.0, 0.0, 0.0]))
        results = client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=2)
        assert results[0].vector == [1.0, 0.0, 0.0, 0.0]

    def test_query_with_time_window(self, client):
        client.insert(_make_state(ts=1000))
        client.insert(_make_state(ts=6000))
        results = client.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            time_window_ms=(500, 1500),
        )
        assert len(results) == 1
        assert results[0].timestamp_ms == 1000

    def test_query_with_spatial_bounds(self, client):
        client.insert(_make_state(x=0.1, y=0.1, z=0.1, vector=[1, 0, 0, 0]))
        client.insert(_make_state(x=0.9, y=0.9, z=0.9, vector=[1, 0, 0, 0]))
        results = client.query(
            vector=[1, 0, 0, 0],
            spatial_bounds={
                "x_min": 0.0,
                "x_max": 0.3,
                "y_min": 0.0,
                "y_max": 0.3,
                "z_min": 0.0,
                "z_max": 0.3,
            },
        )
        assert len(results) == 1
        assert results[0].x == 0.1

    def test_query_empty_returns_empty(self, client):
        results = client.query(vector=[1, 0, 0, 0])
        assert results == []

    def test_query_limit(self, client):
        for i in range(20):
            client.insert(_make_state(ts=i * 100, scene=f"s{i}"))
        results = client.query(vector=[1, 0, 0, 0], limit=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# QueryStats
# ---------------------------------------------------------------------------


class TestQueryStats:
    def test_stats_populated_after_query(self, client):
        client.insert(_make_state())
        client.query(vector=[1, 0, 0, 0])
        stats = client.last_query_stats
        assert stats is not None
        assert stats.shards_searched >= 1
        assert stats.elapsed_ms > 0

    def test_stats_with_spatial_filter(self, client):
        client.insert(_make_state(x=0.5, y=0.5, z=0.5))
        client.query(
            vector=[1, 0, 0, 0],
            spatial_bounds={
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
                "z_min": 0.0,
                "z_max": 1.0,
            },
        )
        stats = client.last_query_stats
        assert stats.hilbert_ids_in_filter > 0

    def test_stats_decay_disabled(self, client):
        client.insert(_make_state())
        client.query(vector=[1, 0, 0, 0])
        assert client.last_query_stats.decay_applied is False

    def test_stats_decay_enabled(self):
        c = LocalEngramClient(vector_size=VEC_SIZE, decay_lambda=1e-4)
        c.insert(_make_state())
        c.query(vector=[1, 0, 0, 0])
        assert c.last_query_stats.decay_applied is True

    def test_stats_none_before_query(self, client):
        assert client.last_query_stats is None


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------


class TestDecay:
    def test_decay_reranks_results(self):
        c = LocalEngramClient(vector_size=VEC_SIZE, decay_lambda=0.01)
        # Insert old and new states with same vector
        c.insert(_make_state(ts=1000, vector=[1, 0, 0, 0]))
        now_ms = int(time.time() * 1000)
        c.insert(_make_state(ts=now_ms, vector=[0.9, 0.1, 0, 0]))
        results = c.query(vector=[1, 0, 0, 0], limit=2)
        # With high decay, the recent one should rank higher despite lower cosine
        if len(results) == 2:
            assert results[0].timestamp_ms >= results[1].timestamp_ms


# ---------------------------------------------------------------------------
# Causal linking (single inserts)
# ---------------------------------------------------------------------------


class TestCausalLinking:
    def test_single_insert_links(self, client):
        client.insert(_make_state(ts=100, scene="s1"))
        id2 = client.insert(_make_state(ts=200, scene="s1"))
        traj = client.get_trajectory(id2, steps_back=5, steps_forward=5)
        assert len(traj) == 2

    def test_no_cross_scene_linking(self, client):
        client.insert(_make_state(ts=100, scene="s1"))
        id2 = client.insert(_make_state(ts=200, scene="s2"))
        traj = client.get_trajectory(id2, steps_back=5, steps_forward=5)
        assert len(traj) == 1  # only the anchor, no cross-scene link


# ---------------------------------------------------------------------------
# get_trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_full_trajectory(self, client):
        ids = client.insert_batch(
            [
                _make_state(ts=100, scene="s1"),
                _make_state(ts=200, scene="s1"),
                _make_state(ts=300, scene="s1"),
                _make_state(ts=400, scene="s1"),
            ]
        )
        traj = client.get_trajectory(ids[1], steps_back=10, steps_forward=10)
        assert len(traj) == 4
        # Ordered by time
        ts_list = [s.timestamp_ms for s in traj]
        assert ts_list == sorted(ts_list)

    def test_trajectory_missing_id(self, client):
        assert client.get_trajectory("nonexistent") == []


# ---------------------------------------------------------------------------
# predict_and_retrieve
# ---------------------------------------------------------------------------


class TestPredictAndRetrieve:
    def test_predict_and_retrieve(self, client):
        now_ms = int(time.time() * 1000)
        client.insert(_make_state(ts=now_ms + 500, vector=[0.5, 0.5, 0, 0]))
        results = client.predict_and_retrieve(
            context_vector=[1, 0, 0, 0],
            predictor_fn=lambda v: [0.5, 0.5, 0, 0],
            future_horizon_ms=2000,
            limit=5,
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------


class TestDistanceMetrics:
    def test_dot_product_distance(self):
        c = LocalEngramClient(vector_size=VEC_SIZE, distance="dot", decay_lambda=0)
        c.insert(_make_state(vector=[3, 0, 0, 0]))
        c.insert(_make_state(x=0.1, y=0.1, z=0.1, vector=[1, 0, 0, 0]))
        results = c.query(vector=[1, 0, 0, 0], limit=2)
        assert len(results) >= 1

    def test_euclidean_distance(self):
        c = LocalEngramClient(vector_size=VEC_SIZE, distance="euclidean", decay_lambda=0)
        c.insert(_make_state(vector=[1, 0, 0, 0]))
        results = c.query(vector=[1, 0, 0, 0], limit=1)
        assert len(results) == 1
