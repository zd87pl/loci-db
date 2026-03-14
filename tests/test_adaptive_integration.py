"""Tests for adaptive Hilbert resolution integration in clients."""

from __future__ import annotations

import pytest

from engram.local_client import LocalEngramClient
from engram.schema import WorldState
from engram.spatial.adaptive import DensityStats

VEC_SIZE = 4


def _make_state(
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.5,
    ts: int = 1000,
    vector: list[float] | None = None,
) -> WorldState:
    return WorldState(
        x=x,
        y=y,
        z=z,
        timestamp_ms=ts,
        vector=vector or [1.0, 0.0, 0.0, 0.0],
    )


class TestAdaptiveDisabled:
    def test_density_stats_none_when_disabled(self):
        c = LocalEngramClient(vector_size=VEC_SIZE, decay_lambda=0)
        assert c.density_stats is None

    def test_works_normally_without_adaptive(self):
        c = LocalEngramClient(vector_size=VEC_SIZE, decay_lambda=0)
        sid = c.insert(_make_state())
        assert isinstance(sid, str)
        results = c.query(vector=[1, 0, 0, 0])
        assert len(results) == 1


class TestAdaptiveEnabled:
    @pytest.fixture()
    def client(self):
        return LocalEngramClient(
            vector_size=VEC_SIZE,
            decay_lambda=0,
            adaptive=True,
        )

    def test_density_stats_available(self, client):
        stats = client.density_stats
        assert isinstance(stats, DensityStats)
        assert stats.total_points == 0

    def test_density_tracks_inserts(self, client):
        for _ in range(5):
            client.insert(_make_state())
        stats = client.density_stats
        assert stats.total_points == 5

    def test_density_tracks_batch_inserts(self, client):
        states = [_make_state(ts=100 * i) for i in range(10)]
        client.insert_batch(states)
        stats = client.density_stats
        assert stats.total_points == 10

    def test_query_still_works_with_adaptive(self, client):
        client.insert(_make_state(vector=[1, 0, 0, 0]))
        results = client.query(vector=[1, 0, 0, 0])
        assert len(results) == 1

    def test_spatial_query_with_adaptive(self, client):
        client.insert(_make_state(x=0.1, y=0.1, z=0.1, vector=[1, 0, 0, 0]))
        client.insert(_make_state(x=0.9, y=0.9, z=0.9, vector=[0, 1, 0, 0]))
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

    def test_hot_cells_detected(self):
        # Use a low density threshold so we can trigger hot cells easily
        from engram.spatial.adaptive import AdaptiveResolution

        c = LocalEngramClient(
            vector_size=VEC_SIZE,
            decay_lambda=0,
            adaptive=True,
        )
        # Override the adaptive resolution with a lower threshold
        c._adaptive = AdaptiveResolution(base_order=4, max_order=6, density_threshold=3)
        # Insert many points into same spatial cell
        for i in range(10):
            c.insert(_make_state(ts=i * 100))
        stats = c.density_stats
        assert stats.hot_cells >= 1
