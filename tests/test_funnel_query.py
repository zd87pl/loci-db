"""Tests for funnel_query (multi-scale coarse-to-fine search)."""

from __future__ import annotations

import pytest

from engram.local_client import LocalEngramClient
from engram.schema import WorldState

VEC_SIZE = 4


def _make_state(
    scale: str = "patch",
    ts: int = 1000,
    vector: list[float] | None = None,
) -> WorldState:
    return WorldState(
        x=0.5,
        y=0.5,
        z=0.5,
        timestamp_ms=ts,
        vector=vector or [1.0, 0.0, 0.0, 0.0],
        scale_level=scale,
    )


@pytest.fixture()
def client():
    return LocalEngramClient(
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VEC_SIZE,
        decay_lambda=0.0,
    )


class TestFunnelQuery:
    def test_returns_patch_when_all_scales_present(self, client):
        client.insert(_make_state(scale="sequence", vector=[1, 0, 0, 0]))
        client.insert(_make_state(scale="frame", vector=[0.9, 0.1, 0, 0]))
        client.insert(_make_state(scale="patch", vector=[0.8, 0.2, 0, 0]))
        results = client.funnel_query(vector=[1, 0, 0, 0], limit=5)
        assert len(results) >= 1
        assert results[0].scale_level == "patch"

    def test_returns_frame_when_no_patch(self, client):
        client.insert(_make_state(scale="sequence", vector=[1, 0, 0, 0]))
        client.insert(_make_state(scale="frame", vector=[0.9, 0.1, 0, 0]))
        results = client.funnel_query(vector=[1, 0, 0, 0], limit=5)
        assert len(results) >= 1
        assert results[0].scale_level == "frame"

    def test_returns_sequence_when_only_coarse(self, client):
        client.insert(_make_state(scale="sequence", vector=[1, 0, 0, 0]))
        results = client.funnel_query(vector=[1, 0, 0, 0], limit=5)
        assert len(results) == 1
        assert results[0].scale_level == "sequence"

    def test_empty_when_no_data(self, client):
        results = client.funnel_query(vector=[1, 0, 0, 0])
        assert results == []

    def test_respects_limit(self, client):
        for i in range(10):
            client.insert(_make_state(scale="patch", ts=100 * i))
        results = client.funnel_query(vector=[1, 0, 0, 0], limit=3)
        assert len(results) <= 3

    def test_with_spatial_bounds(self, client):
        client.insert(_make_state(scale="patch", vector=[1, 0, 0, 0]))
        results = client.funnel_query(
            vector=[1, 0, 0, 0],
            spatial_bounds={
                "x_min": 0.4,
                "x_max": 0.6,
                "y_min": 0.4,
                "y_max": 0.6,
                "z_min": 0.4,
                "z_max": 0.6,
            },
            limit=5,
        )
        assert len(results) >= 1

    def test_with_time_window(self, client):
        client.insert(_make_state(scale="patch", ts=1000))
        client.insert(_make_state(scale="patch", ts=6000))
        results = client.funnel_query(
            vector=[1, 0, 0, 0],
            time_window_ms=(500, 1500),
            limit=5,
        )
        assert len(results) == 1
