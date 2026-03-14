"""Tests for adaptive Hilbert resolution with density estimation."""

from __future__ import annotations

import pytest

from engram.spatial.adaptive import AdaptiveResolution


class TestAdaptiveResolution:
    def test_default_resolution(self):
        ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=10)
        # Before any data, should return base resolution
        assert ar.resolution_for(0.5, 0.5, 0.5, 0.5) == 4

    def test_escalation_after_threshold(self):
        ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=5)
        # Insert 5 points into the same cell
        for _ in range(5):
            ar.record(0.5, 0.5, 0.5, 0.5)
        assert ar.resolution_for(0.5, 0.5, 0.5, 0.5) == 5  # base + 1

    def test_further_escalation(self):
        ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=5)
        for _ in range(10):
            ar.record(0.5, 0.5, 0.5, 0.5)
        assert ar.resolution_for(0.5, 0.5, 0.5, 0.5) == 6  # base + 2

    def test_max_order_cap(self):
        ar = AdaptiveResolution(base_order=4, max_order=5, density_threshold=5)
        for _ in range(100):
            ar.record(0.5, 0.5, 0.5, 0.5)
        assert ar.resolution_for(0.5, 0.5, 0.5, 0.5) == 5  # capped

    def test_sparse_region_stays_base(self):
        ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=10)
        ar.record(0.1, 0.1, 0.1, 0.1)
        ar.record(0.9, 0.9, 0.9, 0.9)
        assert ar.resolution_for(0.1, 0.1, 0.1, 0.1) == 4
        assert ar.resolution_for(0.9, 0.9, 0.9, 0.9) == 4

    def test_cell_density(self):
        ar = AdaptiveResolution(base_order=4, density_threshold=100)
        for _ in range(7):
            ar.record(0.3, 0.3, 0.3, 0.3)
        assert ar.cell_density(0.3, 0.3, 0.3, 0.3) == 7
        assert ar.cell_density(0.9, 0.9, 0.9, 0.9) == 0

    def test_stats(self):
        ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=3)
        for _ in range(5):
            ar.record(0.5, 0.5, 0.5, 0.5)
        ar.record(0.1, 0.1, 0.1, 0.1)

        s = ar.stats()
        assert s.total_points == 6
        assert s.num_cells == 2
        assert s.max_density == 5
        assert s.hot_cells == 1  # only the (0.5, 0.5) cell is hot

    def test_stats_empty(self):
        ar = AdaptiveResolution()
        s = ar.stats()
        assert s.total_points == 0
        assert s.num_cells == 0

    def test_reset(self):
        ar = AdaptiveResolution(base_order=4, density_threshold=3)
        for _ in range(10):
            ar.record(0.5, 0.5, 0.5, 0.5)
        ar.reset()
        assert ar.cell_density(0.5, 0.5, 0.5, 0.5) == 0
        assert ar.resolution_for(0.5, 0.5, 0.5, 0.5) == 4
        assert ar.stats().total_points == 0

    def test_invalid_max_order(self):
        with pytest.raises(ValueError, match="max_order"):
            AdaptiveResolution(base_order=6, max_order=4)

    def test_invalid_density_threshold(self):
        with pytest.raises(ValueError, match="density_threshold"):
            AdaptiveResolution(density_threshold=0)

    def test_different_regions_different_resolutions(self):
        ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=5)
        # Dense region
        for _ in range(10):
            ar.record(0.5, 0.5, 0.5, 0.5)
        # Sparse region
        ar.record(0.1, 0.1, 0.1, 0.1)

        assert ar.resolution_for(0.5, 0.5, 0.5, 0.5) > ar.resolution_for(0.1, 0.1, 0.1, 0.1)
