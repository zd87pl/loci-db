"""Adaptive Hilbert resolution via density estimation.

Standard Hilbert encoding uses a fixed resolution (e.g. p=4 → 16 bins/axis).
This wastes precision in dense regions and filter cardinality in sparse ones.

AdaptiveResolution tracks insertion density per Hilbert cell and recommends
higher resolution for cells that exceed a density threshold.  This lets
Loci self-tune its spatial indexing as data arrives — no manual knob-turning.

Design principles:
- Zero-copy: operates on counters, never touches stored vectors
- Online: updates incrementally with each insert (O(1) amortized)
- Bounded: max resolution capped to prevent combinatorial explosion
- Compatible: produces standard Hilbert IDs at varying resolutions

Usage:
    ar = AdaptiveResolution(base_order=4, max_order=6, density_threshold=50)
    resolution = ar.resolution_for(x, y, z, t_norm)
    hid = hilbert_encode(x, y, z, t_norm, resolution_order=resolution)
    ar.record(x, y, z, t_norm)  # update density counters
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from loci.spatial.hilbert import encode as hilbert_encode


@dataclass
class DensityStats:
    """Snapshot of density estimation state.

    Attributes:
        total_points: Total points recorded.
        num_cells: Number of non-empty cells at base resolution.
        max_density: Maximum point count in any single cell.
        mean_density: Mean point count across non-empty cells.
        hot_cells: Number of cells above the density threshold.
        resolution_map: Mapping of cell_id → active resolution for hot cells.
    """

    total_points: int = 0
    num_cells: int = 0
    max_density: int = 0
    mean_density: float = 0.0
    hot_cells: int = 0
    resolution_map: dict[int, int] = field(default_factory=dict)


class AdaptiveResolution:
    """Density-aware Hilbert resolution selector.

    Tracks per-cell density at the base resolution.  When a cell exceeds
    ``density_threshold`` points, queries and inserts into that region
    use a higher resolution (``base_order + 1``, up to ``max_order``).

    This is a pure heuristic — it trades slightly larger Hilbert ID sets
    in dense regions for better spatial discrimination.

    Args:
        base_order: Default Hilbert resolution (p value).
        max_order: Maximum resolution to escalate to.
        density_threshold: Point count per cell that triggers escalation.
        max_cache_size: Maximum number of entries in the resolution LRU cache.
            Older entries are evicted when the limit is reached.  Defaults to
            4096, which bounds memory to roughly 100 KB for typical workloads.
    """

    _DEFAULT_MAX_CACHE_SIZE = 4096

    def __init__(
        self,
        base_order: int = 4,
        max_order: int = 6,
        density_threshold: int = 50,
        max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE,
    ) -> None:
        if max_order < base_order:
            raise ValueError(f"max_order ({max_order}) must be >= base_order ({base_order})")
        if density_threshold < 1:
            raise ValueError("density_threshold must be >= 1")
        if max_cache_size < 1:
            raise ValueError("max_cache_size must be >= 1")

        self._base_order = base_order
        self._max_order = max_order
        self._density_threshold = density_threshold
        self._max_cache_size = max_cache_size

        # Density counters at base resolution
        self._cell_counts: Counter[int] = Counter()
        self._total_points = 0

        # LRU cache: base cell_id → escalated resolution (bounded by max_cache_size)
        self._resolution_cache: OrderedDict[int, int] = OrderedDict()

    @property
    def base_order(self) -> int:
        return self._base_order

    @property
    def max_order(self) -> int:
        return self._max_order

    @property
    def density_threshold(self) -> int:
        return self._density_threshold

    def _cache_set(self, cell_id: int, resolution: int) -> None:
        """Insert or update *cell_id* in the LRU cache, evicting the oldest entry if full."""
        if cell_id in self._resolution_cache:
            self._resolution_cache.move_to_end(cell_id)
        self._resolution_cache[cell_id] = resolution
        if len(self._resolution_cache) > self._max_cache_size:
            self._resolution_cache.popitem(last=False)

    def record(self, x: float, y: float, z: float, t_norm: float) -> None:
        """Record a point insertion for density tracking.

        Call this after each insert to keep density estimates current.
        """
        cell_id = hilbert_encode(x, y, z, t_norm, resolution_order=self._base_order)
        self._cell_counts[cell_id] += 1
        self._total_points += 1

        # Check if this cell crossed or re-evaluated the threshold
        count = self._cell_counts[cell_id]
        if count >= self._density_threshold:
            escalation = min(
                self._max_order - self._base_order,
                count // self._density_threshold,
            )
            self._cache_set(cell_id, self._base_order + escalation)

    def resolution_for(self, x: float, y: float, z: float, t_norm: float) -> int:
        """Return the recommended Hilbert resolution for a coordinate.

        Returns the base resolution unless the region is dense, in which
        case it returns an escalated resolution.
        """
        cell_id = hilbert_encode(x, y, z, t_norm, resolution_order=self._base_order)
        if cell_id in self._resolution_cache:
            self._resolution_cache.move_to_end(cell_id)
            return self._resolution_cache[cell_id]
        return self._base_order

    def cell_density(self, x: float, y: float, z: float, t_norm: float) -> int:
        """Return the current point count for the cell containing the coordinate."""
        cell_id = hilbert_encode(x, y, z, t_norm, resolution_order=self._base_order)
        return self._cell_counts.get(cell_id, 0)

    def stats(self) -> DensityStats:
        """Return a snapshot of density estimation state."""
        if not self._cell_counts:
            return DensityStats()

        counts = list(self._cell_counts.values())
        return DensityStats(
            total_points=self._total_points,
            num_cells=len(self._cell_counts),
            max_density=max(counts),
            mean_density=sum(counts) / len(counts),
            hot_cells=len(self._resolution_cache),
            resolution_map=dict(self._resolution_cache),  # snapshot (not LRU-ordered)
        )

    def reset(self) -> None:
        """Clear all density counters and cached resolutions."""
        self._cell_counts.clear()
        self._resolution_cache.clear()
        self._total_points = 0
