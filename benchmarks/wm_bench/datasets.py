"""Trajectory datasets for the wm_bench world-model memory benchmark.

ALL DATA IN v1 IS SYNTHETIC.  This environment cannot download the embodied
datasets named in RFC-0001 R3 (Habitat/AI2-THOR rollouts, TartanAir,
nuScenes), so v1 ships principled synthetic generators plus a dataset-adapter
protocol so real datasets plug in later without touching the tasks.

Realism limitations of the synthetic generators (be explicit, no spin):

- Embeddings are random low-dimensional manifolds + Gaussian noise, not the
  output of a trained world model.  Real V-JEPA/DINO-style embeddings have
  heavy-tailed structure, semantic clusters, and view-dependent aliasing that
  these generators do not reproduce.
- Positions are smooth momentum walks in the unit cube, not SLAM output;
  there is no drift, no loop-closure error, no kidnapped-robot events.
- OOD segments are *constructed* to be near-orthogonal to the base manifold.
  Real out-of-distribution observations are usually much closer to the
  training distribution — treat the novelty-AUC numbers as an upper bound
  on separability, not a field result.
- Scene revisits produce embeddings drawn from the same anchor distribution;
  real revisits differ by lighting, time of day, and dynamic obstacles.

Adapter protocol — to plug in a real dataset, implement
:class:`TrajectoryDataset`: expose ``name``, ``vector_dim``,
``synthetic = False``, and iterate :class:`TrajectoryPoint` in timestamp
order (strictly increasing ``timestamp_ms``, positions normalised to
[0, 1]^3, ``is_ood`` labels where ground truth exists, else ``False``).
Every task in :mod:`benchmarks.wm_bench.tasks` consumes only this protocol.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# All synthetic streams are future-dated (year ~2096).  This is deliberate:
# the LOCI clients clock maintenance (consolidation, retention) off
# max(wall clock, newest inserted timestamp), so a future-dated stream makes
# memory aging advance deterministically with the stream itself instead of
# with the machine's wall clock.  Metrics never read the wall clock.
BASE_TIMESTAMP_MS = 4_000_000_000_000
DEFAULT_STEP_MS = 100


@dataclass(frozen=True)
class TrajectoryPoint:
    """One embodied-trajectory sample: 4D address + embedding + labels."""

    timestamp_ms: int
    x: float
    y: float
    z: float
    embedding: tuple[float, ...]
    scene_id: str
    is_ood: bool = False


@runtime_checkable
class TrajectoryDataset(Protocol):
    """Protocol every dataset (synthetic or real adapter) implements."""

    name: str
    vector_dim: int
    synthetic: bool

    def points(self) -> list[TrajectoryPoint]:
        """Materialised points in strictly increasing timestamp order."""
        ...

    def __iter__(self) -> Iterator[TrajectoryPoint]: ...


# ---------------------------------------------------------------------------
# Generator building blocks
# ---------------------------------------------------------------------------


def _smooth_walk(
    rng: np.random.Generator,
    n: int,
    dims: int,
    lo: float,
    hi: float,
    *,
    accel: float,
    damping: float,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Momentum random walk with reflecting bounds; returns an (n, dims) array."""
    pos = np.empty((n, dims), dtype=np.float64)
    p = start.astype(np.float64).copy() if start is not None else rng.uniform(lo, hi, dims)
    v = np.zeros(dims, dtype=np.float64)
    for i in range(n):
        v = damping * v + accel * rng.standard_normal(dims)
        p = p + v
        # Reflect at the bounds so the walk stays inside [lo, hi].
        for d in range(dims):
            if p[d] < lo:
                p[d] = 2 * lo - p[d]
                v[d] = -v[d]
            elif p[d] > hi:
                p[d] = 2 * hi - p[d]
                v[d] = -v[d]
            p[d] = min(hi, max(lo, p[d]))
        pos[i] = p
    return pos


def _orthonormal_basis(rng: np.random.Generator, dim: int, m: int) -> np.ndarray:
    """Deterministic random orthonormal basis of shape (dim, m)."""
    q, _ = np.linalg.qr(rng.standard_normal((dim, m)))
    return q[:, :m]


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def _embed_latents(
    basis: np.ndarray,
    latents: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Project latents through the manifold basis, add noise, L2-normalise."""
    emb = latents @ basis.T
    emb = emb + noise_scale * rng.standard_normal((latents.shape[0], basis.shape[0]))
    return _normalise_rows(emb)


class _MaterialisedDataset:
    """Base for generators: cached point list + iteration."""

    name: str = "unnamed"
    vector_dim: int = 0
    synthetic: bool = True

    def __init__(self) -> None:
        self._points: list[TrajectoryPoint] = []

    def points(self) -> list[TrajectoryPoint]:
        return list(self._points)

    def __iter__(self) -> Iterator[TrajectoryPoint]:
        return iter(self._points)

    def __len__(self) -> int:
        return len(self._points)


# ---------------------------------------------------------------------------
# 1. Smooth random-walk patrol
# ---------------------------------------------------------------------------


class SmoothPatrolDataset(_MaterialisedDataset):
    """SYNTHETIC single-scene patrol: smooth spatial walk + manifold embedding.

    The embedding drifts along a low-dimensional manifold (a fixed random
    orthonormal basis applied to a smooth latent walk) plus isotropic noise —
    a crude stand-in for a world model watching a continuous traversal.
    """

    def __init__(
        self,
        *,
        n_points: int = 1500,
        vector_dim: int = 32,
        seed: int = 7,
        step_ms: int = DEFAULT_STEP_MS,
        manifold_dim: int = 4,
        noise_scale: float = 0.02,
        scene_id: str = "patrol",
        base_timestamp_ms: int = BASE_TIMESTAMP_MS,
    ) -> None:
        super().__init__()
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        self.name = f"synthetic:smooth_patrol(n={n_points},dim={vector_dim},seed={seed})"
        self.vector_dim = vector_dim
        self.step_ms = step_ms
        rng = np.random.default_rng(seed)

        self.manifold_basis = _orthonormal_basis(rng, vector_dim, manifold_dim)
        positions = _smooth_walk(rng, n_points, 3, 0.05, 0.95, accel=0.004, damping=0.92)
        latents = _smooth_walk(rng, n_points, manifold_dim, -1.0, 1.0, accel=0.05, damping=0.90)
        embeddings = _embed_latents(self.manifold_basis, latents, noise_scale, rng)

        self._points = [
            TrajectoryPoint(
                timestamp_ms=base_timestamp_ms + i * step_ms,
                x=float(positions[i, 0]),
                y=float(positions[i, 1]),
                z=float(positions[i, 2]),
                embedding=tuple(float(v) for v in embeddings[i]),
                scene_id=scene_id,
                is_ood=False,
            )
            for i in range(n_points)
        ]


# ---------------------------------------------------------------------------
# 2. Multi-scene warehouse with revisits
# ---------------------------------------------------------------------------


class WarehouseDataset(_MaterialisedDataset):
    """SYNTHETIC multi-scene warehouse: scenes are revisited in rotation.

    Each scene has a fixed latent anchor and spatial centre; every visit
    produces embeddings near that anchor, so re-encountered scenes yield
    re-encountered embeddings (revisit similarity > cross-scene similarity).
    One continuous timeline covers all visits.
    """

    def __init__(
        self,
        *,
        n_scenes: int = 4,
        n_visits: int = 3,
        points_per_visit: int = 40,
        vector_dim: int = 32,
        seed: int = 11,
        step_ms: int = DEFAULT_STEP_MS,
        manifold_dim: int = 4,
        noise_scale: float = 0.02,
        base_timestamp_ms: int = BASE_TIMESTAMP_MS,
    ) -> None:
        super().__init__()
        self.name = (
            f"synthetic:warehouse(scenes={n_scenes},visits={n_visits},"
            f"ppv={points_per_visit},dim={vector_dim},seed={seed})"
        )
        self.vector_dim = vector_dim
        self.step_ms = step_ms
        self.n_scenes = n_scenes
        self.n_visits = n_visits
        rng = np.random.default_rng(seed)

        self.manifold_basis = _orthonormal_basis(rng, vector_dim, manifold_dim)
        anchors = rng.standard_normal((n_scenes, manifold_dim))
        centres = rng.uniform(0.2, 0.8, (n_scenes, 3))

        self._points = []
        i = 0
        for _visit in range(n_visits):
            for scene in range(n_scenes):
                local_pos = _smooth_walk(
                    rng,
                    points_per_visit,
                    3,
                    -0.08,
                    0.08,
                    accel=0.003,
                    damping=0.9,
                    start=np.zeros(3),
                )
                pos = np.clip(centres[scene] + local_pos, 0.0, 1.0)
                local_lat = _smooth_walk(
                    rng,
                    points_per_visit,
                    manifold_dim,
                    -0.25,
                    0.25,
                    accel=0.02,
                    damping=0.9,
                    start=np.zeros(manifold_dim),
                )
                emb = _embed_latents(
                    self.manifold_basis, anchors[scene] + local_lat, noise_scale, rng
                )
                for j in range(points_per_visit):
                    self._points.append(
                        TrajectoryPoint(
                            timestamp_ms=base_timestamp_ms + i * step_ms,
                            x=float(pos[j, 0]),
                            y=float(pos[j, 1]),
                            z=float(pos[j, 2]),
                            embedding=tuple(float(v) for v in emb[j]),
                            scene_id=f"scene_{scene}",
                            is_ood=False,
                        )
                    )
                    i += 1


# ---------------------------------------------------------------------------
# 3. OOD-segment injection
# ---------------------------------------------------------------------------


class OodPatrolDataset(_MaterialisedDataset):
    """SYNTHETIC patrol with injected out-of-distribution excursions.

    Contiguous segments of the base patrol are replaced by excursions into a
    novel spatial corner with embeddings built from directions orthogonal to
    the base manifold (plus noise), labelled ``is_ood=True``.  Segments start
    only after ``ood_start_frac`` of the stream so novelty scorers have clean
    history to calibrate on.

    Honesty note: constructed orthogonality makes this an EASY OOD detection
    problem — treat the resulting AUC as an upper bound (see module docstring).
    """

    OOD_BOX = (0.90, 0.99)  # spatial corner used by excursions

    def __init__(
        self,
        *,
        n_points: int = 800,
        vector_dim: int = 32,
        seed: int = 13,
        n_ood_segments: int = 3,
        ood_segment_len: int = 40,
        ood_start_frac: float = 0.3,
        step_ms: int = DEFAULT_STEP_MS,
        manifold_dim: int = 4,
        noise_scale: float = 0.02,
        base_timestamp_ms: int = BASE_TIMESTAMP_MS,
    ) -> None:
        super().__init__()
        base = SmoothPatrolDataset(
            n_points=n_points,
            vector_dim=vector_dim,
            seed=seed,
            step_ms=step_ms,
            manifold_dim=manifold_dim,
            noise_scale=noise_scale,
            base_timestamp_ms=base_timestamp_ms,
        )
        self.name = (
            f"synthetic:ood_patrol(n={n_points},dim={vector_dim},seed={seed},"
            f"segments={n_ood_segments}x{ood_segment_len})"
        )
        self.vector_dim = vector_dim
        self.step_ms = step_ms
        self.manifold_basis = base.manifold_basis

        first_ood = int(n_points * ood_start_frac)
        usable = n_points - first_ood
        needed = n_ood_segments * ood_segment_len
        if needed > usable:
            raise ValueError(
                f"{n_ood_segments} segments of {ood_segment_len} points do not fit "
                f"in the last {usable} points of an {n_points}-point stream"
            )

        rng = np.random.default_rng(seed + 1)
        pts = base.points()
        gap = usable // n_ood_segments
        for s in range(n_ood_segments):
            start = first_ood + s * gap
            self._inject_segment(pts, start, ood_segment_len, rng)
        self._points = pts

    def _inject_segment(
        self,
        pts: list[TrajectoryPoint],
        start: int,
        length: int,
        rng: np.random.Generator,
    ) -> None:
        dim = self.vector_dim
        basis = self.manifold_basis
        # Novel direction: random vector minus its projection onto the base
        # manifold, i.e. (near-)orthogonal to everything seen in-distribution.
        raw = rng.standard_normal(dim)
        ortho = raw - basis @ (basis.T @ raw)
        ortho = ortho / np.linalg.norm(ortho)

        lo, hi = self.OOD_BOX
        pos = _smooth_walk(rng, length, 3, lo, hi, accel=0.002, damping=0.9)
        emb = ortho[None, :] + 0.05 * rng.standard_normal((length, dim))
        emb = _normalise_rows(emb)

        for j in range(length):
            i = start + j
            old = pts[i]
            pts[i] = TrajectoryPoint(
                timestamp_ms=old.timestamp_ms,
                x=float(pos[j, 0]),
                y=float(pos[j, 1]),
                z=float(pos[j, 2]),
                embedding=tuple(float(v) for v in emb[j]),
                scene_id=old.scene_id,
                is_ood=True,
            )

    @property
    def ood_fraction(self) -> float:
        return sum(1 for p in self._points if p.is_ood) / len(self._points)
