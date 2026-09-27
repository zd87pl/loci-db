"""Deterministic mock embedding generation for the warehouse demo.

Key properties:

* Same (x, y) + same visible objects = same vector, so revisiting an
  unchanged place is recognised as familiar (novelty ~0).
* Place features vary smoothly across the floor, so neighbouring cells look
  alike and a first visit next to familiar territory is only mildly novel.
* Each visible object adds its own direction to the view. An anomaly (an
  object the robot has never seen) adds a strong one, so a scene containing
  it is clearly unlike anything in memory.
"""

from __future__ import annotations

import hashlib
import math
import struct

EMBEDDING_DIM = 128
GRID_MAX = 19.0

# How strongly each visible object shapes the view, relative to the (unit)
# place component. Anomalies dominate: they are what surprise detection is for.
OBJECT_WEIGHTS = {"anomaly": 6.0}
DEFAULT_OBJECT_WEIGHT = 0.1


def _seeded_unit_vector(key: str, dim: int) -> list[float]:
    """Deterministic pseudo-random unit vector derived from ``key``."""
    seed = struct.unpack("<I", hashlib.sha256(key.encode()).digest()[:4])[0]
    vec = []
    for _ in range(dim):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        vec.append((seed / 0x7FFFFFFF) * 2.0 - 1.0)
    return _normalize(vec)


def _place_params(dim: int) -> list[tuple[float, float, float]]:
    """Low random spatial frequencies and phases (fixed per dimension count)."""
    raw = _seeded_unit_vector(f"place-params:{dim}", dim * 3)
    params = []
    for i in range(dim):
        kx, ky, phase = raw[3 * i : 3 * i + 3]
        # Scale to at most ~1 cycle across the floor: smooth, not aliased.
        params.append((kx * 4.0, ky * 4.0, phase * math.pi * 8.0))
    return params


_PLACE_PARAMS: dict[int, list[tuple[float, float, float]]] = {}


def _place_features(grid_x: int, grid_y: int, dim: int) -> list[float]:
    params = _PLACE_PARAMS.setdefault(dim, _place_params(dim))
    u, v = grid_x / GRID_MAX, grid_y / GRID_MAX
    return _normalize([math.cos(2 * math.pi * (kx * u + ky * v) + ph) for kx, ky, ph in params])


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 1e-8 else vec


def generate_embedding(
    grid_x: int,
    grid_y: int,
    visible_objects: list[str],
    dim: int = EMBEDDING_DIM,
) -> list[float]:
    """Generate a deterministic embedding from position + visible objects.

    ``visible_objects`` are keys like ``"shelf@3,4"``; the part before ``@``
    selects the object's weight.
    """
    vec = _place_features(grid_x, grid_y, dim)
    for key in sorted(visible_objects):
        weight = OBJECT_WEIGHTS.get(key.split("@", 1)[0], DEFAULT_OBJECT_WEIGHT)
        obj = _seeded_unit_vector(key, dim)
        vec = [a + weight * b for a, b in zip(vec, obj, strict=True)]
    return _normalize(vec)
