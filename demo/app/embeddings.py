"""Deterministic mock embedding generation for the warehouse demo.

Key property: same (x, y) + same visible objects = same vector.
This makes novelty detection actually work — revisiting a known area
produces low novelty, while new/changed areas produce high novelty.
"""

from __future__ import annotations

import hashlib
import math
import struct

EMBEDDING_DIM = 128


def generate_embedding(
    grid_x: int,
    grid_y: int,
    visible_objects: list[str],
    dim: int = EMBEDDING_DIM,
) -> list[float]:
    """Generate a deterministic embedding from position + visible objects.

    Uses sin/cos positional encoding for the first dimensions (captures
    spatial structure), then fills remaining dimensions from a seeded PRNG
    derived from the hash of visible objects.
    """
    key = f"{grid_x},{grid_y}|{'|'.join(sorted(visible_objects))}"
    digest = hashlib.sha256(key.encode()).digest()
    seed = struct.unpack("<I", digest[:4])[0]

    vec: list[float] = []
    for i in range(dim):
        if i < 8:
            # Positional encoding: sin/cos at increasing frequencies
            freq = (i // 2 + 1) * math.pi
            if i % 2 == 0:
                vec.append(math.sin(grid_x / 19.0 * freq))
            else:
                vec.append(math.cos(grid_y / 19.0 * freq))
        else:
            # Deterministic pseudo-random from seed
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            val = (seed / 0x7FFFFFFF) * 2.0 - 1.0  # [-1, 1]
            vec.append(val)

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-8:
        vec = [v / norm for v in vec]

    return vec
