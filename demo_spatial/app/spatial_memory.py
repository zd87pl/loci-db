"""LOCI-DB spatial memory wrapper for the assistive AI demo.

Provides a high-level API for tracking object positions over time and
answering natural-language location queries ("where did I leave my keys?").
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass

from loci import LocalLociClient
from loci.schema import WorldState

EMBED_DIM = 128


def _label_embedding(label: str) -> list[float]:
    """Deterministic 128-dim embedding from an object label.

    Uses positional-encoding style construction: first 8 dims are
    sin/cos of character code frequencies, remainder from a seeded PRNG.
    Same label always produces the same vector (enables semantic recall).
    """
    normalized = label.strip().lower()
    key = f"label:{normalized}"
    digest = hashlib.sha256(key.encode()).digest()
    seed = struct.unpack("<I", digest[:4])[0]

    vec: list[float] = []
    for i in range(EMBED_DIM):
        if i < 8:
            char_val = sum(ord(c) for c in normalized) / max(len(normalized), 1)
            freq = (i // 2 + 1) * math.pi
            if i % 2 == 0:
                vec.append(math.sin(char_val / 128.0 * freq))
            else:
                vec.append(math.cos(char_val / 128.0 * freq))
        else:
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            val = (seed / 0x7FFFFFFF) * 2.0 - 1.0
            vec.append(val)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-8:
        vec = [v / norm for v in vec]
    return vec


def _object_embedding(label: str, cx: float, cy: float) -> list[float]:
    """Blend label semantics (80%) with position (20%) into a single vector."""
    label_vec = _label_embedding(label)
    pos_key = f"pos:{cx:.3f},{cy:.3f}"
    digest = hashlib.sha256(pos_key.encode()).digest()
    seed = struct.unpack("<I", digest[:4])[0]

    pos_vec: list[float] = []
    for i in range(EMBED_DIM):
        if i < 4:
            vals = [cx * 2 - 1, cy * 2 - 1, math.sin(cx * math.pi), math.cos(cy * math.pi)]
            pos_vec.append(vals[i])
        else:
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            pos_vec.append((seed / 0x7FFFFFFF) * 2.0 - 1.0)

    p_norm = math.sqrt(sum(v * v for v in pos_vec))
    if p_norm > 1e-8:
        pos_vec = [v / p_norm for v in pos_vec]

    blended = [0.8 * l + 0.2 * p for l, p in zip(label_vec, pos_vec)]
    b_norm = math.sqrt(sum(v * v for v in blended))
    if b_norm > 1e-8:
        blended = [v / b_norm for v in blended]
    return blended


@dataclass
class ObjectObservation:
    """A single tracked observation of a physical object."""

    label: str
    cx: float       # normalized [0, 1] — horizontal position in camera frame
    cy: float       # normalized [0, 1] — vertical position in camera frame
    confidence: float
    timestamp_ms: int
    state_id: str
    depth_m: float | None = None   # LiDAR depth in meters (None if 2D only)

    @property
    def age_seconds(self) -> float:
        return (int(time.time() * 1000) - self.timestamp_ms) / 1000.0

    def to_dict(self) -> dict:
        result = {
            "label": self.label,
            "cx": round(self.cx, 3),
            "cy": round(self.cy, 3),
            "confidence": round(self.confidence, 3),
            "timestamp_ms": self.timestamp_ms,
            "age_seconds": round(self.age_seconds, 1),
            "state_id": self.state_id,
        }
        if self.depth_m is not None:
            result["depth_m"] = round(self.depth_m, 3)
        return result


class SpatialMemory:
    """Thread-compatible wrapper around LocalLociClient for object tracking.

    Uses LOCI-DB's spatial + temporal indexing to answer questions like:
      - "Where is my water bottle?" (latest known position)
      - "Where was my phone last Tuesday?" (temporal history)
      - "What objects are in the left side of the room?" (spatial range query)
    """

    def __init__(self, epoch_size_ms: int = 5000) -> None:
        self._client = LocalLociClient(
            epoch_size_ms=epoch_size_ms,
            vector_size=EMBED_DIM,
            distance="cosine",
        )
        # Cache of the latest known observation per label
        self._latest: dict[str, ObjectObservation] = {}
        self._observation_count = 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def observe(
        self,
        label: str,
        cx: float,
        cy: float,
        confidence: float = 1.0,
        timestamp_ms: int | None = None,
        depth_m: float | None = None,
    ) -> str:
        """Record an object sighting in spatial memory. Returns the state_id."""
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        # Clamp coords to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        confidence = max(0.0, min(1.0, confidence))

        # Normalize depth to [0, 1] range (0m = 0.0, 5m+ = 1.0)
        z_normalized = min(depth_m / 5.0, 1.0) if depth_m is not None else 0.0

        vector = _object_embedding(label, cx, cy)
        state = WorldState(
            x=cx,
            y=cy,
            z=z_normalized,
            timestamp_ms=ts,
            vector=vector,
            scene_id=label.strip().lower(),  # scene_id groups observations by object
            confidence=confidence,
        )
        state_id = self._client.insert(state)
        obs = ObjectObservation(
            label=label, cx=cx, cy=cy,
            confidence=confidence, timestamp_ms=ts, state_id=state_id,
            depth_m=depth_m,
        )
        self._latest[label.strip().lower()] = obs
        self._observation_count += 1
        return state_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def where_is(self, label: str, limit: int = 5) -> list[ObjectObservation]:
        """Find recent observations of an object by semantic label match.

        Returns results sorted newest-first. Combines vector similarity
        (semantic label match) with a full-history time window for recall.
        """
        query_vec = _label_embedding(label)
        normalized = label.strip().lower()

        # First try: exact scene_id match via extra filter (scalar = exact match)
        results = self._client.query(
            vector=query_vec,
            limit=limit * 2,
            _extra_payload_filter={"scene_id": normalized},
        )

        # Fallback: semantic similarity search without filter
        if not results:
            results = self._client.query(vector=query_vec, limit=limit * 3)
            results = [r for r in results if r.scene_id == normalized]

        observations = [
            ObjectObservation(
                label=r.scene_id or label,
                cx=r.x, cy=r.y,
                confidence=r.confidence,
                timestamp_ms=r.timestamp_ms,
                state_id=r.id,
            )
            for r in results
        ]
        # Sort newest-first, deduplicate by very close positions
        observations.sort(key=lambda o: o.timestamp_ms, reverse=True)
        return observations[:limit]

    def history(self, label: str, limit: int = 20) -> list[ObjectObservation]:
        """Return full temporal history of an object's positions."""
        return self.where_is(label, limit=limit)

    def objects_in_region(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        limit: int = 20,
    ) -> list[ObjectObservation]:
        """Find all objects observed in a spatial region (normalized coords)."""
        import numpy as np

        query_vec = [0.0] * EMBED_DIM  # neutral vector — let spatial filter dominate
        spatial_bounds = {
            "x_min": max(0.0, x_min),
            "x_max": min(1.0, x_max),
            "y_min": max(0.0, y_min),
            "y_max": min(1.0, y_max),
            "z_min": 0.0,
            "z_max": 1.0,
        }
        results = self._client.query(
            vector=query_vec,
            spatial_bounds=spatial_bounds,
            limit=limit,
        )
        seen_labels: set[str] = set()
        observations = []
        for r in results:
            if r.scene_id not in seen_labels:
                seen_labels.add(r.scene_id)
                observations.append(ObjectObservation(
                    label=r.scene_id or "unknown",
                    cx=r.x, cy=r.y,
                    confidence=r.confidence,
                    timestamp_ms=r.timestamp_ms,
                    state_id=r.id,
                ))
        return observations

    def current_objects(self) -> list[ObjectObservation]:
        """Return the most recently observed position of every tracked object."""
        return sorted(self._latest.values(), key=lambda o: o.timestamp_ms, reverse=True)

    def recent_changes(self, window_seconds: float = 30.0) -> list[ObjectObservation]:
        """Objects observed in the last N seconds."""
        cutoff_ms = int(time.time() * 1000) - int(window_seconds * 1000)
        return [o for o in self._latest.values() if o.timestamp_ms >= cutoff_ms]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def tracked_labels(self) -> list[str]:
        return list(self._latest.keys())

    def stats(self) -> dict:
        qs = self._client.last_query_stats
        return {
            "observation_count": self._observation_count,
            "tracked_objects": len(self._latest),
            "labels": self.tracked_labels,
            "last_query": {
                "shards_searched": qs.shards_searched if qs else 0,
                "elapsed_ms": round(qs.elapsed_ms, 2) if qs else 0,
            },
        }
