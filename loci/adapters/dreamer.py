"""DreamerV3 RSSM state adapter for LOCI.

Converts DreamerV3 RSSM states (deterministic h_t + stochastic z_t)
to WorldState objects for storage in the spatiotemporal database.
"""

from __future__ import annotations

import uuid

import numpy as np

from loci.schema import WorldState


class DreamerV3Adapter:
    """Converts DreamerV3 RSSM states to WorldState objects.

    DreamerV3 state = deterministic (h_t) + stochastic (z_t) concatenated.
    The stochastic component captures uncertainty — a key signal
    that LOCI can use to weight predictions.

    Args:
        default_scale_level: Scale level for generated WorldStates.
    """

    def __init__(self, default_scale_level: str = "frame") -> None:
        self._scale_level = default_scale_level

    def rssm_to_world_state(
        self,
        h_t: np.ndarray,
        z_t: np.ndarray,
        position: tuple[float, float, float],
        timestamp_ms: int,
        scene_id: str,
        confidence: float | None = None,
    ) -> WorldState:
        """Convert an RSSM state to a WorldState.

        Concatenates h_t and z_t into a single vector for storage.
        If confidence is not provided, it is estimated from the
        entropy of the stochastic component.

        Args:
            h_t: Deterministic GRU hidden state (e.g., 512-dim or 1024-dim).
            z_t: Stochastic categorical state (32x32 = 1024 dims typically).
            position: (x, y, z) in normalized [0, 1] coordinates.
            timestamp_ms: Timestamp in milliseconds.
            scene_id: Scene identifier for causal linking.
            confidence: Optional confidence override. If None, estimated
                from z_t entropy (low entropy = high confidence).

        Returns:
            A WorldState with concatenated [h_t, z_t] as the vector.
        """
        if h_t.ndim != 1:
            raise ValueError(f"h_t must be 1D, got shape {h_t.shape}")
        if z_t.ndim != 1:
            raise ValueError(f"z_t must be 1D, got shape {z_t.shape}")

        combined = np.concatenate([h_t, z_t])

        if confidence is None:
            # Estimate confidence from stochastic component
            # Higher max values in z_t indicate more certain categorical choices
            z_max = float(np.max(np.abs(z_t)))
            confidence = min(1.0, max(0.0, z_max / (z_max + 1.0)))

        x, y, z = position
        return WorldState(
            x=x,
            y=y,
            z=z,
            timestamp_ms=timestamp_ms,
            vector=combined.tolist(),
            scene_id=scene_id,
            scale_level=self._scale_level,
            confidence=confidence,
            id=uuid.uuid4().hex,
        )
