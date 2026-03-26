"""V-JEPA 2 embedding adapter for LOCI.

Converts V-JEPA 2 patch tokens (tubelets) to WorldState objects.

V-JEPA 2 produces tubelets: (2 frames x 16x16 pixels) -> 1408-dim vector.
This adapter handles patch -> WorldState mapping including position extraction
from the tubelet grid coordinates.
"""

from __future__ import annotations

import uuid

import numpy as np

from loci.schema import WorldState


class VJEPA2Adapter:
    """Converts V-JEPA 2 patch tokens to WorldState objects.

    V-JEPA 2 produces tubelets: (2 frames x 16x16 pixels) -> 1408-dim vector.
    This adapter handles patch -> WorldState mapping including
    3D-RoPE-aware position extraction.

    Args:
        default_scale_level: Scale level for generated WorldStates.
        default_confidence: Default confidence for generated WorldStates.
    """

    def __init__(
        self,
        default_scale_level: str = "patch",
        default_confidence: float = 1.0,
    ) -> None:
        self._scale_level = default_scale_level
        self._confidence = default_confidence

    def tubelet_to_world_state(
        self,
        tubelet_embedding: np.ndarray,
        patch_position: tuple[int, int, int],
        grid_shape: tuple[int, int, int],
        timestamp_ms: int,
        scene_id: str,
    ) -> WorldState:
        """Convert a single V-JEPA 2 tubelet to a WorldState.

        Maps patch grid position to normalized (x, y, z) using the grid
        dimensions so that grid indices are uniformly distributed over [0, 1].

        Args:
            tubelet_embedding: Tubelet embedding vector, shape (1408,).
            patch_position: (time_idx, h_idx, w_idx) in the patch grid.
            grid_shape: (T, H, W) dimensions of the patch grid.
                Used to normalize patch positions to [0, 1].
            timestamp_ms: Timestamp in milliseconds.
            scene_id: Scene identifier for causal linking.

        Returns:
            A WorldState with normalized spatial coordinates.
        """
        if tubelet_embedding.ndim != 1:
            raise ValueError(
                f"Expected 1D embedding, got shape {tubelet_embedding.shape}"
            )

        t_idx, h_idx, w_idx = patch_position
        T, H, W = grid_shape

        # Normalize grid positions to [0, 1]
        x = min(1.0, max(0.0, w_idx / max(W - 1, 1)))
        y = min(1.0, max(0.0, h_idx / max(H - 1, 1)))
        z = min(1.0, max(0.0, t_idx / max(T - 1, 1)))

        return WorldState(
            x=x,
            y=y,
            z=z,
            timestamp_ms=timestamp_ms,
            vector=tubelet_embedding.tolist(),
            scene_id=scene_id,
            scale_level=self._scale_level,
            confidence=self._confidence,
            id=uuid.uuid4().hex,
        )

    def batch_clip_to_states(
        self,
        clip_embeddings: np.ndarray,
        start_timestamp_ms: int,
        scene_id: str,
        frame_interval_ms: int = 33,
    ) -> list[WorldState]:
        """Convert an entire V-JEPA 2 clip output to a list of WorldStates.

        Args:
            clip_embeddings: Full clip output, shape (T, H, W, D) where
                T=temporal patches, H=height patches, W=width patches,
                D=embedding dimension (typically 1408).
            start_timestamp_ms: Timestamp of the first frame.
            scene_id: Scene identifier.
            frame_interval_ms: Time between frames in ms (default 33 ~ 30fps).

        Returns:
            List of WorldState objects, one per tubelet.
        """
        if clip_embeddings.ndim != 4:
            raise ValueError(
                f"Expected 4D array (T, H, W, D), got shape {clip_embeddings.shape}"
            )

        T, H, W, D = clip_embeddings.shape
        grid_shape = (T, H, W)
        states: list[WorldState] = []

        for t in range(T):
            ts = start_timestamp_ms + t * frame_interval_ms
            for h in range(H):
                for w in range(W):
                    states.append(
                        self.tubelet_to_world_state(
                            tubelet_embedding=clip_embeddings[t, h, w],
                            patch_position=(t, h, w),
                            grid_shape=grid_shape,
                            timestamp_ms=ts,
                            scene_id=scene_id,
                        )
                    )

        return states
