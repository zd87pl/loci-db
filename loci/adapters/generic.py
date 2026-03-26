"""Generic adapter for any world model producing numpy arrays or torch tensors.

Handles normalization, dimensionality validation, and type conversion.
"""

from __future__ import annotations

import uuid

import numpy as np

from loci.schema import WorldState


class GenericAdapter:
    """Adapter for any world model that produces numpy arrays or torch tensors.

    Handles normalization, dimensionality validation, and type conversion.

    Args:
        expected_dim: Expected embedding dimensionality (None = any).
    """

    def __init__(self, expected_dim: int | None = None) -> None:
        self._expected_dim = expected_dim

    def from_numpy(
        self,
        embedding: np.ndarray,
        position: tuple[float, float, float],
        timestamp_ms: int,
        scene_id: str,
        scale_level: str = "patch",
        confidence: float = 1.0,
    ) -> WorldState:
        """Convert a numpy array embedding to a WorldState.

        Args:
            embedding: 1D numpy array of any dtype (will be converted to float).
            position: (x, y, z) in normalized [0, 1] coordinates.
            timestamp_ms: Timestamp in milliseconds.
            scene_id: Scene identifier.
            scale_level: Spatial scale ("patch", "frame", or "sequence").
            confidence: Confidence score [0, 1].

        Returns:
            A WorldState with the embedding as vector.
        """
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {embedding.shape}")
        if self._expected_dim is not None and embedding.shape[0] != self._expected_dim:
            raise ValueError(f"Expected {self._expected_dim}-dim, got {embedding.shape[0]}-dim")

        x, y, z = position
        return WorldState(
            x=x,
            y=y,
            z=z,
            timestamp_ms=timestamp_ms,
            vector=embedding.tolist(),
            scene_id=scene_id,
            scale_level=scale_level,
            confidence=confidence,
            id=uuid.uuid4().hex,
        )

    def from_torch(
        self,
        embedding,  # torch.Tensor — not type-hinted to keep torch optional
        position: tuple[float, float, float],
        timestamp_ms: int,
        scene_id: str,
        scale_level: str = "patch",
        confidence: float = 1.0,
    ) -> WorldState:
        """Convert a torch tensor embedding to a WorldState.

        Args:
            embedding: 1D torch tensor (will be detached and converted).
            position: (x, y, z) in normalized [0, 1] coordinates.
            timestamp_ms: Timestamp in milliseconds.
            scene_id: Scene identifier.
            scale_level: Spatial scale ("patch", "frame", or "sequence").
            confidence: Confidence score [0, 1].

        Returns:
            A WorldState with the embedding as vector.
        """
        arr = embedding.detach().cpu().numpy()
        return self.from_numpy(
            embedding=arr,
            position=position,
            timestamp_ms=timestamp_ms,
            scene_id=scene_id,
            scale_level=scale_level,
            confidence=confidence,
        )
