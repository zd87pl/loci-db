"""Core data model for Engram — the WorldState dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldState:
    """A single spatiotemporal embedding with a 4D address.

    Every vector stored in Engram has a position in (x, y, z, t) space
    plus an arbitrary-dimension embedding vector produced by a world model.

    Attributes:
        x: Normalised spatial coordinate in [0, 1].
        y: Normalised spatial coordinate in [0, 1].
        z: Normalised spatial coordinate in [0, 1].
        timestamp_ms: Unix epoch timestamp in milliseconds.
        vector: Embedding vector (e.g. 512-d, 1024-d, 1408-d).
        scene_id: Optional scene/environment identifier.
        scale_level: Granularity — ``"patch"``, ``"frame"``, or ``"sequence"``.
        confidence: Confidence score for this state, in [0, 1].
        prev_state_id: ID of the causally preceding state (populated after insert).
        next_state_id: ID of the causally following state (populated after insert).
        id: Unique identifier assigned by the store on insert.
    """

    # 4D spatiotemporal address
    x: float
    y: float
    z: float
    timestamp_ms: int

    # embedding
    vector: list[float]

    # optional metadata
    scene_id: str = ""
    scale_level: str = "patch"
    confidence: float = 1.0

    # causal links (populated by the store after insert)
    prev_state_id: str | None = None
    next_state_id: str | None = None

    # store-assigned id
    id: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0):
            raise ValueError(f"x must be in [0, 1], got {self.x}")
        if not (0.0 <= self.y <= 1.0):
            raise ValueError(f"y must be in [0, 1], got {self.y}")
        if not (0.0 <= self.z <= 1.0):
            raise ValueError(f"z must be in [0, 1], got {self.z}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.timestamp_ms < 0:
            raise ValueError(f"timestamp_ms must be non-negative, got {self.timestamp_ms}")
        if self.scale_level not in ("patch", "frame", "sequence"):
            raise ValueError(
                f"scale_level must be 'patch', 'frame', or 'sequence', got {self.scale_level!r}"
            )
