"""Type stubs for the loci_core Rust extension."""

from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Hilbert encoding
# ---------------------------------------------------------------------------

def encode_hilbert_3d(x: float, y: float, z: float, order: int) -> int: ...
def encode_hilbert_4d(x: float, y: float, z: float, t: float, order: int) -> int: ...
def decode_hilbert_3d(h: int, order: int) -> tuple[float, float, float]: ...
def batch_encode_hilbert_3d(
    coords: npt.NDArray[np.float64], order: int
) -> npt.NDArray[np.uint64]: ...
def batch_encode_hilbert_4d(
    coords: npt.NDArray[np.float64], order: int
) -> npt.NDArray[np.uint64]: ...
def spatial_bounds_to_hilbert_buckets_3d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    order: int,
    overlap_factor: float,
) -> list[int]: ...
def spatial_bounds_to_hilbert_buckets_4d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    order: int,
    overlap_factor: float,
) -> list[int]: ...

# ---------------------------------------------------------------------------
# Temporal sharding
# ---------------------------------------------------------------------------

def compute_epoch_id(timestamp_ms: int, epoch_size_ms: int) -> int: ...
def epoch_collection_name(epoch_id: int) -> str: ...
def epochs_for_time_window(start_ms: int, end_ms: int, epoch_size_ms: int) -> list[int]: ...
def normalise_timestamp_in_epoch(
    timestamp_ms: int, epoch_id: int, epoch_size_ms: int
) -> float: ...
def batch_compute_epoch_ids(
    timestamps_ms: npt.NDArray[np.int64], epoch_size_ms: int
) -> npt.NDArray[np.int64]: ...

# ---------------------------------------------------------------------------
# Spatial utilities
# ---------------------------------------------------------------------------

def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float: ...
def point_in_bounds(
    x: float,
    y: float,
    z: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> bool: ...
def batch_distances_3d(
    points: npt.NDArray[np.float64],
    query_x: float,
    query_y: float,
    query_z: float,
) -> npt.NDArray[np.float64]: ...
def adaptive_hilbert_order(density: float) -> int: ...

# ---------------------------------------------------------------------------
# Novelty scoring
# ---------------------------------------------------------------------------

def cosine_similarity(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> float: ...
def compute_novelty_score(
    predicted: npt.NDArray[np.float32], retrieved: npt.NDArray[np.float32]
) -> float: ...
def batch_novelty_scores(
    predicted: npt.NDArray[np.float32], retrieved: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]: ...
def temporal_decay_weight(
    observation_ms: int, query_ms: int, decay_factor: float
) -> float: ...

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_prepare_world_states(
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    zs: npt.NDArray[np.float64],
    timestamps_ms: npt.NDArray[np.int64],
    epoch_size_ms: int,
    hilbert_order: int,
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.int64],
]: ...

# ---------------------------------------------------------------------------
# In-memory vector store (RFC-0001 R5 stage (a))
# ---------------------------------------------------------------------------

class LociStore:
    """Native in-memory vector store; points/payloads/filters cross as dicts."""

    def __init__(self) -> None: ...
    def create_collection(
        self, name: str, vector_size: int, distance: str = "cosine"
    ) -> None: ...
    def collection_exists(self, name: str) -> bool: ...
    def delete_collection(self, name: str) -> None: ...
    def create_payload_index(self, collection: str, field_name: str) -> None: ...
    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None: ...
    def set_payload(
        self, collection: str, point_id: str, payload: dict[str, Any]
    ) -> None: ...
    def delete_points(self, collection: str, ids: list[str]) -> int: ...
    def delete_points_in_time_range(
        self,
        collection: str,
        start_ms: int,
        end_ms_exclusive: int,
        field: str = "timestamp_ms",
    ) -> int: ...
    def retrieve(self, collection: str, ids: list[str]) -> list[dict[str, Any]]: ...
    def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def scroll(
        self,
        collection: str,
        payload_filter: dict[str, Any] | None = None,
        limit: int = 10,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]: ...
    @property
    def total_points(self) -> int: ...
    def collection_count(self, name: str) -> int: ...
    def payload_value_range(
        self, collection: str, field: str
    ) -> tuple[Any, Any] | None: ...
