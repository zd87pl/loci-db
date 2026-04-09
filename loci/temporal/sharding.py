"""Shard key computation and epoch management for temporal partitioning.

Vectors are routed to per-epoch Qdrant collections named ``loci_{epoch_id}``.
An epoch is a fixed-width time window (default 5 000 ms).

When the ``loci_core`` Rust extension is available, all functions
delegate to the native implementation.
"""

from __future__ import annotations

try:
    import loci_core as _rust

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def epoch_id(timestamp_ms: int, epoch_size_ms: int) -> int:
    """Return the epoch index for a given timestamp.

    Args:
        timestamp_ms: Unix epoch timestamp in milliseconds.
        epoch_size_ms: Width of each temporal shard in milliseconds.

    Returns:
        Non-negative epoch index.
    """
    if _RUST_AVAILABLE:
        return _rust.compute_epoch_id(timestamp_ms=timestamp_ms, epoch_size_ms=epoch_size_ms)
    return timestamp_ms // epoch_size_ms


def collection_name(ep_id: int) -> str:
    """Return the Qdrant collection name for an epoch.

    Args:
        ep_id: Epoch index from :func:`epoch_id`.

    Returns:
        Collection name string, e.g. ``"loci_42"``.
    """
    if _RUST_AVAILABLE:
        return _rust.epoch_collection_name(epoch_id=ep_id)
    return f"loci_{ep_id}"


def epochs_in_range(
    start_ms: int,
    end_ms: int,
    epoch_size_ms: int,
) -> list[int]:
    """Return all epoch IDs that overlap a time window.

    Args:
        start_ms: Start of the time window (inclusive).
        end_ms: End of the time window (inclusive).
        epoch_size_ms: Width of each temporal shard in milliseconds.

    Returns:
        Sorted list of epoch IDs.
    """
    if _RUST_AVAILABLE:
        return _rust.epochs_for_time_window(
            start_ms=start_ms, end_ms=end_ms, epoch_size_ms=epoch_size_ms
        )
    first = epoch_id(start_ms, epoch_size_ms)
    last = epoch_id(end_ms, epoch_size_ms)
    return list(range(first, last + 1))
