"""Epoch computation for logical temporal partitioning.

An epoch is a fixed-width time window (default 5 000 ms).  Epochs are a
purely logical concept: the unit of consolidation granularity and of
Hilbert t-normalisation.  Storage uses a bounded collection set (see
:mod:`loci.temporal.consolidation`), not per-epoch collections.

When the ``loci_core`` Rust extension is available, all functions
delegate to the native implementation.
"""

from __future__ import annotations

try:
    import loci_core as _rust

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# Guard against accidentally materialising an astronomically large epoch list
# (e.g. an all-time window with a 5-second epoch size is ~350M epochs).
_MAX_EPOCHS_IN_RANGE = 10_000_000


def epoch_id(timestamp_ms: int, epoch_size_ms: int) -> int:
    """Return the epoch index for a given timestamp.

    Negative timestamps are clamped to 0 so both the Rust and Python
    backends honour the documented non-negative contract.

    Args:
        timestamp_ms: Unix epoch timestamp in milliseconds.
        epoch_size_ms: Width of each temporal shard in milliseconds.

    Returns:
        Non-negative epoch index.
    """
    timestamp_ms = max(0, timestamp_ms)
    if _RUST_AVAILABLE:
        return int(_rust.compute_epoch_id(timestamp_ms=timestamp_ms, epoch_size_ms=epoch_size_ms))
    return timestamp_ms // epoch_size_ms


def collection_name(ep_id: int) -> str:
    """LEGACY: return the old-layout per-epoch collection name.

    Used only by the migration tool for the old one-collection-per-epoch
    layout (and by the not-yet-migrated Qdrant clients).  New code stores
    raw points in the single data collection
    (:func:`loci.temporal.consolidation.data_collection_name`).

    Args:
        ep_id: Epoch index from :func:`epoch_id`.

    Returns:
        Collection name string, e.g. ``"loci_42"``.
    """
    if _RUST_AVAILABLE:
        return str(_rust.epoch_collection_name(epoch_id=ep_id))
    return f"loci_{ep_id}"


def epochs_in_range(
    start_ms: int,
    end_ms: int,
    epoch_size_ms: int,
) -> list[int]:
    """Return all epoch IDs that overlap a time window.

    Negative bounds are clamped to 0 (matching :func:`epoch_id`). The
    range is capped at ``_MAX_EPOCHS_IN_RANGE`` epochs: materialising a
    wider window (e.g. an all-time query with a small ``epoch_size_ms``)
    would exhaust memory. Callers with unbounded windows should intersect
    ``epoch_id(start)..epoch_id(end)`` with their known epochs instead.

    Args:
        start_ms: Start of the time window (inclusive).
        end_ms: End of the time window (inclusive).
        epoch_size_ms: Width of each temporal shard in milliseconds.

    Returns:
        Sorted list of epoch IDs.

    Raises:
        ValueError: If the window spans more than ``_MAX_EPOCHS_IN_RANGE``
            epochs. Increase ``epoch_size_ms`` or narrow the window.
    """
    start_ms = max(0, start_ms)
    end_ms = max(0, end_ms)
    first = epoch_id(start_ms, epoch_size_ms)
    last = epoch_id(end_ms, epoch_size_ms)
    if last - first + 1 > _MAX_EPOCHS_IN_RANGE:
        raise ValueError(
            f"time window spans {last - first + 1} epochs, exceeding the "
            f"{_MAX_EPOCHS_IN_RANGE} cap; increase epoch_size_ms or narrow the window"
        )
    if _RUST_AVAILABLE:
        return [
            int(x)
            for x in _rust.epochs_for_time_window(
                start_ms=start_ms, end_ms=end_ms, epoch_size_ms=epoch_size_ms
            )
        ]
    return list(range(first, last + 1))
