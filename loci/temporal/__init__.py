"""Temporal epoch logic, decay, retention, and consolidation utilities."""

from loci.temporal.consolidation import (
    ConsolidationPolicy,
    coarse_id,
    coarse_time_range,
    consolidate_states,
    data_collection_name,
    epochs_to_consolidate,
    fold_cutoff_ms,
    summary_collection_name,
)
from loci.temporal.decay import apply_decay, decay_score
from loci.temporal.retention import RetentionManager, RetentionPolicy, retention_cutoff_ms
from loci.temporal.sharding import epoch_id, epochs_in_range

__all__ = [
    "apply_decay",
    "coarse_id",
    "coarse_time_range",
    "consolidate_states",
    "ConsolidationPolicy",
    "data_collection_name",
    "decay_score",
    "epoch_id",
    "epochs_in_range",
    "epochs_to_consolidate",
    "fold_cutoff_ms",
    "retention_cutoff_ms",
    "RetentionManager",
    "RetentionPolicy",
    "summary_collection_name",
]
