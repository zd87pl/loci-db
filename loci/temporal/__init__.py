"""Temporal sharding, decay, retention, and consolidation utilities."""

from loci.temporal.consolidation import (
    ConsolidationPolicy,
    consolidate_states,
    epochs_to_consolidate,
    is_summary_collection,
    summary_coarse_range,
    summary_collection_name,
)
from loci.temporal.decay import apply_decay, decay_score
from loci.temporal.retention import RetentionManager, RetentionPolicy, epochs_to_drop
from loci.temporal.sharding import collection_name, epoch_id, epochs_in_range

__all__ = [
    "apply_decay",
    "collection_name",
    "consolidate_states",
    "ConsolidationPolicy",
    "decay_score",
    "epoch_id",
    "epochs_in_range",
    "epochs_to_consolidate",
    "epochs_to_drop",
    "is_summary_collection",
    "RetentionManager",
    "RetentionPolicy",
    "summary_coarse_range",
    "summary_collection_name",
]
