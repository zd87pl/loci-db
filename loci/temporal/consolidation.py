"""Memory consolidation — episodic-to-semantic aging of old epochs.

Per tenant/store there are exactly two collections: raw data lives in
``{prefix}loci_data`` and consolidated summaries live in
``{prefix}loci_summary``.  Epochs (``timestamp_ms // epoch_size_ms``) are a
purely logical concept: the unit of consolidation granularity and of
Hilbert t-normalisation.  When a raw epoch leaves the raw window it is
summarised into a small number of per-scene centroid "summary states" and
its raw points are deleted.  Recent memory stays raw and high-fidelity;
storage stays bounded while old data remains findable.

One *coarse group* spans ``summary_epoch_ratio`` raw epochs
(``coarse_id = raw_epoch_id // summary_epoch_ratio``).  All summaries share
the one summary collection; a coarse group's summaries are addressed by
timestamp range (:func:`coarse_time_range`).  When a raw epoch is folded
in, the existing summaries for its coarse group are selected by that
range, combined with the epoch's raw states, and re-consolidated, so each
coarse group holds at most ``max_states_per_scene`` states per scene at
all times — the flight-recorder property.

Everything here is a backend-agnostic pure function over
:class:`~loci.schema.WorldState` lists so the Qdrant clients can reuse it;
clients wire the reads, writes, and deletes around these functions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from loci.schema import WorldState
from loci.temporal.sharding import epoch_id

_DATA_SUFFIX = "loci_data"
_SUMMARY_SUFFIX = "loci_summary"
_KMEANS_ITERATIONS = 8
_ZERO_NORM_EPS = 1e-12


@dataclass
class ConsolidationPolicy:
    """Policy controlling episodic-to-semantic consolidation of old epochs.

    Attributes:
        raw_window_epochs: Epochs newer than ``now - raw_window_epochs``
            (in epoch units) stay raw.  Older epochs are summarised into
            coarse summary collections and their raw collections dropped.
        summary_epoch_ratio: One summary collection spans this many raw
            epochs (``coarse_id = raw_epoch_id // summary_epoch_ratio``).
        max_states_per_scene: Maximum number of centroid summary states kept
            per scene in each coarse summary collection.
    """

    raw_window_epochs: int
    summary_epoch_ratio: int = 100
    max_states_per_scene: int = 4

    def __post_init__(self) -> None:
        if self.raw_window_epochs < 1:
            raise ValueError(f"raw_window_epochs must be >= 1, got {self.raw_window_epochs}")
        if self.summary_epoch_ratio < 1:
            raise ValueError(f"summary_epoch_ratio must be >= 1, got {self.summary_epoch_ratio}")
        if self.max_states_per_scene < 1:
            raise ValueError(f"max_states_per_scene must be >= 1, got {self.max_states_per_scene}")


def data_collection_name(prefix: str = "") -> str:
    """Return the raw-data collection name for a tenant/store.

    Args:
        prefix: Optional collection prefix (multi-tenant namespacing).

    Returns:
        Collection name, e.g. ``"loci_data"`` or ``"t_loci_data"``.
    """
    return f"{prefix}{_DATA_SUFFIX}"


def summary_collection_name(prefix: str = "") -> str:
    """Return the consolidated-summary collection name for a tenant/store.

    Constant per tenant — summaries for every coarse group live in this one
    collection and are addressed by timestamp range
    (:func:`coarse_time_range`), not by collection name.

    Args:
        prefix: Optional collection prefix (multi-tenant namespacing).

    Returns:
        Collection name, e.g. ``"loci_summary"`` or ``"t_loci_summary"``.
    """
    return f"{prefix}{_SUMMARY_SUFFIX}"


def coarse_id(epoch: int, policy: ConsolidationPolicy) -> int:
    """Return the coarse group index covering raw epoch *epoch*."""
    return epoch // policy.summary_epoch_ratio


def coarse_time_range(
    coarse: int,
    policy: ConsolidationPolicy,
    epoch_size_ms: int,
) -> tuple[int, int]:
    """Return the inclusive ``(t_min_ms, t_max_ms)`` a coarse group spans.

    Summaries for coarse group *coarse* always carry ``timestamp_ms``
    inside this range (member timestamps of its raw epochs do, and summary
    timestamps are member means), so refold lookups and replacements select
    a coarse group's summaries with a timestamp Range over this window.
    """
    span_ms = policy.summary_epoch_ratio * epoch_size_ms
    t_min = coarse * span_ms
    return t_min, t_min + span_ms - 1


def fold_cutoff_ms(
    now_ms: int,
    epoch_size_ms: int,
    policy: ConsolidationPolicy,
) -> int:
    """Return the start of the raw window as an epoch-aligned timestamp.

    Raw points with ``timestamp_ms < fold_cutoff_ms`` have left the raw
    window and must be folded into summaries.  Epochs
    ``<= epoch_id(now_ms) - raw_window_epochs`` fold — parity with
    :func:`epochs_to_consolidate`.  Clamped to 0 while the raw window still
    reaches back to the beginning of time.
    """
    cutoff_epoch = epoch_id(now_ms, epoch_size_ms) - policy.raw_window_epochs + 1
    return max(0, cutoff_epoch * epoch_size_ms)


def epochs_to_consolidate(
    active_epochs: list[int],
    *,
    now_ms: int,
    epoch_size_ms: int,
    policy: ConsolidationPolicy,
) -> list[int]:
    """Return the raw epoch IDs that have left the raw window.

    Args:
        active_epochs: Raw epoch IDs currently stored (summary collections
            must not be included).
        now_ms: Current timestamp in milliseconds.
        epoch_size_ms: Width of each raw epoch in milliseconds.
        policy: Active consolidation policy.

    Returns:
        Sorted list of epoch IDs older than the raw window.
    """
    cutoff = epoch_id(now_ms, epoch_size_ms) - policy.raw_window_epochs
    return sorted(ep for ep in active_epochs if ep <= cutoff)


def consolidate_states(
    states: list[WorldState],
    policy: ConsolidationPolicy,
    *,
    seed: int,
) -> list[WorldState]:
    """Consolidate raw states into per-scene centroid summary states.

    States are grouped by ``scene_id``.  Groups with at most
    ``max_states_per_scene`` members pass through marked as summaries;
    larger groups are reduced with a small deterministic k-means
    (``k = max_states_per_scene``) over the embedding vectors.

    Each summary state carries:

    - ``vector``: L2-normalised centroid mean (plain mean when the norm is
      ~0, e.g. opposing vectors cancelling out);
    - ``x``/``y``/``z``: member means;
    - ``timestamp_ms``: integer mean of member timestamps;
    - ``scene_id``: preserved;
    - ``confidence``: mean member confidence;
    - ``metadata``: ``{"consolidated": True, "source_count": ...,
      "t_min_ms": ..., "t_max_ms": ...}``.

    Inputs that are themselves summaries compose: their ``source_count``
    accumulates and their ``t_min_ms``/``t_max_ms`` extend the range, so
    re-consolidating a coarse collection stays lossless in the bookkeeping.

    IDs are left blank — fresh IDs are assigned by the insert path.

    Args:
        states: Raw (and/or previously summarised) states to consolidate.
        policy: Active consolidation policy.
        seed: RNG seed; the same ``(states, policy, seed)`` triple always
            yields an identical result.

    Returns:
        List of summary :class:`WorldState`, at most
        ``max_states_per_scene`` per scene, ordered by scene then time.
    """
    if not states:
        return []

    rng = np.random.default_rng(seed)
    groups: dict[str, list[WorldState]] = {}
    for state in states:
        groups.setdefault(state.scene_id, []).append(state)

    summaries: list[WorldState] = []
    for scene_id in sorted(groups):
        members = groups[scene_id]
        if len(members) <= policy.max_states_per_scene:
            ordered = sorted(members, key=lambda s: s.timestamp_ms)
            summaries.extend(_passthrough_summary(m) for m in ordered)
            continue
        clusters = _kmeans_clusters([m.vector for m in members], policy.max_states_per_scene, rng)
        scene_summaries = [_cluster_summary([members[i] for i in idx]) for idx in clusters]
        scene_summaries.sort(key=lambda s: s.timestamp_ms)
        summaries.extend(scene_summaries)
    return summaries


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _kmeans_clusters(
    vectors: list[list[float]],
    k: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Deterministic mini k-means; returns member-index lists per cluster.

    Empty-cluster-safe: an empty cluster is reseeded with the point farthest
    from its assigned centroid, chosen only among clusters that can spare a
    member.  Requires ``len(vectors) > k`` (guaranteed by the caller).
    """
    x = np.asarray(vectors, dtype=np.float64)
    n = x.shape[0]
    init = np.sort(rng.choice(n, size=k, replace=False))
    centroids = x[init].copy()

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(_KMEANS_ITERATIONS):
        # (n, k) pairwise euclidean distances
        dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)

        for c in range(k):
            if np.any(assignments == c):
                continue
            # Reseed the empty cluster with the farthest point whose current
            # cluster has more than one member (never re-empties a cluster).
            counts = np.bincount(assignments, minlength=k)
            own_dist = dists[np.arange(n), assignments]
            eligible = counts[assignments] > 1
            own_dist = np.where(eligible, own_dist, -np.inf)
            far = int(np.argmax(own_dist))
            assignments[far] = c

        for c in range(k):
            mask = assignments == c
            if np.any(mask):
                centroids[c] = x[mask].mean(axis=0)

    return [np.flatnonzero(assignments == c).tolist() for c in range(k) if np.any(assignments == c)]


def _passthrough_summary(state: WorldState) -> WorldState:
    """Mark a single state as a summary of itself (small-group pass-through)."""
    return WorldState(
        x=state.x,
        y=state.y,
        z=state.z,
        timestamp_ms=state.timestamp_ms,
        vector=list(state.vector),
        scene_id=state.scene_id,
        scale_level=state.scale_level,
        confidence=state.confidence,
        metadata={
            "consolidated": True,
            "source_count": _source_count(state),
            "t_min_ms": _t_min(state),
            "t_max_ms": _t_max(state),
        },
    )


def _cluster_summary(members: list[WorldState]) -> WorldState:
    """Collapse one k-means cluster into a single centroid summary state."""
    vectors = np.asarray([m.vector for m in members], dtype=np.float64)
    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > _ZERO_NORM_EPS:
        centroid = centroid / norm

    timestamps = [m.timestamp_ms for m in members]
    return WorldState(
        x=_clamp01(float(np.mean([m.x for m in members]))),
        y=_clamp01(float(np.mean([m.y for m in members]))),
        z=_clamp01(float(np.mean([m.z for m in members]))),
        timestamp_ms=int(sum(timestamps) / len(timestamps)),
        vector=[float(v) for v in centroid],
        scene_id=members[0].scene_id,
        scale_level=_modal_scale_level(members),
        confidence=_clamp01(float(np.mean([m.confidence for m in members]))),
        metadata={
            "consolidated": True,
            "source_count": sum(_source_count(m) for m in members),
            "t_min_ms": min(_t_min(m) for m in members),
            "t_max_ms": max(_t_max(m) for m in members),
        },
    )


def _source_count(state: WorldState) -> int:
    """Number of raw states a member represents (1 unless already a summary)."""
    metadata = state.metadata or {}
    if metadata.get("consolidated"):
        count = metadata.get("source_count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            return count
    return 1


def _t_min(state: WorldState) -> int:
    metadata = state.metadata or {}
    if metadata.get("consolidated"):
        t_min = metadata.get("t_min_ms")
        if isinstance(t_min, int) and not isinstance(t_min, bool):
            return t_min
    return state.timestamp_ms


def _t_max(state: WorldState) -> int:
    metadata = state.metadata or {}
    if metadata.get("consolidated"):
        t_max = metadata.get("t_max_ms")
        if isinstance(t_max, int) and not isinstance(t_max, bool):
            return t_max
    return state.timestamp_ms


def _modal_scale_level(members: list[WorldState]) -> str:
    """Most common scale_level; ties break by first occurrence (deterministic)."""
    counts = Counter(m.scale_level for m in members)
    return counts.most_common(1)[0][0]


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
