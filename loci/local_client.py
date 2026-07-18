"""LocalLociClient — full Loci with zero external dependencies.

Uses the in-memory backend instead of Qdrant.  Identical API surface
to :class:`LociClient`, so code that works with LocalLociClient
works with the real client by swapping the constructor.

Storage layout (bounded collection set): raw points live in one
``{prefix}loci_data`` collection and consolidated summaries in one
``{prefix}loci_summary`` collection.  Epochs are a purely logical
concept — the unit of consolidation granularity and of Hilbert
t-normalisation — never a collection.

Use cases:
- Tests without Docker
- Benchmarks that isolate Loci's indexing overhead
- Demos and prototyping
- CI environments
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loci.backends.memory import MemoryStore
from loci.payload_filters import extra_filter_to_memory
from loci.retrieval.predict import PredictRetrieveResult
from loci.schema import ScoredWorldState, WorldState
from loci.spatial.adaptive import AdaptiveResolution
from loci.spatial.filtering import exact_payload_match
from loci.spatial.hilbert import HilbertIndex
from loci.spatial.query_plan import bounds_for_epoch, choose_query_resolution
from loci.temporal.consolidation import (
    ConsolidationPolicy,
    coarse_id,
    coarse_time_range,
    consolidate_states,
    data_collection_name,
    fold_cutoff_ms,
    summary_collection_name,
)
from loci.temporal.decay import DEFAULT_DECAY_LAMBDA, apply_decay
from loci.temporal.retention import RetentionManager, RetentionPolicy
from loci.temporal.sharding import epoch_id

logger = logging.getLogger(__name__)

_EXACT_FILTER_OVERFETCH = 3


@dataclass
class QueryStats:
    """Statistics from a single query execution.

    Attributes:
        shards_searched: Number of collections searched (data and/or summary).
        total_candidates: Points that passed payload filters before ANN.
        hilbert_ids_in_filter: Size of the Hilbert MatchAny set (0 if no spatial filter).
        decay_applied: Whether temporal decay re-ranking was applied.
        elapsed_ms: Wall-clock time for the query in milliseconds.
    """

    shards_searched: int = 0
    total_candidates: int = 0
    hilbert_ids_in_filter: int = 0
    decay_applied: bool = False
    elapsed_ms: float = 0.0


class LocalLociClient:
    """Full Loci client backed by an in-memory store.

    API-compatible with :class:`LociClient`.  No Qdrant required.

    Args:
        epoch_size_ms: Width of each logical temporal epoch in milliseconds.
        spatial_resolution: Hilbert curve resolution order used as the
            default (coarsest) query resolution. Ignored when an explicit
            ``resolutions`` list is provided — ``resolutions`` wins.
        vector_size: Dimensionality of embedding vectors.
        decay_lambda: Temporal decay rate (per ms; defaults to a one-hour
            half-life).
        distance: Distance metric — ``"cosine"``, ``"dot"``, or ``"euclidean"``.
        retention_policy: Optional policy that purges expired raw points
            from the data collection (cutoff-based; never touches the
            summary collection).
        consolidation_policy: Optional policy that summarises raw epochs
            leaving the raw window into the ``loci_summary`` collection,
            then deletes their raw points (episodic-to-semantic memory
            aging).  Works standalone or combined with ``retention_policy``;
            when both are set, consolidation runs first on each maintenance
            pass and retention's purge applies to whatever raw points
            remain.  Summaries are never purged or re-consolidated by
            retention.  Note: a retention policy tighter than the raw
            window (fewer than ``raw_window_epochs + 1`` epochs retained)
            purges raw points before consolidation can fold them.
    """

    def __init__(
        self,
        epoch_size_ms: int = 5000,
        spatial_resolution: int = 4,
        vector_size: int = 512,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
        distance: str = "cosine",
        adaptive: bool = False,
        resolutions: list[int] | None = None,
        retention_policy: RetentionPolicy | None = None,
        consolidation_policy: ConsolidationPolicy | None = None,
        collection_prefix: str = "",
    ) -> None:
        if epoch_size_ms <= 0:
            raise ValueError(f"epoch_size_ms must be positive, got {epoch_size_ms}")
        if distance not in {"cosine", "dot", "euclidean"}:
            raise ValueError("distance must be one of ['cosine', 'dot', 'euclidean']")
        self._store = MemoryStore()
        self._epoch_size_ms = epoch_size_ms
        self._spatial_resolution = spatial_resolution
        self._vector_size = vector_size
        self._decay_lambda = decay_lambda
        self._distance = distance
        self._collection_prefix = collection_prefix
        self._data_collection = data_collection_name(collection_prefix)
        self._summary_collection = summary_collection_name(collection_prefix)
        self._data_ready = False
        self._summary_ready = False
        self._last_query_stats: QueryStats | None = None
        # Frontier of time this store has seen; maintenance clocks off
        # max(wall clock, frontier) so future-dated streams still age out.
        self._newest_timestamp_ms = 0
        # Explicit resolutions win; otherwise spatial_resolution becomes the
        # coarsest (default) query resolution alongside the finer defaults.
        self._hilbert = HilbertIndex(
            resolutions=resolutions
            if resolutions is not None
            else sorted({spatial_resolution, 8, 12})
        )
        self._adaptive = (
            AdaptiveResolution(
                base_order=self._hilbert.resolutions[0],
                max_order=max(self._hilbert.resolutions),
                density_threshold=50,
            )
            if adaptive
            else None
        )
        self._retention_policy = retention_policy
        self._retention = (
            RetentionManager(policy=retention_policy, epoch_size_ms=self._epoch_size_ms)
            if retention_policy is not None
            else None
        )
        self._consolidation_policy = consolidation_policy

    @property
    def density_stats(self):
        """Return adaptive resolution density stats, or None if not enabled."""
        return self._adaptive.stats() if self._adaptive is not None else None

    @property
    def last_query_stats(self) -> QueryStats | None:
        """Statistics from the most recent query() call."""
        return self._last_query_stats

    @property
    def store(self) -> MemoryStore:
        """Direct access to the underlying memory store (for introspection)."""
        return self._store

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_data_collection(self) -> None:
        if self._data_ready:
            return
        self._create_collection(self._data_collection, hilbert_indices=True)
        self._data_ready = True

    def _ensure_summary_collection(self) -> None:
        if self._summary_ready:
            return
        # Summaries carry no Hilbert payload; spatial queries reach them
        # via the exact post-filter on x/y/z.
        self._create_collection(self._summary_collection, hilbert_indices=False)
        self._summary_ready = True

    def _create_collection(self, name: str, *, hilbert_indices: bool) -> None:
        self._store.create_collection(name, self._vector_size, self._distance)
        if hilbert_indices:
            for r in self._hilbert.resolutions:
                self._store.create_payload_index(name, f"hilbert_r{r}")
        self._store.create_payload_index(name, "timestamp_ms")
        self._store.create_payload_index(name, "scale_level")
        self._store.create_payload_index(name, "scene_id")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, state: WorldState) -> str:
        """Insert a single WorldState. Input is not mutated."""
        self._validate_vector(state.vector)
        point_id = str(uuid.uuid4())

        ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
        self._ensure_data_collection()

        t_norm = self._normalise_time(state.timestamp_ms, ep)
        hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)

        if self._adaptive is not None:
            self._adaptive.record(state.x, state.y, state.z, t_norm)

        payload = _state_to_payload(state, hilbert_ids)

        # Causal linking
        predecessor = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
        if predecessor is not None:
            payload["prev_state_id"] = predecessor
            self._store.set_payload(self._data_collection, predecessor, {"next_state_id": point_id})

        self._store.upsert(
            self._data_collection,
            [{"id": point_id, "vector": state.vector, "payload": payload}],
        )
        self._newest_timestamp_ms = max(self._newest_timestamp_ms, state.timestamp_ms)
        self._maybe_purge()
        return point_id

    def insert_batch(self, states: list[WorldState]) -> list[str]:
        """Insert a batch with intra-batch causal linking. Input is not mutated."""
        for state in states:
            self._validate_vector(state.vector)
        self._ensure_data_collection()

        id_by_index: dict[int, str] = {}
        scene_chains: dict[str, str] = {}
        points: list[dict] = []

        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))

        for orig_idx, state in indexed:
            point_id = str(uuid.uuid4())
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            t_norm = self._normalise_time(state.timestamp_ms, ep)
            hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)
            if self._adaptive is not None:
                self._adaptive.record(state.x, state.y, state.z, t_norm)

            payload = _state_to_payload(state, hilbert_ids)

            # Causal link within the batch; the first state per scene links to
            # the latest predecessor already in the store (matching the
            # sequential-insert behaviour).
            if state.scene_id:
                if state.scene_id in scene_chains:
                    prev_id: str | None = scene_chains[state.scene_id]
                else:
                    prev_id = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
                if prev_id is not None:
                    payload["prev_state_id"] = prev_id
                scene_chains[state.scene_id] = point_id

            points.append({"id": point_id, "vector": state.vector, "payload": payload})

        self._store.upsert(self._data_collection, points)

        # Patch next links
        for point in points:
            prev_id = point["payload"].get("prev_state_id")
            if prev_id:
                self._store.set_payload(
                    self._data_collection, prev_id, {"next_state_id": point["id"]}
                )

        if states:
            self._newest_timestamp_ms = max(
                self._newest_timestamp_ms, max(s.timestamp_ms for s in states)
            )
        self._maybe_purge()
        return [id_by_index[i] for i in range(len(states))]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        *,
        _extra_payload_filter: dict | None = None,
        _epoch_ids: set[int] | None = None,
        overlap_factor: float = 1.2,
        min_confidence: float | None = None,
    ) -> list[WorldState]:
        """Search with Hilbert pre-filtering, timestamp filtering, and decay.

        After each call, inspect :attr:`last_query_stats` for diagnostics.
        """
        return [
            candidate.state
            for candidate in self.query_scored(
                vector,
                spatial_bounds,
                time_window_ms,
                limit,
                _extra_payload_filter=_extra_payload_filter,
                _epoch_ids=_epoch_ids,
                overlap_factor=overlap_factor,
                min_confidence=min_confidence,
            )
        ]

    def query_scored(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
        *,
        _extra_payload_filter: dict | None = None,
        _epoch_ids: set[int] | None = None,
        overlap_factor: float = 1.2,
        min_confidence: float | None = None,
    ) -> list[ScoredWorldState]:
        """Search and return scored results for downstream reranking.

        Runs ONE search over the ``loci_data`` collection with
        must-conditions [timestamp Range (when a window or epoch
        restriction is given), Hilbert MatchAny (when spatial bounds are
        given)], PLUS one search over ``loci_summary`` (vector + timestamp
        Range only — summaries carry no Hilbert payload), then merges both
        result sets before the exact post-filter and decay re-rank.
        Summaries are always searched with the timestamp filter; when no
        summaries exist the extra search is a cheap in-process no-op.

        Hilbert bucket selection: when the time window maps to a SINGLE
        epoch, the cover uses that epoch's normalised t-bounds (full
        4D selectivity).  When the window spans multiple epochs, or there
        is no window, the cover is computed with the full t range [0, 1]
        — spatial-only selectivity.  This is deliberate: Hilbert
        t-encoding is epoch-relative, so a multi-epoch t-cover is
        meaningless, and the timestamp Range condition carries the
        t-dimension instead.

        Scores follow the higher-is-better convention for every distance
        metric (the memory backend negates euclidean distances), so decay
        re-ranking, merging, and truncation behave identically across
        metrics and backends.
        """
        t_start = time.perf_counter()
        stats = QueryStats()

        if _epoch_ids is not None and not _epoch_ids:
            # Explicit empty epoch restriction (funnel dead-end): no results.
            stats.elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._last_query_stats = stats
            return []

        # Timestamp Range: intersection of the query window and the epoch
        # restriction's envelope (funnel narrowing).
        ts_lo, ts_hi = self._timestamp_range(time_window_ms, _epoch_ids)

        # Over-fetch whenever post-search filtering (spatial exact match,
        # min_confidence, epoch membership) or decay re-ranking can
        # reorder/drop hits, else truncation could evict the top-k.
        needs_overfetch = (
            spatial_bounds is not None
            or min_confidence is not None
            or _epoch_ids is not None
            or self._decay_lambda > 0
        )
        fetch_limit = limit * _EXACT_FILTER_OVERFETCH if needs_overfetch else limit

        base_filter: dict = {}
        if ts_lo is not None:
            base_filter["timestamp_ms"] = {"gte": ts_lo, "lte": ts_hi}
        if min_confidence is not None:
            base_filter["confidence"] = {"gte": min_confidence}
        base_filter.update(extra_filter_to_memory(_extra_payload_filter))

        all_results: list[dict] = []

        # --- Raw data search (Hilbert pre-filter + timestamp Range) ---
        data_filter = dict(base_filter)
        skip_data = False
        if spatial_bounds is not None:
            hids, field = self._hilbert_cover(spatial_bounds, time_window_ms, overlap_factor)
            if hids:
                data_filter[field] = {"any": hids}
                stats.hilbert_ids_in_filter = len(hids)
            else:
                skip_data = True  # no bucket overlaps the requested region
        if not skip_data and self._store.collection_exists(self._data_collection):
            stats.shards_searched += 1
            hits = self._store.search(
                collection=self._data_collection,
                query_vector=vector,
                limit=fetch_limit,
                payload_filter=data_filter if data_filter else None,
            )
            stats.total_candidates += len(hits)
            all_results.extend(_hit_to_result(hit, summary=False) for hit in hits)

        # --- Summary search (no Hilbert condition) ---
        if self._store.collection_exists(self._summary_collection):
            stats.shards_searched += 1
            hits = self._store.search(
                collection=self._summary_collection,
                query_vector=vector,
                limit=fetch_limit,
                payload_filter=base_filter if base_filter else None,
            )
            stats.total_candidates += len(hits)
            all_results.extend(_hit_to_result(hit, summary=True) for hit in hits)

        if spatial_bounds is not None or time_window_ms is not None or min_confidence is not None:
            all_results = [
                r
                for r in all_results
                if exact_payload_match(
                    r["payload"],
                    spatial_bounds=spatial_bounds,
                    time_window_ms=time_window_ms,
                    min_confidence=min_confidence,
                )
            ]
        if _epoch_ids is not None:
            all_results = [r for r in all_results if self._epoch_restriction_ok(r, _epoch_ids)]

        # Decay and re-rank
        now_ms = int(time.time() * 1000)
        apply_decay(all_results, now_ms, self._decay_lambda)
        stats.decay_applied = self._decay_lambda > 0
        all_results = all_results[:limit]

        stats.elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._last_query_stats = stats

        return [
            ScoredWorldState(
                state=_payload_to_state(r["payload"], r["id"], r["vector"]),
                score=float(r["score"]),
                decayed_score=float(r.get("decayed_score", r["score"])),
            )
            for r in all_results
        ]

    def predict_and_retrieve(
        self,
        context_vector: list[float],
        predictor_fn: Callable[[list[float]], list[float]],
        future_horizon_ms: int = 1000,
        limit: int = 5,
        current_position: tuple[float, float, float] | None = None,
        current_timestamp_ms: int | None = None,
        spatial_search_radius: float = 0.3,
        alpha: float = 0.7,
        return_prediction: bool = False,
        *,
        calibrator: Any = None,
        search_time_window_ms: tuple[int, int] | None = None,
    ) -> list[WorldState] | PredictRetrieveResult:
        """Predict-then-retrieve using the local backend.

        When ``current_position`` is provided, returns a full
        :class:`PredictRetrieveResult` with novelty scoring.  By default,
        retrieval searches stored history for analogs; pass
        ``search_time_window_ms`` to restrict it to an absolute timestamp range.
        """
        if current_position is not None or return_prediction:
            from loci.retrieval.predict import PredictThenRetrieve

            ptr = PredictThenRetrieve(self, calibrator=calibrator)
            return ptr.retrieve(
                context_vector=context_vector,
                predictor_fn=predictor_fn,
                future_horizon_ms=future_horizon_ms,
                current_position=current_position,
                current_timestamp_ms=current_timestamp_ms,
                spatial_search_radius=spatial_search_radius,
                limit=limit,
                alpha=alpha,
                return_prediction=return_prediction,
                search_time_window_ms=search_time_window_ms,
            )
        predicted = predictor_fn(context_vector)
        return self.query(
            vector=predicted,
            time_window_ms=search_time_window_ms,
            limit=limit,
        )

    def funnel_query(
        self,
        vector: list[float],
        spatial_bounds: dict | None = None,
        time_window_ms: tuple[int, int] | None = None,
        limit: int = 10,
    ) -> list[WorldState]:
        """Multi-scale coarse-to-fine search across scale levels.

        Cascades from sequence → frame → patch, returning results at the
        finest scale that produced hits.

        Args:
            vector: Query embedding vector.
            spatial_bounds: Optional spatial bounding box.
            time_window_ms: Optional ``(start_ms, end_ms)`` time window.
            limit: Maximum number of results.

        Returns:
            List of :class:`WorldState` at the finest available scale.
        """
        from loci.retrieval.funnel import funnel_search

        return funnel_search(self, vector, spatial_bounds, time_window_ms, limit)

    def get_trajectory(
        self,
        state_id: str,
        steps_back: int = 10,
        steps_forward: int = 10,
    ) -> list[WorldState]:
        """Reconstruct a trajectory using scroll with scene_id filter.

        Scans the raw data collection only — summaries are naturally
        excluded by collection.
        """
        anchor = self._get_state_by_id(state_id)
        if anchor is None:
            return []
        if not anchor.scene_id:
            return [anchor]

        hits = self._scroll_all(
            collection=self._data_collection,
            payload_filter={"scene_id": anchor.scene_id},
            order_by="timestamp_ms",
        )
        all_states = [_payload_to_state(hit["payload"], hit["id"], hit["vector"]) for hit in hits]

        anchor_idx = None
        target = _normalize_id(state_id)
        for i, s in enumerate(all_states):
            if _normalize_id(s.id) == target:
                anchor_idx = i
                break

        if anchor_idx is None:
            return [anchor]

        start = max(0, anchor_idx - steps_back)
        end = min(len(all_states), anchor_idx + steps_forward + 1)
        return all_states[start:end]

    def get_causal_context(
        self,
        state_id: str,
        window_ms: int = 5000,
    ) -> list[WorldState]:
        """Return all states within ±window_ms of the given state in the same scene.

        Scans the raw data collection only — summaries are naturally
        excluded by collection.
        """
        anchor = self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms

        hits = self._scroll_all(
            collection=self._data_collection,
            payload_filter={
                "scene_id": anchor.scene_id,
                "timestamp_ms": {"gte": t_min, "lte": t_max},
            },
            order_by="timestamp_ms",
        )
        return [_payload_to_state(hit["payload"], hit["id"], hit["vector"]) for hit in hits]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _timestamp_range(
        self,
        time_window_ms: tuple[int, int] | None,
        epoch_ids: set[int] | None,
    ) -> tuple[int | None, int | None]:
        """Inclusive timestamp Range for the search must-conditions.

        Intersects the explicit query window with the envelope of an
        internal epoch restriction (funnel narrowing).  The epoch envelope
        is widened to whole coarse groups so summaries representing a
        requested epoch stay reachable; exact membership is enforced by
        :meth:`_epoch_restriction_ok` afterwards.
        """
        lo: int | None = None
        hi: int | None = None
        if time_window_ms is not None:
            lo, hi = time_window_ms
        if epoch_ids:
            e_lo = min(epoch_ids) * self._epoch_size_ms
            e_hi = (max(epoch_ids) + 1) * self._epoch_size_ms - 1
            policy = self._consolidation_policy
            if policy is not None:
                first = coarse_id(min(epoch_ids), policy)
                last = coarse_id(max(epoch_ids), policy)
                e_lo = min(e_lo, coarse_time_range(first, policy, self._epoch_size_ms)[0])
                e_hi = max(e_hi, coarse_time_range(last, policy, self._epoch_size_ms)[1])
            lo = e_lo if lo is None else max(lo, e_lo)
            hi = e_hi if hi is None else min(hi, e_hi)
        return lo, hi

    def _hilbert_cover(
        self,
        spatial_bounds: dict,
        time_window_ms: tuple[int, int] | None,
        overlap_factor: float,
    ) -> tuple[list[int], str]:
        """Hilbert bucket cover and payload field for a spatial query.

        Single-epoch windows use that epoch's normalised t-bounds; anything
        else covers the full t range [0, 1] (see :meth:`query_scored`).
        """
        single_epoch = time_window_ms is not None and epoch_id(
            time_window_ms[0], self._epoch_size_ms
        ) == epoch_id(time_window_ms[1], self._epoch_size_ms)
        if single_epoch:
            assert time_window_ms is not None
            ep = epoch_id(time_window_ms[0], self._epoch_size_ms)
            window: tuple[int, int] | None = time_window_ms
        else:
            ep = 0
            window = None  # bounds_for_epoch then yields t in [0, 1]
        resolution = choose_query_resolution(
            self._hilbert,
            self._adaptive,
            spatial_bounds,
            window,
            ep,
            self._epoch_size_ms,
            overlap_factor,
        )
        hids = self._hilbert.query_buckets(
            bounds_for_epoch(spatial_bounds, window, ep, self._epoch_size_ms),
            resolution=resolution,
            overlap_factor=overlap_factor,
        )
        return hids, self._hilbert.payload_field(resolution)

    def _epoch_restriction_ok(self, result: dict, epoch_ids: set[int]) -> bool:
        """Exact epoch-membership post-filter for an internal restriction.

        Raw points must lie in a requested epoch.  Summaries represent a
        whole coarse group, so they pass when their coarse group covers any
        requested epoch (matching the old per-collection pruning).
        """
        ts = result["payload"].get("timestamp_ms", 0)
        ep = epoch_id(ts, self._epoch_size_ms)
        if not result["summary"]:
            return ep in epoch_ids
        policy = self._consolidation_policy
        if policy is None:
            return False
        return coarse_id(ep, policy) in {coarse_id(e, policy) for e in epoch_ids}

    # ------------------------------------------------------------------
    # Maintenance (consolidation first, then retention)
    # ------------------------------------------------------------------

    def _maybe_purge(self) -> None:
        """Run maintenance: consolidation first, then retention purge."""
        now_ms = self._maintenance_now_ms()
        self._maybe_consolidate(now_ms)
        if self._retention is None:
            return
        try:
            self._retention.maybe_purge(now_ms, self._delete_raw_before)
        except Exception as exc:
            logger.warning("Retention purge failed: %s", exc)

    def _maintenance_now_ms(self) -> int:
        """Maintenance clock: max(wall clock, newest inserted timestamp).

        Future-dated streams (simulations, replays) advance the clock so
        their old points still leave the raw/retention windows.
        """
        return max(int(time.time() * 1000), self._newest_timestamp_ms)

    def _delete_raw_before(self, cutoff_ms: int) -> int:
        """Retention deleter: purge raw points below the cutoff.

        Never touches the summary collection.
        """
        return self._store.delete_points_in_time_range(self._data_collection, 0, cutoff_ms)

    def _maybe_consolidate(self, now_ms: int) -> None:
        """Fold raw epochs that left the raw window into the summary collection."""
        policy = self._consolidation_policy
        if policy is None:
            return
        cutoff_ms = fold_cutoff_ms(now_ms, self._epoch_size_ms, policy)
        if cutoff_ms <= 0:
            return
        stale_filter = {"timestamp_ms": {"lt": cutoff_ms}}
        # Cheap staleness probe (limit-1) before the full scan.
        if not self._store.scroll(self._data_collection, stale_filter, limit=1):
            return
        stale = self._scroll_all(collection=self._data_collection, payload_filter=stale_filter)
        by_epoch: dict[int, list[dict]] = {}
        for hit in stale:
            ep = epoch_id(hit["payload"]["timestamp_ms"], self._epoch_size_ms)
            by_epoch.setdefault(ep, []).append(hit)
        for ep in sorted(by_epoch):
            try:
                self._consolidate_epoch(ep, by_epoch[ep], policy)
            except Exception:
                logger.warning("Consolidation failed for epoch %d", ep, exc_info=True)

    def _consolidate_epoch(
        self, ep: int, raw_hits: list[dict], policy: ConsolidationPolicy
    ) -> None:
        """Fold one raw epoch into its coarse summary group.

        The coarse group's existing summaries (selected by timestamp Range
        over its coarse time range) are combined with the epoch's raw
        states and re-consolidated, keeping the group at
        ``max_states_per_scene`` states per scene; the old summaries are
        replaced and the epoch's raw points deleted.
        """
        coarse = coarse_id(ep, policy)
        t_min, t_max = coarse_time_range(coarse, policy, self._epoch_size_ms)
        existing = self._scroll_all(
            collection=self._summary_collection,
            payload_filter={"timestamp_ms": {"gte": t_min, "lte": t_max}},
        )
        combined = [
            _payload_to_state(hit["payload"], hit["id"], hit["vector"])
            for hit in [*existing, *raw_hits]
        ]
        summaries = consolidate_states(combined, policy, seed=ep)
        self._store.delete_points_in_time_range(self._summary_collection, t_min, t_max + 1)
        if summaries:
            self._ensure_summary_collection()
            points = [
                {
                    "id": str(uuid.uuid4()),
                    "vector": s.vector,
                    "payload": _state_to_payload(s, {}),
                }
                for s in summaries
            ]
            self._store.upsert(self._summary_collection, points)
        self._store.delete_points_in_time_range(
            self._data_collection, ep * self._epoch_size_ms, (ep + 1) * self._epoch_size_ms
        )
        logger.info("Consolidated epoch %d into %s", ep, self._summary_collection)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._vector_size:
            raise ValueError(f"vector has dimension {len(vector)}, expected {self._vector_size}")

    def _normalise_time(self, timestamp_ms: int, ep: int) -> float:
        epoch_start = ep * self._epoch_size_ms
        offset = timestamp_ms - epoch_start
        return min(1.0, max(0.0, offset / self._epoch_size_ms))

    def _get_state_by_id(self, state_id: str) -> WorldState | None:
        for col in (self._data_collection, self._summary_collection):
            results = self._store.retrieve(col, [state_id])
            if results:
                r = results[0]
                return _payload_to_state(r["payload"], r["id"], r["vector"])
        return None

    def _find_latest_predecessor(self, scene_id: str, before_ms: int) -> str | None:
        """Latest raw point in *scene_id* strictly before *before_ms*.

        A single filtered scroll over the data collection — summaries are
        naturally excluded by collection and never become predecessors.
        """
        if not scene_id:
            return None
        results = self._scroll_all(
            collection=self._data_collection,
            payload_filter={
                "scene_id": scene_id,
                "timestamp_ms": {"lt": before_ms},
            },
            order_by="timestamp_ms",
        )
        if results:
            # Scroll returns ascending order; last item is the latest predecessor
            return str(results[-1]["id"])
        return None

    def _scroll_all(
        self,
        *,
        collection: str,
        payload_filter: dict | None = None,
        order_by: str | None = None,
    ) -> list[dict]:
        """Return the full ordered scroll result for a collection."""
        limit = self._store.collection_count(collection)
        if limit <= 0:
            return []
        return self._store.scroll(
            collection=collection,
            payload_filter=payload_filter,
            limit=limit,
            order_by=order_by,
        )


# ------------------------------------------------------------------
# Shared payload helpers
# ------------------------------------------------------------------


def _normalize_id(point_id: object) -> str:
    """Normalise a point ID for comparison (lowercase, hyphens stripped)."""
    return str(point_id).lower().replace("-", "")


def _hit_to_result(hit: dict, *, summary: bool) -> dict:
    return {
        "score": hit["score"],
        "timestamp_ms": hit["payload"].get("timestamp_ms", 0),
        "payload": hit["payload"],
        "vector": hit["vector"],
        "id": hit["id"],
        "summary": summary,
    }


def _state_to_payload(state: WorldState, hilbert_ids: dict[str, int]) -> dict:
    payload = {
        "x": state.x,
        "y": state.y,
        "z": state.z,
        "timestamp_ms": state.timestamp_ms,
        "scene_id": state.scene_id,
        "scale_level": state.scale_level,
        "confidence": state.confidence,
        "prev_state_id": state.prev_state_id,
        "next_state_id": state.next_state_id,
        "metadata": state.metadata,
    }
    payload.update(hilbert_ids)
    return payload


def _payload_to_state(
    payload: dict, point_id: str, vector: list[float] | None = None
) -> WorldState:
    return WorldState(
        x=payload["x"],
        y=payload["y"],
        z=payload["z"],
        timestamp_ms=payload["timestamp_ms"],
        vector=vector if vector is not None else [],
        scene_id=payload.get("scene_id", ""),
        scale_level=payload.get("scale_level", "patch"),
        confidence=payload.get("confidence", 1.0),
        prev_state_id=payload.get("prev_state_id"),
        next_state_id=payload.get("next_state_id"),
        metadata=payload.get("metadata") or {},
        id=str(point_id),
    )
