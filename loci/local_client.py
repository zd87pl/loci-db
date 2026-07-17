"""LocalLociClient — full Loci with zero external dependencies.

Uses the in-memory backend instead of Qdrant.  Identical API surface
to :class:`LociClient`, so code that works with LocalLociClient
works with the real client by swapping the constructor.

Use cases:
- Tests without Docker
- Benchmarks that isolate Loci's indexing overhead
- Demos and prototyping
- CI environments
"""

from __future__ import annotations

import contextlib
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
    consolidate_states,
    epochs_to_consolidate,
    is_summary_collection,
    summary_coarse_range,
    summary_collection_name,
)
from loci.temporal.decay import DEFAULT_DECAY_LAMBDA, apply_decay
from loci.temporal.retention import RetentionManager, RetentionPolicy
from loci.temporal.sharding import collection_name, epoch_id

logger = logging.getLogger(__name__)

_EXACT_FILTER_OVERFETCH = 3


@dataclass
class QueryStats:
    """Statistics from a single query execution.

    Attributes:
        shards_searched: Number of temporal shards (collections) searched.
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
        epoch_size_ms: Width of each temporal shard in milliseconds.
        spatial_resolution: Hilbert curve resolution order used as the
            default (coarsest) query resolution. Ignored when an explicit
            ``resolutions`` list is provided — ``resolutions`` wins.
        vector_size: Dimensionality of embedding vectors.
        decay_lambda: Temporal decay rate (per ms; defaults to a one-hour
            half-life).
        distance: Distance metric — ``"cosine"``, ``"dot"``, or ``"euclidean"``.
        retention_policy: Optional policy that purges expired raw epochs.
        consolidation_policy: Optional policy that summarises raw epochs
            leaving the raw window into coarse ``loci_sum_*`` collections,
            then drops their raw collections (episodic-to-semantic memory
            aging).  Works standalone or combined with ``retention_policy``;
            when both are set, consolidation runs first on each maintenance
            pass and retention's purge applies to whatever raw epochs remain.
            Summary collections are never purged or re-consolidated by
            retention.  Note: a retention policy tighter than the raw window
            (fewer than ``raw_window_epochs + 1`` epochs retained) drops raw
            epochs before consolidation can fold them.
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
        self._known_collections: set[str] = set()
        self._last_query_stats: QueryStats | None = None
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
            RetentionManager(
                policy=retention_policy,
                epoch_size_ms=self._epoch_size_ms,
                collection_prefix=self._collection_prefix,
            )
            if retention_policy is not None
            else None
        )
        self._consolidation_policy = consolidation_policy
        # Epochs already folded into a summary collection but whose raw
        # collection drop failed (guards against double-counting the same
        # epoch's states on the retry of the drop).
        self._consolidated_epochs: set[int] = set()

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

    def _col_name(self, ep: int) -> str:
        base = collection_name(ep)
        return f"{self._collection_prefix}{base}" if self._collection_prefix else base

    def _ensure_collection(self, name: str) -> None:
        if name in self._known_collections:
            return
        self._store.create_collection(name, self._vector_size, self._distance)
        for r in self._hilbert.resolutions:
            self._store.create_payload_index(name, f"hilbert_r{r}")
        self._store.create_payload_index(name, "timestamp_ms")
        self._store.create_payload_index(name, "scale_level")
        self._store.create_payload_index(name, "scene_id")
        self._known_collections.add(name)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(self, state: WorldState) -> str:
        """Insert a single WorldState. Input is not mutated."""
        self._validate_vector(state.vector)
        point_id = str(uuid.uuid4())

        ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
        col = self._col_name(ep)
        self._ensure_collection(col)

        t_norm = self._normalise_time(state.timestamp_ms, ep)
        hilbert_ids = self._hilbert.encode(state.x, state.y, state.z, t_norm)

        if self._adaptive is not None:
            self._adaptive.record(state.x, state.y, state.z, t_norm)

        payload = _state_to_payload(state, hilbert_ids)

        # Causal linking
        predecessor = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
        if predecessor is not None:
            prev_id, prev_col = predecessor
            payload["prev_state_id"] = prev_id
            self._store.set_payload(prev_col, prev_id, {"next_state_id": point_id})

        self._store.upsert(col, [{"id": point_id, "vector": state.vector, "payload": payload}])
        self._maybe_purge()
        return point_id

    def insert_batch(self, states: list[WorldState]) -> list[str]:
        """Insert a batch with intra-batch causal linking. Input is not mutated."""
        for state in states:
            self._validate_vector(state.vector)

        id_by_index: dict[int, str] = {}
        scene_chains: dict[str, tuple[str, str]] = {}
        prev_collection_by_point: dict[str, str] = {}
        groups: dict[str, list[dict]] = {}

        indexed = sorted(enumerate(states), key=lambda it: (it[1].scene_id, it[1].timestamp_ms))

        for orig_idx, state in indexed:
            point_id = str(uuid.uuid4())
            id_by_index[orig_idx] = point_id

            ep = epoch_id(state.timestamp_ms, self._epoch_size_ms)
            col = self._col_name(ep)
            self._ensure_collection(col)

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
                    prev_link: tuple[str, str] | None = scene_chains[state.scene_id]
                else:
                    prev_link = self._find_latest_predecessor(state.scene_id, state.timestamp_ms)
                if prev_link is not None:
                    prev_id, prev_col = prev_link
                    payload["prev_state_id"] = prev_id
                    prev_collection_by_point[point_id] = prev_col
                scene_chains[state.scene_id] = (point_id, col)

            groups.setdefault(col, []).append(
                {"id": point_id, "vector": state.vector, "payload": payload}
            )

        for col, points in groups.items():
            self._store.upsert(col, points)

        # Patch next links
        for col, points in groups.items():
            for point in points:
                prev_id = point["payload"].get("prev_state_id")
                if prev_id:
                    self._store.set_payload(
                        prev_collection_by_point.get(point["id"], col),
                        prev_id,
                        {"next_state_id": point["id"]},
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
        """Search with Hilbert pre-filtering, temporal sharding, and decay.

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

        Scores follow the higher-is-better convention for every distance
        metric (the memory backend negates euclidean distances), so decay
        re-ranking, cross-shard merging, and truncation behave identically
        across metrics and backends.
        """
        t_start = time.perf_counter()
        stats = QueryStats()

        if time_window_ms is not None:
            start_ms, end_ms = time_window_ms
            first_ep = epoch_id(start_ms, self._epoch_size_ms)
            last_ep = epoch_id(end_ms, self._epoch_size_ms)
            epochs = [ep for ep in self._list_active_epochs() if first_ep <= ep <= last_ep]
        else:
            epochs = self._list_active_epochs()
        if _epoch_ids is not None:
            epochs = [ep for ep in epochs if ep in _epoch_ids]

        epoch_collections = [
            (e, self._col_name(e)) for e in epochs if self._col_name(e) in self._known_collections
        ]
        # Summary collections join the fan-out transparently: time-windowed
        # queries include only those whose coarse range overlaps the window,
        # unwindowed queries include all.  A ``None`` epoch marks a summary
        # collection — it has no single raw epoch, so the Hilbert pre-filter
        # is skipped and spatial bounds rely on the exact post-filter.
        search_targets: list[tuple[int | None, str]] = list(epoch_collections)
        search_targets.extend(
            (None, col) for col in self._summary_search_collections(time_window_ms, _epoch_ids)
        )
        stats.shards_searched = len(search_targets)

        # Search across shards; over-fetch whenever post-search filtering
        # (spatial exact match, min_confidence) or decay re-ranking can
        # reorder/drop hits, else per-shard truncation could evict the top-k.
        needs_overfetch = (
            spatial_bounds is not None or min_confidence is not None or self._decay_lambda > 0
        )
        shard_limit = limit * _EXACT_FILTER_OVERFETCH if needs_overfetch else limit
        all_results: list[dict] = []
        for ep, col in search_targets:
            payload_filter: dict = {}
            if spatial_bounds is not None and ep is not None:
                query_resolution = choose_query_resolution(
                    self._hilbert,
                    self._adaptive,
                    spatial_bounds,
                    time_window_ms,
                    ep,
                    self._epoch_size_ms,
                    overlap_factor,
                )
                hids = self._hilbert.query_buckets(
                    bounds_for_epoch(spatial_bounds, time_window_ms, ep, self._epoch_size_ms),
                    resolution=query_resolution,
                    overlap_factor=overlap_factor,
                )
                if not hids:
                    continue
                field = self._hilbert.payload_field(query_resolution)
                payload_filter[field] = {"any": hids}
                stats.hilbert_ids_in_filter += len(hids)

            if time_window_ms is not None:
                payload_filter["timestamp_ms"] = {"gte": start_ms, "lte": end_ms}

            if min_confidence is not None:
                payload_filter["confidence"] = {"gte": min_confidence}

            payload_filter.update(extra_filter_to_memory(_extra_payload_filter))

            hits = self._store.search(
                collection=col,
                query_vector=vector,
                limit=shard_limit,
                payload_filter=payload_filter if payload_filter else None,
            )
            stats.total_candidates += len(hits)
            for hit in hits:
                all_results.append(
                    {
                        "score": hit["score"],
                        "timestamp_ms": hit["payload"].get("timestamp_ms", 0),
                        "payload": hit["payload"],
                        "vector": hit["vector"],
                        "id": hit["id"],
                    }
                )

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
        """Reconstruct a trajectory using scroll with scene_id filter."""
        anchor = self._get_state_by_id(state_id)
        if anchor is None:
            return []
        if not anchor.scene_id:
            return [anchor]

        all_states: list[WorldState] = []
        for col in self._raw_collections():
            hits = self._scroll_all(
                collection=col,
                payload_filter={"scene_id": anchor.scene_id},
                order_by="timestamp_ms",
            )
            for hit in hits:
                all_states.append(_payload_to_state(hit["payload"], hit["id"], hit["vector"]))

        all_states.sort(key=lambda s: s.timestamp_ms)
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
        """Return all states within ±window_ms of the given state in the same scene."""
        anchor = self._get_state_by_id(state_id)
        if anchor is None or not anchor.scene_id:
            return []

        t_min = anchor.timestamp_ms - window_ms
        t_max = anchor.timestamp_ms + window_ms

        context: list[WorldState] = []
        for col in self._raw_collections():
            hits = self._scroll_all(
                collection=col,
                payload_filter={
                    "scene_id": anchor.scene_id,
                    "timestamp_ms": {"gte": t_min, "lte": t_max},
                },
                order_by="timestamp_ms",
            )
            for hit in hits:
                context.append(_payload_to_state(hit["payload"], hit["id"], hit["vector"]))

        context.sort(key=lambda s: s.timestamp_ms)
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_purge(self) -> None:
        """Run epoch maintenance: consolidation first, then retention purge."""
        self._maybe_consolidate(int(time.time() * 1000))
        if self._retention is None:
            return
        try:
            dropped = self._retention.maybe_purge(
                active_epochs=self._list_active_epochs(),
                now_ms=int(time.time() * 1000),
                delete_fn=self._store.delete_collection,
            )
            # Forget purged collections so a late insert into a purged epoch
            # simply recreates the collection instead of crashing.
            for name in dropped:
                self._known_collections.discard(name)
        except Exception as exc:
            logger.warning("Retention purge failed: %s", exc)

    def _maybe_consolidate(self, now_ms: int) -> None:
        """Fold raw epochs that left the raw window into summary collections."""
        policy = self._consolidation_policy
        if policy is None:
            return
        stale = epochs_to_consolidate(
            self._list_active_epochs(),
            now_ms=now_ms,
            epoch_size_ms=self._epoch_size_ms,
            policy=policy,
        )
        for ep in stale:
            try:
                self._consolidate_epoch(ep, policy)
            except Exception:
                logger.warning("Consolidation failed for epoch %d", ep, exc_info=True)

    def _consolidate_epoch(self, ep: int, policy: ConsolidationPolicy) -> None:
        """Fold one raw epoch into its summary collection, then drop it.

        The coarse collection's existing summaries are combined with the
        epoch's raw states and re-consolidated, keeping the collection at
        ``max_states_per_scene`` states per scene.  Mirrors the retention
        flow: the raw collection is deleted and forgotten from
        ``_known_collections`` so a late insert simply recreates it.
        """
        raw_col = self._col_name(ep)
        sum_col = summary_collection_name(ep, self._epoch_size_ms, policy, self._collection_prefix)
        if ep not in self._consolidated_epochs:
            combined = [
                _payload_to_state(hit["payload"], hit["id"], hit["vector"])
                for col in (sum_col, raw_col)
                if col in self._known_collections
                for hit in self._scroll_all(collection=col)
            ]
            summaries = consolidate_states(combined, policy, seed=ep)
            self._store.delete_collection(sum_col)
            self._known_collections.discard(sum_col)
            if summaries:
                self._ensure_collection(sum_col)
                # Summaries bypass epoch routing and carry no Hilbert payload;
                # spatial queries reach them via the exact post-filter on x/y/z.
                points = [
                    {
                        "id": str(uuid.uuid4()),
                        "vector": s.vector,
                        "payload": _state_to_payload(s, {}),
                    }
                    for s in summaries
                ]
                self._store.upsert(sum_col, points)
            self._consolidated_epochs.add(ep)
        self._store.delete_collection(raw_col)
        self._known_collections.discard(raw_col)
        self._consolidated_epochs.discard(ep)
        logger.info("Consolidated epoch %d into %s", ep, sum_col)

    def _summary_search_collections(
        self,
        time_window_ms: tuple[int, int] | None,
        epoch_ids: set[int] | None,
    ) -> list[str]:
        """Summary collections to include in a query fan-out.

        Time-windowed queries keep only collections whose coarse time range
        overlaps the window; unwindowed queries keep all.  An explicit
        ``epoch_ids`` restriction (funnel narrowing) keeps only collections
        whose coarse range covers at least one requested epoch.
        """
        policy = self._consolidation_policy
        if policy is None:
            return []
        matched: list[tuple[int, str]] = []
        for col in self._known_collections:
            coarse = is_summary_collection(col, self._collection_prefix)
            if coarse is None:
                continue
            if time_window_ms is not None:
                t_min, t_max = summary_coarse_range(coarse, policy, self._epoch_size_ms)
                if t_max < time_window_ms[0] or t_min > time_window_ms[1]:
                    continue
            if epoch_ids is not None and not any(
                ep // policy.summary_epoch_ratio == coarse for ep in epoch_ids
            ):
                continue
            matched.append((coarse, col))
        return [col for _, col in sorted(matched)]

    def _raw_collections(self) -> list[str]:
        """Known collections excluding summaries (for trajectory/causal scans)."""
        return sorted(
            col
            for col in self._known_collections
            if is_summary_collection(col, self._collection_prefix) is None
        )

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._vector_size:
            raise ValueError(f"vector has dimension {len(vector)}, expected {self._vector_size}")

    def _normalise_time(self, timestamp_ms: int, ep: int) -> float:
        epoch_start = ep * self._epoch_size_ms
        offset = timestamp_ms - epoch_start
        return min(1.0, max(0.0, offset / self._epoch_size_ms))

    def _get_state_by_id(self, state_id: str) -> WorldState | None:
        for col in list(self._known_collections):
            results = self._store.retrieve(col, [state_id])
            if results:
                r = results[0]
                return _payload_to_state(r["payload"], r["id"], r["vector"])
        return None

    def _find_latest_predecessor(self, scene_id: str, before_ms: int) -> tuple[str, str] | None:
        if not scene_id:
            return None
        for collection in self._predecessor_search_collections(before_ms):
            results = self._scroll_all(
                collection=collection,
                payload_filter={
                    "scene_id": scene_id,
                    "timestamp_ms": {"lt": before_ms},
                },
                order_by="timestamp_ms",
            )
            if results:
                # Scroll returns ascending order; last item is the latest predecessor
                return str(results[-1]["id"]), collection
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

    def _list_active_epochs(self) -> list[int]:
        epochs: list[int] = []
        prefix = f"{self._collection_prefix}loci_" if self._collection_prefix else "loci_"
        for col in self._known_collections:
            if col.startswith(prefix):
                with contextlib.suppress(ValueError):
                    epochs.append(int(col[len(prefix) :]))
        return sorted(epochs) if epochs else []

    def _predecessor_search_collections(self, before_ms: int) -> list[str]:
        target_epoch = epoch_id(before_ms, self._epoch_size_ms)
        epochs = [ep for ep in self._list_active_epochs() if ep <= target_epoch]
        return [self._col_name(ep) for ep in sorted(epochs, reverse=True)]


# ------------------------------------------------------------------
# Shared payload helpers
# ------------------------------------------------------------------


def _normalize_id(point_id: object) -> str:
    """Normalise a point ID for comparison (lowercase, hyphens stripped)."""
    return str(point_id).lower().replace("-", "")


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
