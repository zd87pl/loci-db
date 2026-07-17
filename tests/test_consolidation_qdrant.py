"""Wiring tests for memory consolidation on the Qdrant clients (mock-based).

Mirrors the LocalLociClient contract in tests/test_consolidation.py at the
wiring level, using the mocked-Qdrant patterns of tests/test_client.py and
tests/test_async_client.py, against the bounded two-collection layout:
raw points in ``{prefix}loci_data``, summaries in ``{prefix}loci_summary``.

Covered: fold triggered for raw points leaving the raw window (grouped by
logical epoch), refolds against a coarse group's existing summaries selected
by timestamp Range, summary collection created lazily with the client's
vector params and WITHOUT Hilbert indexes, raw epoch Ranges deleted after
the fold, query filters (Hilbert on data only, timestamp Range on both),
funnel epoch-restriction envelope + exact post-filter, exclusion of
summaries from trajectory/causal/predecessor scans, retention deleting raw
points only, and tenant prefixes applied throughout.

All timestamps are injected and ``time.time`` is patched at each client's
module seam, so nothing here depends on the wall clock.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from loci.async_client import AsyncLociClient
from loci.client import LociClient
from loci.temporal.consolidation import ConsolidationPolicy
from loci.temporal.retention import RetentionPolicy

VEC_SIZE = 4
EPOCH_MS = 5000

# raw_window_epochs=2 / summary_epoch_ratio=4: with "now" inside epoch 5,
# raw points in epochs <= 3 are stale; epochs 0-3 share coarse group 0
# (timestamp range [0, 4*EPOCH_MS - 1]).
POLICY = ConsolidationPolicy(raw_window_epochs=2, summary_epoch_ratio=4, max_states_per_scene=3)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _not_found() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=404, reason_phrase="Not Found", content=b"", headers=httpx.Headers()
    )


@pytest.fixture()
def mock_qdrant():
    """Patch QdrantClient so no real Qdrant is needed."""
    with patch("loci.client.QdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance

        instance.get_collection.side_effect = _not_found()
        _empty_qr = MagicMock()
        _empty_qr.points = []
        instance.query_points.return_value = _empty_qr
        instance.scroll.return_value = ([], None)

        yield instance


@pytest.fixture()
def mock_async_qdrant():
    """Patch AsyncQdrantClient so no real Qdrant is needed."""
    with patch("loci.async_client.AsyncQdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance

        instance.get_collection = AsyncMock(side_effect=_not_found())
        instance.create_collection = AsyncMock()
        instance.create_payload_index = AsyncMock()
        instance.upsert = AsyncMock()
        _empty_qr = MagicMock()
        _empty_qr.points = []
        instance.query_points = AsyncMock(return_value=_empty_qr)
        instance.scroll = AsyncMock(return_value=([], None))
        instance.set_payload = AsyncMock()
        instance.retrieve = AsyncMock(return_value=[])
        instance.delete = AsyncMock()
        instance.delete_collection = AsyncMock()
        instance.close = AsyncMock()

        yield instance


def _make_client(**overrides) -> LociClient:
    kwargs: dict = {
        "qdrant_url": "http://fake:6333",
        "epoch_size_ms": EPOCH_MS,
        "vector_size": VEC_SIZE,
        "decay_lambda": 0.0,
        "consolidation_policy": POLICY,
    }
    kwargs.update(overrides)
    return LociClient(**kwargs)


def _make_async_client(**overrides) -> AsyncLociClient:
    kwargs: dict = {
        "qdrant_url": "http://fake:6333",
        "epoch_size_ms": EPOCH_MS,
        "vector_size": VEC_SIZE,
        "decay_lambda": 0.0,
        "consolidation_policy": POLICY,
    }
    kwargs.update(overrides)
    return AsyncLociClient(**kwargs)


@contextlib.contextmanager
def _now(module: str, ts_ms: int):
    """Pin the client module's wall clock (maintenance + decay) to *ts_ms*."""
    with patch(f"{module}.time.time", return_value=ts_ms / 1000.0):
        yield


def _record(
    pid: str,
    ts: int,
    scene: str = "a",
    vector: list[float] | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    """A scrolled Qdrant point (Record-shaped) with a full Loci payload."""
    point = MagicMock()
    point.id = pid
    point.score = 0.9
    point.vector = list(vector) if vector is not None else [1.0, 0.0, 0.0, 0.0]
    point.payload = {
        "x": 0.5,
        "y": 0.5,
        "z": 0.5,
        "timestamp_ms": ts,
        "scene_id": scene,
        "scale_level": "patch",
        "confidence": 1.0,
        "prev_state_id": None,
        "next_state_id": None,
        "metadata": metadata or {},
    }
    return point


def _record_from_struct(point_struct) -> MagicMock:
    point = MagicMock()
    point.id = point_struct.id
    point.vector = point_struct.vector
    point.payload = point_struct.payload
    return point


def _filter_matches(payload: dict, qfilter) -> bool:
    """Evaluate a Qdrant Filter's must-conditions against a payload."""
    if qfilter is None:
        return True
    for cond in qfilter.must or []:
        key = cond.key
        rng = getattr(cond, "range", None)
        if rng is not None:
            val = payload.get(key, 0)
            if rng.gte is not None and not val >= rng.gte:
                return False
            if rng.gt is not None and not val > rng.gt:
                return False
            if rng.lte is not None and not val <= rng.lte:
                return False
            if rng.lt is not None and not val < rng.lt:
                return False
            continue
        match = getattr(cond, "match", None)
        if match is not None:
            any_vals = getattr(match, "any", None)
            if any_vals is not None:
                if payload.get(key) not in any_vals:
                    return False
                continue
            if payload.get(key) != getattr(match, "value", None):
                return False
    return True


def _wire_store(mock, initial: dict[str, list[MagicMock]]) -> dict[str, list[MagicMock]]:
    """Route upsert/scroll/delete through one dict so consolidation reads back
    exactly what it wrote.  Scroll honours filters and limits (the fold path
    scans by timestamp Range) and delete honours FilterSelector timestamp
    Ranges.  Works for sync and async mocks: AsyncMock runs a plain-function
    side_effect and returns its value.
    """
    stored: dict[str, list[MagicMock]] = {k: list(v) for k, v in initial.items()}

    def _upsert(collection_name, points, **kwargs):
        stored.setdefault(collection_name, []).extend(_record_from_struct(p) for p in points)

    def _delete(collection_name, points_selector=None, **kwargs):
        qfilter = getattr(points_selector, "filter", None)
        stored[collection_name] = [
            p for p in stored.get(collection_name, []) if not _filter_matches(p.payload, qfilter)
        ]
        return True

    def _scroll(collection_name, scroll_filter=None, limit=None, **kwargs):
        points = [
            p for p in stored.get(collection_name, []) if _filter_matches(p.payload, scroll_filter)
        ]
        if limit is not None:
            points = points[:limit]
        return (points, None)

    mock.upsert.side_effect = _upsert
    mock.delete.side_effect = _delete
    mock.scroll.side_effect = _scroll
    return stored


def _timestamps(points: list[MagicMock]) -> list[int]:
    return sorted(p.payload["timestamp_ms"] for p in points)


def _raw_points(prefix: str = "") -> dict[str, list[MagicMock]]:
    """One data collection holding two stale epochs plus one in-window point:
    3 scene-a + 2 scene-b points in epoch 0, 2 scene-a points in epoch 1
    (7 stale source states, all in coarse group 0), and 1 point in epoch 5.
    """
    return {
        f"{prefix}loci_data": [
            _record("a0", 100, scene="a", vector=[1.0, 0.0, 0.0, 0.0]),
            _record("a1", 200, scene="a", vector=[1.0, 0.0, 0.1, 0.0]),
            _record("a2", 300, scene="a", vector=[1.0, 0.0, 0.2, 0.0]),
            _record("b0", 150, scene="b", vector=[0.0, 1.0, 0.0, 0.0]),
            _record("b1", 250, scene="b", vector=[0.0, 1.0, 0.1, 0.0]),
            _record("a3", EPOCH_MS + 100, scene="a", vector=[1.0, 0.0, 0.3, 0.0]),
            _record("a4", EPOCH_MS + 200, scene="a", vector=[1.0, 0.0, 0.4, 0.0]),
            _record("keep", 5 * EPOCH_MS + 100, scene="a", vector=[1.0, 0.0, 0.5, 0.0]),
        ],
    }


# ---------------------------------------------------------------------------
# Sync: consolidation trigger, fold, and raw-range delete
# ---------------------------------------------------------------------------


class TestSyncConsolidationTrigger:
    def test_stale_epochs_folded_and_raw_points_deleted(self, mock_qdrant):
        client = _make_client()
        stored = _wire_store(mock_qdrant, _raw_points())

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        # Raw points beyond the window are gone; the in-window point survives.
        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]

        # Summaries land in the one summary collection, bounded per scene,
        # with lossless source-count bookkeeping across the refold.
        summaries = stored["loci_summary"]
        assert 0 < len(summaries) <= 2 * POLICY.max_states_per_scene
        assert all(p.payload["metadata"]["consolidated"] is True for p in summaries)
        assert sum(p.payload["metadata"]["source_count"] for p in summaries) == 7
        # Summary timestamps stay inside coarse group 0's range.
        assert all(0 <= ts < 4 * EPOCH_MS for ts in _timestamps(summaries))

        # No collection is ever dropped — only point Ranges are deleted.
        mock_qdrant.delete_collection.assert_not_called()

    def test_summary_collection_created_lazily_with_client_vector_params(self, mock_qdrant):
        client = _make_client()
        _wire_store(mock_qdrant, _raw_points())

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        created = {
            c.kwargs["collection_name"]: c.kwargs["vectors_config"]
            for c in mock_qdrant.create_collection.call_args_list
        }
        assert set(created) == {"loci_summary"}
        assert created["loci_summary"].size == VEC_SIZE
        # No Hilbert indexes on summaries — timestamp/keyword indexes only.
        fields = [
            c.kwargs["field_name"]
            for c in mock_qdrant.create_payload_index.call_args_list
            if c.kwargs["collection_name"] == "loci_summary"
        ]
        assert sorted(fields) == ["scale_level", "scene_id", "timestamp_ms"]

    def test_refold_merges_existing_summaries_selected_by_coarse_range(self, mock_qdrant):
        client = _make_client()
        prior = _record(
            "sum-old",
            300,
            scene="a",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"consolidated": True, "source_count": 5, "t_min_ms": 0, "t_max_ms": 400},
        )
        stored = _wire_store(
            mock_qdrant,
            {
                "loci_data": [
                    _record("late", 3 * EPOCH_MS + 100, scene="a", vector=[1.0, 0.0, 0.1, 0.0])
                ],
                "loci_summary": [prior],
            },
        )
        client._collection_ready[client._summary_collection] = True

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        assert stored["loci_data"] == []
        summaries = stored["loci_summary"]
        assert 0 < len(summaries) <= POLICY.max_states_per_scene
        # The old summary was replaced, its source count composed losslessly.
        assert "sum-old" not in {p.id for p in summaries}
        assert sum(p.payload["metadata"]["source_count"] for p in summaries) == 6

    def test_in_window_points_left_alone(self, mock_qdrant):
        client = _make_client()
        stored = _wire_store(
            mock_qdrant,
            {"loci_data": [_record("keep", 5 * EPOCH_MS + 100, scene="a")]},
        )

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        mock_qdrant.upsert.assert_not_called()
        mock_qdrant.delete.assert_not_called()
        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]

    def test_no_policy_no_consolidation(self, mock_qdrant):
        client = _make_client(consolidation_policy=None)
        stored = _wire_store(mock_qdrant, _raw_points())

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        mock_qdrant.upsert.assert_not_called()
        mock_qdrant.delete.assert_not_called()
        assert len(stored["loci_data"]) == 8


# ---------------------------------------------------------------------------
# Sync: query filters over the two collections
# ---------------------------------------------------------------------------


class TestSyncQueryFilters:
    def _client(self, mock_qdrant) -> LociClient:
        client = _make_client()
        client._collection_ready["loci_data"] = True
        client._collection_ready["loci_summary"] = True
        return client

    def test_both_collections_searched(self, mock_qdrant):
        client = self._client(mock_qdrant)
        client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = [c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list]
        assert searched == ["loci_data", "loci_summary"]

    def test_hilbert_prefilter_on_data_only_time_filter_on_both(self, mock_qdrant):
        client = self._client(mock_qdrant)
        bounds = {
            "x_min": 0.4,
            "x_max": 0.6,
            "y_min": 0.4,
            "y_max": 0.6,
            "z_min": 0.4,
            "z_max": 0.6,
        }
        client.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            spatial_bounds=bounds,
            time_window_ms=(0, 19_999),
            limit=5,
        )
        filters = {
            c.kwargs["collection_name"]: c.kwargs["query_filter"]
            for c in mock_qdrant.query_points.call_args_list
        }
        data_keys = {condition.key for condition in filters["loci_data"].must}
        summary_keys = {condition.key for condition in filters["loci_summary"].must}
        assert "timestamp_ms" in data_keys
        assert any(key.startswith("hilbert_r") for key in data_keys)
        assert summary_keys == {"timestamp_ms"}  # no hilbert_r* condition on summaries

    def test_exact_post_filter_applies_to_summary_hits(self, mock_qdrant):
        client = self._client(mock_qdrant)

        inside = _record("inside", 10_000)
        inside.score = 0.8
        outside = _record("outside", 25_000)  # beyond the window
        outside.score = 0.99
        qr = MagicMock()
        qr.points = [outside, inside]
        mock_qdrant.query_points.return_value = qr

        results = client.query(vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(0, 19_999), limit=5)
        assert {r.id for r in results} == {"inside"}

    def test_epoch_restriction_widens_range_to_coarse_group(self, mock_qdrant):
        """Funnel narrowing: envelope Range + exact epoch/coarse post-filter."""
        client = self._client(mock_qdrant)

        raw_in = _record("raw-ep0", 100)
        raw_out = _record("raw-ep2", 2 * EPOCH_MS + 100)  # not in the requested epoch
        summary_hit = _record(
            "sum-coarse0",
            2 * EPOCH_MS,  # inside coarse group 0's range, outside epoch 0
            metadata={"consolidated": True, "source_count": 3, "t_min_ms": 0, "t_max_ms": 100},
        )

        def _query_points(collection_name, **kwargs):
            qr = MagicMock()
            qr.points = [raw_in, raw_out] if collection_name == "loci_data" else [summary_hit]
            return qr

        mock_qdrant.query_points.side_effect = _query_points

        results = client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=10, _epoch_ids={0})

        # Raw hits need exact epoch membership; summaries pass when their
        # coarse group covers a requested epoch.
        assert {r.id for r in results} == {"raw-ep0", "sum-coarse0"}
        # The pushed-down Range is the coarse-group envelope [0, 4*EPOCH-1].
        filters = {
            c.kwargs["collection_name"]: c.kwargs["query_filter"]
            for c in mock_qdrant.query_points.call_args_list
        }
        for qfilter in filters.values():
            (ts_condition,) = [c for c in qfilter.must if c.key == "timestamp_ms"]
            assert ts_condition.range.gte == 0
            assert ts_condition.range.lte == 4 * EPOCH_MS - 1

    def test_empty_epoch_restriction_short_circuits(self, mock_qdrant):
        client = self._client(mock_qdrant)
        assert client.query(vector=[1.0, 0.0, 0.0, 0.0], _epoch_ids=set()) == []
        mock_qdrant.query_points.assert_not_called()


# ---------------------------------------------------------------------------
# Sync: trajectory / causal / predecessor exclusion
# ---------------------------------------------------------------------------


class TestSyncNavigationExclusion:
    def _client_with_anchor(self, mock_qdrant) -> LociClient:
        client = _make_client()
        anchor = _record("anchor", 8 * EPOCH_MS + 100)
        mock_qdrant.retrieve.side_effect = lambda collection_name, **kw: (
            [anchor] if collection_name == "loci_data" else []
        )
        mock_qdrant.scroll.side_effect = lambda collection_name, **kw: (
            ([anchor], None) if collection_name == "loci_data" else ([], None)
        )
        return client

    def test_trajectory_scans_only_the_data_collection(self, mock_qdrant):
        client = self._client_with_anchor(mock_qdrant)
        trajectory = client.get_trajectory("anchor", steps_back=5, steps_forward=5)
        assert [s.id for s in trajectory] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_data"}

    def test_causal_context_scans_only_the_data_collection(self, mock_qdrant):
        client = self._client_with_anchor(mock_qdrant)
        context = client.get_causal_context("anchor", window_ms=1000)
        assert [s.id for s in context] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_data"}

    def test_predecessor_search_never_touches_summaries(self, mock_qdrant):
        client = _make_client()
        assert client._find_latest_predecessor("a", before_ms=8 * EPOCH_MS) is None
        scrolled = {c.kwargs["collection_name"] for c in mock_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_data"}


# ---------------------------------------------------------------------------
# Sync: retention interplay
# ---------------------------------------------------------------------------


class TestSyncRetentionInterplay:
    def test_retention_alone_deletes_raw_points_never_summaries(self, mock_qdrant):
        client = _make_client(
            consolidation_policy=None, retention_policy=RetentionPolicy(max_epochs=1)
        )
        stored = _wire_store(
            mock_qdrant,
            {
                "loci_data": [_record("old", 100), _record("new", 5 * EPOCH_MS + 100)],
                "loci_summary": [_record("sum", 200, metadata={"consolidated": True})],
            },
        )

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        deleted = {c.kwargs["collection_name"] for c in mock_qdrant.delete.call_args_list}
        assert deleted == {"loci_data"}
        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]
        assert [p.id for p in stored["loci_summary"]] == ["sum"]

    def test_combined_consolidation_first_then_retention_on_raw_only(self, mock_qdrant):
        client = _make_client(retention_policy=RetentionPolicy(max_epochs=1))
        stored = _wire_store(
            mock_qdrant,
            {
                "loci_data": [
                    _record("p3", 3 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.0, 0.0]),
                    _record("p4", 4 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.1, 0.0]),
                    _record("p5", 5 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.2, 0.0]),
                ]
            },
        )

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        # Consolidation folds epoch 3 (stale) into the summary collection;
        # retention then purges epoch 4 (over max_epochs) but never touches
        # summaries.
        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]
        summaries = stored["loci_summary"]
        assert len(summaries) == 1
        assert summaries[0].payload["metadata"]["source_count"] == 1
        deleted = {c.kwargs["collection_name"] for c in mock_qdrant.delete.call_args_list}
        assert deleted == {"loci_data"}


# ---------------------------------------------------------------------------
# Sync: tenant prefix
# ---------------------------------------------------------------------------


class TestSyncPrefix:
    def test_two_prefixed_collections_per_tenant(self, mock_qdrant):
        client = _make_client(collection_prefix="t_")
        stored = _wire_store(mock_qdrant, _raw_points(prefix="t_"))

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        assert "t_loci_summary" in stored
        assert "loci_summary" not in stored
        assert sum(p.payload["metadata"]["source_count"] for p in stored["t_loci_summary"]) == 7

        # The prefixed pair is what queries search for this tenant.
        client._collection_ready["t_loci_data"] = True
        client._collection_ready["t_loci_summary"] = True
        client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert searched == {"t_loci_data", "t_loci_summary"}


# ---------------------------------------------------------------------------
# Async: consolidation trigger, fold, and raw-range delete
# ---------------------------------------------------------------------------


class TestAsyncConsolidationTrigger:
    @pytest.mark.asyncio
    async def test_stale_epochs_folded_and_raw_points_deleted(self, mock_async_qdrant):
        client = _make_async_client()
        stored = _wire_store(mock_async_qdrant, _raw_points())

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]

        summaries = stored["loci_summary"]
        assert 0 < len(summaries) <= 2 * POLICY.max_states_per_scene
        assert all(p.payload["metadata"]["consolidated"] is True for p in summaries)
        assert sum(p.payload["metadata"]["source_count"] for p in summaries) == 7
        assert all(0 <= ts < 4 * EPOCH_MS for ts in _timestamps(summaries))

        mock_async_qdrant.delete_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_summary_collection_created_lazily_with_client_vector_params(
        self, mock_async_qdrant
    ):
        client = _make_async_client()
        _wire_store(mock_async_qdrant, _raw_points())

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        created = {
            c.kwargs["collection_name"]: c.kwargs["vectors_config"]
            for c in mock_async_qdrant.create_collection.call_args_list
        }
        assert set(created) == {"loci_summary"}
        assert created["loci_summary"].size == VEC_SIZE
        fields = [
            c.kwargs["field_name"]
            for c in mock_async_qdrant.create_payload_index.call_args_list
            if c.kwargs["collection_name"] == "loci_summary"
        ]
        assert sorted(fields) == ["scale_level", "scene_id", "timestamp_ms"]

    @pytest.mark.asyncio
    async def test_in_window_points_left_alone(self, mock_async_qdrant):
        client = _make_async_client()
        stored = _wire_store(
            mock_async_qdrant,
            {"loci_data": [_record("keep", 5 * EPOCH_MS + 100, scene="a")]},
        )

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        mock_async_qdrant.upsert.assert_not_called()
        mock_async_qdrant.delete.assert_not_called()
        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]


# ---------------------------------------------------------------------------
# Async: query filters over the two collections
# ---------------------------------------------------------------------------


class TestAsyncQueryFilters:
    def _client(self, mock_async_qdrant) -> AsyncLociClient:
        client = _make_async_client()
        client._collection_ready["loci_data"] = True
        client._collection_ready["loci_summary"] = True
        return client

    @pytest.mark.asyncio
    async def test_both_collections_searched(self, mock_async_qdrant):
        client = self._client(mock_async_qdrant)
        await client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = [
            c.kwargs["collection_name"] for c in mock_async_qdrant.query_points.call_args_list
        ]
        assert searched == ["loci_data", "loci_summary"]

    @pytest.mark.asyncio
    async def test_hilbert_prefilter_on_data_only_time_filter_on_both(self, mock_async_qdrant):
        client = self._client(mock_async_qdrant)
        bounds = {
            "x_min": 0.4,
            "x_max": 0.6,
            "y_min": 0.4,
            "y_max": 0.6,
            "z_min": 0.4,
            "z_max": 0.6,
        }
        await client.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            spatial_bounds=bounds,
            time_window_ms=(0, 19_999),
            limit=5,
        )
        filters = {
            c.kwargs["collection_name"]: c.kwargs["query_filter"]
            for c in mock_async_qdrant.query_points.call_args_list
        }
        data_keys = {condition.key for condition in filters["loci_data"].must}
        summary_keys = {condition.key for condition in filters["loci_summary"].must}
        assert "timestamp_ms" in data_keys
        assert any(key.startswith("hilbert_r") for key in data_keys)
        assert summary_keys == {"timestamp_ms"}  # no hilbert_r* condition on summaries

    @pytest.mark.asyncio
    async def test_epoch_restriction_widens_range_to_coarse_group(self, mock_async_qdrant):
        client = self._client(mock_async_qdrant)

        raw_in = _record("raw-ep0", 100)
        raw_out = _record("raw-ep2", 2 * EPOCH_MS + 100)
        summary_hit = _record(
            "sum-coarse0",
            2 * EPOCH_MS,
            metadata={"consolidated": True, "source_count": 3, "t_min_ms": 0, "t_max_ms": 100},
        )

        def _query_points(collection_name, **kwargs):
            qr = MagicMock()
            qr.points = [raw_in, raw_out] if collection_name == "loci_data" else [summary_hit]
            return qr

        mock_async_qdrant.query_points = AsyncMock(side_effect=_query_points)

        results = await client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=10, _epoch_ids={0})

        assert {r.id for r in results} == {"raw-ep0", "sum-coarse0"}


# ---------------------------------------------------------------------------
# Async: trajectory / causal / predecessor exclusion
# ---------------------------------------------------------------------------


class TestAsyncNavigationExclusion:
    def _client_with_anchor(self, mock_async_qdrant) -> AsyncLociClient:
        client = _make_async_client()
        anchor = _record("anchor", 8 * EPOCH_MS + 100)
        mock_async_qdrant.retrieve = AsyncMock(
            side_effect=lambda collection_name, **kw: (
                [anchor] if collection_name == "loci_data" else []
            )
        )
        mock_async_qdrant.scroll = AsyncMock(
            side_effect=lambda collection_name, **kw: (
                ([anchor], None) if collection_name == "loci_data" else ([], None)
            )
        )
        return client

    @pytest.mark.asyncio
    async def test_trajectory_scans_only_the_data_collection(self, mock_async_qdrant):
        client = self._client_with_anchor(mock_async_qdrant)
        trajectory = await client.get_trajectory("anchor", steps_back=5, steps_forward=5)
        assert [s.id for s in trajectory] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_async_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_data"}

    @pytest.mark.asyncio
    async def test_causal_context_scans_only_the_data_collection(self, mock_async_qdrant):
        client = self._client_with_anchor(mock_async_qdrant)
        context = await client.get_causal_context("anchor", window_ms=1000)
        assert [s.id for s in context] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_async_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_data"}

    @pytest.mark.asyncio
    async def test_predecessor_search_never_touches_summaries(self, mock_async_qdrant):
        client = _make_async_client()
        assert await client._find_latest_predecessor("a", before_ms=8 * EPOCH_MS) is None
        scrolled = {c.kwargs["collection_name"] for c in mock_async_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_data"}


# ---------------------------------------------------------------------------
# Async: retention interplay and tenant prefix
# ---------------------------------------------------------------------------


class TestAsyncRetentionAndPrefix:
    @pytest.mark.asyncio
    async def test_retention_alone_deletes_raw_points_never_summaries(self, mock_async_qdrant):
        client = _make_async_client(
            consolidation_policy=None, retention_policy=RetentionPolicy(max_epochs=1)
        )
        stored = _wire_store(
            mock_async_qdrant,
            {
                "loci_data": [_record("old", 100), _record("new", 5 * EPOCH_MS + 100)],
                "loci_summary": [_record("sum", 200, metadata={"consolidated": True})],
            },
        )

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        deleted = {c.kwargs["collection_name"] for c in mock_async_qdrant.delete.call_args_list}
        assert deleted == {"loci_data"}
        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]
        assert [p.id for p in stored["loci_summary"]] == ["sum"]

    @pytest.mark.asyncio
    async def test_combined_consolidation_first_then_retention_on_raw_only(self, mock_async_qdrant):
        client = _make_async_client(retention_policy=RetentionPolicy(max_epochs=1))
        stored = _wire_store(
            mock_async_qdrant,
            {
                "loci_data": [
                    _record("p3", 3 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.0, 0.0]),
                    _record("p4", 4 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.1, 0.0]),
                    _record("p5", 5 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.2, 0.0]),
                ]
            },
        )

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        assert _timestamps(stored["loci_data"]) == [5 * EPOCH_MS + 100]
        summaries = stored["loci_summary"]
        assert len(summaries) == 1
        assert summaries[0].payload["metadata"]["source_count"] == 1
        deleted = {c.kwargs["collection_name"] for c in mock_async_qdrant.delete.call_args_list}
        assert deleted == {"loci_data"}

    @pytest.mark.asyncio
    async def test_two_prefixed_collections_per_tenant(self, mock_async_qdrant):
        client = _make_async_client(collection_prefix="t_")
        stored = _wire_store(mock_async_qdrant, _raw_points(prefix="t_"))

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        assert "t_loci_summary" in stored
        assert "loci_summary" not in stored
        assert sum(p.payload["metadata"]["source_count"] for p in stored["t_loci_summary"]) == 7

        client._collection_ready["t_loci_data"] = True
        client._collection_ready["t_loci_summary"] = True
        await client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = {
            c.kwargs["collection_name"] for c in mock_async_qdrant.query_points.call_args_list
        }
        assert searched == {"t_loci_data", "t_loci_summary"}
