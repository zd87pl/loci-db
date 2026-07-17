"""Wiring tests for memory consolidation on the Qdrant clients (mock-based).

Mirrors the LocalLociClient contract in tests/test_consolidation.py at the
wiring level, using the mocked-Qdrant patterns of tests/test_client.py and
tests/test_async_client.py: consolidation triggered for epochs leaving the
raw window, summaries upserted into the coarse summary collection (created
lazily with the client's vector params), raw collections dropped and
forgotten, summary collections joining the query fan-out (with coarse-range
pruning), exclusion from trajectory/causal/predecessor scans, retention
never touching summaries, and tenant prefixes applied throughout.

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
# epochs <= 3 are stale and epochs 0-3 all fold into loci_sum_0.
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
        instance.delete_collection = AsyncMock()
        instance.get_collections = AsyncMock(return_value=_collections_response([]))
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


def _record(pid: str, ts: int, scene: str = "a", vector: list[float] | None = None) -> MagicMock:
    """A scrolled Qdrant point (Record-shaped) with a full Loci payload."""
    point = MagicMock()
    point.id = pid
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
        "metadata": {},
    }
    return point


def _record_from_struct(point_struct) -> MagicMock:
    point = MagicMock()
    point.id = point_struct.id
    point.vector = point_struct.vector
    point.payload = point_struct.payload
    return point


def _wire_store(mock, initial: dict[str, list[MagicMock]]) -> dict[str, list[MagicMock]]:
    """Route upsert/delete/scroll through one dict so consolidation reads
    back exactly what it wrote (scroll filters are ignored — the
    consolidation path scrolls unfiltered).  Works for sync and async mocks:
    AsyncMock runs a plain-function side_effect and returns its value.
    """
    stored: dict[str, list[MagicMock]] = {k: list(v) for k, v in initial.items()}

    def _upsert(collection_name, points, **kwargs):
        stored.setdefault(collection_name, []).extend(_record_from_struct(p) for p in points)

    def _delete(collection_name, **kwargs):
        stored.pop(collection_name, None)
        return True

    def _scroll(collection_name, **kwargs):
        return (list(stored.get(collection_name, [])), None)

    mock.upsert.side_effect = _upsert
    mock.delete_collection.side_effect = _delete
    mock.scroll.side_effect = _scroll
    return stored


def _collections_response(names: list[str]) -> MagicMock:
    response = MagicMock()
    cols = []
    for name in names:
        col = MagicMock()
        col.name = name
        cols.append(col)
    response.collections = cols
    return response


def _raw_points(prefix: str = "") -> dict[str, list[MagicMock]]:
    """Two stale epochs: 3 scene-a + 2 scene-b points in epoch 0, 2 scene-a
    points in epoch 1 — 7 source states in coarse collection 0 overall."""
    return {
        f"{prefix}loci_0": [
            _record("a0", 100, scene="a", vector=[1.0, 0.0, 0.0, 0.0]),
            _record("a1", 200, scene="a", vector=[1.0, 0.0, 0.1, 0.0]),
            _record("a2", 300, scene="a", vector=[1.0, 0.0, 0.2, 0.0]),
            _record("b0", 150, scene="b", vector=[0.0, 1.0, 0.0, 0.0]),
            _record("b1", 250, scene="b", vector=[0.0, 1.0, 0.1, 0.0]),
        ],
        f"{prefix}loci_1": [
            _record("a3", EPOCH_MS + 100, scene="a", vector=[1.0, 0.0, 0.3, 0.0]),
            _record("a4", EPOCH_MS + 200, scene="a", vector=[1.0, 0.0, 0.4, 0.0]),
        ],
    }


# ---------------------------------------------------------------------------
# Sync: consolidation trigger, fold, and drop
# ---------------------------------------------------------------------------


class TestSyncConsolidationTrigger:
    def test_out_of_window_epochs_folded_dropped_and_forgotten(self, mock_qdrant):
        client = _make_client()
        client._known_collections = {"loci_0", "loci_1", "loci_5"}
        client._discovered = True
        stored = _wire_store(mock_qdrant, _raw_points())

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        # Raw collections beyond the window are dropped and forgotten; the
        # in-window epoch survives untouched.
        assert client._known_collections == {"loci_5", "loci_sum_0"}
        assert "loci_0" not in stored
        assert "loci_1" not in stored
        assert "loci_0" not in client._collection_locks
        assert "loci_1" not in client._collection_locks

        # Summaries land in the coarse collection, bounded per scene, with
        # lossless source-count bookkeeping across the refold.
        summaries = stored["loci_sum_0"]
        assert 0 < len(summaries) <= 2 * POLICY.max_states_per_scene
        assert all(p.payload["metadata"]["consolidated"] is True for p in summaries)
        assert sum(p.payload["metadata"]["source_count"] for p in summaries) == 7

    def test_summary_collection_created_with_client_vector_params(self, mock_qdrant):
        client = _make_client()
        client._known_collections = {"loci_0", "loci_5"}
        client._discovered = True
        _wire_store(mock_qdrant, _raw_points())

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        created = {
            c.kwargs["collection_name"]: c.kwargs["vectors_config"]
            for c in mock_qdrant.create_collection.call_args_list
        }
        assert "loci_sum_0" in created
        assert created["loci_sum_0"].size == VEC_SIZE

    def test_in_window_epochs_left_alone(self, mock_qdrant):
        client = _make_client()
        client._known_collections = {"loci_4", "loci_5"}
        client._discovered = True

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        mock_qdrant.delete_collection.assert_not_called()
        mock_qdrant.upsert.assert_not_called()
        assert client._known_collections == {"loci_4", "loci_5"}

    def test_no_policy_no_consolidation(self, mock_qdrant):
        client = _make_client(consolidation_policy=None)
        client._known_collections = {"loci_0", "loci_5"}
        client._discovered = True

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        mock_qdrant.delete_collection.assert_not_called()
        assert client._known_collections == {"loci_0", "loci_5"}


# ---------------------------------------------------------------------------
# Sync: query fan-out over summary collections
# ---------------------------------------------------------------------------


class TestSyncQueryFanOut:
    def _client(self, mock_qdrant) -> LociClient:
        client = _make_client()
        client._known_collections = {"loci_8", "loci_9", "loci_sum_0", "loci_sum_1"}
        client._discovered = True
        mock_qdrant.get_collections.return_value = _collections_response(
            sorted(client._known_collections)
        )
        return client

    def test_unwindowed_query_includes_all_summaries(self, mock_qdrant):
        client = self._client(mock_qdrant)
        client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert searched == {"loci_8", "loci_9", "loci_sum_0", "loci_sum_1"}

    def test_time_window_prunes_summaries_by_coarse_range(self, mock_qdrant):
        client = self._client(mock_qdrant)
        # ratio=4 x 5000ms: coarse 0 spans [0, 19999], coarse 1 [20000, 39999].
        client.query(vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(20_000, 39_999), limit=5)
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert searched == {"loci_sum_1"}

    def test_time_window_mixes_raw_and_overlapping_summary(self, mock_qdrant):
        client = self._client(mock_qdrant)
        client.query(vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(35_000, 44_999), limit=5)
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert searched == {"loci_8", "loci_sum_1"}

    def test_summary_shard_skips_hilbert_prefilter_keeps_time_filter(self, mock_qdrant):
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
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert searched == {"loci_sum_0"}
        filt = mock_qdrant.query_points.call_args.kwargs["query_filter"]
        keys = {condition.key for condition in filt.must}
        assert keys == {"timestamp_ms"}  # no hilbert_r* condition on summaries

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
        assert [r.id for r in results] == ["inside"]

    def test_no_policy_ignores_summary_collections(self, mock_qdrant):
        client = _make_client(consolidation_policy=None)
        client._known_collections = {"loci_8", "loci_sum_0"}
        client._discovered = True
        client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert searched == {"loci_8"}


# ---------------------------------------------------------------------------
# Sync: trajectory / causal / predecessor exclusion
# ---------------------------------------------------------------------------


class TestSyncNavigationExclusion:
    def _client_with_anchor(self, mock_qdrant) -> LociClient:
        client = _make_client()
        client._known_collections = {"loci_8", "loci_sum_0"}
        client._discovered = True
        anchor = _record("anchor", 8 * EPOCH_MS + 100)
        mock_qdrant.retrieve.side_effect = lambda collection_name, **kw: (
            [anchor] if collection_name == "loci_8" else []
        )
        mock_qdrant.scroll.side_effect = lambda collection_name, **kw: (
            ([anchor], None) if collection_name == "loci_8" else ([], None)
        )
        return client

    def test_trajectory_scans_only_raw_collections(self, mock_qdrant):
        client = self._client_with_anchor(mock_qdrant)
        trajectory = client.get_trajectory("anchor", steps_back=5, steps_forward=5)
        assert [s.id for s in trajectory] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_8"}

    def test_causal_context_scans_only_raw_collections(self, mock_qdrant):
        client = self._client_with_anchor(mock_qdrant)
        context = client.get_causal_context("anchor", window_ms=1000)
        assert [s.id for s in context] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_8"}

    def test_predecessor_search_ignores_summaries(self, mock_qdrant):
        client = _make_client()
        client._known_collections = {"loci_sum_0"}
        client._discovered = True
        assert client._find_latest_predecessor("a", before_ms=8 * EPOCH_MS) is None
        mock_qdrant.scroll.assert_not_called()


# ---------------------------------------------------------------------------
# Sync: retention interplay
# ---------------------------------------------------------------------------


class TestSyncRetentionInterplay:
    def test_retention_alone_never_purges_summary_collections(self, mock_qdrant):
        client = _make_client(
            consolidation_policy=None, retention_policy=RetentionPolicy(max_epochs=1)
        )
        client._known_collections = {"loci_0", "loci_5", "loci_sum_0"}
        client._discovered = True

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        deleted = [c.args[0] for c in mock_qdrant.delete_collection.call_args_list]
        assert deleted == ["loci_0"]
        assert client._known_collections == {"loci_5", "loci_sum_0"}

    def test_combined_consolidation_first_then_retention_on_raw_only(self, mock_qdrant):
        client = _make_client(retention_policy=RetentionPolicy(max_epochs=1))
        client._known_collections = {"loci_3", "loci_4", "loci_5"}
        client._discovered = True
        _wire_store(
            mock_qdrant,
            {"loci_3": [_record("p3", 3 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.0, 0.0])]},
        )

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        # Consolidation folds epoch 3 (stale); retention then purges epoch 4
        # (over max_epochs) but never touches the summary collection.
        deleted = [c.args[0] for c in mock_qdrant.delete_collection.call_args_list]
        assert deleted == ["loci_3", "loci_4"]
        assert client._known_collections == {"loci_5", "loci_sum_0"}


# ---------------------------------------------------------------------------
# Sync: tenant prefix
# ---------------------------------------------------------------------------


class TestSyncPrefix:
    def test_summary_collections_carry_the_tenant_prefix(self, mock_qdrant):
        client = _make_client(collection_prefix="t_")
        client._known_collections = {"t_loci_0", "t_loci_1", "t_loci_5"}
        client._discovered = True
        stored = _wire_store(mock_qdrant, _raw_points(prefix="t_"))

        with _now("loci.client", 5 * EPOCH_MS + 100):
            client._maybe_purge()

        assert client._known_collections == {"t_loci_5", "t_loci_sum_0"}
        assert "t_loci_sum_0" in stored
        assert "loci_sum_0" not in stored

        # The prefixed summary joins the fan-out for this tenant.
        client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert searched == {"t_loci_5", "t_loci_sum_0"}


# ---------------------------------------------------------------------------
# Async: consolidation trigger, fold, and drop
# ---------------------------------------------------------------------------


class TestAsyncConsolidationTrigger:
    @pytest.mark.asyncio
    async def test_out_of_window_epochs_folded_dropped_and_forgotten(self, mock_async_qdrant):
        client = _make_async_client()
        client._known_collections = {"loci_0", "loci_1", "loci_5"}
        client._discovered = True
        stored = _wire_store(mock_async_qdrant, _raw_points())

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        assert client._known_collections == {"loci_5", "loci_sum_0"}
        assert "loci_0" not in stored
        assert "loci_1" not in stored
        assert "loci_0" not in client._collection_locks
        assert "loci_1" not in client._collection_locks

        summaries = stored["loci_sum_0"]
        assert 0 < len(summaries) <= 2 * POLICY.max_states_per_scene
        assert all(p.payload["metadata"]["consolidated"] is True for p in summaries)
        assert sum(p.payload["metadata"]["source_count"] for p in summaries) == 7

    @pytest.mark.asyncio
    async def test_summary_collection_created_with_client_vector_params(self, mock_async_qdrant):
        client = _make_async_client()
        client._known_collections = {"loci_0", "loci_5"}
        client._discovered = True
        _wire_store(mock_async_qdrant, _raw_points())

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        created = {
            c.kwargs["collection_name"]: c.kwargs["vectors_config"]
            for c in mock_async_qdrant.create_collection.call_args_list
        }
        assert "loci_sum_0" in created
        assert created["loci_sum_0"].size == VEC_SIZE

    @pytest.mark.asyncio
    async def test_in_window_epochs_left_alone(self, mock_async_qdrant):
        client = _make_async_client()
        client._known_collections = {"loci_4", "loci_5"}
        client._discovered = True

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        mock_async_qdrant.delete_collection.assert_not_called()
        mock_async_qdrant.upsert.assert_not_called()
        assert client._known_collections == {"loci_4", "loci_5"}


# ---------------------------------------------------------------------------
# Async: query fan-out over summary collections
# ---------------------------------------------------------------------------


class TestAsyncQueryFanOut:
    def _client(self, mock_async_qdrant) -> AsyncLociClient:
        client = _make_async_client()
        client._known_collections = {"loci_8", "loci_9", "loci_sum_0", "loci_sum_1"}
        client._discovered = True
        mock_async_qdrant.get_collections = AsyncMock(
            return_value=_collections_response(sorted(client._known_collections))
        )
        return client

    @pytest.mark.asyncio
    async def test_unwindowed_query_includes_all_summaries(self, mock_async_qdrant):
        client = self._client(mock_async_qdrant)
        await client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = {
            c.kwargs["collection_name"] for c in mock_async_qdrant.query_points.call_args_list
        }
        assert searched == {"loci_8", "loci_9", "loci_sum_0", "loci_sum_1"}

    @pytest.mark.asyncio
    async def test_time_window_prunes_summaries_by_coarse_range(self, mock_async_qdrant):
        client = self._client(mock_async_qdrant)
        await client.query(vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(20_000, 39_999), limit=5)
        searched = {
            c.kwargs["collection_name"] for c in mock_async_qdrant.query_points.call_args_list
        }
        assert searched == {"loci_sum_1"}

    @pytest.mark.asyncio
    async def test_summary_shard_skips_hilbert_prefilter_keeps_time_filter(self, mock_async_qdrant):
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
        searched = {
            c.kwargs["collection_name"] for c in mock_async_qdrant.query_points.call_args_list
        }
        assert searched == {"loci_sum_0"}
        filt = mock_async_qdrant.query_points.call_args.kwargs["query_filter"]
        keys = {condition.key for condition in filt.must}
        assert keys == {"timestamp_ms"}  # no hilbert_r* condition on summaries


# ---------------------------------------------------------------------------
# Async: trajectory / causal / predecessor exclusion
# ---------------------------------------------------------------------------


class TestAsyncNavigationExclusion:
    def _client_with_anchor(self, mock_async_qdrant) -> AsyncLociClient:
        client = _make_async_client()
        client._known_collections = {"loci_8", "loci_sum_0"}
        client._discovered = True
        anchor = _record("anchor", 8 * EPOCH_MS + 100)
        mock_async_qdrant.retrieve = AsyncMock(
            side_effect=lambda collection_name, **kw: (
                [anchor] if collection_name == "loci_8" else []
            )
        )
        mock_async_qdrant.scroll = AsyncMock(
            side_effect=lambda collection_name, **kw: (
                ([anchor], None) if collection_name == "loci_8" else ([], None)
            )
        )
        return client

    @pytest.mark.asyncio
    async def test_trajectory_scans_only_raw_collections(self, mock_async_qdrant):
        client = self._client_with_anchor(mock_async_qdrant)
        trajectory = await client.get_trajectory("anchor", steps_back=5, steps_forward=5)
        assert [s.id for s in trajectory] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_async_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_8"}

    @pytest.mark.asyncio
    async def test_causal_context_scans_only_raw_collections(self, mock_async_qdrant):
        client = self._client_with_anchor(mock_async_qdrant)
        context = await client.get_causal_context("anchor", window_ms=1000)
        assert [s.id for s in context] == ["anchor"]
        scrolled = {c.kwargs["collection_name"] for c in mock_async_qdrant.scroll.call_args_list}
        assert scrolled == {"loci_8"}

    @pytest.mark.asyncio
    async def test_predecessor_search_ignores_summaries(self, mock_async_qdrant):
        client = _make_async_client()
        client._known_collections = {"loci_sum_0"}
        client._discovered = True
        assert await client._find_latest_predecessor("a", before_ms=8 * EPOCH_MS) is None
        mock_async_qdrant.scroll.assert_not_called()


# ---------------------------------------------------------------------------
# Async: retention interplay and tenant prefix
# ---------------------------------------------------------------------------


class TestAsyncRetentionAndPrefix:
    @pytest.mark.asyncio
    async def test_retention_alone_never_purges_summary_collections(self, mock_async_qdrant):
        client = _make_async_client(
            consolidation_policy=None, retention_policy=RetentionPolicy(max_epochs=1)
        )
        client._known_collections = {"loci_0", "loci_5", "loci_sum_0"}
        client._discovered = True

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        deleted = [c.args[0] for c in mock_async_qdrant.delete_collection.call_args_list]
        assert deleted == ["loci_0"]
        assert client._known_collections == {"loci_5", "loci_sum_0"}

    @pytest.mark.asyncio
    async def test_combined_consolidation_first_then_retention_on_raw_only(self, mock_async_qdrant):
        client = _make_async_client(retention_policy=RetentionPolicy(max_epochs=1))
        client._known_collections = {"loci_3", "loci_4", "loci_5"}
        client._discovered = True
        _wire_store(
            mock_async_qdrant,
            {"loci_3": [_record("p3", 3 * EPOCH_MS + 100, vector=[1.0, 0.0, 0.0, 0.0])]},
        )

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        deleted = [c.args[0] for c in mock_async_qdrant.delete_collection.call_args_list]
        assert deleted == ["loci_3", "loci_4"]
        assert client._known_collections == {"loci_5", "loci_sum_0"}

    @pytest.mark.asyncio
    async def test_summary_collections_carry_the_tenant_prefix(self, mock_async_qdrant):
        client = _make_async_client(collection_prefix="t_")
        client._known_collections = {"t_loci_0", "t_loci_1", "t_loci_5"}
        client._discovered = True
        stored = _wire_store(mock_async_qdrant, _raw_points(prefix="t_"))

        with _now("loci.async_client", 5 * EPOCH_MS + 100):
            await client._maybe_purge()

        assert client._known_collections == {"t_loci_5", "t_loci_sum_0"}
        assert "t_loci_sum_0" in stored
        assert "loci_sum_0" not in stored

        await client.query(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        searched = {
            c.kwargs["collection_name"] for c in mock_async_qdrant.query_points.call_args_list
        }
        assert searched == {"t_loci_5", "t_loci_sum_0"}
