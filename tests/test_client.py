"""Integration tests for LociClient with a mocked Qdrant backend."""

from __future__ import annotations

import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from loci.client import LociClient
from loci.retrieval.predict import PredictRetrieveResult
from loci.schema import ScoredWorldState, WorldState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_qdrant():
    """Patch QdrantClient so no real Qdrant is needed."""
    with patch("loci.client.QdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance

        # get_collection raises 404 by default → triggers creation
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        resp_404 = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=httpx.Headers(),
        )
        instance.get_collection.side_effect = resp_404

        # query_points returns empty by default
        _empty_qr = MagicMock()
        _empty_qr.points = []
        instance.query_points.return_value = _empty_qr
        # scroll returns empty by default
        instance.scroll.return_value = ([], None)

        yield instance


@pytest.fixture()
def client(mock_qdrant):
    return LociClient(
        qdrant_url="http://fake:6333",
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=4,
        decay_lambda=0.0,  # disable decay for deterministic tests
    )


def _make_state(**overrides) -> WorldState:
    defaults = dict(
        x=0.5,
        y=0.5,
        z=0.5,
        timestamp_ms=10_000,
        vector=[1.0, 2.0, 3.0, 4.0],
        scene_id="test_scene",
    )
    defaults.update(overrides)
    return WorldState(**defaults)


# ---------------------------------------------------------------------------
# _ensure_collection
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    def test_creates_collection_on_404(self, client, mock_qdrant):
        """Collection should be created when Qdrant returns 404."""
        client._ensure_collection("loci_0")

        mock_qdrant.create_collection.assert_called_once()
        name_arg = mock_qdrant.create_collection.call_args.kwargs["collection_name"]
        assert name_arg == "loci_0"

    def test_idempotent_after_first_call(self, client, mock_qdrant):
        """Second call should not hit Qdrant again."""
        client._ensure_collection("loci_0")
        mock_qdrant.reset_mock()

        client._ensure_collection("loci_0")
        mock_qdrant.get_collection.assert_not_called()
        mock_qdrant.create_collection.assert_not_called()

    def test_skips_create_when_collection_exists(self, mock_qdrant):
        """If get_collection succeeds, don't create."""
        mock_qdrant.get_collection.side_effect = None
        mock_qdrant.get_collection.return_value = MagicMock()

        c = LociClient.__new__(LociClient)
        c._qdrant = mock_qdrant
        c._vector_size = 4
        c._known_collections = set()
        c._collection_locks = {}
        c._locks_mutex = threading.Lock()

        c._ensure_collection("loci_0")
        mock_qdrant.create_collection.assert_not_called()

    def test_propagates_non_404_errors(self, mock_qdrant):
        """Non-404 errors from get_collection should propagate."""
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_qdrant.get_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal",
            content=b"",
            headers=httpx.Headers(),
        )
        c = LociClient.__new__(LociClient)
        c._qdrant = mock_qdrant
        c._vector_size = 4
        c._known_collections = set()
        c._collection_locks = {}
        c._locks_mutex = threading.Lock()

        with pytest.raises(UnexpectedResponse):
            c._ensure_collection("loci_0")

    def test_create_conflict_treated_as_success(self, client, mock_qdrant):
        """A concurrent-writer 409 on create_collection should not raise."""
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_qdrant.create_collection.side_effect = UnexpectedResponse(
            status_code=409,
            reason_phrase="Conflict",
            content=b"already exists",
            headers=httpx.Headers(),
        )

        client._ensure_collection("loci_0")

        assert "loci_0" in client._known_collections
        # The race winner creates the indexes, not us.
        mock_qdrant.create_payload_index.assert_not_called()

    def test_concurrent_threads_race_single_create(self, client, mock_qdrant):
        """Two threads racing _ensure_collection must not double-create."""
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def _run():
            barrier.wait()
            try:
                client._ensure_collection("loci_7")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=_run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert mock_qdrant.create_collection.call_count == 1
        assert "loci_7" in client._known_collections

    def test_creates_scale_level_index(self, client, mock_qdrant):
        """scale_level should get a KEYWORD payload index."""
        client._ensure_collection("loci_0")

        index_calls = mock_qdrant.create_payload_index.call_args_list
        field_names = [c.kwargs["field_name"] for c in index_calls]
        assert "scale_level" in field_names
        assert "scene_id" in field_names


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_returns_id(self, client, mock_qdrant):
        state = _make_state()
        result = client.insert(state)
        assert isinstance(result, str)
        # Canonical hyphenated UUID so IDs round-trip through real Qdrant
        # servers (which serialise UUID point IDs in hyphenated form).
        assert len(result) == 36
        assert str(uuid.UUID(result)) == result

    def test_rejects_wrong_vector_dimension(self, client, mock_qdrant):
        state = _make_state(vector=[1.0, 2.0, 3.0])  # client has vector_size=4
        with pytest.raises(ValueError, match="dimension"):
            client.insert(state)
        mock_qdrant.upsert.assert_not_called()

    def test_batch_rejects_wrong_vector_dimension(self, client, mock_qdrant):
        states = [_make_state(), _make_state(vector=[1.0])]
        with pytest.raises(ValueError, match="dimension"):
            client.insert_batch(states)
        mock_qdrant.upsert.assert_not_called()

    def test_does_not_mutate_input(self, client, mock_qdrant):
        state = _make_state()
        original_id = state.id
        client.insert(state)
        assert state.id == original_id  # should not be modified

    def test_upserts_to_correct_collection(self, client, mock_qdrant):
        # timestamp 10_000 with epoch_size 5000 → epoch 2 → "loci_2"
        state = _make_state(timestamp_ms=10_000)
        client.insert(state)

        upsert_call = mock_qdrant.upsert.call_args
        assert upsert_call.kwargs["collection_name"] == "loci_2"

    def test_payload_includes_hilbert_ids(self, client, mock_qdrant):
        state = _make_state()
        client.insert(state)

        point = mock_qdrant.upsert.call_args.kwargs["points"][0]
        assert "hilbert_r4" in point.payload
        assert isinstance(point.payload["hilbert_r4"], int)
        assert "hilbert_r8" in point.payload
        assert "hilbert_r12" in point.payload

    def test_payload_stores_all_fields(self, client, mock_qdrant):
        state = _make_state(scene_id="s1", scale_level="frame", confidence=0.9)
        client.insert(state)

        payload = mock_qdrant.upsert.call_args.kwargs["points"][0].payload
        assert payload["scene_id"] == "s1"
        assert payload["scale_level"] == "frame"
        assert payload["confidence"] == 0.9
        assert payload["x"] == 0.5
        assert payload["y"] == 0.5
        assert payload["z"] == 0.5

    def test_find_latest_predecessor_paginates_scroll_results(self, client, mock_qdrant):
        client._known_collections = {"loci_0"}
        client._discovered = True

        def _point(i: int):
            point = MagicMock()
            point.id = f"p{i}"
            point.payload = {"timestamp_ms": i}
            return point

        page_1 = [_point(i) for i in range(256)]
        page_2 = [_point(i) for i in range(256, 300)]

        mock_qdrant.scroll.side_effect = [
            (page_1, "page-2"),
            (page_2, None),
        ]

        predecessor = client._find_latest_predecessor("scene_a", 20_000)

        assert predecessor == ("p299", "loci_0")
        assert mock_qdrant.scroll.call_args_list[1].kwargs["offset"] == "page-2"

    def test_find_latest_predecessor_unordered_pages(self, client, mock_qdrant):
        """The latest predecessor is chosen by timestamp, not page position."""
        client._known_collections = {"loci_0"}
        client._discovered = True

        def _point(pid: str, ts: int):
            point = MagicMock()
            point.id = pid
            point.payload = {"timestamp_ms": ts}
            return point

        # Unordered scroll: latest timestamp arrives mid-page.
        points = [_point("a", 100), _point("latest", 900), _point("b", 500)]
        mock_qdrant.scroll.side_effect = [(points, None)]

        predecessor = client._find_latest_predecessor("scene_a", 20_000)

        assert predecessor == ("latest", "loci_0")
        # Ordered scrolls break Qdrant pagination; must scroll unordered.
        assert mock_qdrant.scroll.call_args.kwargs.get("order_by") is None


# ---------------------------------------------------------------------------
# insert_batch
# ---------------------------------------------------------------------------


class TestInsertBatch:
    def test_returns_correct_count(self, client, mock_qdrant):
        states = [_make_state(timestamp_ms=10_000 + i * 50) for i in range(5)]
        ids = client.insert_batch(states)
        assert len(ids) == 5
        assert len(set(ids)) == 5  # all unique

    def test_groups_by_epoch(self, client, mock_qdrant):
        # epoch_size=5000: ts 3000 → epoch 0, ts 8000 → epoch 1
        states = [
            _make_state(timestamp_ms=3000),
            _make_state(timestamp_ms=8000),
        ]
        client.insert_batch(states)

        upsert_calls = mock_qdrant.upsert.call_args_list
        collections = {c.kwargs["collection_name"] for c in upsert_calls}
        assert collections == {"loci_0", "loci_1"}

    def test_single_upsert_per_epoch(self, client, mock_qdrant):
        # All in same epoch → exactly one upsert call
        states = [_make_state(timestamp_ms=10_000 + i) for i in range(10)]
        client.insert_batch(states)

        # Filter to only upsert calls for loci_2
        upsert_calls = [
            c for c in mock_qdrant.upsert.call_args_list if c.kwargs["collection_name"] == "loci_2"
        ]
        assert len(upsert_calls) == 1
        assert len(upsert_calls[0].kwargs["points"]) == 10

    def test_does_not_mutate_inputs(self, client, mock_qdrant):
        states = [_make_state(timestamp_ms=10_000 + i) for i in range(3)]
        original_ids = [s.id for s in states]
        client.insert_batch(states)
        assert [s.id for s in states] == original_ids


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_returns_empty_when_no_collections(self, client, mock_qdrant):
        results = client.query(vector=[1.0, 2.0, 3.0, 4.0])
        assert results == []

    def test_returns_world_states_with_vectors(self, client, mock_qdrant):
        # Insert first so the collection is known
        client.insert(_make_state())

        hit = MagicMock()
        hit.score = 0.95
        hit.id = "abc123"
        hit.vector = [1.0, 2.0, 3.0, 4.0]
        hit.payload = {
            "x": 0.5,
            "y": 0.5,
            "z": 0.5,
            "timestamp_ms": 10_000,
            "scene_id": "s1",
            "scale_level": "patch",
            "confidence": 1.0,
        }
        qr = MagicMock()
        qr.points = [hit]
        mock_qdrant.query_points.return_value = qr

        results = client.query(
            vector=[1.0, 2.0, 3.0, 4.0],
            time_window_ms=(10_000, 15_000),
        )
        assert len(results) == 1
        assert results[0].vector == [1.0, 2.0, 3.0, 4.0]
        assert results[0].id == "abc123"

    def test_search_called_with_vectors_flag(self, client, mock_qdrant):
        client.insert(_make_state())

        client.query(
            vector=[1.0, 2.0, 3.0, 4.0],
            time_window_ms=(10_000, 15_000),
        )

        search_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert search_kwargs.get("with_vectors") is True

    def test_spatial_filter_uses_match_any(self, client, mock_qdrant):
        client.insert(_make_state())

        client.query(
            vector=[1.0, 2.0, 3.0, 4.0],
            spatial_bounds={
                "x_min": 0.2,
                "x_max": 0.8,
                "y_min": 0.2,
                "y_max": 0.8,
                "z_min": 0.0,
                "z_max": 1.0,
            },
            time_window_ms=(10_000, 15_000),
        )

        search_kwargs = mock_qdrant.query_points.call_args.kwargs
        filt = search_kwargs["query_filter"]
        assert filt is not None
        # Should have at least a hilbert_id MatchAny and a timestamp Range
        assert len(filt.must) >= 2

    def test_spatial_query_applies_exact_post_filter(self, client, mock_qdrant):
        client.insert(_make_state())

        inside = MagicMock()
        inside.score = 0.8
        inside.id = "inside"
        inside.vector = [1.0, 0.0, 0.0, 0.0]
        inside.payload = {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
            "timestamp_ms": 10_000,
            "scene_id": "s1",
            "scale_level": "patch",
            "confidence": 1.0,
        }

        outside = MagicMock()
        outside.score = 0.99
        outside.id = "outside"
        outside.vector = [1.0, 0.0, 0.0, 0.0]
        outside.payload = {
            "x": 0.08,
            "y": 0.08,
            "z": 0.08,
            "timestamp_ms": 10_000,
            "scene_id": "s1",
            "scale_level": "patch",
            "confidence": 1.0,
        }

        qr = MagicMock()
        qr.points = [outside, inside]
        mock_qdrant.query_points.return_value = qr

        results = client.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            spatial_bounds={
                "x_min": 0.0,
                "x_max": 0.03,
                "y_min": 0.0,
                "y_max": 0.03,
                "z_min": 0.0,
                "z_max": 0.03,
            },
            time_window_ms=(9_000, 11_000),
            limit=5,
        )

        assert [result.id for result in results] == ["inside"]

    def test_adaptive_query_uses_finer_hilbert_field(self, mock_qdrant):
        from loci.spatial.adaptive import AdaptiveResolution

        client = LociClient(
            qdrant_url="http://fake:6333",
            epoch_size_ms=5000,
            spatial_resolution=4,
            vector_size=4,
            decay_lambda=0.0,
            adaptive=True,
        )
        client._adaptive = AdaptiveResolution(base_order=4, max_order=12, density_threshold=3)

        for i in range(10):
            client.insert(_make_state(timestamp_ms=1000 + i * 10))

        mock_qdrant.query_points.reset_mock()
        client.query(
            vector=[1.0, 2.0, 3.0, 4.0],
            spatial_bounds={
                "x_min": 0.49,
                "x_max": 0.51,
                "y_min": 0.49,
                "y_max": 0.51,
                "z_min": 0.49,
                "z_max": 0.51,
            },
            time_window_ms=(1000, 1100),
            limit=5,
        )

        filt = mock_qdrant.query_points.call_args.kwargs["query_filter"]
        keys = {condition.key for condition in filt.must}
        assert "hilbert_r8" in keys

    def test_query_scored_returns_scores(self, client, mock_qdrant):
        client.insert(_make_state())

        hit = MagicMock()
        hit.score = 0.42
        hit.id = "abc123"
        hit.vector = [1.0, 2.0, 3.0, 4.0]
        hit.payload = {
            "x": 0.5,
            "y": 0.5,
            "z": 0.5,
            "timestamp_ms": 10_000,
            "scene_id": "s1",
            "scale_level": "patch",
            "confidence": 1.0,
        }
        qr = MagicMock()
        qr.points = [hit]
        mock_qdrant.query_points.return_value = qr

        results = client.query_scored(
            vector=[1.0, 2.0, 3.0, 4.0],
            time_window_ms=(10_000, 15_000),
        )

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.42)
        assert results[0].decayed_score == pytest.approx(0.42)
        assert results[0].state.id == "abc123"


# ---------------------------------------------------------------------------
# Collection discovery
# ---------------------------------------------------------------------------


def _mock_collections_response(names: list[str]) -> MagicMock:
    response = MagicMock()
    cols = []
    for name in names:
        col = MagicMock()
        col.name = name
        cols.append(col)
    response.collections = cols
    return response


class TestDiscovery:
    def test_insert_then_query_sees_preexisting_collections(self, client, mock_qdrant):
        """_ensure_collection populating the cache must not suppress discovery."""
        mock_qdrant.get_collections.return_value = _mock_collections_response(["loci_5"])

        client.insert(_make_state(timestamp_ms=10_000))  # creates loci_2
        client.query(vector=[1.0, 2.0, 3.0, 4.0], time_window_ms=(25_000, 29_999))

        assert "loci_5" in client._known_collections
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert "loci_5" in searched

    def test_rediscovers_when_window_epoch_unknown(self, client, mock_qdrant):
        """A query for an unknown epoch re-runs discovery for that call."""
        mock_qdrant.get_collections.return_value = _mock_collections_response(["loci_2"])
        client.query(vector=[1.0, 2.0, 3.0, 4.0], time_window_ms=(10_000, 14_999))
        assert mock_qdrant.get_collections.call_count == 1

        # Another writer creates loci_9; a query covering epoch 9 must find it.
        mock_qdrant.get_collections.return_value = _mock_collections_response(["loci_2", "loci_9"])
        client.query(vector=[1.0, 2.0, 3.0, 4.0], time_window_ms=(45_000, 49_999))

        assert mock_qdrant.get_collections.call_count == 2
        searched = {c.kwargs["collection_name"] for c in mock_qdrant.query_points.call_args_list}
        assert "loci_9" in searched

    def test_discovery_merges_instead_of_replacing(self, client, mock_qdrant):
        client._known_collections = {"loci_1"}
        client._discovered = False
        mock_qdrant.get_collections.return_value = _mock_collections_response(["loci_3"])

        client._discover_collections()

        assert client._known_collections == {"loci_1", "loci_3"}


# ---------------------------------------------------------------------------
# min_confidence
# ---------------------------------------------------------------------------


class TestMinConfidence:
    def _hit(self, i: int, confidence: float = 1.0) -> MagicMock:
        hit = MagicMock()
        hit.score = 0.9 - i * 0.01
        hit.id = f"hit{i}"
        hit.vector = [1.0, 0.0, 0.0, 0.0]
        hit.payload = {
            "x": 0.5,
            "y": 0.5,
            "z": 0.5,
            "timestamp_ms": 10_000,
            "scene_id": "s1",
            "scale_level": "patch",
            "confidence": confidence,
        }
        return hit

    def test_min_confidence_pushed_down_and_overfetched(self, client, mock_qdrant):
        client.insert(_make_state())

        qr = MagicMock()
        qr.points = [self._hit(i) for i in range(10)]
        mock_qdrant.query_points.return_value = qr

        results = client.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            time_window_ms=(10_000, 14_999),
            limit=5,
            min_confidence=0.5,
        )

        # More qualifying matches than limit → the full limit is returned.
        assert len(results) == 5

        kwargs = mock_qdrant.query_points.call_args.kwargs
        # Overfetch beyond limit so the post-filter cannot starve results.
        assert kwargs["limit"] == 15
        conf_conditions = [
            c for c in kwargs["query_filter"].must if getattr(c, "key", None) == "confidence"
        ]
        assert len(conf_conditions) == 1
        assert conf_conditions[0].range.gte == 0.5

    def test_min_confidence_filters_low_confidence(self, client, mock_qdrant):
        client.insert(_make_state())

        qr = MagicMock()
        qr.points = [self._hit(0, confidence=0.2), self._hit(1, confidence=0.9)]
        mock_qdrant.query_points.return_value = qr

        results = client.query(
            vector=[1.0, 0.0, 0.0, 0.0],
            time_window_ms=(10_000, 14_999),
            limit=5,
            min_confidence=0.5,
        )
        assert [r.id for r in results] == ["hit1"]


# ---------------------------------------------------------------------------
# Decay-aware overfetch
# ---------------------------------------------------------------------------


class TestDecayOverfetch:
    def test_decay_active_triggers_overfetch(self, mock_qdrant):
        client = LociClient(
            qdrant_url="http://fake:6333",
            epoch_size_ms=5000,
            vector_size=4,
            decay_lambda=1e-3,
        )
        client.insert(_make_state())
        client.query(vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(10_000, 14_999), limit=5)
        assert mock_qdrant.query_points.call_args.kwargs["limit"] == 15

    def test_no_overfetch_without_filters_or_decay(self, client, mock_qdrant):
        client.insert(_make_state())
        client.query(vector=[1.0, 0.0, 0.0, 0.0], time_window_ms=(10_000, 14_999), limit=5)
        assert mock_qdrant.query_points.call_args.kwargs["limit"] == 5


# ---------------------------------------------------------------------------
# Trajectory anchor ID normalisation
# ---------------------------------------------------------------------------


class TestTrajectoryIdNormalisation:
    def test_hyphenless_anchor_matches_hyphenated_server_ids(self, client, mock_qdrant):
        """Real Qdrant returns canonical hyphenated UUIDs; hex IDs must match."""
        canonical = "0e864cb1-9b3c-4713-9f44-0a3a4e0e6f13"
        hyphenless = canonical.replace("-", "")

        def _pt(pid: str, ts: int):
            point = MagicMock()
            point.id = pid
            point.vector = [1.0, 0.0, 0.0, 0.0]
            point.payload = {
                "x": 0.5,
                "y": 0.5,
                "z": 0.5,
                "timestamp_ms": ts,
                "scene_id": "s1",
                "scale_level": "patch",
                "confidence": 1.0,
            }
            return point

        client._known_collections = {"loci_2"}
        client._discovered = True
        mock_qdrant.retrieve.return_value = [_pt(canonical, 10_050)]
        mock_qdrant.scroll.return_value = (
            [_pt("11111111-2222-3333-4444-555555555555", 10_000), _pt(canonical, 10_050)],
            None,
        )

        traj = client.get_trajectory(hyphenless, steps_back=1, steps_forward=1)

        # Anchor matched → both states returned, not just the anchor fallback.
        assert len(traj) == 2


# ---------------------------------------------------------------------------
# Shard failure logging
# ---------------------------------------------------------------------------


class TestShardFailureLogging:
    def test_partial_shard_failure_logged_at_warning(self, client, mock_qdrant, caplog):
        client.insert(_make_state(timestamp_ms=3000))  # loci_0
        client.insert(_make_state(timestamp_ms=8000))  # loci_1

        good = MagicMock()
        good.points = []
        mock_qdrant.query_points.side_effect = [RuntimeError("shard down"), good]

        import logging

        with caplog.at_level(logging.WARNING, logger="loci.client"):
            results = client.query(vector=[1.0, 2.0, 3.0, 4.0], time_window_ms=(0, 9_999))

        assert results == []
        assert any(
            "loci_0" in rec.message and "shard down" in rec.message for rec in caplog.records
        )

    def test_all_shards_failed_summary_warning(self, client, mock_qdrant, caplog):
        client.insert(_make_state(timestamp_ms=3000))
        client.insert(_make_state(timestamp_ms=8000))
        mock_qdrant.query_points.side_effect = RuntimeError("shard down")

        import logging

        with caplog.at_level(logging.WARNING, logger="loci.client"):
            results = client.query(vector=[1.0, 2.0, 3.0, 4.0], time_window_ms=(0, 9_999))

        assert results == []
        assert any("All 2 shard searches failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# close() / context manager
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_delegates_to_qdrant(self, client, mock_qdrant):
        client.close()
        mock_qdrant.close.assert_called_once()

    def test_context_manager_closes(self, mock_qdrant):
        with LociClient(qdrant_url="http://fake:6333", vector_size=4) as client:
            assert isinstance(client, LociClient)
        mock_qdrant.close.assert_called_once()


# ---------------------------------------------------------------------------
# Retention cache invalidation
# ---------------------------------------------------------------------------


class TestRetentionCache:
    def test_purged_collections_forgotten_and_recreatable(self, mock_qdrant):
        from loci.temporal.retention import RetentionPolicy

        client = LociClient(
            qdrant_url="http://fake:6333",
            epoch_size_ms=5000,
            vector_size=4,
            decay_lambda=0.0,
            retention_policy=RetentionPolicy(max_epochs=1),
        )
        client.insert(_make_state(timestamp_ms=1000))  # loci_0
        client.insert(_make_state(timestamp_ms=6000))  # loci_1 → loci_0 purged

        assert "loci_0" not in client._known_collections
        assert "loci_0" not in client._collection_locks

        # A late insert into the purged epoch recreates the collection.
        mock_qdrant.create_collection.reset_mock()
        client.insert(_make_state(timestamp_ms=2000))
        created = {
            c.kwargs["collection_name"] for c in mock_qdrant.create_collection.call_args_list
        }
        assert "loci_0" in created


# ---------------------------------------------------------------------------
# spatial_resolution constructor parameter
# ---------------------------------------------------------------------------


class TestSpatialResolution:
    def test_spatial_resolution_used_for_payload_keys(self, mock_qdrant):
        client = LociClient(
            qdrant_url="http://fake:6333",
            vector_size=4,
            spatial_resolution=6,
        )
        client.insert(_make_state())
        payload = mock_qdrant.upsert.call_args.kwargs["points"][0].payload
        assert "hilbert_r6" in payload
        assert "hilbert_r4" not in payload

    def test_explicit_resolutions_win(self, mock_qdrant):
        client = LociClient(
            qdrant_url="http://fake:6333",
            vector_size=4,
            spatial_resolution=6,
            resolutions=[5, 9],
        )
        assert client._hilbert.resolutions == [5, 9]


# ---------------------------------------------------------------------------
# predict_and_retrieve
# ---------------------------------------------------------------------------


class TestPredictAndRetrieve:
    def test_uses_predicted_vector(self, client, mock_qdrant):
        # Insert a state at "now" so the future-horizon collection is known
        import time as _time

        now_ms = int(_time.time() * 1000)
        client.insert(_make_state(timestamp_ms=now_ms))
        predicted = [9.0, 8.0, 7.0, 6.0]
        predictor = MagicMock(return_value=predicted)

        client.predict_and_retrieve(
            context_vector=[1.0, 2.0, 3.0, 4.0],
            predictor_fn=predictor,
            future_horizon_ms=2000,
        )

        predictor.assert_called_once_with([1.0, 2.0, 3.0, 4.0])
        # query_points should have been called with the predicted vector
        assert mock_qdrant.query_points.called
        search_kwargs = mock_qdrant.query_points.call_args.kwargs
        assert search_kwargs["query"] == predicted

    def test_extended_path_uses_real_scores_instead_of_rank_proxy(self, client):
        low = _make_state(timestamp_ms=10_500)
        low.id = "low"
        high = _make_state(timestamp_ms=10_500)
        high.id = "high"

        client.query_scored = MagicMock(
            return_value=[
                ScoredWorldState(state=low, score=0.1, decayed_score=0.1),
                ScoredWorldState(state=high, score=0.9, decayed_score=0.9),
            ]
        )

        with patch("loci.retrieval.predict.time.time", return_value=10.0):
            result = client.predict_and_retrieve(
                context_vector=[1.0, 2.0, 3.0, 4.0],
                predictor_fn=lambda _: [9.0, 8.0, 7.0, 6.0],
                future_horizon_ms=2000,
                limit=2,
                current_position=(0.5, 0.5, 0.5),
            )

        assert isinstance(result, PredictRetrieveResult)
        assert [state.id for state in result.results] == ["high", "low"]


# ---------------------------------------------------------------------------
# Bounding-box quantisation consistency
# ---------------------------------------------------------------------------


class TestBoundingBoxConsistency:
    def test_point_inside_box_is_found(self):
        """A point encoded with round() must be in the Hilbert IDs
        produced by expand_bounding_box for a box that contains it."""
        from loci.spatial.buckets import expand_bounding_box
        from loci.spatial.hilbert import encode

        # The bug: x=0.53, side=15 → round(7.95)=8 but int(7.95)=7
        x, y, z, t = 0.53, 0.5, 0.5, 0.5
        hid = encode(x, y, z, t, resolution_order=4)

        box_ids = expand_bounding_box(
            0.0,
            0.53,  # x range includes the point
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            resolution_order=4,
        )
        assert hid in box_ids, (
            f"Hilbert ID {hid} for point ({x},{y},{z},{t}) not found "
            f"in bounding box IDs (range: {min(box_ids)}-{max(box_ids)})"
        )
