"""Integration tests for EngramClient with a mocked Qdrant backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from engram.client import EngramClient
from engram.schema import WorldState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_qdrant():
    """Patch QdrantClient so no real Qdrant is needed."""
    with patch("engram.client.QdrantClient") as MockCls:
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
    return EngramClient(
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
        client._ensure_collection("engram_0")

        mock_qdrant.create_collection.assert_called_once()
        name_arg = mock_qdrant.create_collection.call_args.kwargs["collection_name"]
        assert name_arg == "engram_0"

    def test_idempotent_after_first_call(self, client, mock_qdrant):
        """Second call should not hit Qdrant again."""
        client._ensure_collection("engram_0")
        mock_qdrant.reset_mock()

        client._ensure_collection("engram_0")
        mock_qdrant.get_collection.assert_not_called()
        mock_qdrant.create_collection.assert_not_called()

    def test_skips_create_when_collection_exists(self, mock_qdrant):
        """If get_collection succeeds, don't create."""
        mock_qdrant.get_collection.side_effect = None
        mock_qdrant.get_collection.return_value = MagicMock()

        c = EngramClient.__new__(EngramClient)
        c._qdrant = mock_qdrant
        c._vector_size = 4
        c._known_collections = set()

        c._ensure_collection("engram_0")
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
        c = EngramClient.__new__(EngramClient)
        c._qdrant = mock_qdrant
        c._vector_size = 4
        c._known_collections = set()

        with pytest.raises(UnexpectedResponse):
            c._ensure_collection("engram_0")

    def test_creates_scale_level_index(self, client, mock_qdrant):
        """scale_level should get a KEYWORD payload index."""
        client._ensure_collection("engram_0")

        index_calls = mock_qdrant.create_payload_index.call_args_list
        field_names = [c.kwargs["field_name"] for c in index_calls]
        assert "scale_level" in field_names


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


class TestInsert:
    def test_returns_id(self, client, mock_qdrant):
        state = _make_state()
        result = client.insert(state)
        assert isinstance(result, str)
        assert len(result) == 32  # uuid4 hex

    def test_does_not_mutate_input(self, client, mock_qdrant):
        state = _make_state()
        original_id = state.id
        client.insert(state)
        assert state.id == original_id  # should not be modified

    def test_upserts_to_correct_collection(self, client, mock_qdrant):
        # timestamp 10_000 with epoch_size 5000 → epoch 2 → "engram_2"
        state = _make_state(timestamp_ms=10_000)
        client.insert(state)

        upsert_call = mock_qdrant.upsert.call_args
        assert upsert_call.kwargs["collection_name"] == "engram_2"

    def test_payload_includes_hilbert_id(self, client, mock_qdrant):
        state = _make_state()
        client.insert(state)

        point = mock_qdrant.upsert.call_args.kwargs["points"][0]
        assert "hilbert_id" in point.payload
        assert isinstance(point.payload["hilbert_id"], int)

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
        assert collections == {"engram_0", "engram_1"}

    def test_single_upsert_per_epoch(self, client, mock_qdrant):
        # All in same epoch → exactly one upsert call
        states = [_make_state(timestamp_ms=10_000 + i) for i in range(10)]
        client.insert_batch(states)

        # Filter to only upsert calls for engram_2
        upsert_calls = [
            c
            for c in mock_qdrant.upsert.call_args_list
            if c.kwargs["collection_name"] == "engram_2"
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


# ---------------------------------------------------------------------------
# Bounding-box quantisation consistency
# ---------------------------------------------------------------------------


class TestBoundingBoxConsistency:
    def test_point_inside_box_is_found(self):
        """A point encoded with round() must be in the Hilbert IDs
        produced by expand_bounding_box for a box that contains it."""
        from engram.spatial.buckets import expand_bounding_box
        from engram.spatial.hilbert import encode

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
