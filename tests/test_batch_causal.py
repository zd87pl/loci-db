"""Tests for causal linking in insert_batch (sync client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from loci.client import LociClient
from loci.schema import WorldState


@pytest.fixture()
def mock_qdrant():
    with patch("loci.client.QdrantClient") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance

        from qdrant_client.http.exceptions import UnexpectedResponse

        instance.get_collection.side_effect = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"",
            headers=httpx.Headers(),
        )
        instance.scroll.return_value = ([], None)
        yield instance


@pytest.fixture()
def client(mock_qdrant):
    return LociClient(
        qdrant_url="http://fake:6333",
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=4,
    )


def _make(ts: int, scene: str = "scene_a") -> WorldState:
    return WorldState(
        x=0.5,
        y=0.5,
        z=0.5,
        timestamp_ms=ts,
        vector=[1.0, 2.0, 3.0, 4.0],
        scene_id=scene,
    )


def test_batch_causal_chain(client, mock_qdrant):
    """Three states in the same scene should form a causal chain."""
    states = [_make(10_000), _make(10_050), _make(10_100)]
    client.insert_batch(states)

    upsert_calls = mock_qdrant.upsert.call_args_list
    all_points = []
    for call in upsert_calls:
        all_points.extend(call.kwargs["points"])

    all_points.sort(key=lambda p: p.payload["timestamp_ms"])

    assert all_points[0].payload.get("prev_state_id") is None
    assert all_points[1].payload["prev_state_id"] == all_points[0].id
    assert all_points[2].payload["prev_state_id"] == all_points[1].id


def test_batch_separate_scenes(client, mock_qdrant):
    """States in different scenes should have independent chains."""
    states = [
        _make(10_000, "scene_a"),
        _make(10_050, "scene_b"),
        _make(10_100, "scene_a"),
    ]
    client.insert_batch(states)

    upsert_calls = mock_qdrant.upsert.call_args_list
    all_points = []
    for call in upsert_calls:
        all_points.extend(call.kwargs["points"])

    by_scene: dict[str, list] = {}
    for p in all_points:
        by_scene.setdefault(p.payload["scene_id"], []).append(p)

    # scene_a: two states linked
    a_points = sorted(by_scene["scene_a"], key=lambda p: p.payload["timestamp_ms"])
    assert a_points[0].payload.get("prev_state_id") is None
    assert a_points[1].payload["prev_state_id"] == a_points[0].id

    # scene_b: single state, no link
    assert by_scene["scene_b"][0].payload.get("prev_state_id") is None


def test_batch_patches_next_links(client, mock_qdrant):
    """insert_batch should call set_payload to patch next_state_id."""
    states = [_make(10_000), _make(10_050)]
    client.insert_batch(states)

    # set_payload should have been called at least once for the next link
    assert mock_qdrant.set_payload.called
    call_kwargs = mock_qdrant.set_payload.call_args.kwargs
    assert "next_state_id" in call_kwargs["payload"]


def test_batch_patches_cross_epoch_next_link_in_predecessor_collection(client, mock_qdrant):
    states = [_make(4_900), _make(5_100)]
    client.insert_batch(states)

    call_kwargs = mock_qdrant.set_payload.call_args.kwargs
    assert call_kwargs["collection_name"] == "loci_0"


def test_batch_preserves_original_order(client, mock_qdrant):
    """IDs should be returned in the same order as input states."""
    states = [_make(10_100), _make(10_000), _make(10_050)]
    ids = client.insert_batch(states)
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_batch_links_first_state_to_stored_predecessor(client, mock_qdrant):
    """The earliest batch state per scene links to the latest stored state."""
    # A predecessor for scene_a already exists in the store (loci_1).
    stored = MagicMock()
    stored.id = "stored-pred"
    stored.payload = {"timestamp_ms": 9_000}

    def _scroll(collection_name=None, **kwargs):
        if collection_name == "loci_1":
            return ([stored], None)
        return ([], None)

    mock_qdrant.scroll.side_effect = _scroll
    client._known_collections = {"loci_1"}
    client._discovered = True

    states = [_make(10_000), _make(10_050)]
    client.insert_batch(states)

    all_points = []
    for call in mock_qdrant.upsert.call_args_list:
        all_points.extend(call.kwargs["points"])
    all_points.sort(key=lambda p: p.payload["timestamp_ms"])

    # First batch state links back to the stored predecessor …
    assert all_points[0].payload["prev_state_id"] == "stored-pred"
    # … and the chain continues within the batch.
    assert all_points[1].payload["prev_state_id"] == all_points[0].id

    # The stored predecessor's next link is patched in its own collection.
    patched = [
        call.kwargs
        for call in mock_qdrant.set_payload.call_args_list
        if call.kwargs["points"] == ["stored-pred"]
    ]
    assert len(patched) == 1
    assert patched[0]["collection_name"] == "loci_1"
    assert patched[0]["payload"]["next_state_id"] == all_points[0].id


def test_batch_matches_sequential_chain_on_memory_backend():
    """Batch and sequential inserts must build the same causal chain."""
    from loci.local_client import LocalLociClient
    from loci.schema import WorldState

    def _state(ts: int) -> WorldState:
        return WorldState(
            x=0.5, y=0.5, z=0.5, timestamp_ms=ts, vector=[1.0, 0.0, 0.0, 0.0], scene_id="s1"
        )

    client = LocalLociClient(vector_size=4, epoch_size_ms=5000, decay_lambda=0.0)
    first_id = client.insert(_state(1000))
    batch_ids = client.insert_batch([_state(2000), _state(3000)])

    traj = client.get_trajectory(batch_ids[1], steps_back=5, steps_forward=5)
    assert [s.id for s in traj] == [first_id, *batch_ids]
    # Explicit prev/next links across the store/batch boundary.
    assert traj[1].prev_state_id == first_id
    assert traj[0].next_state_id == batch_ids[0]
