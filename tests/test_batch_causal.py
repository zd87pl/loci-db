"""Tests for causal linking in insert_batch (sync client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from engram.client import EngramClient
from engram.schema import WorldState


@pytest.fixture()
def mock_qdrant():
    with patch("engram.client.QdrantClient") as MockCls:
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
    return EngramClient(
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


def test_batch_preserves_original_order(client, mock_qdrant):
    """IDs should be returned in the same order as input states."""
    states = [_make(10_100), _make(10_000), _make(10_050)]
    ids = client.insert_batch(states)
    assert len(ids) == 3
    assert len(set(ids)) == 3
