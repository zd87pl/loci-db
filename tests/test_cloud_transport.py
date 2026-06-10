"""Unit tests for the cloud HTTP transport helpers and CloudTransport.

These tests are network-free: the sync ``_request`` path mocks
``urllib.request.urlopen`` so no socket is ever opened, and the payload/parse
helpers are pure functions exercised directly. The async (httpx) path is not
touched here.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from loci.cloud_transport import (
    CloudTransport,
    _insert_payload,
    _parse_query_results,
    _query_payload,
)
from loci.schema import WorldState

# ── _insert_payload ──────────────────────────────────────────────────────────


def test_insert_payload_has_all_eight_keys():
    state = WorldState(
        x=0.1,
        y=0.2,
        z=0.3,
        timestamp_ms=1000,
        vector=[0.0, 1.0, 2.0],
        scene_id="scene-a",
        scale_level="frame",
        confidence=0.5,
    )
    payload = _insert_payload(state)

    assert set(payload) == {
        "x",
        "y",
        "z",
        "timestamp_ms",
        "vector",
        "scene_id",
        "scale_level",
        "confidence",
    }
    assert payload["x"] == 0.1
    assert payload["y"] == 0.2
    assert payload["z"] == 0.3
    assert payload["timestamp_ms"] == 1000
    assert payload["vector"] == [0.0, 1.0, 2.0]
    assert payload["scene_id"] == "scene-a"
    assert payload["scale_level"] == "frame"
    assert payload["confidence"] == 0.5


# ── _query_payload ───────────────────────────────────────────────────────────


def test_query_payload_without_spatial_bounds():
    body = _query_payload([0.1, 0.2], None, None, limit=5, overlap_factor=1.2)

    assert body["vector"] == [0.1, 0.2]
    assert body["limit"] == 5
    assert body["overlap_factor"] == 1.2
    for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        assert key not in body
    assert "time_start_ms" not in body
    assert "time_end_ms" not in body


def test_query_payload_with_spatial_bounds():
    bounds = {
        "x_min": 0.1,
        "x_max": 0.9,
        "y_min": 0.2,
        "y_max": 0.8,
        "z_min": 0.3,
        "z_max": 0.7,
    }
    body = _query_payload([0.0], bounds, None, limit=3, overlap_factor=1.0)

    assert body["x_min"] == 0.1
    assert body["x_max"] == 0.9
    assert body["y_min"] == 0.2
    assert body["y_max"] == 0.8
    assert body["z_min"] == 0.3
    assert body["z_max"] == 0.7


def test_query_payload_with_time_window():
    body = _query_payload([0.0], None, (1000, 2000), limit=3, overlap_factor=1.0)

    assert body["time_start_ms"] == 1000
    assert body["time_end_ms"] == 2000


def test_query_payload_include_vectors_false_adds_flag():
    body = _query_payload([0.0], None, None, limit=3, overlap_factor=1.0, include_vectors=False)
    assert body["include_vectors"] is False


def test_query_payload_include_vectors_default_omits_flag():
    body = _query_payload([0.0], None, None, limit=3, overlap_factor=1.0)
    assert "include_vectors" not in body


# ── _parse_query_results ─────────────────────────────────────────────────────


def test_parse_query_results_with_vector_populates_world_state():
    payload = {
        "results": [
            {
                "id": "p1",
                "x": 0.5,
                "y": 0.5,
                "z": 0.5,
                "timestamp_ms": 123,
                "scene_id": "s",
                "vector": [0.1, 0.2, 0.3],
            }
        ]
    }
    results = _parse_query_results(payload)

    assert len(results) == 1
    assert results[0].vector == [0.1, 0.2, 0.3]
    assert results[0].id == "p1"
    assert results[0].scene_id == "s"


def test_parse_query_results_without_vector_yields_empty_list():
    payload = {
        "results": [
            {
                "id": "p2",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
                "timestamp_ms": 456,
                "scene_id": "s",
            }
        ]
    }
    results = _parse_query_results(payload)

    assert len(results) == 1
    assert results[0].vector == []


def test_parse_query_results_missing_scene_id_defaults_empty():
    payload = {
        "results": [
            {
                "id": "p3",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
                "timestamp_ms": 789,
            }
        ]
    }
    results = _parse_query_results(payload)

    assert len(results) == 1
    assert results[0].scene_id == ""


# ── CloudTransport._request scheme validation ────────────────────────────────


def test_request_rejects_non_http_scheme():
    transport = CloudTransport(base_url="ftp://example.com", api_key="loci_x")
    with pytest.raises(ValueError, match="base_url must be http"):
        transport._request("POST", "/insert", {"x": 0.0})


# ── CloudTransport.insert / query over a mocked urlopen ──────────────────────


def _fake_urlopen(response_body: dict):
    """Build a context-manager stand-in for urllib.request.urlopen."""

    class _FakeResp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *exc):
            return False

        def read(self_inner):
            return json.dumps(response_body).encode()

    return lambda req, timeout=None: _FakeResp()


def test_insert_returns_str_id_over_mocked_urlopen():
    transport = CloudTransport(base_url="https://api.example.com", api_key="loci_x")
    state = WorldState(x=0.1, y=0.2, z=0.3, timestamp_ms=1000, vector=[0.0, 1.0], scene_id="s")

    with patch("urllib.request.urlopen", _fake_urlopen({"id": 42})):
        result_id = transport.insert(state)

    assert result_id == "42"
    assert isinstance(result_id, str)


def test_query_parses_results_over_mocked_urlopen():
    transport = CloudTransport(base_url="https://api.example.com", api_key="loci_x")
    body = {
        "results": [
            {
                "id": "pA",
                "x": 0.5,
                "y": 0.5,
                "z": 0.5,
                "timestamp_ms": 123,
                "scene_id": "s",
                "vector": [0.4, 0.5],
            }
        ]
    }

    with patch("urllib.request.urlopen", _fake_urlopen(body)):
        hits = transport.query(vector=[0.0, 1.0], limit=1)

    assert len(hits) == 1
    assert hits[0].id == "pA"
    assert hits[0].x == 0.5
    assert hits[0].vector == [0.4, 0.5]
