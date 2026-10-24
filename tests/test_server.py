"""Integration tests for the LOCI REST API server (server.py).

Uses httpx + FastAPI TestClient so no real Qdrant instance is needed — the
LociClient is monkey-patched at the module level before the app is imported.

The root ``server.py`` is loaded by explicit file path under a private module
name: the cloud test suite (cloud/tests/conftest.py) puts ``cloud/api`` on
``sys.path`` where another module named ``server`` lives, so a bare
``import server`` is ambiguous when both suites run in one pytest session.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from loci.schema import ScoredWorldState, WorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_SIZE = 4  # small dimension for tests

_SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
_MODULE_NAME = "loci_root_rest_server"


def _make_client() -> tuple[TestClient, Any, MagicMock]:
    """Return a TestClient with the root server freshly loaded at VECTOR_SIZE=4."""
    mock_loci_client = MagicMock()
    mock_loci_client.return_value.insert.return_value = "test-uuid-1234"
    mock_loci_client.return_value.query.return_value = []
    mock_loci_client.return_value.query_scored.return_value = []

    with (
        patch.dict("os.environ", {"LOCI_VECTOR_SIZE": str(VECTOR_SIZE)}),
        patch("loci.LociClient", mock_loci_client),
    ):
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SERVER_PATH)
        assert spec is not None and spec.loader is not None
        srv = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = srv
        spec.loader.exec_module(srv)
        return TestClient(srv.app), srv, mock_loci_client.return_value


def _insert_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "timestamp_ms": 1000,
        "vector": [0.1, 0.2, 0.3, 0.4],
        "scene_id": "scene-a",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self) -> None:
        client, _, _ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "vector_size" in data

    def test_health_does_not_leak_qdrant_url(self) -> None:
        """Internal connection details must not be echoed to callers."""
        client, _, _ = _make_client()
        data = client.get("/health").json()
        assert "qdrant_url" not in data
        assert not any("qdrant" in str(v) for v in data.values())


class TestGetClientSingleton:
    def test_concurrent_first_calls_build_one_client(self) -> None:
        """The lazy singleton must not build duplicate clients under races."""
        _, srv, _ = _make_client()
        srv._client = None  # force re-initialisation

        built: list[object] = []
        barrier = threading.Barrier(8)

        class _SlowClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                time.sleep(0.05)  # widen the race window
                built.append(self)

        results: list[object] = []
        with patch.object(srv, "LociClient", _SlowClient):

            def _get() -> None:
                barrier.wait()
                results.append(srv.get_client())

            threads = [threading.Thread(target=_get) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(built) == 1
        assert all(r is results[0] for r in results)


class TestInsert:
    def test_insert_valid(self) -> None:
        client, srv, mock = _make_client()
        resp = client.post("/insert", json=_insert_payload())
        assert resp.status_code == 200
        assert resp.json() == {"id": "test-uuid-1234"}
        mock.insert.assert_called_once()

    def test_insert_wrong_vector_size(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/insert", json=_insert_payload(vector=[0.1, 0.2]))
        # 422 for parity with the cloud API (was 400).
        assert resp.status_code == 422
        assert any("dimensions" in str(d) for d in resp.json()["detail"])

    def test_insert_out_of_range_coordinates_rejected_as_422(self) -> None:
        """Coordinates outside WorldState's [0,1] contract must 422, not 500."""
        client, _, _ = _make_client()
        for field in ("x", "y", "z"):
            for bad in (-0.5, 1.5):
                resp = client.post("/insert", json=_insert_payload(**{field: bad}))
                assert resp.status_code == 422, (field, bad, resp.text)

    def test_insert_nan_coordinate_rejected_as_422(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/insert", json=_insert_payload(x="NaN"))
        assert resp.status_code == 422

    def test_insert_negative_timestamp_rejected_as_422(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/insert", json=_insert_payload(timestamp_ms=-1))
        assert resp.status_code == 422

    def test_insert_invalid_scale_level_rejected_as_422(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/insert", json=_insert_payload(scale_level="galaxy"))
        assert resp.status_code == 422

    def test_insert_confidence_out_of_range_rejected_as_422(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/insert", json=_insert_payload(confidence=1.5))
        assert resp.status_code == 422

    def test_insert_confidence_passed_through(self) -> None:
        client, _, mock = _make_client()
        resp = client.post("/insert", json=_insert_payload(confidence=0.25))
        assert resp.status_code == 200
        assert mock.insert.call_args.args[0].confidence == 0.25

    def test_insert_metadata_isolation(self) -> None:
        """Successive requests must not share metadata across instances."""
        client, _, mock = _make_client()
        base = _insert_payload(x=0.0, y=0.0, z=0.0, timestamp_ms=0, vector=[0.0] * VECTOR_SIZE)
        resp1 = client.post("/insert", json={**base, "metadata": {"key": "val"}})
        resp2 = client.post("/insert", json=base)  # no metadata
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert mock.insert.call_args.args[0].metadata == {}

    def test_insert_passes_metadata_to_world_state(self) -> None:
        """Client-supplied metadata must reach the stored WorldState, not be dropped."""
        client, _, mock = _make_client()
        payload = _insert_payload(metadata={"label": "doorway", "source": "lidar"})
        resp = client.post("/insert", json=payload)
        assert resp.status_code == 200
        state = mock.insert.call_args.args[0]
        assert state.metadata == {"label": "doorway", "source": "lidar"}


class TestQuery:
    def test_query_valid(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.1, 0.2, 0.3, 0.4],
            "limit": 5,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 200
        assert "results" in resp.json()

    def test_query_serializes_scored_results(self) -> None:
        client, _, mock = _make_client()
        state = WorldState(
            x=0.1,
            y=0.2,
            z=0.3,
            timestamp_ms=1000,
            vector=[0.1, 0.2, 0.3, 0.4],
            scene_id="scene-a",
            metadata={"label": "doorway"},
            id="state-1",
        )
        mock.query_scored.return_value = [
            ScoredWorldState(state=state, score=0.8, decayed_score=0.7)
        ]

        resp = client.post("/query", json={"vector": [0.1, 0.2, 0.3, 0.4]})

        assert resp.status_code == 200
        assert resp.json()["results"] == [
            {
                "id": "state-1",
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
                "timestamp_ms": 1000,
                "scene_id": "scene-a",
                "metadata": {"label": "doorway"},
                "score": 0.8,
                "decayed_score": 0.7,
            }
        ]

    def test_query_wrong_vector_size(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/query", json={"vector": [0.1, 0.2]})
        # 422 for parity with the cloud API (was 400).
        assert resp.status_code == 422
        assert any("dimensions" in str(d) for d in resp.json()["detail"])

    def test_query_limit_too_large(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "limit": 9999,  # exceeds max of 1000
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_query_limit_zero_invalid(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "limit": 0,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 422

    def test_query_negative_limit_invalid(self) -> None:
        client, _, _ = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "limit": -1,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 422

    # ── Bounds validation ──────────────────────────────────────────────────

    def test_query_nan_bound_rejected(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/query", json={"vector": [0.0] * VECTOR_SIZE, "x_min": "NaN"})
        assert resp.status_code == 422

    def test_query_out_of_range_bound_rejected(self) -> None:
        client, _, _ = _make_client()
        resp = client.post("/query", json={"vector": [0.0] * VECTOR_SIZE, "y_max": 2.0})
        assert resp.status_code == 422

    def test_query_inverted_spatial_bounds_rejected(self) -> None:
        client, _, _ = _make_client()
        for axis in ("x", "y", "z"):
            resp = client.post(
                "/query",
                json={
                    "vector": [0.0] * VECTOR_SIZE,
                    f"{axis}_min": 0.9,
                    f"{axis}_max": 0.1,
                },
            )
            assert resp.status_code == 422, axis

    def test_query_inverted_time_bounds_rejected(self) -> None:
        client, _, _ = _make_client()
        resp = client.post(
            "/query",
            json={"vector": [0.0] * VECTOR_SIZE, "time_start_ms": 1000, "time_end_ms": 500},
        )
        assert resp.status_code == 422

    # ── Time windows (including half-open) ─────────────────────────────────

    def test_query_with_time_window(self) -> None:
        client, _, mock = _make_client()
        payload: dict[str, Any] = {
            "vector": [0.0] * VECTOR_SIZE,
            "time_start_ms": 0,
            "time_end_ms": 5000,
        }
        resp = client.post("/query", json=payload)
        assert resp.status_code == 200
        call_kwargs = mock.query_scored.call_args.kwargs
        assert call_kwargs["time_window_ms"] == (0, 5000)

    def test_query_start_only_time_window_is_half_open(self) -> None:
        """A start without an end must filter, not be silently dropped."""
        client, srv, mock = _make_client()
        resp = client.post(
            "/query",
            json={"vector": [0.0] * VECTOR_SIZE, "time_start_ms": 12345},
        )
        assert resp.status_code == 200
        call_kwargs = mock.query_scored.call_args.kwargs
        assert call_kwargs["time_window_ms"] == (12345, srv._MAX_TIME_MS)

    def test_query_end_only_time_window_is_half_open(self) -> None:
        client, _, mock = _make_client()
        resp = client.post(
            "/query",
            json={"vector": [0.0] * VECTOR_SIZE, "time_end_ms": 5000},
        )
        assert resp.status_code == 200
        call_kwargs = mock.query_scored.call_args.kwargs
        assert call_kwargs["time_window_ms"] == (0, 5000)

    def test_query_no_time_fields_means_no_window(self) -> None:
        client, _, mock = _make_client()
        resp = client.post("/query", json={"vector": [0.0] * VECTOR_SIZE})
        assert resp.status_code == 200
        assert mock.query_scored.call_args.kwargs["time_window_ms"] is None

    # ── Spatial bounds resolution ──────────────────────────────────────────

    def test_query_without_bounds_passes_no_spatial_filter(self) -> None:
        """Omitting all six bounds must reach the client as spatial_bounds=None."""
        client, _, mock = _make_client()
        resp = client.post("/query", json={"vector": [0.0] * VECTOR_SIZE})
        assert resp.status_code == 200
        assert mock.query_scored.call_args.kwargs["spatial_bounds"] is None

    def test_query_explicit_full_unit_box_passes_no_spatial_filter(self) -> None:
        """The full unit box has zero selectivity — treat it as 'no filter'."""
        client, _, mock = _make_client()
        resp = client.post(
            "/query",
            json={
                "vector": [0.0] * VECTOR_SIZE,
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
                "z_min": 0.0,
                "z_max": 1.0,
            },
        )
        assert resp.status_code == 200
        assert mock.query_scored.call_args.kwargs["spatial_bounds"] is None

    def test_query_partial_bounds_filled_with_defaults(self) -> None:
        client, _, mock = _make_client()
        resp = client.post(
            "/query",
            json={"vector": [0.0] * VECTOR_SIZE, "x_min": 0.25, "z_max": 0.75},
        )
        assert resp.status_code == 200
        assert mock.query_scored.call_args.kwargs["spatial_bounds"] == {
            "x_min": 0.25,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 0.75,
        }
