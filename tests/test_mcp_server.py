"""Tests for the Loci MCP server tools (loci/mcp/server.py).

The tool implementations are plain functions, so they are exercised directly
against a local in-memory client — no MCP transport involved.  The MCP SDK is
only required for the ``build_server`` registration tests, which importorskip.
"""

from __future__ import annotations

import time

import pytest

from loci.mcp import server as mcp_server

VEC_SIZE = 4

V_A = [1.0, 0.0, 0.0, 0.0]
V_B = [0.0, 1.0, 0.0, 0.0]
V_C = [0.0, 0.0, 1.0, 0.0]
V_ORTHO = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture()
def local_env(monkeypatch: pytest.MonkeyPatch):
    """Fresh local-mode server with a tiny vector size, torn down after."""
    monkeypatch.setenv("LOCI_MCP_MODE", "local")
    monkeypatch.setenv("LOCI_VECTOR_SIZE", str(VEC_SIZE))
    monkeypatch.setenv("LOCI_EPOCH_SIZE_MS", "1000")
    monkeypatch.delenv("LOCI_SCENE_ID", raising=False)
    monkeypatch.delenv("LOCI_DISTANCE", raising=False)
    mcp_server.reset_client()
    yield
    mcp_server.reset_client()


def _seed_three(scene: str = "default") -> list[dict]:
    """Remember three orthogonal observations at distinct places/times."""
    return [
        mcp_server.remember(
            V_A, 0.1, 0.1, 0.1, timestamp_ms=1_000, scene_id=scene, metadata={"label": "a"}
        ),
        mcp_server.remember(V_B, 0.9, 0.9, 0.9, timestamp_ms=2_000, scene_id=scene),
        mcp_server.remember(V_C, 0.5, 0.5, 0.5, timestamp_ms=3_000, scene_id=scene),
    ]


# ---------------------------------------------------------------------------
# Configuration & client lifecycle
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "LOCI_MCP_MODE",
            "LOCI_VECTOR_SIZE",
            "LOCI_EPOCH_SIZE_MS",
            "LOCI_DISTANCE",
            "LOCI_SCENE_ID",
        ):
            monkeypatch.delenv(var, raising=False)
        config = mcp_server.load_config()
        assert config.mode == "local"
        assert config.vector_size == 512
        assert config.epoch_size_ms == 5000
        assert config.distance == "cosine"
        assert config.default_scene_id == "default"

    def test_invalid_mode_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCI_MCP_MODE", "sqlite")
        with pytest.raises(RuntimeError, match="LOCI_MCP_MODE"):
            mcp_server.load_config()

    def test_invalid_vector_size_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCI_VECTOR_SIZE", "zero")
        with pytest.raises(RuntimeError, match="LOCI_VECTOR_SIZE"):
            mcp_server.load_config()

    def test_cloud_mode_requires_url_and_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCI_MCP_MODE", "cloud")
        monkeypatch.delenv("LOCI_CLOUD_URL", raising=False)
        monkeypatch.delenv("LOCI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="LOCI_CLOUD_URL"):
            mcp_server.load_config()

    def test_reset_client_rereads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCI_MCP_MODE", "local")
        monkeypatch.setenv("LOCI_VECTOR_SIZE", "4")
        mcp_server.reset_client()
        first = mcp_server.get_client()
        assert mcp_server.get_config().vector_size == 4
        monkeypatch.setenv("LOCI_VECTOR_SIZE", "8")
        mcp_server.reset_client()
        assert mcp_server.get_client() is not first
        assert mcp_server.get_config().vector_size == 8
        mcp_server.reset_client()

    def test_get_client_is_cached(self, local_env: None) -> None:
        assert mcp_server.get_client() is mcp_server.get_client()


# ---------------------------------------------------------------------------
# remember / recall round-trip
# ---------------------------------------------------------------------------


class TestRememberRecall:
    def test_round_trip_by_similarity(self, local_env: None) -> None:
        remembered = _seed_three()
        for entry in remembered:
            assert set(entry) == {"id", "epoch"}

        results = mcp_server.recall(vector=V_A, limit=2)
        assert isinstance(results, list)
        assert results[0]["id"] == remembered[0]["id"]
        assert results[0]["score"] == pytest.approx(1.0)
        assert results[0]["metadata"] == {"label": "a"}
        assert results[0]["score"] >= results[1]["score"]

    def test_epoch_matches_timestamp_shard(self, local_env: None) -> None:
        entry = mcp_server.remember(V_A, 0.1, 0.1, 0.1, timestamp_ms=2_500)
        assert entry["epoch"] == 2  # 2500 // LOCI_EPOCH_SIZE_MS(1000)

    def test_default_timestamp_is_now(self, local_env: None) -> None:
        before = int(time.time() * 1000)
        entry = mcp_server.remember(V_A, 0.1, 0.1, 0.1)
        after = int(time.time() * 1000)
        results = mcp_server.recall(vector=V_A, limit=1)
        assert isinstance(results, list)
        assert results[0]["id"] == entry["id"]
        assert before <= results[0]["timestamp_ms"] <= after

    def test_place_only_recall(self, local_env: None) -> None:
        remembered = _seed_three()
        results = mcp_server.recall(x=0.1, y=0.1, z=0.1, radius=0.05)
        assert isinstance(results, list)
        assert [r["id"] for r in results] == [remembered[0]["id"]]
        # No query vector -> no similarity score, ordered by recency.
        assert results[0]["score"] is None

    def test_place_only_recall_orders_by_recency(self, local_env: None) -> None:
        _seed_three()
        results = mcp_server.recall(x=0.5, y=0.5, z=0.5, radius=1.0)
        assert isinstance(results, list)
        timestamps = [r["timestamp_ms"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)
        assert all(r["score"] is None for r in results)

    def test_time_window_recall(self, local_env: None) -> None:
        remembered = _seed_three()
        results = mcp_server.recall(time_start_ms=1_500, time_end_ms=2_500)
        assert isinstance(results, list)
        assert [r["id"] for r in results] == [remembered[1]["id"]]

    def test_open_ended_time_window(self, local_env: None) -> None:
        remembered = _seed_three()
        results = mcp_server.recall(time_start_ms=2_500)  # end defaults to now
        assert isinstance(results, list)
        assert [r["id"] for r in results] == [remembered[2]["id"]]

    def test_scene_filter(self, local_env: None) -> None:
        mcp_server.remember(V_A, 0.1, 0.1, 0.1, timestamp_ms=1_000, scene_id="kitchen")
        mcp_server.remember(V_A, 0.2, 0.2, 0.2, timestamp_ms=2_000, scene_id="garage")
        results = mcp_server.recall(vector=V_A, scene_id="kitchen")
        assert isinstance(results, list)
        assert [r["scene_id"] for r in results] == ["kitchen"]

    def test_recall_result_shape(self, local_env: None) -> None:
        _seed_three()
        results = mcp_server.recall(vector=V_A, limit=1)
        assert isinstance(results, list)
        assert set(results[0]) == {
            "id",
            "x",
            "y",
            "z",
            "timestamp_ms",
            "scene_id",
            "score",
            "metadata",
        }


# ---------------------------------------------------------------------------
# Friendly errors (never tracebacks)
# ---------------------------------------------------------------------------


class TestErrors:
    def test_remember_dim_mismatch(self, local_env: None) -> None:
        result = mcp_server.remember([1.0, 0.0], 0.1, 0.1, 0.1)
        assert set(result) == {"error"}
        assert "dimension 2" in result["error"]
        assert str(VEC_SIZE) in result["error"]
        assert "Traceback" not in result["error"]

    def test_recall_dim_mismatch(self, local_env: None) -> None:
        result = mcp_server.recall(vector=[1.0] * (VEC_SIZE + 1))
        assert isinstance(result, dict)
        assert "dimension" in result["error"]

    def test_remember_out_of_range_position(self, local_env: None) -> None:
        result = mcp_server.remember(V_A, 1.5, 0.1, 0.1)
        assert isinstance(result, dict)
        assert "must be in [0, 1]" in result["error"]

    def test_recall_needs_some_criterion(self, local_env: None) -> None:
        result = mcp_server.recall()
        assert isinstance(result, dict)
        assert "at least one" in result["error"]

    def test_recall_partial_position(self, local_env: None) -> None:
        result = mcp_server.recall(x=0.5)
        assert isinstance(result, dict)
        assert "all three" in result["error"]

    def test_recall_empty_time_window(self, local_env: None) -> None:
        result = mcp_server.recall(time_start_ms=2_000, time_end_ms=1_000)
        assert isinstance(result, dict)
        assert "time window is empty" in result["error"]

    def test_recall_bad_limit_and_radius(self, local_env: None) -> None:
        bad_limit = mcp_server.recall(vector=V_A, limit=0)
        assert isinstance(bad_limit, dict) and "limit" in bad_limit["error"]
        bad_radius = mcp_server.recall(x=0.5, y=0.5, z=0.5, radius=0.0)
        assert isinstance(bad_radius, dict) and "radius" in bad_radius["error"]

    def test_novelty_dim_mismatch(self, local_env: None) -> None:
        result = mcp_server.novelty([1.0])
        assert set(result) == {"error"}
        assert "dimension" in result["error"]


# ---------------------------------------------------------------------------
# novelty
# ---------------------------------------------------------------------------


class TestNovelty:
    def test_empty_memory_is_maximally_novel(self, local_env: None) -> None:
        result = mcp_server.novelty(V_A)
        assert result["novelty"] == pytest.approx(1.0)
        assert result["best_cosine"] == pytest.approx(0.0)
        assert result["nearest"] == []

    def test_re_remembered_vector_is_not_novel(self, local_env: None) -> None:
        _seed_three()
        result = mcp_server.novelty(V_A)
        assert result["novelty"] == pytest.approx(0.0, abs=1e-6)
        assert result["best_cosine"] == pytest.approx(1.0)
        assert len(result["nearest"]) <= 3
        assert result["nearest"][0]["score"] == pytest.approx(1.0)

    def test_orthogonal_vector_is_novel(self, local_env: None) -> None:
        _seed_three()
        result = mcp_server.novelty(V_ORTHO)
        assert result["novelty"] == pytest.approx(1.0, abs=1e-6)
        assert result["best_cosine"] == pytest.approx(0.0)

    def test_position_restricts_comparison(self, local_env: None) -> None:
        _seed_three()
        # V_A was seen at (0.1, 0.1, 0.1); near (0.9, 0.9, 0.9) it is novel.
        near_home = mcp_server.novelty(V_A, x=0.1, y=0.1, z=0.1)
        far_away = mcp_server.novelty(V_A, x=0.9, y=0.9, z=0.9)
        assert near_home["novelty"] == pytest.approx(0.0, abs=1e-6)
        assert far_away["novelty"] == pytest.approx(1.0, abs=1e-6)

    def test_partial_position_rejected(self, local_env: None) -> None:
        result = mcp_server.novelty(V_A, x=0.5, y=0.5)
        assert "all three" in result["error"]


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_chronological_ordering_around_anchor(self, local_env: None) -> None:
        remembered = _seed_three(scene="walk")
        result = mcp_server.trajectory(remembered[1]["id"], steps_back=5, steps_forward=5)
        assert isinstance(result, list)
        assert [s["id"] for s in result] == [r["id"] for r in remembered]
        timestamps = [s["timestamp_ms"] for s in result]
        assert timestamps == sorted(timestamps)
        assert set(result[0]) == {"id", "x", "y", "z", "timestamp_ms"}

    def test_steps_limits_respected(self, local_env: None) -> None:
        remembered = _seed_three(scene="walk")
        result = mcp_server.trajectory(remembered[1]["id"], steps_back=0, steps_forward=1)
        assert isinstance(result, list)
        assert [s["id"] for s in result] == [remembered[1]["id"], remembered[2]["id"]]

    def test_unknown_id_is_friendly_error(self, local_env: None) -> None:
        result = mcp_server.trajectory("does-not-exist")
        assert isinstance(result, dict)
        assert "does-not-exist" in result["error"]

    def test_negative_steps_rejected(self, local_env: None) -> None:
        remembered = _seed_three()
        result = mcp_server.trajectory(remembered[0]["id"], steps_back=-1)
        assert isinstance(result, dict)
        assert "non-negative" in result["error"]


# ---------------------------------------------------------------------------
# memory_stats
# ---------------------------------------------------------------------------


class TestMemoryStats:
    def test_shape_and_counts(self, local_env: None) -> None:
        _seed_three()
        stats = mcp_server.memory_stats()
        assert set(stats) == {
            "mode",
            "vector_size",
            "distance",
            "epoch_size_ms",
            "default_scene_id",
            "total_states",
            "oldest_timestamp_ms",
            "newest_timestamp_ms",
        }
        assert stats["mode"] == "local"
        assert stats["vector_size"] == VEC_SIZE
        assert stats["epoch_size_ms"] == 1000
        assert stats["total_states"] == 3
        assert stats["oldest_timestamp_ms"] == 1_000
        assert stats["newest_timestamp_ms"] == 3_000

    def test_empty_memory(self, local_env: None) -> None:
        stats = mcp_server.memory_stats()
        assert stats["total_states"] == 0
        assert stats["oldest_timestamp_ms"] is None
        assert stats["newest_timestamp_ms"] is None


# ---------------------------------------------------------------------------
# Server assembly & entry point (needs the MCP SDK)
# ---------------------------------------------------------------------------


class TestServerAssembly:
    def test_build_server_registers_all_tools(self, local_env: None) -> None:
        pytest.importorskip("mcp")
        import anyio

        server = mcp_server.build_server()
        assert server.name == "loci-memory"
        tools = anyio.run(server.list_tools)
        assert sorted(t.name for t in tools) == [
            "memory_stats",
            "novelty",
            "recall",
            "remember",
            "trajectory",
        ]
        by_name = {t.name: t for t in tools}
        assert "Store an observation" in (by_name["remember"].description or "")

    def test_no_forget_tool(self, local_env: None) -> None:
        # The client API has no safe targeted deletion; the contract is to
        # omit `forget` entirely rather than ship a stub.
        assert not hasattr(mcp_server, "forget")
        assert all(fn.__name__ != "forget" for fn in mcp_server._TOOLS)

    def test_main_help_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest.importorskip("mcp")
        monkeypatch.setattr("sys.argv", ["loci-mcp", "--help"])
        with pytest.raises(SystemExit) as excinfo:
            mcp_server.main()
        assert excinfo.value.code == 0
        assert "loci-mcp" in capsys.readouterr().out

    def test_main_fails_fast_on_bad_config(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pytest.importorskip("mcp")
        monkeypatch.setattr("sys.argv", ["loci-mcp"])
        monkeypatch.setenv("LOCI_MCP_MODE", "bogus")
        with pytest.raises(SystemExit) as excinfo:
            mcp_server.main()
        assert excinfo.value.code == 2
        assert "LOCI_MCP_MODE" in capsys.readouterr().err
