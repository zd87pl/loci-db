"""Tests for memory consolidation — pure core and LocalLociClient wiring.

All timestamps are injected and ``time.time`` is patched at the client's
module seam, so nothing here depends on the wall clock.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import numpy as np
import pytest

from loci.local_client import LocalLociClient
from loci.schema import WorldState
from loci.temporal.consolidation import (
    ConsolidationPolicy,
    consolidate_states,
    epochs_to_consolidate,
    is_summary_collection,
    summary_coarse_range,
    summary_collection_name,
)
from loci.temporal.retention import RetentionPolicy

VEC_SIZE = 4
EPOCH_MS = 5000


def _state(
    ts: int,
    scene: str = "a",
    vector: list[float] | None = None,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.5,
    confidence: float = 1.0,
    metadata: dict | None = None,
) -> WorldState:
    return WorldState(
        x=x,
        y=y,
        z=z,
        timestamp_ms=ts,
        vector=vector if vector is not None else [1.0, 0.0, 0.0, 0.0],
        scene_id=scene,
        confidence=confidence,
        metadata=metadata or {},
    )


@contextlib.contextmanager
def _now(ts_ms: int):
    """Pin the client's wall clock (maintenance + decay) to *ts_ms*."""
    with patch("loci.local_client.time.time", return_value=ts_ms / 1000.0):
        yield


# ---------------------------------------------------------------------------
# ConsolidationPolicy validation
# ---------------------------------------------------------------------------


class TestConsolidationPolicy:
    def test_defaults(self):
        policy = ConsolidationPolicy(raw_window_epochs=2)
        assert policy.raw_window_epochs == 2
        assert policy.summary_epoch_ratio == 100
        assert policy.max_states_per_scene == 4

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"raw_window_epochs": 0},
            {"raw_window_epochs": -1},
            {"raw_window_epochs": 2, "summary_epoch_ratio": 0},
            {"raw_window_epochs": 2, "max_states_per_scene": 0},
        ],
    )
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ConsolidationPolicy(**kwargs)


# ---------------------------------------------------------------------------
# Naming / range helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_summary_collection_name(self):
        policy = ConsolidationPolicy(raw_window_epochs=2, summary_epoch_ratio=100)
        assert summary_collection_name(0, EPOCH_MS, policy) == "loci_sum_0"
        assert summary_collection_name(250, EPOCH_MS, policy) == "loci_sum_2"
        assert summary_collection_name(250, EPOCH_MS, policy, prefix="t_") == "t_loci_sum_2"

    def test_summary_coarse_range(self):
        policy = ConsolidationPolicy(raw_window_epochs=2, summary_epoch_ratio=100)
        span = 100 * EPOCH_MS
        assert summary_coarse_range(0, policy, EPOCH_MS) == (0, span - 1)
        assert summary_coarse_range(2, policy, EPOCH_MS) == (2 * span, 3 * span - 1)

    def test_range_matches_member_epochs(self):
        """Every epoch mapping to a coarse id has its timestamps inside the range."""
        policy = ConsolidationPolicy(raw_window_epochs=1, summary_epoch_ratio=4)
        for ep in range(12):
            coarse = ep // 4
            t_min, t_max = summary_coarse_range(coarse, policy, EPOCH_MS)
            assert t_min <= ep * EPOCH_MS
            assert (ep + 1) * EPOCH_MS - 1 <= t_max

    def test_is_summary_collection(self):
        assert is_summary_collection("loci_sum_7") == 7
        assert is_summary_collection("loci_sum_0") == 0
        assert is_summary_collection("loci_42") is None
        assert is_summary_collection("loci_sum_x") is None
        assert is_summary_collection("other") is None
        assert is_summary_collection("t_loci_sum_3", prefix="t_") == 3
        assert is_summary_collection("t_loci_sum_3") is None
        assert is_summary_collection("loci_sum_3", prefix="t_") is None

    def test_epochs_to_consolidate(self):
        policy = ConsolidationPolicy(raw_window_epochs=2)
        now_ms = 10 * EPOCH_MS + 100  # inside epoch 10
        stale = epochs_to_consolidate(
            list(range(11)), now_ms=now_ms, epoch_size_ms=EPOCH_MS, policy=policy
        )
        assert stale == list(range(9))  # epochs <= 10 - 2

    def test_epochs_to_consolidate_empty(self):
        policy = ConsolidationPolicy(raw_window_epochs=2)
        assert epochs_to_consolidate([], now_ms=10**9, epoch_size_ms=EPOCH_MS, policy=policy) == []


# ---------------------------------------------------------------------------
# Pure consolidate_states
# ---------------------------------------------------------------------------


def _snapshot(states: list[WorldState]) -> list[tuple]:
    return [
        (s.scene_id, s.timestamp_ms, tuple(s.vector), s.x, s.y, s.z, s.confidence, s.metadata)
        for s in states
    ]


class TestConsolidateStates:
    def test_empty_input(self):
        policy = ConsolidationPolicy(raw_window_epochs=1)
        assert consolidate_states([], policy, seed=0) == []

    def test_deterministic_for_same_seed(self):
        rng = np.random.default_rng(42)
        states = [
            _state(ts=1000 + i * 37, scene="a" if i % 2 else "b", vector=list(rng.random(VEC_SIZE)))
            for i in range(30)
        ]
        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=3)
        first = consolidate_states(states, policy, seed=7)
        second = consolidate_states(states, policy, seed=7)
        assert _snapshot(first) == _snapshot(second)

    def test_groups_by_scene_and_bounds_per_scene(self):
        rng = np.random.default_rng(0)
        states = (
            [_state(ts=i, scene="big", vector=list(rng.random(VEC_SIZE))) for i in range(10)]
            + [_state(ts=i, scene="mid", vector=list(rng.random(VEC_SIZE))) for i in range(4)]
            + [_state(ts=i, scene="tiny", vector=list(rng.random(VEC_SIZE))) for i in range(2)]
        )
        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=3)
        result = consolidate_states(states, policy, seed=1)

        per_scene: dict[str, int] = {}
        for s in result:
            per_scene[s.scene_id] = per_scene.get(s.scene_id, 0) + 1
        assert set(per_scene) == {"big", "mid", "tiny"}
        assert per_scene["big"] == 3  # k-means down to k
        assert per_scene["mid"] == 3
        assert per_scene["tiny"] == 2  # pass-through
        # Bookkeeping is lossless: source counts add up to the inputs per scene.
        totals = {"big": 0, "mid": 0, "tiny": 0}
        for s in result:
            totals[s.scene_id] += s.metadata["source_count"]
        assert totals == {"big": 10, "mid": 4, "tiny": 2}

    def test_small_group_passthrough(self):
        states = [
            _state(ts=2000, scene="a", vector=[0.0, 1.0, 0.0, 0.0], confidence=0.5),
            _state(ts=1000, scene="a", vector=[1.0, 0.0, 0.0, 0.0], confidence=0.9),
        ]
        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=4)
        result = consolidate_states(states, policy, seed=0)

        assert len(result) == 2
        # Ordered by time, vectors/coords/confidence preserved exactly.
        assert [s.timestamp_ms for s in result] == [1000, 2000]
        assert result[0].vector == [1.0, 0.0, 0.0, 0.0]
        assert result[0].confidence == 0.9
        for s in result:
            assert s.metadata == {
                "consolidated": True,
                "source_count": 1,
                "t_min_ms": s.timestamp_ms,
                "t_max_ms": s.timestamp_ms,
            }

    def test_k_bound_on_large_group(self):
        rng = np.random.default_rng(3)
        states = [
            _state(ts=1000 + i, scene="a", vector=list(rng.random(VEC_SIZE))) for i in range(50)
        ]
        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=4)
        result = consolidate_states(states, policy, seed=5)
        assert len(result) == 4
        assert sum(s.metadata["source_count"] for s in result) == 50

    def test_centroid_math(self):
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.0],
            [0.2, 0.1, 0.7, 0.0],
            [0.9, 0.3, 0.1, 0.0],
        ]
        states = [
            _state(
                ts=1000 + 100 * i,
                scene="a",
                vector=v,
                x=0.1 * (i + 1),
                y=0.05 * (i + 1),
                z=0.5,
                confidence=0.5 + 0.05 * i,
            )
            for i, v in enumerate(vectors)
        ]
        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=1)
        (summary,) = consolidate_states(states, policy, seed=0)

        mean_vec = np.mean(np.asarray(vectors), axis=0)
        expected_vec = mean_vec / np.linalg.norm(mean_vec)
        assert np.allclose(summary.vector, expected_vec)
        assert np.isclose(np.linalg.norm(summary.vector), 1.0)
        assert np.isclose(summary.x, np.mean([s.x for s in states]))
        assert np.isclose(summary.y, np.mean([s.y for s in states]))
        assert np.isclose(summary.z, 0.5)
        assert summary.timestamp_ms == int(np.mean([s.timestamp_ms for s in states]))
        assert np.isclose(summary.confidence, np.mean([s.confidence for s in states]))
        assert summary.scene_id == "a"
        assert summary.metadata == {
            "consolidated": True,
            "source_count": 6,
            "t_min_ms": 1000,
            "t_max_ms": 1500,
        }

    def test_zero_norm_centroid_falls_back_to_plain_mean(self):
        states = [
            _state(ts=1000, scene="a", vector=[1.0, 0.0, 0.0, 0.0]),
            _state(ts=2000, scene="a", vector=[-1.0, 0.0, 0.0, 0.0]),
        ]
        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=1)
        (summary,) = consolidate_states(states, policy, seed=0)
        assert summary.vector == [0.0, 0.0, 0.0, 0.0]

    def test_resummarising_composes_metadata(self):
        prior_summary = _state(
            ts=50,
            scene="a",
            vector=[1.0, 0.0, 0.0, 0.0],
            metadata={"consolidated": True, "source_count": 5, "t_min_ms": 0, "t_max_ms": 100},
        )
        raw = _state(ts=200, scene="a", vector=[0.0, 1.0, 0.0, 0.0])
        policy = ConsolidationPolicy(raw_window_epochs=1, max_states_per_scene=1)
        (summary,) = consolidate_states([prior_summary, raw], policy, seed=0)
        assert summary.metadata == {
            "consolidated": True,
            "source_count": 6,
            "t_min_ms": 0,
            "t_max_ms": 200,
        }


# ---------------------------------------------------------------------------
# End-to-end: LocalLociClient wiring
# ---------------------------------------------------------------------------

SCENE_VECTORS = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
N_EPOCHS = 10
STATES_PER_SCENE_PER_EPOCH = 5


def _scene_vector(scene: str, i: int) -> list[float]:
    base = SCENE_VECTORS[scene]
    return [base[0], base[1], 0.01 * i, 0.0]


def _build_client(
    retention_policy: RetentionPolicy | None = None,
) -> tuple[LocalLociClient, dict[int, list[str]]]:
    """Insert 10 epochs x 2 scenes x 5 states, advancing the pinned clock.

    With raw_window_epochs=2 and summary_epoch_ratio=4, epochs 0-7 end up
    consolidated into loci_sum_0 (epochs 0-3) and loci_sum_1 (epochs 4-7),
    leaving epochs 8 and 9 raw.
    """
    policy = ConsolidationPolicy(raw_window_epochs=2, summary_epoch_ratio=4, max_states_per_scene=3)
    client = LocalLociClient(
        vector_size=VEC_SIZE,
        epoch_size_ms=EPOCH_MS,
        decay_lambda=0.0,
        consolidation_policy=policy,
        retention_policy=retention_policy,
    )
    ids_by_epoch: dict[int, list[str]] = {}
    for e in range(N_EPOCHS):
        for scene in ("a", "b"):
            for i in range(STATES_PER_SCENE_PER_EPOCH):
                ts = e * EPOCH_MS + i * 100
                with _now(ts):
                    sid = client.insert(_state(ts, scene=scene, vector=_scene_vector(scene, i)))
                ids_by_epoch.setdefault(e, []).append(sid)
    return client, ids_by_epoch


class TestClientConsolidation:
    def test_raw_epochs_beyond_window_dropped(self):
        client, _ = _build_client()
        assert client._list_active_epochs() == [8, 9]
        for e in range(8):
            assert f"loci_{e}" not in client._known_collections
            assert not client.store.collection_exists(f"loci_{e}")

    def test_summary_collections_exist_and_bounded(self):
        client, _ = _build_client()
        assert client.store.collection_exists("loci_sum_0")
        assert client.store.collection_exists("loci_sum_1")
        # Bounded: at most max_states_per_scene per scene per coarse collection.
        for name in ("loci_sum_0", "loci_sum_1"):
            assert 0 < client.store.collection_count(name) <= 2 * 3
            for point in client.store.scroll(name, limit=100):
                assert point["payload"]["metadata"]["consolidated"] is True

    def test_old_data_findable_via_summaries(self):
        client, _ = _build_client()
        old_window = (0, 4 * EPOCH_MS - 1)  # epochs 0-3, all raw collections dropped
        with _now(9 * EPOCH_MS + 400):
            results = client.query(
                vector=_scene_vector("a", 0), time_window_ms=old_window, limit=10
            )
        assert results
        for s in results:
            assert s.metadata["consolidated"] is True
            assert old_window[0] <= s.timestamp_ms <= old_window[1]
            assert s.scene_id in {"a", "b"}
            assert s.metadata["source_count"] >= 1

    def test_recent_data_returns_raw(self):
        client, ids_by_epoch = _build_client()
        recent_window = (8 * EPOCH_MS, 10 * EPOCH_MS)
        with _now(9 * EPOCH_MS + 400):
            results = client.query(
                vector=_scene_vector("a", 0), time_window_ms=recent_window, limit=20
            )
        assert results
        raw_ids = set(ids_by_epoch[8]) | set(ids_by_epoch[9])
        for s in results:
            assert not s.metadata.get("consolidated")
            assert s.id in raw_ids
        # No summary collection overlaps this window: 2 raw shards searched.
        assert client.last_query_stats.shards_searched == 2

    def test_unwindowed_query_spans_raw_and_summaries(self):
        client, _ = _build_client()
        with _now(9 * EPOCH_MS + 400):
            results = client.query(vector=_scene_vector("a", 0), limit=50)
        flags = {bool(s.metadata.get("consolidated")) for s in results}
        assert flags == {True, False}
        # 2 raw epochs + 2 summary collections in the fan-out.
        assert client.last_query_stats.shards_searched == 4

    def test_time_window_prunes_summary_collections_by_coarse_range(self):
        client, _ = _build_client()
        mid_window = (4 * EPOCH_MS, 8 * EPOCH_MS - 1)  # exactly coarse id 1
        with _now(9 * EPOCH_MS + 400):
            results = client.query(
                vector=_scene_vector("b", 0), time_window_ms=mid_window, limit=10
            )
        # Only loci_sum_1 overlaps; no raw epochs are inside the window.
        assert client.last_query_stats.shards_searched == 1
        assert results
        for s in results:
            assert s.metadata["consolidated"] is True
            assert mid_window[0] <= s.timestamp_ms <= mid_window[1]

    def test_spatial_query_reaches_summaries_via_exact_post_filter(self):
        client, _ = _build_client()
        bounds = {
            k: v
            for k, v in zip(
                ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
                (0.4, 0.6, 0.4, 0.6, 0.4, 0.6),
                strict=True,
            )
        }
        with _now(9 * EPOCH_MS + 400):
            results = client.query(
                vector=_scene_vector("a", 0),
                spatial_bounds=bounds,
                time_window_ms=(0, 4 * EPOCH_MS - 1),
                limit=10,
            )
        assert results
        assert all(s.metadata["consolidated"] is True for s in results)

    def test_trajectory_excludes_summaries(self):
        client, ids_by_epoch = _build_client()
        anchor_id = ids_by_epoch[9][0]  # scene "a", epoch 9
        trajectory = client.get_trajectory(anchor_id, steps_back=100, steps_forward=100)
        assert trajectory
        for s in trajectory:
            assert not s.metadata.get("consolidated")
            assert s.timestamp_ms >= 8 * EPOCH_MS  # only raw epochs remain

    def test_causal_context_excludes_summaries(self):
        client, ids_by_epoch = _build_client()
        anchor_id = ids_by_epoch[9][0]
        context = client.get_causal_context(anchor_id, window_ms=10 * EPOCH_MS)
        assert context
        for s in context:
            assert not s.metadata.get("consolidated")

    def test_predecessor_search_ignores_summaries(self):
        client, _ = _build_client()
        # All raw states before epoch 8 are gone; the summaries that remain for
        # scene "a" must not be offered as causal predecessors.
        assert client._find_latest_predecessor("a", before_ms=8 * EPOCH_MS) is None

    def test_list_active_epochs_ignores_summary_collections(self):
        client, _ = _build_client()
        assert "loci_sum_0" in client._known_collections
        assert client._list_active_epochs() == [8, 9]

    def test_combined_retention_and_consolidation(self):
        # Retention must keep at least raw_window_epochs + 1 epochs, otherwise
        # it drops raw epochs before consolidation can fold them.  With a
        # compatible pairing, consolidation performs the epoch drops and
        # retention acts as a backstop without ever touching summaries.
        client, _ = _build_client(retention_policy=RetentionPolicy(max_epochs=3))
        assert client._list_active_epochs() == [8, 9]
        for e in range(8):
            assert not client.store.collection_exists(f"loci_{e}")
        # Summary collections are never purged by retention.
        assert client.store.collection_exists("loci_sum_0")
        assert client.store.collection_exists("loci_sum_1")
        with _now(9 * EPOCH_MS + 400):
            old = client.query(
                vector=_scene_vector("a", 0), time_window_ms=(0, 4 * EPOCH_MS - 1), limit=10
            )
        assert old
        assert all(s.metadata["consolidated"] is True for s in old)

    def test_late_insert_into_consolidated_epoch_is_refolded(self):
        client, _ = _build_client()
        source_count_before = sum(
            p["payload"]["metadata"]["source_count"]
            for p in client.store.scroll("loci_sum_0", limit=100)
        )
        # A straggler lands in long-gone epoch 0 while "now" stays recent: the
        # recreated raw collection is folded back into loci_sum_0 immediately.
        with _now(9 * EPOCH_MS + 400):
            client.insert(_state(100, scene="a", vector=_scene_vector("a", 0)))
        assert client._list_active_epochs() == [8, 9]
        assert client.store.collection_count("loci_sum_0") <= 2 * 3
        source_count_after = sum(
            p["payload"]["metadata"]["source_count"]
            for p in client.store.scroll("loci_sum_0", limit=100)
        )
        assert source_count_after == source_count_before + 1

    def test_summaries_deterministic_across_identical_builds(self):
        def _summary_snapshot(client: LocalLociClient) -> set[tuple]:
            points = []
            for name in ("loci_sum_0", "loci_sum_1"):
                for p in client.store.scroll(name, limit=100):
                    payload = p["payload"]
                    points.append(
                        (
                            name,
                            payload["scene_id"],
                            payload["timestamp_ms"],
                            tuple(round(v, 12) for v in p["vector"]),
                            payload["metadata"]["source_count"],
                        )
                    )
            return set(points)

        first, _ = _build_client()
        second, _ = _build_client()
        assert _summary_snapshot(first) == _summary_snapshot(second)

    def test_collection_prefix_summaries(self):
        policy = ConsolidationPolicy(
            raw_window_epochs=1, summary_epoch_ratio=2, max_states_per_scene=2
        )
        client = LocalLociClient(
            vector_size=VEC_SIZE,
            epoch_size_ms=EPOCH_MS,
            decay_lambda=0.0,
            consolidation_policy=policy,
            collection_prefix="t_",
        )
        for e in range(4):
            ts = e * EPOCH_MS + 50
            with _now(ts):
                client.insert(_state(ts, scene="a", vector=_scene_vector("a", e)))
        # raw_window_epochs=1 keeps only the current epoch raw.
        assert client._list_active_epochs() == [3]
        assert client.store.collection_exists("t_loci_sum_0")
        assert client.store.collection_exists("t_loci_sum_1")
        with _now(3 * EPOCH_MS + 50):
            old = client.query(
                vector=_scene_vector("a", 0), time_window_ms=(0, 2 * EPOCH_MS - 1), limit=5
            )
        assert old
        assert all(s.metadata["consolidated"] is True for s in old)


# ---------------------------------------------------------------------------
# Bounded-memory property across many epochs
# ---------------------------------------------------------------------------


class TestBoundedMemory:
    def test_many_epochs_stay_bounded(self):
        epoch_ms = 1000
        n_epochs = 200
        states_per_scene = 4
        policy = ConsolidationPolicy(
            raw_window_epochs=3, summary_epoch_ratio=10, max_states_per_scene=2
        )
        client = LocalLociClient(
            vector_size=VEC_SIZE,
            epoch_size_ms=epoch_ms,
            decay_lambda=0.0,
            consolidation_policy=policy,
        )
        rng = np.random.default_rng(11)
        inserted = 0
        for e in range(n_epochs):
            for scene in ("a", "b"):
                for i in range(states_per_scene):
                    ts = e * epoch_ms + i * 10
                    with _now(ts):
                        client.insert(_state(ts, scene=scene, vector=list(rng.random(VEC_SIZE))))
                    inserted += 1
        assert inserted == n_epochs * 2 * states_per_scene  # 1600

        # Raw window: epochs 197-199 survive (cutoff = 199 - 3 = 196).
        assert client._list_active_epochs() == [197, 198, 199]
        raw_points = sum(client.store.collection_count(f"loci_{e}") for e in (197, 198, 199))
        assert raw_points == 3 * 2 * states_per_scene  # 24

        summary_collections = sorted(
            c for c in client._known_collections if c.startswith("loci_sum_")
        )
        # Epochs 0-196 span coarse ids 0-19: 20 summary collections.
        assert len(summary_collections) == 20
        summary_points = sum(client.store.collection_count(c) for c in summary_collections)
        # Each coarse collection holds <= 2 scenes * 2 states.
        assert all(client.store.collection_count(c) <= 4 for c in summary_collections)
        assert summary_points <= 20 * 4

        total = client.store.total_points
        assert total == raw_points + summary_points
        assert total <= 24 + 80  # 1600 inserted -> at most 104 resident
        # Nothing was silently lost: summarised source counts + raw = inserted.
        summarised_sources = sum(
            p["payload"]["metadata"]["source_count"]
            for c in summary_collections
            for p in client.store.scroll(c, limit=1000)
        )
        assert summarised_sources + raw_points == inserted


# ---------------------------------------------------------------------------
# Predict-and-retrieve over a mixed (raw + summary) store
# ---------------------------------------------------------------------------


class TestPredictRetrieveOverMixedStore:
    def test_familiar_prediction_low_novelty(self):
        client, _ = _build_client()
        with _now(9 * EPOCH_MS + 400):
            result = client.predict_and_retrieve(
                context_vector=_scene_vector("a", 0),
                predictor_fn=lambda v: v,
                current_position=(0.5, 0.5, 0.5),
                current_timestamp_ms=9 * EPOCH_MS + 400,
                limit=5,
            )
        assert result.results
        assert result.prediction_novelty < 0.2

    def test_novel_prediction_high_novelty(self):
        client, _ = _build_client()
        with _now(9 * EPOCH_MS + 400):
            result = client.predict_and_retrieve(
                context_vector=[0.0, 0.0, 0.0, 1.0],  # orthogonal to everything stored
                predictor_fn=lambda v: v,
                current_position=(0.5, 0.5, 0.5),
                current_timestamp_ms=9 * EPOCH_MS + 400,
                limit=5,
            )
        assert result.prediction_novelty > 0.8

    def test_familiar_old_scene_matches_summary(self):
        """A prediction resembling long-consolidated history still finds analogs."""
        client, _ = _build_client()
        with _now(9 * EPOCH_MS + 400):
            result = client.predict_and_retrieve(
                context_vector=_scene_vector("b", 2),
                predictor_fn=lambda v: v,
                current_position=(0.5, 0.5, 0.5),
                current_timestamp_ms=9 * EPOCH_MS + 400,
                limit=10,
            )
        assert result.results
        assert result.prediction_novelty < 0.2
        assert any(s.scene_id == "b" for s in result.results)
