"""Tests for the wm_bench world-model memory benchmark suite.

Covers: generator determinism and OOD label sanity, each task's metric
contract on tiny configs (LociLocal + BruteForce), the BruteForce-vs-itself
recall identity, the consolidated flight-recorder properties, and the
runner's --quick end-to-end JSON artifact.  Configured small so the whole
module stays well under a minute.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.wm_bench.datasets import (
    OodPatrolDataset,
    SmoothPatrolDataset,
    WarehouseDataset,
)
from benchmarks.wm_bench.runner import main as runner_main
from benchmarks.wm_bench.systems import (
    BruteForceSystem,
    LociLocalSystem,
    NaiveQdrantSystem,
)
from benchmarks.wm_bench.tasks import (
    future_analog_recall,
    novelty_auc,
    recall_vs_age,
    roc_auc,
    sustained_load,
    trajectory_fidelity,
)
from loci.temporal.consolidation import ConsolidationPolicy

DIM = 16


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class TestGenerators:
    def test_patrol_deterministic_per_seed(self):
        a = SmoothPatrolDataset(n_points=60, vector_dim=DIM, seed=3).points()
        b = SmoothPatrolDataset(n_points=60, vector_dim=DIM, seed=3).points()
        c = SmoothPatrolDataset(n_points=60, vector_dim=DIM, seed=4).points()
        assert a == b
        assert a != c

    def test_patrol_shape_and_bounds(self):
        pts = SmoothPatrolDataset(n_points=50, vector_dim=DIM, seed=1).points()
        assert len(pts) == 50
        ts = [p.timestamp_ms for p in pts]
        assert ts == sorted(ts) and len(set(ts)) == 50  # strictly increasing, unique
        for p in pts:
            assert 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 and 0.0 <= p.z <= 1.0
            assert len(p.embedding) == DIM
            assert not p.is_ood

    def test_warehouse_deterministic_and_revisits_reencounter(self):
        ds1 = WarehouseDataset(n_scenes=2, n_visits=2, points_per_visit=15, vector_dim=DIM, seed=5)
        ds2 = WarehouseDataset(n_scenes=2, n_visits=2, points_per_visit=15, vector_dim=DIM, seed=5)
        assert ds1.points() == ds2.points()

        pts = ds1.points()
        by_scene: dict[str, list[np.ndarray]] = {}
        for p in pts:
            by_scene.setdefault(p.scene_id, []).append(np.asarray(p.embedding))
        assert set(by_scene) == {"scene_0", "scene_1"}
        # Revisited scenes re-encounter embeddings: mean intra-scene cosine
        # (across visits) beats mean cross-scene cosine.
        s0, s1 = by_scene["scene_0"], by_scene["scene_1"]
        intra = np.mean([np.dot(s0[0], v) for v in s0[15:]])  # visit 1 vs visit 2
        cross = np.mean([np.dot(s0[0], v) for v in s1])
        assert intra > cross

    def test_ood_labels_sane(self):
        ds = OodPatrolDataset(
            n_points=200,
            vector_dim=DIM,
            seed=9,
            n_ood_segments=2,
            ood_segment_len=15,
            ood_start_frac=0.3,
        )
        pts = ds.points()
        ood = [p for p in pts if p.is_ood]
        assert len(ood) == 2 * 15
        assert 0.0 < ds.ood_fraction < 0.5
        # No OOD before the calibration prefix.
        first_ood_idx = next(i for i, p in enumerate(pts) if p.is_ood)
        assert first_ood_idx >= int(200 * 0.3)
        # Excursions live in the novel spatial corner.
        lo, hi = OodPatrolDataset.OOD_BOX
        for p in ood:
            assert lo <= p.x <= hi and lo <= p.y <= hi and lo <= p.z <= hi
        # OOD embeddings are near-orthogonal to the base manifold; base
        # embeddings live almost entirely inside it.
        basis = ds.manifold_basis
        ood_proj = [np.linalg.norm(basis.T @ np.asarray(p.embedding)) for p in ood]
        base_proj = [np.linalg.norm(basis.T @ np.asarray(p.embedding)) for p in pts if not p.is_ood]
        assert max(ood_proj) < 0.5
        assert min(base_proj) > 0.7

    def test_ood_segments_must_fit(self):
        with pytest.raises(ValueError, match="do not fit"):
            OodPatrolDataset(
                n_points=100, vector_dim=DIM, seed=1, n_ood_segments=5, ood_segment_len=50
            )


# ---------------------------------------------------------------------------
# roc_auc helper
# ---------------------------------------------------------------------------


class TestRocAuc:
    def test_perfect_reversed_and_ties(self):
        labels = [False, False, True, True]
        assert roc_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
        assert roc_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
        assert roc_auc(labels, [0.5, 0.5, 0.5, 0.5]) == 0.5

    def test_requires_both_classes(self):
        with pytest.raises(ValueError):
            roc_auc([True, True], [0.1, 0.2])


# ---------------------------------------------------------------------------
# Tasks on tiny configs (LociLocal + BruteForce)
# ---------------------------------------------------------------------------

TINY_RECALL = dict(n_points=200, vector_dim=DIM, k=5, n_queries=8, horizon_steps=5, seed=3)
TINY_NOVELTY = dict(
    n_points=150, vector_dim=DIM, n_ood_segments=1, ood_segment_len=15, warmup=30, seed=3
)
TINY_TRAJ = dict(n_scenes=2, n_visits=2, points_per_visit=15, vector_dim=DIM, steps=5, seed=3)
TINY_AGE = dict(n_points=400, vector_dim=DIM, k=5, n_age_buckets=4, queries_per_bucket=4, seed=3)
TINY_LOAD = dict(n_points=80, vector_dim=DIM, query_every=4, k=5, seed=3)


class TestFutureAnalogRecall:
    def test_brute_force_against_itself_is_one(self):
        metrics = future_analog_recall(BruteForceSystem(), **TINY_RECALL)
        assert metrics["recall_at_k"] == 1.0
        assert metrics["recall_at_k_windowed"] == 1.0
        assert metrics["oracle"] == "brute_force"
        assert metrics["synthetic"] is True

    def test_loci_local_keys_and_ranges(self):
        metrics = future_analog_recall(LociLocalSystem(), **TINY_RECALL)
        for key in ("task", "dataset", "k", "n_queries", "recall_at_k", "recall_at_k_windowed"):
            assert key in metrics
        assert 0.0 <= metrics["recall_at_k"] <= 1.0
        assert 0.0 <= metrics["recall_at_k_windowed"] <= 1.0
        # In-memory backends are exact, so LOCI should match the oracle here.
        assert metrics["recall_at_k"] == 1.0


class TestNoveltyAuc:
    def test_loci_local_scores_ood(self):
        metrics = novelty_auc(LociLocalSystem(), **TINY_NOVELTY)
        assert metrics["supported"] is True
        assert 0.0 <= metrics["auc"] <= 1.0
        assert 0.0 <= metrics["auc_onset"] <= 1.0
        assert metrics["n_ood"] > 0
        assert metrics["n_scored"] > metrics["n_ood"]
        # The constructed OOD problem should be clearly better than chance.
        assert metrics["auc"] > 0.6

    def test_brute_force_matches_definition_bounds(self):
        metrics = novelty_auc(BruteForceSystem(), **TINY_NOVELTY)
        assert metrics["supported"] is True
        assert metrics["median_familiar"] is not None
        assert metrics["median_ood"] >= 0.0

    def test_naive_qdrant_reports_null(self):
        metrics = novelty_auc(NaiveQdrantSystem(), **TINY_NOVELTY)
        assert metrics["supported"] is False
        assert metrics["auc"] is None
        assert metrics["auc_onset"] is None
        assert metrics["median_familiar"] is None


class TestTrajectoryFidelity:
    @pytest.mark.parametrize("system_cls", [BruteForceSystem, LociLocalSystem])
    def test_full_recovery_on_tiny_config(self, system_cls):
        metrics = trajectory_fidelity(system_cls(), **TINY_TRAJ)
        assert metrics["supported"] is True
        assert metrics["coverage"] == 1.0
        assert metrics["order_fidelity"] == 1.0
        assert metrics["n_scenes"] == 2
        assert metrics["method"] in {"array_slice", "causal_scene_scan"}


class TestRecallVsAge:
    def test_unconsolidated_keeps_everything(self):
        metrics = recall_vs_age(LociLocalSystem(), **TINY_AGE)
        assert metrics["compression_ratio"] == 1.0
        assert metrics["resident_points"] == metrics["inserted_points"] == 400
        assert metrics["consolidation_active"] is False
        assert all(r == 1.0 for r in metrics["recall_strict_by_age"])
        assert all(r == 1.0 for r in metrics["recall_covered_by_age"])

    def test_consolidated_flight_recorder_curve(self):
        policy = ConsolidationPolicy(
            raw_window_epochs=2, summary_epoch_ratio=2, max_states_per_scene=3
        )
        system = LociLocalSystem(consolidation_policy=policy, name="loci_local_consolidated")
        metrics = recall_vs_age(system, **TINY_AGE)

        assert metrics["consolidation_active"] is True
        assert metrics["resident_points"] < metrics["inserted_points"]
        assert metrics["compression_ratio"] > 1.0

        strict = metrics["recall_strict_by_age"]
        covered = metrics["recall_covered_by_age"]
        assert len(strict) == len(covered) == 4
        assert all(0.0 <= r <= 1.0 for r in strict + covered)
        # Monotone-ish: the newest bucket (raw window) beats the oldest
        # (fully consolidated), and old buckets keep gist via summaries.
        assert strict[-1] > strict[0]
        assert covered[-1] >= covered[0]
        assert covered[0] >= strict[0]


class TestSustainedLoad:
    @pytest.mark.parametrize("system_cls", [BruteForceSystem, LociLocalSystem])
    def test_percentiles_and_counts(self, system_cls):
        metrics = sustained_load(system_cls(), **TINY_LOAD)
        assert metrics["n_inserts"] == 80
        assert metrics["n_queries"] == 19  # i = 4, 8, ..., 76
        assert 0.0 <= metrics["insert_p50_ms"] <= metrics["insert_p95_ms"]
        assert 0.0 <= metrics["query_p50_ms"] <= metrics["query_p95_ms"]


# ---------------------------------------------------------------------------
# Runner end-to-end (--quick)
# ---------------------------------------------------------------------------


class TestRunner:
    def test_quick_end_to_end_writes_valid_json(self, tmp_path, capsys):
        output = tmp_path / "wm_bench.json"
        result = runner_main(
            [
                "--quick",
                "--systems",
                "brute_force,loci_local",
                "--output",
                str(output),
            ]
        )

        assert output.exists()
        persisted = json.loads(output.read_text())
        assert persisted == result
        assert persisted["benchmark"] == "wm_bench"
        assert persisted["schema_version"] == 1
        assert persisted["data"] == "synthetic"
        assert persisted["quick"] is True
        assert "synthetic_notice" in persisted
        assert set(persisted["results"]) == {
            "future_analog_recall",
            "novelty_auc",
            "trajectory_fidelity",
            "recall_vs_age",
            "sustained_load",
        }
        for per_system in persisted["results"].values():
            assert set(per_system) == {"brute_force", "loci_local"}
            for metrics in per_system.values():
                assert metrics["synthetic"] is True
                assert metrics["dataset"].startswith("synthetic:")
        assert persisted["versions"]["python"]

        markdown = capsys.readouterr().out
        assert "SYNTHETIC" in markdown
        assert "brute_force" in markdown

    def test_unknown_task_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown task"):
            runner_main(["--tasks", "not_a_task", "--no-write"])
