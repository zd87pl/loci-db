"""Tests for ConformalNoveltyCalibrator (RFC-0001 R2 conformal novelty).

Covers: constructor validation, hand-computed p-values, the finite-sample
false-alarm guarantee on exchangeable data, warm-up fallback, window
eviction, monotonicity, end-to-end use through LocalLociClient's
predict_and_retrieve via the duck-typed calibrator interface, and drift
adaptation of the sliding window.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from loci.local_client import LocalLociClient
from loci.retrieval.novelty import ConformalNoveltyCalibrator
from loci.retrieval.predict import PredictRetrieveResult
from loci.schema import WorldState

VEC_SIZE = 4


def _local_client() -> LocalLociClient:
    return LocalLociClient(
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VEC_SIZE,
        decay_lambda=0.0,
        distance="cosine",
    )


class TestConstructorValidation:
    def test_defaults_valid(self) -> None:
        calibrator = ConformalNoveltyCalibrator()
        assert calibrator.alpha == 0.05
        assert len(calibrator) == 0
        assert not calibrator.warmed_up

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            ConformalNoveltyCalibrator(alpha=alpha)

    def test_min_samples_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="min_samples"):
            ConformalNoveltyCalibrator(min_samples=1)

    def test_window_smaller_than_min_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            ConformalNoveltyCalibrator(window=10, min_samples=20)


class TestPValueHandComputed:
    """p = (1 + #{stored nonconformity >= current}) / (n + 1)."""

    def _calibrator(self) -> ConformalNoveltyCalibrator:
        calibrator = ConformalNoveltyCalibrator(alpha=0.25, window=16, min_samples=2)
        # Stored nonconformity scores: 0.1, 0.2, 0.3, 0.4
        for score in (0.9, 0.8, 0.7, 0.6):
            calibrator.observe(score)
        return calibrator

    def test_empty_window_p_is_one(self) -> None:
        calibrator = ConformalNoveltyCalibrator()
        assert calibrator.p_value(0.0) == 1.0
        assert not calibrator.is_novel(0.0)

    def test_mid_window_value(self) -> None:
        # current nonconformity 0.25 → {0.3, 0.4} exceed → p = 3/5
        assert self._calibrator().p_value(0.75) == pytest.approx(3 / 5)

    def test_worse_than_all(self) -> None:
        # current nonconformity 0.5 → none stored >= 0.5 → p = 1/5
        assert self._calibrator().p_value(0.5) == pytest.approx(1 / 5)

    def test_better_than_all(self) -> None:
        # current nonconformity 0.05 → all 4 exceed → p = 5/5
        assert self._calibrator().p_value(0.95) == pytest.approx(1.0)

    def test_tie_counts_against_novelty(self) -> None:
        # current nonconformity 0.2 → {0.2, 0.3, 0.4} exceed (>=) → p = 4/5
        assert self._calibrator().p_value(0.8) == pytest.approx(4 / 5)

    def test_alarm_threshold(self) -> None:
        calibrator = self._calibrator()
        assert calibrator.is_novel(0.5)  # p = 0.2 <= alpha = 0.25
        assert not calibrator.is_novel(0.75)  # p = 0.6

    def test_score_clamped_defensively(self) -> None:
        calibrator = self._calibrator()
        # Scores outside [0, 1] clamp: -3.0 behaves like 0.0, 7.0 like 1.0.
        assert calibrator.p_value(-3.0) == calibrator.p_value(0.0)
        assert calibrator.p_value(7.0) == calibrator.p_value(1.0)


class TestFalseAlarmGuarantee:
    """On exchangeable (i.i.d.) data, alarms fire on <= alpha of samples."""

    @pytest.mark.parametrize("alpha", [0.05, 0.1])
    def test_far_within_one_percent_of_alpha(self, alpha: float) -> None:
        # Coarse pin of the RFC-0001 R2 success metric: empirical FAR within
        # +/-1% (absolute) of the configured alpha on held-out i.i.d. data.
        rng = np.random.default_rng(1234)
        calibrator = ConformalNoveltyCalibrator(alpha=alpha, window=512, min_samples=30)
        for score in np.clip(rng.normal(0.85, 0.05, 512), 0.0, 1.0):
            calibrator.observe(float(score))

        alarms = 0
        trials = 10_000
        for score in np.clip(rng.normal(0.85, 0.05, trials), 0.0, 1.0):
            if calibrator.is_novel(float(score)):
                alarms += 1
            calibrator.observe(float(score))
        far = alarms / trials
        assert far <= alpha + 0.01
        assert far >= alpha - 0.01

    def test_far_bounded_for_small_alpha(self) -> None:
        rng = np.random.default_rng(99)
        calibrator = ConformalNoveltyCalibrator(alpha=0.01, window=512, min_samples=30)
        for score in np.clip(rng.normal(0.8, 0.08, 512), 0.0, 1.0):
            calibrator.observe(float(score))

        alarms = 0
        trials = 10_000
        for score in np.clip(rng.normal(0.8, 0.08, trials), 0.0, 1.0):
            if calibrator.is_novel(float(score)):
                alarms += 1
            calibrator.observe(float(score))
        assert alarms / trials <= 0.02

    def test_alarm_cannot_fire_on_tiny_window(self) -> None:
        # With n < ceil(1/alpha) - 1 the minimum p-value 1/(n+1) exceeds alpha.
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=64, min_samples=2)
        for score in (0.9, 0.9, 0.9):
            calibrator.observe(score)
        assert not calibrator.is_novel(0.0)  # p = 1/4 > 0.05


class TestWarmupFallback:
    def test_below_min_samples_uses_raw_novelty(self) -> None:
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=64, min_samples=5)
        for score in (0.9, 0.8, 0.7):
            calibrator.observe(score)
        assert not calibrator.warmed_up
        assert calibrator.calibrated_novelty(0.9) == pytest.approx(0.1)
        assert calibrator.calibrated_novelty(-0.5) == pytest.approx(1.0)  # clamped

    def test_at_min_samples_switches_to_conformal(self) -> None:
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=64, min_samples=4)
        for score in (0.9, 0.8, 0.7, 0.6):
            calibrator.observe(score)
        assert calibrator.warmed_up
        # nonconformity 0.25 → p = 3/5 → novelty = 1 - p = 0.4
        assert calibrator.calibrated_novelty(0.75) == pytest.approx(0.4)

    def test_thresholding_novelty_reproduces_is_novel(self) -> None:
        rng = np.random.default_rng(7)
        calibrator = ConformalNoveltyCalibrator(alpha=0.1, window=128, min_samples=10)
        for score in np.clip(rng.normal(0.8, 0.1, 128), 0.0, 1.0):
            calibrator.observe(float(score))
        for probe in np.linspace(0.0, 1.0, 101):
            expected = calibrator.calibrated_novelty(float(probe)) >= 1.0 - calibrator.alpha
            assert calibrator.is_novel(float(probe)) == expected


class TestWindowSliding:
    def test_len_caps_at_window(self) -> None:
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=8, min_samples=2)
        for _ in range(20):
            calibrator.observe(0.9)
        assert len(calibrator) == 8

    def test_old_samples_evicted_change_p_value(self) -> None:
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=4, min_samples=2)
        # Fill with poor matches (high nonconformity 0.8).
        for _ in range(4):
            calibrator.observe(0.2)
        # A poor match is unsurprising against poor history: p = 5/5.
        assert calibrator.p_value(0.2) == pytest.approx(1.0)
        # Slide the window entirely to good matches (nonconformity 0.05).
        for _ in range(4):
            calibrator.observe(0.95)
        # Same poor match is now maximally surprising: p = 1/5.
        assert calibrator.p_value(0.2) == pytest.approx(1 / 5)


class TestMonotonicity:
    def test_lower_score_higher_novelty_lower_p(self) -> None:
        rng = np.random.default_rng(42)
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=256, min_samples=30)
        for score in np.clip(rng.normal(0.75, 0.1, 256), 0.0, 1.0):
            calibrator.observe(float(score))

        probes = np.linspace(1.0, 0.0, 51)  # decreasing similarity
        p_values = [calibrator.p_value(float(s)) for s in probes]
        novelties = [calibrator.calibrated_novelty(float(s)) for s in probes]
        for earlier, later in itertools.pairwise(p_values):
            assert later <= earlier
        for earlier, later in itertools.pairwise(novelties):
            assert later >= earlier


class TestEndToEndPredictAndRetrieve:
    """The calibrator must slot into predict_and_retrieve unchanged."""

    def _run_stream(self) -> tuple[ConformalNoveltyCalibrator, LocalLociClient, int, float]:
        rng = np.random.default_rng(2026)
        client = _local_client()

        def jittered(base: np.ndarray) -> list[float]:
            vec = base + rng.normal(0.0, 0.02, VEC_SIZE)
            return [float(v) for v in vec / np.linalg.norm(vec)]

        base = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(60):
            client.insert(
                WorldState(
                    x=0.5,
                    y=0.5,
                    z=0.5,
                    timestamp_ms=1000 + i * 100,
                    vector=jittered(base),
                )
            )

        calibrator = ConformalNoveltyCalibrator(alpha=0.1, window=128, min_samples=20)
        alarms = 0
        stream_len = 120
        last_novelty = 0.0
        for _ in range(stream_len):
            result = client.predict_and_retrieve(
                context_vector=jittered(base),
                predictor_fn=lambda v: v,  # identity predictor
                future_horizon_ms=1000,
                current_position=(0.5, 0.5, 0.5),
                calibrator=calibrator,
            )
            assert isinstance(result, PredictRetrieveResult)
            last_novelty = result.prediction_novelty
            if calibrator.warmed_up and last_novelty >= 1.0 - calibrator.alpha:
                alarms += 1
        return calibrator, client, alarms, last_novelty

    def test_in_distribution_stream_rarely_alarms(self) -> None:
        calibrator, _, alarms, _ = self._run_stream()
        assert calibrator.warmed_up
        assert len(calibrator) > 100  # observe() was called through the pipeline
        assert alarms <= 0.2 * 120  # alpha=0.1 plus generous slack on 120 draws

    def test_orthogonal_probe_alarms(self) -> None:
        calibrator, client, _, _ = self._run_stream()
        probe = [0.0, 0.0, 0.0, 1.0]  # orthogonal to everything stored
        result = client.predict_and_retrieve(
            context_vector=probe,
            predictor_fn=lambda v: v,
            future_horizon_ms=1000,
            current_position=(0.5, 0.5, 0.5),
            calibrator=calibrator,
        )
        assert isinstance(result, PredictRetrieveResult)
        assert result.prediction_novelty >= 1.0 - calibrator.alpha
        assert result.novelty_samples > 0

    def test_scores_before_observing_through_pipeline(self) -> None:
        # First-ever call: novelty must be judged against the empty window
        # (warm-up fallback), then observed — so len grows to exactly 1.
        client = _local_client()
        client.insert(
            WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=1000, vector=[1.0, 0.0, 0.0, 0.0])
        )
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=64, min_samples=2)
        result = client.predict_and_retrieve(
            context_vector=[1.0, 0.0, 0.0, 0.0],
            predictor_fn=lambda v: v,
            future_horizon_ms=1000,
            current_position=(0.5, 0.5, 0.5),
            calibrator=calibrator,
        )
        assert isinstance(result, PredictRetrieveResult)
        assert result.novelty_samples == 0  # scored before the observe
        assert len(calibrator) == 1
        assert result.prediction_novelty == pytest.approx(0.0, abs=1e-9)


class TestDriftAdaptation:
    def test_window_slides_and_alarms_recover(self) -> None:
        rng = np.random.default_rng(11)
        calibrator = ConformalNoveltyCalibrator(alpha=0.05, window=128, min_samples=30)
        # Regime A: high similarity.
        for score in np.clip(rng.normal(0.9, 0.02, 128), 0.0, 1.0):
            calibrator.observe(float(score))

        # Regime B: abruptly lower similarity — alarms at first...
        regime_b = np.clip(rng.normal(0.6, 0.02, 400), 0.0, 1.0)
        early = [calibrator.is_novel(float(s)) for s in regime_b[:10]]
        assert all(early)

        # ...but the sliding window adapts as regime B is observed.
        for score in regime_b:
            calibrator.observe(float(score))
        late_alarms = 0
        probes = np.clip(rng.normal(0.6, 0.02, 500), 0.0, 1.0)
        for score in probes:
            if calibrator.is_novel(float(score)):
                late_alarms += 1
            calibrator.observe(float(score))
        assert late_alarms / len(probes) <= 0.05 + 0.02
