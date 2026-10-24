"""Tests for the predict-then-retrieve pipeline (unit level, no Qdrant)."""

from __future__ import annotations

import math
import statistics
from unittest.mock import MagicMock

import pytest

from loci.local_client import LocalLociClient
from loci.retrieval.novelty import NoveltyCalibrator
from loci.retrieval.predict import (
    PredictRetrieveResult,
    PredictThenRetrieve,
    predict_and_retrieve,
)
from loci.schema import WorldState

VEC_SIZE = 4


def _local_client(distance: str = "cosine") -> LocalLociClient:
    return LocalLociClient(
        epoch_size_ms=5000,
        spatial_resolution=4,
        vector_size=VEC_SIZE,
        decay_lambda=0.0,
        distance=distance,
    )


def _state(vector: list[float], ts: int = 1000) -> WorldState:
    return WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=ts, vector=vector)


def test_predict_and_retrieve_calls_predictor() -> None:
    """predictor_fn should be called with the context vector."""
    context = [1.0, 2.0, 3.0]
    predicted = [4.0, 5.0, 6.0]
    predictor_fn = MagicMock(return_value=predicted)

    mock_client = MagicMock()
    mock_client.query.return_value = [
        WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=9999, vector=predicted, id="r1"),
    ]

    results = predict_and_retrieve(
        mock_client,
        context,
        predictor_fn,
        future_horizon_ms=2000,
        limit=3,
    )

    predictor_fn.assert_called_once_with(context)
    assert mock_client.query.called
    call_kwargs = mock_client.query.call_args
    assert call_kwargs.kwargs["vector"] == predicted
    assert len(results) == 1


def test_predict_and_retrieve_searches_history_by_default() -> None:
    """The legacy helper should not assume future-dated states exist."""
    predictor_fn = MagicMock(return_value=[0.0])
    mock_client = MagicMock()
    mock_client.query.return_value = []

    predict_and_retrieve(mock_client, [0.0], predictor_fn, future_horizon_ms=500)

    call_kwargs = mock_client.query.call_args.kwargs
    assert call_kwargs["time_window_ms"] is None


def test_predict_and_retrieve_accepts_explicit_time_window() -> None:
    predictor_fn = MagicMock(return_value=[0.0])
    mock_client = MagicMock()
    mock_client.query.return_value = []

    predict_and_retrieve(
        mock_client,
        [0.0],
        predictor_fn,
        future_horizon_ms=500,
        search_time_window_ms=(10_000, 10_500),
    )

    call_kwargs = mock_client.query.call_args.kwargs
    assert call_kwargs["time_window_ms"] == (10_000, 10_500)


def test_predict_retrieve_result_defaults() -> None:
    """PredictRetrieveResult should have sensible defaults."""
    result = PredictRetrieveResult()
    assert result.results == []
    assert result.prediction_novelty == 1.0
    assert result.predicted_vector is None
    assert result.retrieval_latency_ms == 0.0
    assert result.predictor_call_ms == 0.0


def test_predict_then_retrieve_returns_novelty() -> None:
    """PredictThenRetrieve should compute novelty score."""
    mock_client = MagicMock()
    # Return some results from query
    mock_client.query.return_value = [
        WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=11000, vector=[1.0, 0.0], id="r1"),
        WorldState(x=0.6, y=0.5, z=0.5, timestamp_ms=11500, vector=[0.9, 0.1], id="r2"),
    ]

    ptr = PredictThenRetrieve(mock_client)
    result = ptr.retrieve(
        context_vector=[1.0, 0.0],
        predictor_fn=lambda v: v,  # identity predictor
        future_horizon_ms=2000,
        current_position=(0.5, 0.5, 0.5),
        current_timestamp_ms=10000,
        limit=5,
        alpha=0.7,
        return_prediction=True,
    )

    assert isinstance(result, PredictRetrieveResult)
    assert 0.0 <= result.prediction_novelty <= 1.0
    assert result.predicted_vector == [1.0, 0.0]
    assert result.retrieval_latency_ms >= 0.0
    assert result.predictor_call_ms >= 0.0
    assert len(result.results) == 2


def test_predict_then_retrieve_no_results_max_novelty() -> None:
    """When no results found, novelty should be 1.0."""
    mock_client = MagicMock()
    mock_client.query.return_value = []

    ptr = PredictThenRetrieve(mock_client)
    result = ptr.retrieve(
        context_vector=[1.0, 0.0],
        predictor_fn=lambda v: v,
        future_horizon_ms=2000,
        current_timestamp_ms=10000,
        limit=5,
    )

    assert isinstance(result, PredictRetrieveResult)
    assert result.prediction_novelty == 1.0
    assert result.results == []


def test_predict_then_retrieve_timestamp_zero_handled_with_explicit_window() -> None:
    """current_timestamp_ms=0 should be accepted when scoring an explicit window."""
    mock_client = MagicMock()
    mock_client.query.return_value = []

    ptr = PredictThenRetrieve(mock_client)
    ptr.retrieve(
        context_vector=[1.0],
        predictor_fn=lambda v: v,
        future_horizon_ms=1000,
        current_timestamp_ms=0,
        limit=5,
        search_time_window_ms=(0, 1000),
    )

    call_kwargs = mock_client.query.call_args.kwargs
    assert call_kwargs["time_window_ms"] == (0, 1000)


def test_empty_scored_response_does_not_requery() -> None:
    """An empty query_scored miss must not trigger a second query."""
    mock_client = MagicMock()
    mock_client.query_scored = MagicMock(return_value=[])

    ptr = PredictThenRetrieve(mock_client)
    result = ptr.retrieve(
        context_vector=[1.0, 0.0],
        predictor_fn=lambda v: v,
        future_horizon_ms=1000,
        current_timestamp_ms=10_000,
    )

    assert result.prediction_novelty == 1.0
    assert result.results == []
    assert not mock_client.query.called


class TestAbsoluteNovelty:
    """prediction_novelty is 1 - best cosine to the prediction: absolute."""

    def test_exact_match_low_novelty(self) -> None:
        client = _local_client()
        client.insert(_state([1.0, 0.0, 0.0, 0.0]))

        ptr = PredictThenRetrieve(client)
        result = ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: v, future_horizon_ms=1000)

        assert result.results
        assert result.prediction_novelty < 0.05

    def test_orthogonal_only_db_high_novelty(self) -> None:
        client = _local_client()
        client.insert(_state([0.0, 1.0, 0.0, 0.0]))
        client.insert(_state([0.0, 0.0, 1.0, 0.0], ts=2000))

        ptr = PredictThenRetrieve(client)
        result = ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: v, future_horizon_ms=1000)

        assert result.results  # matches exist, but none are analogs
        assert result.prediction_novelty > 0.9

    def test_empty_db_novelty_is_one(self) -> None:
        ptr = PredictThenRetrieve(_local_client())
        result = ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: v, future_horizon_ms=1000)

        assert result.results == []
        assert result.prediction_novelty == 1.0

    def test_euclidean_backend_exact_match_low_novelty(self) -> None:
        client = _local_client(distance="euclidean")
        client.insert(_state([1.0, 0.0, 0.0, 0.0]))

        ptr = PredictThenRetrieve(client)
        result = ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: v, future_horizon_ms=1000)

        assert result.results
        assert result.prediction_novelty < 0.05

    def test_euclidean_backend_far_match_high_novelty(self) -> None:
        client = _local_client(distance="euclidean")
        client.insert(_state([-1.0, 0.0, 0.0, 0.0]))

        ptr = PredictThenRetrieve(client)
        result = ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: v, future_horizon_ms=1000)

        assert result.results
        assert result.prediction_novelty > 0.9

    def test_fallback_client_without_query_scored_uses_real_similarity(self) -> None:
        """Duck-typed clients: novelty must come from cosine, not rank position."""

        class DuckClient:
            def query(self, vector, spatial_bounds=None, time_window_ms=None, limit=10):
                return [
                    WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=1000, vector=[0.0, 1.0], id="j1"),
                    WorldState(x=0.5, y=0.5, z=0.5, timestamp_ms=2000, vector=[0.0, -1.0], id="j2"),
                ]

        ptr = PredictThenRetrieve(DuckClient())
        result = ptr.retrieve([1.0, 0.0], lambda v: v, future_horizon_ms=1000)

        assert result.results
        assert result.prediction_novelty > 0.9


class TestPredictorValidation:
    def test_nan_prediction_raises(self) -> None:
        ptr = PredictThenRetrieve(_local_client())
        with pytest.raises(ValueError, match="predictor_fn"):
            ptr.retrieve(
                [1.0, 0.0, 0.0, 0.0],
                lambda v: [float("nan")] * VEC_SIZE,
                future_horizon_ms=1000,
            )

    def test_inf_prediction_raises(self) -> None:
        ptr = PredictThenRetrieve(_local_client())
        with pytest.raises(ValueError, match="predictor_fn"):
            ptr.retrieve(
                [1.0, 0.0, 0.0, 0.0],
                lambda v: [0.0, float("inf"), 0.0, 0.0],
                future_horizon_ms=1000,
            )

    def test_wrong_length_prediction_raises(self) -> None:
        ptr = PredictThenRetrieve(_local_client())
        with pytest.raises(ValueError, match="predictor_fn.*length"):
            ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: [1.0, 0.0], future_horizon_ms=1000)

    def test_non_sequence_prediction_raises(self) -> None:
        ptr = PredictThenRetrieve(_local_client())
        with pytest.raises(ValueError, match="predictor_fn"):
            ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: None, future_horizon_ms=1000)


class TestExplicitWindowScoring:
    def test_temporal_proximity_centered_on_window_midpoint(self) -> None:
        """With an explicit historical window, proximity is measured from the
        window midpoint — not from now + horizon/2."""
        client = _local_client()
        # At the window midpoint, slightly weaker vector match.
        client.insert(_state([0.9, 0.1, 0.0, 0.0], ts=5000))
        # Near the window edge, exact vector match.
        client.insert(_state([1.0, 0.0, 0.0, 0.0], ts=9500))

        ptr = PredictThenRetrieve(client)
        result = ptr.retrieve(
            [1.0, 0.0, 0.0, 0.0],
            lambda v: v,
            future_horizon_ms=1000,
            current_timestamp_ms=20_000,
            search_time_window_ms=(0, 10_000),
            alpha=0.3,  # temporal proximity dominates
            limit=5,
        )

        assert len(result.results) == 2
        # Midpoint candidate wins; centering on now + horizon/2 would zero
        # both proximities and rank the exact vector match first.
        assert result.results[0].timestamp_ms == 5000
        assert result.results[1].timestamp_ms == 9500
        # Novelty stays absolute: the exact match keeps it near zero.
        assert result.prediction_novelty < 0.05


class TestNoveltyCalibrator:
    def test_warmup_uses_raw_heuristic(self) -> None:
        calibrator = NoveltyCalibrator(min_samples=10)
        assert calibrator.calibrated_novelty(0.9) == pytest.approx(0.1)

    def test_average_match_is_midscale(self) -> None:
        calibrator = NoveltyCalibrator(window_size=50, min_samples=5)
        for score in (0.2, 0.4, 0.5, 0.6, 0.8):
            calibrator.observe(score)
        assert calibrator.calibrated_novelty(0.5) == pytest.approx(0.5)

    def test_two_sigma_maps_smoothly(self) -> None:
        calibrator = NoveltyCalibrator(window_size=100, min_samples=5)
        samples = [0.3, 0.4, 0.5, 0.6, 0.7] * 4
        for score in samples:
            calibrator.observe(score)
        mu = statistics.mean(samples)
        sigma = statistics.stdev(samples)

        assert math.isclose(calibrator.calibrated_novelty(mu + 2 * sigma), 0.12, abs_tol=0.02)
        assert math.isclose(calibrator.calibrated_novelty(mu - 2 * sigma), 0.88, abs_tol=0.02)

    def test_stdev_floor_prevents_saturation(self) -> None:
        # Near-constant window (stdev << floor): +/-0.02 wobbles must not
        # swing novelty to the extremes.
        calibrator = NoveltyCalibrator(window_size=100, min_samples=10)
        for i in range(40):
            calibrator.observe(0.80 + (0.005 if i % 2 else -0.005))

        assert 0.3 < calibrator.calibrated_novelty(0.78) < 0.7
        assert 0.3 < calibrator.calibrated_novelty(0.82) < 0.7

    def test_pipeline_scores_before_observing(self) -> None:
        """The current sample must not shift the distribution it is judged by."""
        calls: list[str] = []

        class SpyCalibrator:
            def observe(self, score: float) -> None:
                calls.append("observe")

            def calibrated_novelty(self, score: float) -> float:
                calls.append("score")
                return 0.5

            def __len__(self) -> int:
                return 0

        client = _local_client()
        client.insert(_state([1.0, 0.0, 0.0, 0.0]))
        ptr = PredictThenRetrieve(client, calibrator=SpyCalibrator())
        ptr.retrieve([1.0, 0.0, 0.0, 0.0], lambda v: v, future_horizon_ms=1000)

        assert calls == ["score", "observe"]
