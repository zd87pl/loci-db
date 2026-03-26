"""Tests for the predict-then-retrieve pipeline (unit level, no Qdrant)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from loci.retrieval.predict import (
    PredictRetrieveResult,
    PredictThenRetrieve,
    predict_and_retrieve,
)
from loci.schema import WorldState


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


def test_predict_and_retrieve_time_window() -> None:
    """The query should filter to [now, now + horizon]."""
    predictor_fn = MagicMock(return_value=[0.0])
    mock_client = MagicMock()
    mock_client.query.return_value = []

    with patch("loci.retrieval.predict.time") as mock_time:
        mock_time.time.return_value = 10.0  # 10 000 ms
        mock_time.perf_counter.return_value = 0.0
        predict_and_retrieve(mock_client, [0.0], predictor_fn, future_horizon_ms=500)

    call_kwargs = mock_client.query.call_args.kwargs
    start, end = call_kwargs["time_window_ms"]
    assert start == 10_000
    assert end == 10_500


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


def test_predict_then_retrieve_timestamp_zero_handled() -> None:
    """current_timestamp_ms=0 should be treated as 0, not fallback to time.time()."""
    mock_client = MagicMock()
    mock_client.query.return_value = []

    ptr = PredictThenRetrieve(mock_client)
    ptr.retrieve(
        context_vector=[1.0],
        predictor_fn=lambda v: v,
        future_horizon_ms=1000,
        current_timestamp_ms=0,
        limit=5,
    )

    # Verify the query was called with time_window starting at 0
    call_kwargs = mock_client.query.call_args.kwargs
    start, end = call_kwargs["time_window_ms"]
    assert start == 0
    assert end == 1000
