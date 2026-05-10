"""Tests for the predict-then-retrieve pipeline (unit level, no Qdrant)."""

from __future__ import annotations

from unittest.mock import MagicMock

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
