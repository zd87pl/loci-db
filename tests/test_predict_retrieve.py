"""Tests for the predict-then-retrieve pipeline (unit level, no Qdrant)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from loci.retrieval.predict import predict_and_retrieve
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
        predict_and_retrieve(mock_client, [0.0], predictor_fn, future_horizon_ms=500)

    call_kwargs = mock_client.query.call_args.kwargs
    start, end = call_kwargs["time_window_ms"]
    assert start == 10_000
    assert end == 10_500
