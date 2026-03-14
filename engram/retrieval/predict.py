"""Predict-then-retrieve pipeline.

Atomic operation: take a context vector, run a user-supplied predictor
to generate a hypothetical future-state embedding, then search the
store for nearest neighbours within a configurable future time window.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.client import EngramClient
    from engram.schema import WorldState


def predict_and_retrieve(
    client: EngramClient,
    context_vector: list[float],
    predictor_fn: Callable[[list[float]], list[float]],
    future_horizon_ms: int = 1000,
    limit: int = 5,
) -> list[WorldState]:
    """Run the predict-then-retrieve primitive.

    1. Call ``predictor_fn(context_vector)`` to obtain a predicted
       future-state embedding.
    2. Query the store for nearest neighbours to that prediction,
       filtered to the time window ``[now, now + future_horizon_ms]``.

    Args:
        client: An initialised :class:`EngramClient`.
        context_vector: The current-state embedding vector.
        predictor_fn: A callable that maps an embedding to a predicted
            future embedding.  This is the user's world model.
        future_horizon_ms: How far into the future to search (ms).
        limit: Maximum number of results.

    Returns:
        List of :class:`WorldState` objects ranked by similarity to
        the predicted vector.
    """
    predicted_vector = predictor_fn(context_vector)
    now_ms = int(time.time() * 1000)
    return client.query(
        vector=predicted_vector,
        time_window_ms=(now_ms, now_ms + future_horizon_ms),
        limit=limit,
    )
