"""Temporal decay scoring — recency weighting for search results.

Score formula::

    score = raw_similarity * exp(-lambda_ * age_ms)
"""

from __future__ import annotations

import math


def decay_score(
    raw_similarity: float,
    age_ms: float,
    lambda_: float = 1e-4,
) -> float:
    """Apply exponential temporal decay to a similarity score.

    Args:
        raw_similarity: Original cosine / dot-product similarity.
        age_ms: Age of the vector in milliseconds (``now - timestamp_ms``).
        lambda_: Decay rate.  Larger values penalise older vectors more.

    Returns:
        Decayed similarity score.
    """
    return raw_similarity * math.exp(-lambda_ * age_ms)


def apply_decay(
    results: list[dict],
    now_ms: int,
    lambda_: float = 1e-4,
) -> list[dict]:
    """Re-rank a list of scored results using temporal decay.

    Each element of *results* must have ``"score"`` and ``"timestamp_ms"`` keys.
    The list is sorted in-place by decayed score (descending).

    Args:
        results: Mutable list of result dicts.
        now_ms: Current unix timestamp in milliseconds.
        lambda_: Decay rate.

    Returns:
        The same list, sorted by decayed score descending.
    """
    for r in results:
        age = max(0.0, now_ms - r["timestamp_ms"])
        r["decayed_score"] = decay_score(r["score"], age, lambda_)
    results.sort(key=lambda r: r["decayed_score"], reverse=True)
    return results
