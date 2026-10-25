"""Temporal decay scoring — recency weighting for search results.

Score formula::

    score = raw_similarity * exp(-lambda_ * age_ms)

For the euclidean metric the raw score is a negative distance, so the
similarity fed to the decay is ``1 / (1 + distance)`` (see
:func:`apply_decay`).

``lambda_`` is expressed per millisecond. Because that unit is easy to
misjudge (1e-4/ms is a ~6.9 second half-life), prefer deriving it from a
half-life with :func:`lambda_from_half_life`. The library default is
:data:`DEFAULT_DECAY_LAMBDA` (one-hour half-life).
"""

from __future__ import annotations

import math

# math.exp underflows to exactly 0.0 below ~-745. Clamping the exponent keeps
# the decay factor positive so old results still rank by similarity instead of
# collapsing into an arbitrary-order tie at 0.0.
_MIN_EXPONENT = -700.0


def lambda_from_half_life(half_life_ms: float) -> float:
    """Return the per-millisecond decay rate for a given half-life.

    Args:
        half_life_ms: Time in milliseconds after which a score halves.

    Returns:
        Decay rate ``lambda_`` such that ``exp(-lambda_ * half_life_ms) == 0.5``.
    """
    if half_life_ms <= 0:
        raise ValueError(f"half_life_ms must be positive, got {half_life_ms}")
    return math.log(2) / half_life_ms


#: Default decay rate used by the clients: one-hour half-life.
DEFAULT_DECAY_LAMBDA = lambda_from_half_life(3_600_000)


def decay_score(
    raw_similarity: float,
    age_ms: float,
    lambda_: float = DEFAULT_DECAY_LAMBDA,
) -> float:
    """Apply exponential temporal decay to a similarity score.

    Args:
        raw_similarity: Original cosine / dot-product similarity.
        age_ms: Age of the vector in milliseconds (``now - timestamp_ms``).
        lambda_: Decay rate per millisecond.  Larger values penalise older
            vectors more.  Use :func:`lambda_from_half_life` to derive one.

    Returns:
        Decayed similarity score.  The decay exponent is clamped so the
        factor never underflows to exactly 0.0; beyond the clamp horizon
        ordering degrades gracefully to similarity order.
    """
    return raw_similarity * math.exp(max(-lambda_ * age_ms, _MIN_EXPONENT))


def apply_decay(
    results: list[dict],
    now_ms: int,
    lambda_: float = DEFAULT_DECAY_LAMBDA,
    *,
    distance: str = "cosine",
) -> list[dict]:
    """Re-rank a list of scored results using temporal decay.

    Each element of *results* must have ``"score"`` and ``"timestamp_ms"`` keys.
    The list is sorted in-place by decayed score (descending), with the raw
    similarity as tie-breaker so equal decayed scores never fall back to
    arbitrary insertion order.

    Args:
        results: Mutable list of result dicts.
        now_ms: Current unix timestamp in milliseconds.
        lambda_: Decay rate per millisecond.
        distance: Distance metric the raw scores came from.  Euclidean raw
            scores are *negative distances* (higher-is-better convention);
            multiplying a negative score by the decay factor would make
            older hits *less* negative and rank them higher, so for
            ``"euclidean"`` the decayed score is computed from the bounded
            similarity ``1 / (1 + distance)`` instead.  Raw ``"score"``
            values are never modified.

    Returns:
        The same list, sorted by decayed score descending.
    """
    for r in results:
        age = max(0.0, now_ms - r["timestamp_ms"])
        # Euclidean: score == -distance, so 1 / (1 - score) == 1 / (1 + distance).
        similarity = 1.0 / (1.0 - r["score"]) if distance == "euclidean" else r["score"]
        r["decayed_score"] = decay_score(similarity, age, lambda_)
    results.sort(key=lambda r: (r["decayed_score"], r["score"]), reverse=True)
    return results
