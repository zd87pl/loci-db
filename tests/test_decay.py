"""Tests for temporal decay scoring."""

from __future__ import annotations

import math

import pytest

from loci.temporal.decay import (
    DEFAULT_DECAY_LAMBDA,
    apply_decay,
    decay_score,
    lambda_from_half_life,
)


def test_decay_score_zero_age() -> None:
    assert decay_score(0.9, 0.0) == 0.9


def test_decay_score_decreases_with_age() -> None:
    young = decay_score(0.9, 1000, lambda_=1e-3)
    old = decay_score(0.9, 10000, lambda_=1e-3)
    assert young > old


def test_decay_score_formula() -> None:
    score = decay_score(1.0, 5000, lambda_=2e-4)
    expected = math.exp(-2e-4 * 5000)
    assert abs(score - expected) < 1e-12


def test_decay_score_does_not_underflow_to_zero() -> None:
    # Ages far beyond the clamp horizon must still yield a positive factor.
    assert decay_score(0.9, 1e15, lambda_=1.0) > 0.0


def test_lambda_from_half_life_halves_score() -> None:
    half_life_ms = 42_000.0
    lam = lambda_from_half_life(half_life_ms)
    assert math.isclose(decay_score(0.8, half_life_ms, lam), 0.4, rel_tol=1e-9)


def test_lambda_from_half_life_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        lambda_from_half_life(0)
    with pytest.raises(ValueError):
        lambda_from_half_life(-1.0)


def test_default_lambda_is_one_hour_half_life() -> None:
    assert math.isclose(DEFAULT_DECAY_LAMBDA, lambda_from_half_life(3_600_000))
    assert math.isclose(decay_score(1.0, 3_600_000), 0.5, rel_tol=1e-9)


def test_apply_decay_reranks() -> None:
    results = [
        {"score": 0.8, "timestamp_ms": 1000},  # old → big age
        {"score": 0.7, "timestamp_ms": 9000},  # recent → small age
    ]
    apply_decay(results, now_ms=10_000, lambda_=1e-3)

    # The recent one (score 0.7, age 1000ms) should beat the old one
    # (score 0.8, age 9000ms) after decay
    assert results[0]["timestamp_ms"] == 9000
    assert results[1]["timestamp_ms"] == 1000


def test_apply_decay_ancient_results_rank_by_similarity() -> None:
    # Both ages push the decay exponent past the clamp, so the decay factors
    # are identical: ranking must come from similarity, not insertion order.
    results = [
        {"score": 0.3, "timestamp_ms": 0},
        {"score": 0.9, "timestamp_ms": 0},
    ]
    apply_decay(results, now_ms=10**9, lambda_=1.0)

    assert results[0]["score"] == 0.9
    assert results[0]["decayed_score"] > 0.0
    assert results[1]["decayed_score"] > 0.0


def test_apply_decay_tie_breaks_by_raw_score() -> None:
    # Raw scores so small that even the clamped decay factor underflows the
    # product to exactly 0.0 — the raw-score tie-breaker must order them.
    results = [
        {"score": 1e-30, "timestamp_ms": 0},
        {"score": 3e-30, "timestamp_ms": 0},
    ]
    apply_decay(results, now_ms=10**12, lambda_=1.0)

    assert results[0]["decayed_score"] == 0.0
    assert results[1]["decayed_score"] == 0.0
    assert results[0]["score"] == 3e-30


def test_apply_decay_adds_key() -> None:
    results = [{"score": 0.5, "timestamp_ms": 0}]
    apply_decay(results, now_ms=100)
    assert "decayed_score" in results[0]
