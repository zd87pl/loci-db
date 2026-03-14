"""Tests for temporal decay scoring."""

from __future__ import annotations

import math

from engram.temporal.decay import apply_decay, decay_score


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


def test_apply_decay_adds_key() -> None:
    results = [{"score": 0.5, "timestamp_ms": 0}]
    apply_decay(results, now_ms=100)
    assert "decayed_score" in results[0]
