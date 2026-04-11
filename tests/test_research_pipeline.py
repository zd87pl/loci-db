"""Tests for the auto-research pipeline.

These tests use mocked Anthropic API calls so they run offline.
The integration tests (marked `slow`) call the real API and require
``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from research.models import EvalResult, Thesis, Variant, Verdict
from research.pipeline import PipelineResult, ResearchPipeline
from research.runners.base import BaseRunner
from research.runners.metric import MetricRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_thesis() -> Thesis:
    return Thesis(
        concept_summary="Simple Python addition function",
        hypothesis="Adding type annotations and a docstring will improve maintainability",
        improvement_dimensions=["readability", "type safety"],
        test_strategy="Check for type annotations and docstring presence",
        constraints=["Must remain a pure function", "No third-party imports"],
    )


@pytest.fixture
def sample_variants() -> list[Variant]:
    return [
        Variant(
            id=1,
            content="def add(a: int, b: int) -> int:\n    \"\"\"Return sum of a and b.\"\"\"\n    return a + b",
            rationale="Added type hints and docstring",
            changes_summary="- Added int type hints\n- Added docstring",
        ),
        Variant(
            id=2,
            content="from typing import Union\ndef add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:\n    return a + b",
            rationale="Used Union types for broader compatibility",
            changes_summary="- Added Union type hints\n- Handles floats too",
        ),
    ]


@pytest.fixture
def sample_eval_results() -> list[EvalResult]:
    return [
        EvalResult(variant_id=1, score=0.9, metrics={"readability": 0.9, "type_safety": 0.9}, passed=True),
        EvalResult(variant_id=2, score=0.75, metrics={"readability": 0.7, "type_safety": 0.8}, passed=True),
    ]


# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------


def test_thesis_fields(sample_thesis: Thesis) -> None:
    assert sample_thesis.concept_summary
    assert sample_thesis.hypothesis
    assert len(sample_thesis.improvement_dimensions) >= 1


def test_variant_fields(sample_variants: list[Variant]) -> None:
    for v in sample_variants:
        assert v.id > 0
        assert v.content
        assert v.rationale


def test_eval_result_score_range(sample_eval_results: list[EvalResult]) -> None:
    for r in sample_eval_results:
        assert 0.0 <= r.score <= 1.0


def test_verdict_winner_id() -> None:
    verdict = Verdict(
        winner_id=1,
        reasoning="Variant 1 improved readability the most.",
        scores={1: 0.9, 2: 0.75},
        recommendation="Deploy variant 1.",
    )
    assert verdict.winner_id == 1
    assert verdict.scores[1] > verdict.scores[2]


# ---------------------------------------------------------------------------
# MetricRunner tests (no API calls)
# ---------------------------------------------------------------------------


def test_metric_runner_basic() -> None:
    def word_count_score(text: str) -> float:
        words = len(text.split())
        # Normalize: 10-50 words = 1.0, else scale down
        return min(1.0, words / 50.0)

    runner = MetricRunner(metrics={"word_count": word_count_score})
    variant = Variant(id=1, content="Hello world this is a test", rationale="test", changes_summary="")
    thesis = Thesis(
        concept_summary="test",
        hypothesis="more words",
        improvement_dimensions=["word_count"],
        test_strategy="count words",
    )
    result = runner.evaluate(variant, thesis)
    assert 0.0 <= result.score <= 1.0
    assert result.passed is True
    assert "word_count" in result.metrics


def test_metric_runner_constraint_fail() -> None:
    runner = MetricRunner(
        metrics={"length": lambda t: len(t) / 100.0},
        constraints=[lambda t: len(t) < 10],  # Must be short
    )
    variant = Variant(id=1, content="This is a long piece of text that fails the constraint", rationale="", changes_summary="")
    thesis = Thesis(concept_summary="t", hypothesis="h", improvement_dimensions=[], test_strategy="s")
    result = runner.evaluate(variant, thesis)
    assert result.passed is False


def test_metric_runner_evaluate_all(sample_variants: list[Variant], sample_thesis: Thesis) -> None:
    runner = MetricRunner(metrics={"length": lambda t: min(1.0, len(t) / 200.0)})
    results = runner.evaluate_all(sample_variants, sample_thesis)
    assert len(results) == len(sample_variants)
    assert all(isinstance(r, EvalResult) for r in results)


# ---------------------------------------------------------------------------
# PipelineResult tests
# ---------------------------------------------------------------------------


def test_pipeline_result_summary(
    sample_thesis: Thesis,
    sample_variants: list[Variant],
    sample_eval_results: list[EvalResult],
) -> None:
    verdict = Verdict(
        winner_id=1,
        reasoning="Variant 1 won.",
        scores={1: 0.9, 2: 0.75},
        recommendation="Use variant 1.",
    )
    result = PipelineResult(
        thesis=sample_thesis,
        variants=sample_variants,
        eval_results=sample_eval_results,
        verdict=verdict,
    )
    summary = result.summary()
    assert "Variant 1" in summary
    assert "winner" in summary.lower()


def test_pipeline_result_to_dict(
    sample_thesis: Thesis,
    sample_variants: list[Variant],
    sample_eval_results: list[EvalResult],
) -> None:
    verdict = Verdict(winner_id=1, reasoning=".", scores={1: 0.9}, recommendation="deploy")
    result = PipelineResult(
        thesis=sample_thesis,
        variants=sample_variants,
        eval_results=sample_eval_results,
        verdict=verdict,
    )
    d = result.to_dict()
    assert "thesis" in d
    assert "variants" in d
    assert "eval_results" in d
    assert "verdict" in d
    # Should be JSON-serializable
    json.dumps(d)


def test_pipeline_result_no_winner(
    sample_thesis: Thesis,
    sample_variants: list[Variant],
    sample_eval_results: list[EvalResult],
) -> None:
    verdict = Verdict(winner_id=-1, reasoning="None improved.", scores={}, recommendation="keep original")
    result = PipelineResult(
        thesis=sample_thesis,
        variants=sample_variants,
        eval_results=sample_eval_results,
        verdict=verdict,
    )
    pipeline = ResearchPipeline()
    assert pipeline.get_winner_content(result) is None


# ---------------------------------------------------------------------------
# ResearchPipeline integration test with mocked API
# ---------------------------------------------------------------------------


def _make_mock_message(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock()]
    msg.content[0].text = text
    return msg


@patch("research.agents.analyzer.Anthropic")
@patch("research.agents.optimizer.Anthropic")
@patch("research.agents.judge.Anthropic")
@patch("research.runners.llm.Anthropic")
def test_pipeline_run_mocked(mock_llm_anthropic, mock_judge_anthropic, mock_optimizer_anthropic, mock_analyzer_anthropic) -> None:
    """Full pipeline run with all API calls mocked."""
    # Analyzer response
    analyzer_response = json.dumps({
        "concept_summary": "Simple Python function",
        "hypothesis": "Type hints will improve readability",
        "improvement_dimensions": ["readability", "maintainability"],
        "test_strategy": "Check for type annotations",
        "constraints": ["must be pure function"],
    })
    mock_analyzer_anthropic.return_value.messages.create.return_value = _make_mock_message(analyzer_response)

    # Optimizer response
    optimizer_response = json.dumps([
        {
            "id": 1,
            "content": "def add(a: int, b: int) -> int:\n    return a + b",
            "rationale": "Added type hints",
            "changes_summary": "Added int type hints",
        },
        {
            "id": 2,
            "content": "def add(a: float, b: float) -> float:\n    return a + b",
            "rationale": "Used floats",
            "changes_summary": "Added float type hints",
        },
    ])
    mock_optimizer_anthropic.return_value.messages.create.return_value = _make_mock_message(optimizer_response)

    # LLM runner response
    llm_runner_response = json.dumps({
        "dimension_scores": {"readability": 0.9, "maintainability": 0.85},
        "overall_score": 0.875,
        "constraints_satisfied": True,
        "details": "Good improvement",
    })
    mock_llm_anthropic.return_value.messages.create.return_value = _make_mock_message(llm_runner_response)

    # Judge response
    judge_response = json.dumps({
        "winner_id": 1,
        "reasoning": "Variant 1 best addresses the hypothesis.",
        "scores": {"1": 0.875, "2": 0.8},
        "recommendation": "Use variant 1",
    })
    mock_judge_anthropic.return_value.messages.create.return_value = _make_mock_message(judge_response)

    pipeline = ResearchPipeline(n_variants=2)
    result = pipeline.run(concept="def add(a, b): return a+b", context="Python function")

    assert result.thesis.concept_summary == "Simple Python function"
    assert len(result.variants) == 2
    assert len(result.eval_results) == 2
    assert result.verdict.winner_id == 1
    assert pipeline.get_winner_content(result) is not None
