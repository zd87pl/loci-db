"""Tests for the auto-research pipeline.

These tests use mocked Anthropic API calls so they run offline.
The integration tests (marked `slow`) call the real API and require
``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from research._llm_utils import LLMResponseError
from research.models import EvalResult, Thesis, Variant, Verdict
from research.pipeline import PipelineResult, ResearchPipeline
from research.runners.code import CodeRunner
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
            content=(
                "def add(a: int, b: int) -> int:\n"
                '    """Return sum of a and b."""\n'
                "    return a + b"
            ),
            rationale="Added type hints and docstring",
            changes_summary="- Added int type hints\n- Added docstring",
        ),
        Variant(
            id=2,
            content=(
                "from typing import Union\n"
                "def add(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:\n"
                "    return a + b"
            ),
            rationale="Used Union types for broader compatibility",
            changes_summary="- Added Union type hints\n- Handles floats too",
        ),
    ]


@pytest.fixture
def sample_eval_results() -> list[EvalResult]:
    return [
        EvalResult(
            variant_id=1, score=0.9, metrics={"readability": 0.9, "type_safety": 0.9}, passed=True
        ),
        EvalResult(
            variant_id=2, score=0.75, metrics={"readability": 0.7, "type_safety": 0.8}, passed=True
        ),
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
    variant = Variant(
        id=1, content="Hello world this is a test", rationale="test", changes_summary=""
    )
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
    variant = Variant(
        id=1,
        content="This is a long piece of text that fails the constraint",
        rationale="",
        changes_summary="",
    )
    thesis = Thesis(
        concept_summary="t", hypothesis="h", improvement_dimensions=[], test_strategy="s"
    )
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
    verdict = Verdict(
        winner_id=-1, reasoning="None improved.", scores={}, recommendation="keep original"
    )
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
def test_pipeline_run_mocked(
    mock_llm_anthropic, mock_judge_anthropic, mock_optimizer_anthropic, mock_analyzer_anthropic
) -> None:
    """Full pipeline run with all API calls mocked."""
    # Analyzer response
    analyzer_response = json.dumps(
        {
            "concept_summary": "Simple Python function",
            "hypothesis": "Type hints will improve readability",
            "improvement_dimensions": ["readability", "maintainability"],
            "test_strategy": "Check for type annotations",
            "constraints": ["must be pure function"],
        }
    )
    mock_analyzer_anthropic.return_value.messages.create.return_value = _make_mock_message(
        analyzer_response
    )

    # Optimizer response
    optimizer_response = json.dumps(
        [
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
        ]
    )
    mock_optimizer_anthropic.return_value.messages.create.return_value = _make_mock_message(
        optimizer_response
    )

    # LLM runner response
    llm_runner_response = json.dumps(
        {
            "dimension_scores": {"readability": 0.9, "maintainability": 0.85},
            "overall_score": 0.875,
            "constraints_satisfied": True,
            "details": "Good improvement",
        }
    )
    mock_llm_anthropic.return_value.messages.create.return_value = _make_mock_message(
        llm_runner_response
    )

    # Judge response
    judge_response = json.dumps(
        {
            "winner_id": 1,
            "reasoning": "Variant 1 best addresses the hypothesis.",
            "scores": {"1": 0.875, "2": 0.8},
            "recommendation": "Use variant 1",
        }
    )
    mock_judge_anthropic.return_value.messages.create.return_value = _make_mock_message(
        judge_response
    )

    pipeline = ResearchPipeline(n_variants=2)
    result = pipeline.run(concept="def add(a, b): return a+b", context="Python function")

    assert result.thesis.concept_summary == "Simple Python function"
    assert len(result.variants) == 2
    assert len(result.eval_results) == 2
    assert result.verdict.winner_id == 1
    assert pipeline.get_winner_content(result) is not None


# ---------------------------------------------------------------------------
# CodeRunner isolation tests (no API calls)
# ---------------------------------------------------------------------------

_ORIGINAL_TARGET = "ORIGINAL = True\n"


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture
def code_project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "target.py").write_text(_ORIGINAL_TARGET, encoding="utf-8")
    (proj / "other.py").write_text("OTHER = 1\n", encoding="utf-8")
    return proj


@pytest.fixture
def code_variant() -> Variant:
    return Variant(
        id=1,
        content="VARIANT_MARKER = True\n",
        rationale="test variant",
        changes_summary="replaced content",
    )


def test_code_runner_failing_command_leaves_tree_intact(
    code_project: Path, code_variant: Variant, sample_thesis: Thesis
) -> None:
    """A nonzero-exit test command must not leave any trace in the real tree."""
    before = _tree_snapshot(code_project)
    runner = CodeRunner(
        target_path=code_project / "target.py",
        test_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
        work_dir=code_project,
    )
    result = runner.evaluate(code_variant, sample_thesis)
    assert result.passed is False
    assert _tree_snapshot(code_project) == before


def test_code_runner_killed_command_leaves_tree_intact(
    code_project: Path, code_variant: Variant, sample_thesis: Thesis
) -> None:
    """A test command killed by the wall-clock timeout never touches the real tree."""
    before = _tree_snapshot(code_project)
    runner = CodeRunner(
        target_path=code_project / "target.py",
        test_cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
        work_dir=code_project,
        timeout=1,
    )
    result = runner.evaluate(code_variant, sample_thesis)
    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics.get("error") == "timeout"
    assert _tree_snapshot(code_project) == before


def test_code_runner_evaluates_variant_in_workspace_copy(
    code_project: Path, code_variant: Variant, sample_thesis: Thesis
) -> None:
    """The test command sees the variant content in the copy, not the original."""
    check = (
        "import pathlib, sys; "
        "sys.exit(0 if 'VARIANT_MARKER' in pathlib.Path('target.py').read_text() else 1)"
    )
    runner = CodeRunner(
        target_path=code_project / "target.py",
        test_cmd=[sys.executable, "-c", check],
        work_dir=code_project,
    )
    result = runner.evaluate(code_variant, sample_thesis)
    assert result.passed is True
    # The real file was never rewritten.
    assert (code_project / "target.py").read_text(encoding="utf-8") == _ORIGINAL_TARGET


def test_code_runner_creates_backup_in_workspace(
    code_project: Path, code_variant: Variant, sample_thesis: Thesis
) -> None:
    """The pristine target is preserved as <target>.orig next to the overwritten copy."""
    check = (
        "import pathlib, sys; "
        "orig = pathlib.Path('target.py.orig'); "
        f"sys.exit(0 if orig.exists() and orig.read_text() == {_ORIGINAL_TARGET!r} else 1)"
    )
    runner = CodeRunner(
        target_path=code_project / "target.py",
        test_cmd=[sys.executable, "-c", check],
        work_dir=code_project,
    )
    result = runner.evaluate(code_variant, sample_thesis)
    assert result.passed is True
    # The backup lives only in the throwaway workspace.
    assert not (code_project / "target.py.orig").exists()


def test_code_runner_parses_pytest_style_output(
    code_project: Path, code_variant: Variant, sample_thesis: Thesis
) -> None:
    runner = CodeRunner(
        target_path=code_project / "target.py",
        test_cmd=[sys.executable, "-c", "print('3 passed, 1 failed in 0.12s')"],
        work_dir=code_project,
    )
    result = runner.evaluate(code_variant, sample_thesis)
    assert result.passed is True  # returncode 0
    assert result.metrics["passed"] == 3.0
    assert result.metrics["failed"] == 1.0
    assert result.score == pytest.approx(0.75)


def test_code_runner_string_command_uses_shell(
    code_project: Path, code_variant: Variant, sample_thesis: Thesis
) -> None:
    """Operator-supplied string commands still run (through the shell)."""
    runner = CodeRunner(
        target_path=code_project / "target.py",
        test_cmd="echo '2 passed in 0.01s'",
        work_dir=code_project,
    )
    result = runner.evaluate(code_variant, sample_thesis)
    assert result.passed is True
    assert result.score == pytest.approx(1.0)


def test_code_runner_target_outside_work_dir_raises(tmp_path: Path, code_project: Path) -> None:
    with pytest.raises(ValueError, match="must be inside"):
        CodeRunner(target_path=tmp_path / "elsewhere.py", work_dir=code_project)


# ---------------------------------------------------------------------------
# Optimizer token-budget tests (mocked API)
# ---------------------------------------------------------------------------


@patch("research.agents.optimizer.Anthropic")
def test_optimizer_truncated_response_raises(mock_anthropic, sample_thesis: Thesis) -> None:
    from research.agents.optimizer import optimize

    msg = _make_mock_message('[{"id": 1, "content": "def f(): p')
    msg.stop_reason = "max_tokens"
    mock_anthropic.return_value.messages.create.return_value = msg

    with pytest.raises(LLMResponseError, match="truncated"):
        optimize(thesis=sample_thesis, concept="def f(): pass", n=5)


@patch("research.agents.optimizer.Anthropic")
def test_optimizer_max_tokens_scales_and_caps(mock_anthropic, sample_thesis: Thesis) -> None:
    from research.agents.optimizer import optimize

    msg = _make_mock_message(
        json.dumps([{"id": 1, "content": "x", "rationale": "", "changes_summary": ""}])
    )
    msg.stop_reason = "end_turn"
    create = mock_anthropic.return_value.messages.create
    create.return_value = msg

    # Tiny concept: floor applies.
    optimize(thesis=sample_thesis, concept="tiny", n=2)
    assert create.call_args.kwargs["max_tokens"] == 4096

    # Mid-size concept: n * (len//3 + overhead).
    optimize(thesis=sample_thesis, concept="x" * 6000, n=3)
    assert create.call_args.kwargs["max_tokens"] == 3 * (6000 // 3 + 512)

    # Huge concept: capped at the model limit.
    optimize(thesis=sample_thesis, concept="x" * 600_000, n=5)
    assert create.call_args.kwargs["max_tokens"] == 32000


# ---------------------------------------------------------------------------
# Judge validation tests (mocked API)
# ---------------------------------------------------------------------------


@patch("research.agents.judge.Anthropic")
def test_judge_non_numeric_scores_raise(
    mock_anthropic, sample_thesis: Thesis, sample_eval_results: list[EvalResult]
) -> None:
    from research.agents.judge import judge

    response = json.dumps({"winner_id": 1, "reasoning": "ok", "scores": {"1": "high"}})
    mock_anthropic.return_value.messages.create.return_value = _make_mock_message(response)

    with pytest.raises(LLMResponseError, match="not numeric"):
        judge(sample_thesis, sample_eval_results)


@patch("research.agents.judge.Anthropic")
def test_judge_non_integer_winner_id_raises(
    mock_anthropic, sample_thesis: Thesis, sample_eval_results: list[EvalResult]
) -> None:
    from research.agents.judge import judge

    response = json.dumps(
        {"winner_id": "variant 3", "reasoning": "ok", "scores": {"1": 0.5, "2": 0.4}}
    )
    mock_anthropic.return_value.messages.create.return_value = _make_mock_message(response)

    with pytest.raises(LLMResponseError, match="winner_id"):
        judge(sample_thesis, sample_eval_results)


@patch("research.agents.judge.Anthropic")
def test_judge_hallucinated_winner_id_falls_back_to_original(
    mock_anthropic,
    sample_thesis: Thesis,
    sample_eval_results: list[EvalResult],
    caplog: pytest.LogCaptureFixture,
) -> None:
    from research.agents.judge import judge

    # Only variants 1 and 2 were evaluated; the judge hallucinates 7.
    response = json.dumps(
        {"winner_id": 7, "reasoning": "ok", "scores": {"1": 0.5, "2": 0.4}, "recommendation": "r"}
    )
    mock_anthropic.return_value.messages.create.return_value = _make_mock_message(response)

    with caplog.at_level(logging.WARNING, logger="research.agents.judge"):
        verdict = judge(sample_thesis, sample_eval_results)

    assert verdict.winner_id == -1
    assert "falling back" in caplog.text


# ---------------------------------------------------------------------------
# LLMRunner crash-tolerance tests (mocked API)
# ---------------------------------------------------------------------------


@patch("research.runners.llm.Anthropic")
def test_llm_runner_wrong_types_fall_back_to_zero_score(
    mock_anthropic, sample_thesis: Thesis, sample_variants: list[Variant]
) -> None:
    """Valid JSON with non-numeric fields must hit the zero-score fallback, not crash."""
    from research.runners.llm import LLMRunner

    response = json.dumps(
        {
            "dimension_scores": {"readability": "good"},
            "overall_score": "high",
            "constraints_satisfied": True,
            "details": "wrong types",
        }
    )
    mock_anthropic.return_value.messages.create.return_value = _make_mock_message(response)

    runner = LLMRunner()
    result = runner.evaluate(sample_variants[0], sample_thesis)

    assert result.score == 0.0
    assert result.passed is False
    assert "LLM scoring failed" in result.details


# ---------------------------------------------------------------------------
# CLI runner selection tests
# ---------------------------------------------------------------------------


def test_cli_metric_runner_errors_instead_of_silent_llm_fallback() -> None:
    from research.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--runner", "metric"], input="some concept\n")

    assert result.exit_code == 1
    try:
        stderr = result.stderr
    except (AttributeError, ValueError):
        stderr = ""
    combined = result.output + stderr
    assert "MetricRunner" in combined
    assert "falling back" not in combined.lower()
