"""Auto-research pipeline orchestrator.

Implements the full Analyzer → Optimizer → Test Runner → Judge loop
inspired by Karpathy's autoresearch pattern.

Pipeline stages
---------------
1. **Analyze** — :mod:`research.agents.analyzer` reads the concept
   and produces a :class:`~research.models.Thesis`.
2. **Optimize** — :mod:`research.agents.optimizer` generates *n*
   variants guided by the thesis.
3. **Evaluate** — a :class:`~research.runners.base.BaseRunner` scores
   each variant using a test runner.
4. **Judge** — :mod:`research.agents.judge` reads only the thesis +
   eval results (never the variant content) and picks the winner.

Usage::

    from research.pipeline import ResearchPipeline
    from research.runners.llm import LLMRunner

    pipeline = ResearchPipeline(runner=LLMRunner())
    result = pipeline.run(
        concept="def greet(name):\\n    print('hi ' + name)",
        context="Python function",
    )
    print(result.verdict.recommendation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from research.agents.analyzer import analyze
from research.agents.judge import judge
from research.agents.optimizer import optimize
from research.models import EvalResult, Thesis, Variant, Verdict
from research.runners.base import BaseRunner
from research.runners.llm import LLMRunner


@dataclass
class PipelineResult:
    """Full output of one pipeline run."""

    thesis: Thesis
    variants: list[Variant]
    eval_results: list[EvalResult]
    verdict: Verdict

    def summary(self) -> str:
        """Human-readable run summary."""
        lines = [
            "=== AUTO-RESEARCH PIPELINE RESULT ===",
            f"Concept: {self.thesis.concept_summary}",
            f"Hypothesis: {self.thesis.hypothesis}",
            "",
            f"Generated {len(self.variants)} variants.",
            "",
            "Evaluation scores:",
        ]
        for er in sorted(self.eval_results, key=lambda r: r.score, reverse=True):
            status = "PASS" if er.passed else "FAIL"
            lines.append(f"  Variant {er.variant_id}: {er.score:.3f} [{status}]  {er.details[:80]}")

        lines.append("")
        if self.verdict.winner_id == -1:
            lines.append("Winner: NONE — keep original.")
        else:
            lines.append(f"Winner: Variant {self.verdict.winner_id}")

        lines.append(f"Reasoning: {self.verdict.reasoning[:300]}")
        lines.append(f"Recommendation: {self.verdict.recommendation}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis": {
                "concept_summary": self.thesis.concept_summary,
                "hypothesis": self.thesis.hypothesis,
                "improvement_dimensions": self.thesis.improvement_dimensions,
                "test_strategy": self.thesis.test_strategy,
                "constraints": self.thesis.constraints,
            },
            "variants": [
                {
                    "id": v.id,
                    "rationale": v.rationale,
                    "changes_summary": v.changes_summary,
                }
                for v in self.variants
            ],
            "eval_results": [
                {
                    "variant_id": r.variant_id,
                    "score": r.score,
                    "passed": r.passed,
                    "metrics": r.metrics,
                    "details": r.details,
                }
                for r in self.eval_results
            ],
            "verdict": {
                "winner_id": self.verdict.winner_id,
                "reasoning": self.verdict.reasoning,
                "scores": self.verdict.scores,
                "recommendation": self.verdict.recommendation,
            },
        }


class ResearchPipeline:
    """Orchestrates the four-stage auto-research loop.

    Args:
        runner: Test runner to evaluate variants.  Defaults to
            :class:`~research.runners.llm.LLMRunner` which uses an LLM
            to score variants (no tooling required).
        n_variants: Number of optimization variants to generate.
        analyzer_model: Anthropic model for the Analyzer agent.
        optimizer_model: Anthropic model for the Optimizer agent.
        judge_model: Anthropic model for the Judge agent.
        api_key: Anthropic API key (defaults to ``ANTHROPIC_API_KEY``).
    """

    def __init__(
        self,
        runner: BaseRunner | None = None,
        n_variants: int = 5,
        analyzer_model: str = "claude-opus-4-6",
        optimizer_model: str = "claude-opus-4-6",
        judge_model: str = "claude-opus-4-6",
        api_key: str | None = None,
    ) -> None:
        self.runner = runner or LLMRunner(api_key=api_key)
        self.n_variants = n_variants
        self.analyzer_model = analyzer_model
        self.optimizer_model = optimizer_model
        self.judge_model = judge_model
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        concept: str,
        context: str = "",
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            concept: The artefact to optimize.  Can be code, marketing
                copy, a strategy document, or any other text concept.
            context: Optional background context fed to the Analyzer.

        Returns:
            :class:`PipelineResult` with all intermediate artefacts and
            the final verdict.
        """
        # Stage 1: Analyze
        print("Stage 1/4: Analyzing concept...")
        thesis = analyze(
            concept=concept,
            context=context,
            model=self.analyzer_model,
            api_key=self.api_key,
        )
        print(f"  Hypothesis: {thesis.hypothesis[:100]}...")

        # Stage 2: Optimize
        print(f"Stage 2/4: Generating {self.n_variants} variants...")
        variants = optimize(
            thesis=thesis,
            concept=concept,
            n=self.n_variants,
            model=self.optimizer_model,
            api_key=self.api_key,
        )
        print(f"  Generated {len(variants)} variants.")

        # Stage 3: Evaluate
        print("Stage 3/4: Running test runner...")
        eval_results = self.runner.evaluate_all(variants, thesis)
        for er in eval_results:
            status = "PASS" if er.passed else "FAIL"
            print(f"  Variant {er.variant_id}: score={er.score:.3f} [{status}]")

        # Stage 4: Judge (blind — only sees thesis + results)
        print("Stage 4/4: Judge evaluating results...")
        verdict = judge(
            thesis=thesis,
            eval_results=eval_results,
            model=self.judge_model,
            api_key=self.api_key,
        )
        if verdict.winner_id == -1:
            print("  Judge: no variant beats original.")
        else:
            print(f"  Judge: winner is variant {verdict.winner_id}")

        return PipelineResult(
            thesis=thesis,
            variants=variants,
            eval_results=eval_results,
            verdict=verdict,
        )

    def get_winner_content(self, result: PipelineResult) -> str | None:
        """Return the content of the winning variant, or None if no winner."""
        if result.verdict.winner_id == -1:
            return None
        for v in result.variants:
            if v.id == result.verdict.winner_id:
                return v.content
        return None
