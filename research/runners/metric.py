"""Metric-based test runner.

Evaluates variants using a callable metric function (or a set of
functions).  Suitable for content quality checks, readability scores,
word-count constraints, or any custom scoring logic.

Example usage::

    from research.runners.metric import MetricRunner

    def flesch_score(text: str) -> float:
        # Returns Flesch Reading Ease (normalized to [0, 1])
        ...

    runner = MetricRunner(metrics={"readability": flesch_score})
"""

from __future__ import annotations

from typing import Callable

from research.models import EvalResult, Thesis, Variant
from research.runners.base import BaseRunner

MetricFn = Callable[[str], float]


class MetricRunner(BaseRunner):
    """Scores variants using one or more metric functions.

    Args:
        metrics: Mapping of metric name → callable that takes the
            variant content string and returns a float in [0.0, 1.0].
        weights: Optional mapping of metric name → weight for computing
            the overall score.  Defaults to equal weighting.
        constraints: Optional list of callables that return True if the
            variant content satisfies a hard constraint.
    """

    def __init__(
        self,
        metrics: dict[str, MetricFn],
        weights: dict[str, float] | None = None,
        constraints: list[MetricFn] | None = None,
    ) -> None:
        self.metrics = metrics
        self.weights = weights or {k: 1.0 for k in metrics}
        self.constraints: list[MetricFn] = constraints or []

    def evaluate(self, variant: Variant, thesis: Thesis) -> EvalResult:
        raw_scores: dict[str, float] = {}
        for name, fn in self.metrics.items():
            try:
                raw_scores[name] = float(fn(variant.content))
            except Exception as exc:
                raw_scores[name] = 0.0
                raw_scores[f"{name}_error"] = str(exc)  # type: ignore[assignment]

        # Weighted overall score
        total_weight = sum(self.weights.get(k, 1.0) for k in raw_scores if not k.endswith("_error"))
        if total_weight > 0:
            overall = sum(
                raw_scores[k] * self.weights.get(k, 1.0)
                for k in raw_scores
                if not k.endswith("_error")
            ) / total_weight
        else:
            overall = 0.0

        # Constraint check
        passed = all(bool(fn(variant.content)) for fn in self.constraints)

        metric_names = [k for k in raw_scores if not k.endswith("_error")]
        details = ", ".join(f"{k}={raw_scores[k]:.3f}" for k in metric_names)

        return EvalResult(
            variant_id=variant.id,
            score=overall,
            metrics=raw_scores,
            passed=passed,
            details=details,
        )
