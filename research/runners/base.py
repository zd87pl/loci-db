"""Base test runner interface.

A test runner evaluates a variant and returns an :class:`EvalResult`.
Concrete runners inherit from :class:`BaseRunner` and implement
:meth:`evaluate`.

Built-in runners
----------------
- :class:`~research.runners.llm.LLMRunner` — uses an LLM to score
  the variant against the improvement dimensions (generic, no tooling
  required).
- :class:`~research.runners.code.CodeRunner` — runs ``pytest`` (or a
  custom command) and maps pass/fail/coverage to a score.
- :class:`~research.runners.metric.MetricRunner` — evaluates simple
  metrics like readability (Flesch-Kincaid), word count, or custom
  Python functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from research.models import EvalResult, Thesis, Variant


class BaseRunner(ABC):
    """Abstract base for test runners."""

    @abstractmethod
    def evaluate(self, variant: Variant, thesis: Thesis) -> EvalResult:
        """Evaluate *variant* against the goals in *thesis*.

        Args:
            variant: The optimized artefact to evaluate.
            thesis: The research framing (used for context/scoring).

        Returns:
            An :class:`EvalResult` with a score in [0.0, 1.0].
        """
        ...

    def evaluate_all(
        self, variants: list[Variant], thesis: Thesis
    ) -> list[EvalResult]:
        """Evaluate all variants sequentially.

        Override for parallel evaluation.
        """
        return [self.evaluate(v, thesis) for v in variants]
