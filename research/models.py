"""Data models for the auto-research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Thesis:
    """Analysis output from the Analyzer agent."""

    concept_summary: str
    """Brief summary of what the concept is."""

    hypothesis: str
    """Main hypothesis about how to improve the concept."""

    improvement_dimensions: list[str]
    """Key dimensions to optimize (e.g. 'readability', 'performance', 'conversion rate')."""

    test_strategy: str
    """How to evaluate whether a variant is better than the original."""

    constraints: list[str] = field(default_factory=list)
    """Constraints that must be respected by all variants."""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Variant:
    """A single optimization candidate produced by the Optimizer agent."""

    id: int
    content: str
    """The optimized version of the concept."""

    rationale: str
    """Why this variant was created — maps back to the thesis hypothesis."""

    changes_summary: str
    """Human-readable summary of what changed."""


@dataclass
class EvalResult:
    """Test runner output for a single variant."""

    variant_id: int
    score: float
    """Normalized score in [0.0, 1.0] where higher is better."""

    metrics: dict[str, Any]
    """Raw metrics from the test runner."""

    passed: bool
    """Whether the variant satisfies all hard constraints."""

    details: str = ""
    """Human-readable evaluation summary."""


@dataclass
class Verdict:
    """Final judgment from the Judge agent."""

    winner_id: int
    """ID of the winning variant (-1 means original is best)."""

    reasoning: str
    """Why the winner was chosen — based only on thesis + results."""

    scores: dict[int, float]
    """Judge's assigned scores per variant."""

    recommendation: str
    """Actionable next step based on the winning variant."""
