"""Auto-research pipeline.

A four-stage optimization loop inspired by Karpathy's autoresearch:

  Analyzer → Optimizer → Test Runner → Judge

Works for any text concept: code, marketing copy, content, strategy, etc.
"""

from research.models import EvalResult, Thesis, Variant, Verdict
from research.pipeline import PipelineResult, ResearchPipeline

__all__ = [
    "ResearchPipeline",
    "PipelineResult",
    "Thesis",
    "Variant",
    "EvalResult",
    "Verdict",
]
