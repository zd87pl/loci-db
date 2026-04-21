"""Analyzer agent: reads a concept/code and produces a thesis.

The Analyzer agent is the first stage of the pipeline.  It reads
the concept being optimized—which can be code, marketing copy,
content, a strategy document, or any other artefact—and produces
a :class:`~research.models.Thesis` that frames *what* to improve
and *how* to measure improvement.

The Analyzer never proposes concrete changes.  Its job is to
provide a clear framing that the Optimizer can act on.
"""

from __future__ import annotations

from anthropic import Anthropic

from research._llm_utils import extract_text, parse_json_object, require_fields
from research.models import Thesis

_SYSTEM_PROMPT = """\
You are a rigorous research analyst.  Your task is to read a concept
(code, copy, strategy, content, or any other artefact) and produce a
structured thesis that frames an optimization research loop.

You MUST output a single JSON object with these keys:
  - concept_summary: one-sentence description of what the concept is
  - hypothesis: a clear, falsifiable hypothesis about the single most
    impactful improvement axis
  - improvement_dimensions: list of 2–5 specific measurable dimensions
    to optimize (e.g. "execution speed", "clarity", "conversion rate")
  - test_strategy: one-paragraph description of how a test runner should
    evaluate whether a variant improves on the original
  - constraints: list of 0–5 hard constraints that all variants MUST
    respect (things that must not change)

Be specific and evidence-based.  Reference concrete details from the
concept when forming the hypothesis.

Output ONLY valid JSON.  No markdown, no commentary.
"""


def analyze(
    concept: str,
    context: str = "",
    model: str = "claude-opus-4-6",
    api_key: str | None = None,
) -> Thesis:
    """Analyze a concept and return a :class:`Thesis`.

    Args:
        concept: The artefact to optimize (code, copy, content, etc.).
        context: Optional background context (e.g. "this is a Python web
            handler for an API endpoint", "this is an ad for a B2B SaaS").
        model: Anthropic model to use.
        api_key: Anthropic API key; defaults to ``ANTHROPIC_API_KEY`` env var.

    Returns:
        A :class:`Thesis` with the analysis results.
    """
    client = Anthropic(api_key=api_key) if api_key else Anthropic()

    user_content = concept
    if context:
        user_content = f"Context: {context}\n\n---\n\n{concept}"

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = extract_text(message)
    data = parse_json_object(raw)
    require_fields(data, ["concept_summary", "hypothesis", "test_strategy"], context="Analyzer")

    return Thesis(
        concept_summary=data["concept_summary"],
        hypothesis=data["hypothesis"],
        improvement_dimensions=data.get("improvement_dimensions", []),
        test_strategy=data["test_strategy"],
        constraints=data.get("constraints", []),
    )
