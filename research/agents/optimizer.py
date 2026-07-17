"""Optimizer agent: generates N variants from a thesis.

The Optimizer agent takes a :class:`~research.models.Thesis` and the
original concept, then produces *n* distinct variants that each target
the hypothesis in a different way.

Key design constraint: variants must be diverse.  The Optimizer is
explicitly instructed to use different strategies for each variant
rather than incremental tweaks of the first idea.
"""

from __future__ import annotations

import json

from anthropic import Anthropic

from research._llm_utils import LLMResponseError, extract_text, parse_json_array
from research.models import Thesis, Variant

_SYSTEM_PROMPT = """\
You are a creative optimizer.  You will receive:
1. A THESIS — a structured research framing with a hypothesis and
   improvement dimensions.
2. The ORIGINAL CONCEPT — the artefact to optimize.

Your task is to produce {n} distinct variants of the concept.  Each
variant MUST:
- Address the thesis hypothesis directly.
- Respect ALL listed constraints.
- Be meaningfully different from every other variant (different
  strategy, not just tweaks).
- Be complete and self-contained (not a diff — a full replacement).

You MUST output a single JSON array where each element has:
  - id: integer starting at 1
  - content: the full optimized variant
  - rationale: 1-2 sentences explaining the approach
  - changes_summary: bullet list of concrete changes made

Output ONLY valid JSON.  No markdown fences, no commentary outside JSON.
"""

#: Hard cap on the per-request output budget (model output-token limit).
_MAX_TOKENS_CAP = 32000

#: Fixed per-variant allowance for JSON structure, rationale and summary.
_PER_VARIANT_OVERHEAD_TOKENS = 512

#: Floor so tiny concepts still get a workable budget.
_MIN_MAX_TOKENS = 4096


def _request_max_tokens(concept: str, n: int) -> int:
    """Scale the output budget to fit *n* full-replacement variants.

    Each variant is a complete copy of the concept (~len(concept)//3
    tokens) plus JSON/rationale overhead.  The result is clamped to the
    model output limit; if the model still stops at ``max_tokens`` the
    caller raises instead of parsing a truncated response.  (Splitting
    generation across one request per variant would lift the ceiling
    further, but a single request with a fail-fast truncation check is
    the simpler design and keeps variant diversity in one context.)
    """
    per_variant = len(concept) // 3 + _PER_VARIANT_OVERHEAD_TOKENS
    return max(_MIN_MAX_TOKENS, min(_MAX_TOKENS_CAP, n * per_variant))


def optimize(
    thesis: Thesis,
    concept: str,
    n: int = 5,
    model: str = "claude-opus-4-6",
    api_key: str | None = None,
) -> list[Variant]:
    """Generate *n* optimized variants of *concept* guided by *thesis*.

    Args:
        thesis: The analysis output from the Analyzer.
        concept: The original artefact.
        n: Number of variants to generate.
        model: Anthropic model to use.
        api_key: Anthropic API key.

    Returns:
        List of :class:`Variant` objects.
    """
    client = Anthropic(api_key=api_key) if api_key else Anthropic()

    thesis_block = json.dumps(
        {
            "concept_summary": thesis.concept_summary,
            "hypothesis": thesis.hypothesis,
            "improvement_dimensions": thesis.improvement_dimensions,
            "test_strategy": thesis.test_strategy,
            "constraints": thesis.constraints,
        },
        indent=2,
    )

    user_content = f"THESIS:\n{thesis_block}\n\nORIGINAL CONCEPT:\n{concept}"

    message = client.messages.create(
        model=model,
        max_tokens=_request_max_tokens(concept, n),
        system=_SYSTEM_PROMPT.format(n=n),
        messages=[{"role": "user", "content": user_content}],
    )
    if getattr(message, "stop_reason", None) == "max_tokens":
        raise LLMResponseError(
            "Optimizer: response truncated at the max_tokens limit — the variants "
            "JSON is incomplete. Raise max_tokens or reduce --variants."
        )
    raw = extract_text(message)
    items = parse_json_array(raw)

    variants = []
    for item in items[:n]:
        if not isinstance(item, dict) or "id" not in item or "content" not in item:
            raise LLMResponseError(
                "Optimizer: each variant must be an object with 'id' and 'content' fields"
            )
        variants.append(
            Variant(
                id=int(item["id"]),
                content=item["content"],
                rationale=item.get("rationale", ""),
                changes_summary=item.get("changes_summary", ""),
            )
        )
    return variants
