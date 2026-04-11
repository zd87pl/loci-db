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
import re
from typing import Any

from anthropic import Anthropic

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


def _extract_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


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

    user_content = (
        f"THESIS:\n{thesis_block}\n\n"
        f"ORIGINAL CONCEPT:\n{concept}"
    )

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT.format(n=n),
        messages=[{"role": "user", "content": user_content}],
    )
    raw = message.content[0].text
    items = _extract_json(raw)

    variants = []
    for item in items[:n]:
        variants.append(
            Variant(
                id=int(item["id"]),
                content=item["content"],
                rationale=item.get("rationale", ""),
                changes_summary=item.get("changes_summary", ""),
            )
        )
    return variants
