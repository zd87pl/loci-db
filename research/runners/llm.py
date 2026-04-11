"""LLM-based test runner.

Scores a variant by asking a language model to evaluate it against
the improvement dimensions in the thesis.  This runner requires no
tooling and works for any concept type (code, copy, content, strategy).

It is the default runner when no domain-specific runner is configured.
"""

from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic

from research.models import EvalResult, Thesis, Variant
from research.runners.base import BaseRunner

_SYSTEM_PROMPT = """\
You are an objective evaluator.  You will receive a THESIS and a
VARIANT to score.

Score the variant against each improvement dimension in the thesis.
Return a JSON object with:
  - dimension_scores: object mapping each dimension name to a score
    in [0.0, 1.0] (1.0 = maximally improved, 0.0 = unchanged or worse)
  - overall_score: weighted average (you pick weights based on importance)
  - constraints_satisfied: boolean — true if all constraints are met
  - details: 1-2 sentences summarising the evaluation

Output ONLY valid JSON.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


class LLMRunner(BaseRunner):
    """Scores variants using an LLM.

    Args:
        model: Anthropic model to use for scoring.
        api_key: Anthropic API key.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def evaluate(self, variant: Variant, thesis: Thesis) -> EvalResult:
        thesis_block = json.dumps(
            {
                "concept_summary": thesis.concept_summary,
                "hypothesis": thesis.hypothesis,
                "improvement_dimensions": thesis.improvement_dimensions,
                "constraints": thesis.constraints,
            },
            indent=2,
        )

        user_content = (
            f"THESIS:\n{thesis_block}\n\n"
            f"VARIANT {variant.id}:\n{variant.content}\n\n"
            f"RATIONALE: {variant.rationale}"
        )

        message = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = message.content[0].text
        data = _extract_json(raw)

        return EvalResult(
            variant_id=variant.id,
            score=float(data.get("overall_score", 0.0)),
            metrics=data.get("dimension_scores", {}),
            passed=bool(data.get("constraints_satisfied", True)),
            details=data.get("details", ""),
        )
