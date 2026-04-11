"""Judge agent: blind evaluation of variants.

The Judge agent is deliberately isolated from the original concept.
It receives only:
- The thesis (what we were trying to improve and why)
- The test results for each variant (scores + metrics)

It MUST NOT see the original concept or the variant content.
This design prevents the judge from being influenced by surface
aesthetics and forces it to reason purely from the thesis framing
and measured outcomes.
"""

from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic

from research.models import EvalResult, Thesis, Verdict

_SYSTEM_PROMPT = """\
You are an impartial research judge evaluating the results of an
optimization experiment.

You will receive:
1. THESIS — the research framing: what was being optimized and how.
2. EVAL RESULTS — test runner scores and metrics for each variant.

IMPORTANT: You have NOT seen the original concept or variant code/content.
You are evaluating based on the thesis framing and measured outcomes only.

Your task:
1. Identify which variant best satisfies the thesis hypothesis.
2. If no variant improves on the stated goals, recommend -1 (keep original).
3. Provide clear reasoning grounded in the thesis and metrics.

You MUST output a single JSON object with:
  - winner_id: integer ID of the best variant, or -1 if none win
  - reasoning: 2-3 paragraph explanation of your judgment
  - scores: object mapping variant id (as string) to your 0.0-1.0 score
  - recommendation: one concrete next step (e.g. "deploy variant 3",
    "combine the approach from variant 2 with variant 4",
    "reframe the hypothesis — none of these variants address the root cause")

Output ONLY valid JSON.  No markdown fences, no commentary outside JSON.
"""


def _format_results(eval_results: list[EvalResult]) -> str:
    rows = []
    for r in eval_results:
        rows.append(
            {
                "variant_id": r.variant_id,
                "score": round(r.score, 4),
                "passed": r.passed,
                "metrics": r.metrics,
                "details": r.details,
            }
        )
    return json.dumps(rows, indent=2)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


def judge(
    thesis: Thesis,
    eval_results: list[EvalResult],
    model: str = "claude-opus-4-6",
    api_key: str | None = None,
) -> Verdict:
    """Produce a blind verdict based only on thesis + eval results.

    Args:
        thesis: The Analyzer's thesis.
        eval_results: Scored results from the test runner.
        model: Anthropic model to use.
        api_key: Anthropic API key.

    Returns:
        :class:`Verdict` with winner ID and reasoning.
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
        f"EVAL RESULTS:\n{_format_results(eval_results)}"
    )

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = message.content[0].text
    data = _extract_json(raw)

    scores = {int(k): float(v) for k, v in data.get("scores", {}).items()}

    return Verdict(
        winner_id=int(data["winner_id"]),
        reasoning=data["reasoning"],
        scores=scores,
        recommendation=data.get("recommendation", ""),
    )
