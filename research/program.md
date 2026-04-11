# Auto-Research Pipeline — Autonomous Experiment Protocol

Inspired by Karpathy's autoresearch and adapted for any concept (code,
marketing copy, content, strategy, product specs, etc.).

## What this pipeline does

```
Concept → [Analyzer] → Thesis → [Optimizer] → Variants
                                                   ↓
              Verdict ← [Judge (blind)] ← [Test Runner]
```

1. **Analyzer** — reads the concept, forms a falsifiable hypothesis
   about the single most impactful improvement axis, and defines how to
   measure improvement.

2. **Optimizer** — generates N distinct variants that each attack the
   hypothesis differently.  Variants must be complete replacements (not
   diffs) and must be meaningfully different from each other.

3. **Test Runner** — evaluates each variant against the thesis.  For
   code: runs the test suite.  For content: scores readability, word
   count, or any custom metric.  For anything else: uses an LLM to
   score improvement on each dimension.

4. **Judge** — receives only the thesis + test results.  The judge
   has **NOT** seen the original concept or the variant content.  It
   picks the winner (or recommends keeping the original) based purely
   on whether the metrics support the hypothesis.

## Running a pipeline

```bash
# Install dependencies (anthropic SDK required)
pip install anthropic

# Optimize a Python file using LLM scoring
python -m research.cli run \
    --input path/to/code.py \
    --context "Production web handler for the LOCI vector DB API" \
    --variants 5

# Optimize with a real test suite
python -m research.cli run \
    --input loci/temporal/sharding.py \
    --runner code \
    --test-cmd "pytest tests/ -q" \
    --work-dir /path/to/loci-db \
    --variants 3

# Optimize marketing copy (piped)
cat landing_page_copy.txt | python -m research.cli run \
    --context "B2B SaaS landing page targeting ML engineers" \
    --variants 5 \
    --output results.json
```

## Adding a custom test runner

Implement `BaseRunner.evaluate()`:

```python
from research.runners.base import BaseRunner
from research.models import EvalResult, Thesis, Variant

class MyRunner(BaseRunner):
    def evaluate(self, variant: Variant, thesis: Thesis) -> EvalResult:
        # score in [0.0, 1.0], 1.0 = best possible
        score = my_scoring_fn(variant.content)
        return EvalResult(
            variant_id=variant.id,
            score=score,
            metrics={"my_metric": score},
            passed=True,
        )

from research.pipeline import ResearchPipeline
pipeline = ResearchPipeline(runner=MyRunner(), n_variants=5)
result = pipeline.run(concept=my_text)
```

## Autonomous loop (autoresearch-style)

For long-running autonomous improvement, wrap the pipeline in a loop:

```python
from research.pipeline import ResearchPipeline

pipeline = ResearchPipeline(n_variants=5)
best_concept = original_concept

for iteration in range(max_iterations):
    result = pipeline.run(concept=best_concept, context=context)
    winner = pipeline.get_winner_content(result)
    if winner:
        best_concept = winner  # carry the winner into next iteration
        print(f"Iteration {iteration}: improved (variant {result.verdict.winner_id})")
    else:
        print(f"Iteration {iteration}: no improvement, keeping current best")
```

## Agent roles

| Agent | Sees | Produces | Model |
|-------|------|----------|-------|
| Analyzer | Full concept + context | Thesis | claude-opus-4-6 |
| Optimizer | Thesis + concept | N variants | claude-opus-4-6 |
| Test Runner | Thesis + each variant | EvalResult per variant | configurable |
| Judge | Thesis + EvalResults only | Verdict (winner ID) | claude-opus-4-6 |

The Judge's isolation is the key design invariant: it cannot be biased
by the surface form of variants.  It reasons only from structured
metrics and the original hypothesis.
