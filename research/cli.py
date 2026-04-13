"""CLI entry point for the auto-research pipeline.

Usage::

    # Optimize a file (code, copy, content — anything text)
    python -m research.cli run --input path/to/concept.py \\
        --context "Python web handler for a REST API" \\
        --variants 5

    # Pipe content directly
    echo "def add(a, b): return a+b" | python -m research.cli run --context "Python function"

    # Use a code test runner (runs pytest after writing each variant)
    python -m research.cli run --input src/module.py \\
        --runner code \\
        --test-cmd "pytest tests/ -q" \\
        --work-dir /path/to/project
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """Auto-research pipeline: Analyze → Optimize → Test → Judge."""


@cli.command()
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), default=None,
              help="Path to the concept file.  Reads stdin if not provided.")
@click.option("--context", "-c", default="", help="Optional background context for the Analyzer.")
@click.option("--variants", "-n", default=5, show_default=True,
              help="Number of optimization variants to generate.")
@click.option("--runner", type=click.Choice(["llm", "code", "metric"]), default="llm",
              show_default=True, help="Which test runner to use.")
@click.option("--test-cmd", default="pytest --tb=short -q",
              help="Test command for the 'code' runner.")
@click.option("--work-dir", type=click.Path(), default=None,
              help="Working directory for the 'code' runner.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write full JSON result to this path.")
@click.option("--save-winner", type=click.Path(), default=None,
              help="Write the winning variant content to this path.")
@click.option("--analyzer-model", default="claude-opus-4-6", show_default=True)
@click.option("--optimizer-model", default="claude-opus-4-6", show_default=True)
@click.option("--judge-model", default="claude-opus-4-6", show_default=True)
@click.option("--runner-model", default="claude-haiku-4-5-20251001", show_default=True,
              help="Model for the LLM runner (only used when --runner=llm).")
def run(
    input_path: str | None,
    context: str,
    variants: int,
    runner: str,
    test_cmd: str,
    work_dir: str | None,
    output: str | None,
    save_winner: str | None,
    analyzer_model: str,
    optimizer_model: str,
    judge_model: str,
    runner_model: str,
) -> None:
    """Run the auto-research pipeline on a concept."""
    from research.pipeline import ResearchPipeline
    from research.runners.code import CodeRunner
    from research.runners.llm import LLMRunner

    # Read concept
    concept = Path(input_path).read_text(encoding="utf-8") if input_path else sys.stdin.read()

    if not concept.strip():
        click.echo("Error: no concept provided (empty input).", err=True)
        raise SystemExit(1)

    # Build runner
    if runner == "llm":
        test_runner = LLMRunner(model=runner_model)
    elif runner == "code":
        if not input_path:
            click.echo("Error: --runner=code requires --input.", err=True)
            raise SystemExit(1)
        test_runner = CodeRunner(
            target_path=input_path,
            test_cmd=test_cmd,
            work_dir=work_dir or str(Path(input_path).parent),
        )
    else:
        click.echo(
            f"Runner '{runner}' requires custom setup — falling back to LLM runner.", err=True
        )
        test_runner = LLMRunner(model=runner_model)

    pipeline = ResearchPipeline(
        runner=test_runner,
        n_variants=variants,
        analyzer_model=analyzer_model,
        optimizer_model=optimizer_model,
        judge_model=judge_model,
    )

    result = pipeline.run(concept=concept, context=context)

    click.echo("")
    click.echo(result.summary())

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        click.echo(f"\nFull result written to: {output}")

    if save_winner:
        winner_content = pipeline.get_winner_content(result)
        if winner_content:
            Path(save_winner).write_text(winner_content, encoding="utf-8")
            click.echo(f"Winner written to: {save_winner}")
        else:
            click.echo("No winner to save (judge selected original).")


if __name__ == "__main__":
    cli()
