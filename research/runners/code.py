"""Code test runner.

Writes a variant's content to a temporary file and runs a test command
(default: ``pytest``).  The score is derived from test pass rate and
optional coverage.

Use this runner when the concept being optimized is executable code.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from research.models import EvalResult, Thesis, Variant
from research.runners.base import BaseRunner


class CodeRunner(BaseRunner):
    """Runs a test suite against each variant.

    Args:
        target_path: Path where the variant content should be written
            before running the test command.  E.g. ``"src/module.py"``.
        test_cmd: Shell command to run tests.  Defaults to ``pytest``.
        work_dir: Working directory for the test command.
        timeout: Seconds before killing the test run.
    """

    def __init__(
        self,
        target_path: str | Path,
        test_cmd: str = "pytest --tb=short -q",
        work_dir: str | Path | None = None,
        timeout: int = 120,
    ) -> None:
        self.target_path = Path(target_path)
        self.test_cmd = test_cmd
        self.work_dir = Path(work_dir) if work_dir else self.target_path.parent
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_pytest_output(self, stdout: str) -> dict[str, float]:
        """Extract pass/fail counts from pytest output."""
        metrics: dict[str, float] = {}
        # e.g. "5 passed, 2 failed in 1.23s"
        m = re.search(r"(\d+) passed", stdout)
        if m:
            metrics["passed"] = float(m.group(1))
        m = re.search(r"(\d+) failed", stdout)
        if m:
            metrics["failed"] = float(m.group(1))
        m = re.search(r"(\d+) error", stdout)
        if m:
            metrics["errors"] = float(m.group(1))

        passed = metrics.get("passed", 0.0)
        failed = metrics.get("failed", 0.0)
        errors = metrics.get("errors", 0.0)
        total = passed + failed + errors
        if total > 0:
            metrics["pass_rate"] = passed / total
        else:
            metrics["pass_rate"] = 0.0
        return metrics

    # ------------------------------------------------------------------
    # BaseRunner interface
    # ------------------------------------------------------------------

    def evaluate(self, variant: Variant, thesis: Thesis) -> EvalResult:
        # Write variant content to target path (backup original first)
        backup: bytes | None = None
        if self.target_path.exists():
            backup = self.target_path.read_bytes()

        try:
            self.target_path.write_text(variant.content, encoding="utf-8")

            result = subprocess.run(
                self.test_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.work_dir,
                timeout=self.timeout,
            )
            output = result.stdout + result.stderr
            metrics = self._parse_pytest_output(output)

            passed = result.returncode == 0
            score = metrics.get("pass_rate", 0.0)

            return EvalResult(
                variant_id=variant.id,
                score=score,
                metrics=metrics,
                passed=passed,
                details=output[-500:] if output else "",
            )

        except subprocess.TimeoutExpired:
            return EvalResult(
                variant_id=variant.id,
                score=0.0,
                metrics={"error": "timeout"},
                passed=False,
                details=f"Test run timed out after {self.timeout}s",
            )
        except Exception as exc:
            return EvalResult(
                variant_id=variant.id,
                score=0.0,
                metrics={"error": str(exc)},
                passed=False,
                details=str(exc),
            )
        finally:
            # Restore original
            if backup is not None:
                self.target_path.write_bytes(backup)
