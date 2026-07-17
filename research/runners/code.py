"""Code test runner.

Evaluates each variant in an isolated temporary copy of the project:
the directory tree rooted at ``work_dir`` is copied into a fresh
temporary workspace, the variant content is written into the *copy*,
and the test command runs with the copy as its working directory.
The user's real source tree is never modified.

Inside the workspace, the original file (if it exists) is additionally
preserved as ``<target>.orig`` next to the overwritten copy so a failed
or killed test run can be inspected against the pristine content.

Isolation scope: this protects the real tree from corruption and
accidental in-place damage (including SIGKILL/OOM mid-run — only the
throwaway copy is ever written to).  It is NOT a security sandbox:
model-generated code still executes with the current user's privileges
and network access.  Run under a container/VM when evaluating untrusted
concepts.

The score is derived from test pass rate parsed from pytest-style
output.

Use this runner when the concept being optimized is executable code.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from research.models import EvalResult, Thesis, Variant
from research.runners.base import BaseRunner

#: Directories that are never copied into the temporary workspace.
_DEFAULT_COPY_IGNORE: tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "*.egg-info",
)


class CodeRunner(BaseRunner):
    """Runs a test suite against each variant in an isolated workspace.

    Args:
        target_path: Path of the file the variant content replaces.
            Must live inside ``work_dir``.  May be given relative to
            ``work_dir`` (e.g. ``"src/module.py"``) or absolute.
        test_cmd: Test command.  A sequence of arguments (e.g.
            ``["pytest", "-q"]``) is executed directly without a shell.
            A plain string is executed through the shell (``shell=True``)
            to support pipes/globs; this is safe only because the command
            comes from the pipeline *operator*, never from the model —
            the model-supplied code is confined to the temporary
            workspace via ``cwd``.  Defaults to ``pytest``.
        work_dir: Project root that is copied into the temporary
            workspace and used as the test command's working directory
            (in the copy).  Defaults to the parent of ``target_path``.
        timeout: Seconds before killing the test run.
        copy_ignore: Glob patterns excluded from the workspace copy
            (defaults to VCS/venv/cache directories).
    """

    def __init__(
        self,
        target_path: str | Path,
        test_cmd: str | Sequence[str] = "pytest --tb=short -q",
        work_dir: str | Path | None = None,
        timeout: int = 120,
        copy_ignore: Sequence[str] = _DEFAULT_COPY_IGNORE,
    ) -> None:
        self.target_path = Path(target_path)
        self.test_cmd = test_cmd
        self.work_dir = Path(work_dir).resolve() if work_dir else self.target_path.resolve().parent
        self.timeout = timeout
        self.copy_ignore = tuple(copy_ignore)

        # Locate the target file relative to the project root so it can be
        # addressed inside the workspace copy.
        if self.target_path.is_absolute():
            resolved_target = self.target_path.resolve()
        else:
            resolved_target = (self.work_dir / self.target_path).resolve()
        try:
            self._target_rel = resolved_target.relative_to(self.work_dir)
        except ValueError as exc:
            raise ValueError(
                f"target_path {resolved_target} must be inside work_dir {self.work_dir}"
            ) from exc

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

    def _make_workspace(self, tmp_root: Path) -> Path:
        """Copy the project into *tmp_root* and return the workspace path."""
        workspace = tmp_root / "workspace"
        shutil.copytree(
            self.work_dir,
            workspace,
            ignore=shutil.ignore_patterns(*self.copy_ignore),
            symlinks=True,
        )
        return workspace

    # ------------------------------------------------------------------
    # BaseRunner interface
    # ------------------------------------------------------------------

    def evaluate(self, variant: Variant, thesis: Thesis) -> EvalResult:
        # NOTE: isolation, not sandboxing.  The variant is written into a
        # throwaway copy of the project and the test command runs there, so
        # the real tree is never touched — even if this process is killed
        # (SIGKILL/OOM) mid-run, only the temp copy is affected.  The code
        # still runs unsandboxed on the host; use a container for untrusted
        # input.
        tmp_root = Path(tempfile.mkdtemp(prefix="research-coderunner-"))
        try:
            workspace = self._make_workspace(tmp_root)
            target = workspace / self._target_rel
            target.parent.mkdir(parents=True, exist_ok=True)

            # Belt-and-braces backup alongside the file we overwrite, for
            # post-mortem inspection of failed runs while the workspace lives.
            if target.exists():
                target.with_name(target.name + ".orig").write_bytes(target.read_bytes())

            target.write_text(variant.content, encoding="utf-8")

            result = subprocess.run(  # noqa: S602
                self.test_cmd,
                shell=isinstance(self.test_cmd, str),
                capture_output=True,
                text=True,
                cwd=workspace,
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
            shutil.rmtree(tmp_root, ignore_errors=True)
