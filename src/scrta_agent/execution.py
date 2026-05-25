from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .utils import write_text


@dataclass
class ScriptExecutionResult:
    returncode: int
    stdout_path: str
    stderr_path: str
    attempts: int
    status: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Script Execution",
                "",
                f"- Status: {self.status}",
                f"- Attempts: {self.attempts}",
                f"- Return code: {self.returncode}",
                f"- Stdout: {self.stdout_path}",
                f"- Stderr: {self.stderr_path}",
            ]
        ) + "\n"


def execute_python_script(
    script_path: Path,
    run_dir: Path,
    timeout_seconds: int = 7200,
    repair_attempts: int = 0,
    log_prefix: str = "script",
) -> ScriptExecutionResult:
    """Execute a generated Python script with bounded retry bookkeeping.

    Retries are conservative: they rerun after transient failures, but they do
    not silently mutate generated code. Failed stderr is preserved for the
    code-writer/critic stage.
    """
    attempts = max(1, int(repair_attempts) + 1)
    last_completed: subprocess.CompletedProcess[str] | None = None
    script_path = script_path.resolve()
    run_dir = run_dir.resolve()
    stdout_path = run_dir / f"{log_prefix}_stdout.log"
    stderr_path = run_dir / f"{log_prefix}_stderr.log"

    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=run_dir,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        last_completed = completed
        suffix = "" if attempt == attempts or completed.returncode == 0 else f".attempt{attempt}"
        write_text(run_dir / f"{log_prefix}_stdout{suffix}.log", completed.stdout)
        write_text(run_dir / f"{log_prefix}_stderr{suffix}.log", completed.stderr)
        if completed.returncode == 0:
            stdout_path = run_dir / f"{log_prefix}_stdout{suffix}.log"
            stderr_path = run_dir / f"{log_prefix}_stderr{suffix}.log"
            return ScriptExecutionResult(
                returncode=0,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                attempts=attempt,
                status="completed",
            )

    assert last_completed is not None
    write_text(stdout_path, last_completed.stdout)
    write_text(stderr_path, last_completed.stderr)
    return ScriptExecutionResult(
        returncode=last_completed.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        attempts=attempts,
        status="failed; inspect stderr before interpreting results",
    )
