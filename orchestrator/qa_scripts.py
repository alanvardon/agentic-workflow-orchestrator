"""Scripted QA gate — runs executable scripts before the LLM QA agent.

The orchestrator checks for a `.orchestrator/qa/` directory (configurable
via `qa_scripts_dir` in orchestrator.toml).  If the directory exists every
executable file inside it is run in lexicographic order.  A non-zero exit
from any script aborts QA immediately; the LLM QA agent is never called.

All script output (stdout, stderr) is captured and surfaced in the failure
report so developers can debug locally without re-running the orchestrator.

Windows caveat: `os.access(p, os.X_OK)` always returns True on Windows
because the OS has no executable-bit concept.  On Windows the gate falls
back to checking the file extension (.sh, .bat, .ps1, .py) to decide
whether a file is a script.  Repo owners on Windows should use one of
these extensions for their QA scripts.
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScriptResult:
    """Outcome of running a single QA script."""

    script_path: str   # absolute path to the script
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float


@dataclass
class ScriptedQaOutcome:
    """Aggregated result from running all scripted QA checks."""

    passed: bool
    results: list[ScriptResult] = field(default_factory=list)
    # Human-readable failure block; empty string when passed=True.
    failure_report: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WINDOWS_SCRIPT_EXTENSIONS = {".sh", ".bat", ".ps1", ".py", ".cmd"}


def _is_executable(p: Path) -> bool:
    """Return True if *p* should be treated as an executable QA script.

    On POSIX systems this checks the executable bit via `os.access`.
    On Windows (where the executable-bit concept does not exist) it falls
    back to checking whether the file extension is a known script type.
    """
    if platform.system() == "Windows":
        return p.suffix.lower() in _WINDOWS_SCRIPT_EXTENSIONS
    return os.access(p, os.X_OK)


def _build_failure_report(results: list[ScriptResult]) -> str:
    """Render a human-readable failure report for *results*.

    Includes every script that ran (including the passing ones that
    preceded the failure) so the reader has full context.
    """
    lines: list[str] = [
        "Scripted QA FAILED",
        "==================",
    ]
    for r in results:
        lines.append(
            f"Script: {r.script_path}  exit={r.exit_code}  duration={r.duration_s:.1f}s"
        )
        lines.append("--- stdout ---")
        lines.append(r.stdout if r.stdout.strip() else "(empty)")
        lines.append("--- stderr ---")
        lines.append(r.stderr if r.stderr.strip() else "(empty)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_qa_scripts(repo_root: Path, qa_scripts_dir: str) -> list[Path]:
    """Return sorted list of executable scripts in *qa_scripts_dir*.

    Resolves `repo_root / qa_scripts_dir`.  If the directory does not
    exist, returns an empty list (no scripts = gate passes trivially).
    Only files (not subdirectories) with the executable bit set are
    included.  Sorting is lexicographic by filename.
    """
    scripts_dir = repo_root / qa_scripts_dir
    if not scripts_dir.exists():
        return []

    candidates = [
        p for p in scripts_dir.iterdir()
        if p.is_file() and _is_executable(p)
    ]
    return sorted(candidates, key=lambda p: p.name)


def run_script(script: Path, timeout: int) -> ScriptResult:
    """Run *script* and return its result.

    The script's parent directory is used as cwd so relative-path
    references in scripts resolve predictably.

    `subprocess.TimeoutExpired` is caught and converted to exit code 124
    (the same convention used by the POSIX `timeout(1)` utility).

    `OSError` (the script passed the +x filter but can't actually exec — bad
    shebang, +x lost between glob and run, ENOEXEC) is caught and converted to
    exit code 126 (POSIX "command found but not executable"). Without this the
    OSError would escape `run_qa_scripts` → `qa()` → `qa_task` as a bare crash;
    converting it lets the existing fail-fast/failure_report path report it as
    an ordinary scripted-QA failure (→ QaResult FAIL), the same way
    `steps._run_script_sync` handles it.
    """
    import time

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script.parent,
        )
        duration = time.monotonic() - start
        return ScriptResult(
            script_path=str(script),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_s=duration,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        return ScriptResult(
            script_path=str(script),
            exit_code=124,
            stdout=exc.stdout or "",
            stderr=f"Script timed out after {timeout}s",
            duration_s=duration,
        )
    except OSError as exc:
        duration = time.monotonic() - start
        return ScriptResult(
            script_path=str(script),
            exit_code=126,
            stdout="",
            stderr=f"Script could not be executed: {exc}",
            duration_s=duration,
        )


def run_qa_scripts(
    repo_root: Path,
    qa_scripts_dir: str,
    timeout: int,
) -> ScriptedQaOutcome:
    """Run all scripted QA checks and return the aggregated outcome.

    Scripts are discovered via `find_qa_scripts` and executed in sorted
    order.  Execution stops on the first non-zero exit (fail-fast).

    Returns:
        ScriptedQaOutcome with passed=True if no scripts exist or all pass.
        ScriptedQaOutcome with passed=False and a populated failure_report
        if any script exits non-zero.
    """
    scripts = find_qa_scripts(repo_root, qa_scripts_dir)
    if not scripts:
        return ScriptedQaOutcome(passed=True, results=[], failure_report="")

    results: list[ScriptResult] = []
    for script in scripts:
        result = run_script(script, timeout)
        results.append(result)
        if result.exit_code != 0:
            # Fail-fast: stop after the first failing script.
            failure_report = _build_failure_report(results)
            return ScriptedQaOutcome(
                passed=False,
                results=results,
                failure_report=failure_report,
            )

    # All scripts passed.
    return ScriptedQaOutcome(passed=True, results=results, failure_report="")
