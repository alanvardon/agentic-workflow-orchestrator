"""Pre-flight hook runner — executes user-defined scripts before LLM calls.

The orchestrator checks for a `.orchestrator/pre-hooks/` directory
(configurable via `pre_hooks_dir` in orchestrator.toml). If the directory
exists, every executable file inside it is run in lexicographic order.
A non-zero exit from any script raises PreHookError immediately (fail-fast);
subsequent scripts are not executed.

All script output (stdout, stderr) is captured. On failure, stdout becomes
the displayed abort reason so hook authors can write clear, actionable
messages.

Scripts run with the repo root as their working directory, giving them
stable access to repo files regardless of where the orchestrator was invoked.

Windows caveat: `os.access(p, os.X_OK)` always returns True on Windows
because the OS has no executable-bit concept. On Windows the gate falls
back to checking the file extension (.sh, .bat, .ps1, .py) to decide
whether a file is a script. Repo owners on Windows should use one of
these extensions for their hook scripts.
"""

import os
import platform
import subprocess
from pathlib import Path

from orchestrator.git_ops import REPO_ROOT, PreHookError


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WINDOWS_SCRIPT_EXTENSIONS = {".sh", ".bat", ".ps1", ".py", ".cmd"}


def _is_executable(p: Path) -> bool:
    """Return True if *p* should be treated as an executable hook script.

    On POSIX systems this checks the executable bit via `os.access`.
    On Windows (where the executable-bit concept does not exist) it falls
    back to checking whether the file extension is a known script type.
    """
    if platform.system() == "Windows":
        return p.suffix.lower() in _WINDOWS_SCRIPT_EXTENSIONS
    return os.access(p, os.X_OK)


def _find_hook_scripts(hooks_dir: Path) -> list[Path]:
    """Return sorted list of executable scripts in *hooks_dir*.

    If the directory does not exist or is empty, returns an empty list
    (no hooks = gate passes trivially). Only files (not subdirectories)
    with the executable bit set are included. Sorting is lexicographic
    by filename.
    """
    if not hooks_dir.exists():
        return []

    candidates = [
        p for p in hooks_dir.iterdir()
        if p.is_file() and _is_executable(p)
    ]
    return sorted(candidates, key=lambda p: p.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pre_hooks(hooks_dir: str | Path, timeout: int) -> None:
    """Run all pre-hook scripts; raise PreHookError on the first failure.

    Resolves *hooks_dir* relative to the repo root. If the directory does
    not exist or contains no executable files, this is a silent no-op.

    Scripts are executed in lexicographic order. Execution stops on the
    first non-zero exit (fail-fast). On timeout, PreHookError is raised
    with exit code 124 (the same convention used by the POSIX `timeout(1)`
    utility).

    Args:
        hooks_dir: Path (relative to repo root, or absolute) to the hooks
                   directory. Relative paths are resolved against REPO_ROOT.
        timeout:   Per-script timeout in seconds.

    Raises:
        PreHookError: If any script exits with a non-zero status or times out.
    """
    resolved = REPO_ROOT / hooks_dir
    scripts = _find_hook_scripts(resolved)
    if not scripts:
        return

    for script in scripts:
        try:
            proc = subprocess.run(
                [str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=REPO_ROOT,
            )
            if proc.returncode != 0:
                raise PreHookError(
                    script=script.name,
                    output=proc.stdout,
                    returncode=proc.returncode,
                )
        except subprocess.TimeoutExpired:
            raise PreHookError(
                script=script.name,
                output=f"Script timed out after {timeout}s",
                returncode=124,
            )
