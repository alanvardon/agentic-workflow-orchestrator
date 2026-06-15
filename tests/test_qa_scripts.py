"""Tests for orchestrator.qa_scripts — scripted QA gate (Phase 28).

Uses pytest's tmp_path fixture to create real executable scripts on disk
so the tests exercise the actual subprocess machinery without mocking it.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

from orchestrator.qa_scripts import (
    ScriptedQaOutcome,
    ScriptResult,
    _build_failure_report,
    find_qa_scripts,
    run_qa_scripts,
    run_script,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(directory: Path, name: str, content: str) -> Path:
    """Write *content* to *directory/name*, mark it executable, return path."""
    path = directory / name
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_non_executable_file(directory: Path, name: str, content: str = "") -> Path:
    """Write a regular (non-executable) file to *directory/name*."""
    path = directory / name
    path.write_text(content)
    # Explicitly remove executable bit to be sure.
    current = path.stat().st_mode
    path.chmod(current & ~(stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH))
    return path


def _python_exit(code: int) -> str:
    """Return a shebang script body that exits with *code*."""
    python = sys.executable
    return f"#!{python}\nimport sys\nsys.exit({code})\n"


# ---------------------------------------------------------------------------
# Scenario 1 — qa directory does not exist
# ---------------------------------------------------------------------------


def test_no_qa_directory(tmp_path: Path) -> None:
    """find_qa_scripts returns [] when the qa dir is absent."""
    outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
    assert outcome.passed is True
    assert outcome.results == []
    assert outcome.failure_report == ""


# ---------------------------------------------------------------------------
# Scenario 2 — folder exists but contains no executable files
# ---------------------------------------------------------------------------


def test_no_executable_files(tmp_path: Path) -> None:
    """Gate passes trivially when the folder has no executable files."""
    qa_dir = tmp_path / ".orchestrator" / "qa"
    qa_dir.mkdir(parents=True)
    _make_non_executable_file(qa_dir, "README.txt", "not a script")

    outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
    assert outcome.passed is True
    assert outcome.results == []


# ---------------------------------------------------------------------------
# Scenario 3 — one executable script exits 0
# ---------------------------------------------------------------------------


def test_single_passing_script(tmp_path: Path) -> None:
    """One script that exits 0 → passed=True, one ScriptResult with exit_code=0."""
    qa_dir = tmp_path / ".orchestrator" / "qa"
    qa_dir.mkdir(parents=True)
    _make_script(qa_dir, "10_check.py", _python_exit(0))

    outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
    assert outcome.passed is True
    assert len(outcome.results) == 1
    assert outcome.results[0].exit_code == 0


# ---------------------------------------------------------------------------
# Scenario 4 — two scripts: first passes, second fails
# ---------------------------------------------------------------------------


def test_two_scripts_second_fails(tmp_path: Path) -> None:
    """Both scripts run; second failure → passed=False, two results, non-empty report."""
    qa_dir = tmp_path / ".orchestrator" / "qa"
    qa_dir.mkdir(parents=True)
    _make_script(qa_dir, "10_pass.py", _python_exit(0))
    _make_script(qa_dir, "20_fail.py", _python_exit(1))

    outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
    assert outcome.passed is False
    assert len(outcome.results) == 2
    assert outcome.results[0].exit_code == 0
    assert outcome.results[1].exit_code == 1
    assert outcome.failure_report != ""


# ---------------------------------------------------------------------------
# Scenario 5 — first script fails immediately (fail-fast)
# ---------------------------------------------------------------------------


def test_fail_fast_stops_after_first_failure(tmp_path: Path) -> None:
    """Only the first (failing) script is recorded; second script never runs."""
    qa_dir = tmp_path / ".orchestrator" / "qa"
    qa_dir.mkdir(parents=True)
    _make_script(qa_dir, "10_fail.py", _python_exit(2))
    _make_script(qa_dir, "20_also_fail.py", _python_exit(3))

    outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
    assert outcome.passed is False
    # Only the first script should appear in results.
    assert len(outcome.results) == 1
    assert outcome.results[0].exit_code == 2


# ---------------------------------------------------------------------------
# Scenario 6 — scripts run in sorted (lexicographic) order
# ---------------------------------------------------------------------------


def test_scripts_run_in_sorted_order(tmp_path: Path) -> None:
    """Verify that scripts execute in lexicographic filename order."""
    qa_dir = tmp_path / ".orchestrator" / "qa"
    qa_dir.mkdir(parents=True)

    # Create scripts whose names sort as: 10, 20, 30 — add them out-of-order.
    _make_script(qa_dir, "30_c.py", _python_exit(0))
    _make_script(qa_dir, "10_a.py", _python_exit(0))
    _make_script(qa_dir, "20_b.py", _python_exit(0))

    outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
    assert outcome.passed is True

    names = [Path(r.script_path).name for r in outcome.results]
    assert names == ["10_a.py", "20_b.py", "30_c.py"]


# ---------------------------------------------------------------------------
# Scenario 7 — script exceeds timeout
# ---------------------------------------------------------------------------


def test_script_timeout(tmp_path: Path) -> None:
    """A script that sleeps longer than the timeout → exit_code=124."""
    qa_dir = tmp_path / ".orchestrator" / "qa"
    qa_dir.mkdir(parents=True)

    python = sys.executable
    _make_script(
        qa_dir,
        "slow.py",
        f"#!{python}\nimport time\ntime.sleep(60)\n",
    )

    # Use a 1-second timeout so the test finishes quickly.
    outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=1)
    assert outcome.passed is False
    assert len(outcome.results) == 1
    assert outcome.results[0].exit_code == 124
    assert "timed out" in outcome.results[0].stderr.lower()


# ---------------------------------------------------------------------------
# Scenario 8 — non-executable file in folder is ignored
# ---------------------------------------------------------------------------


def test_non_executable_file_ignored(tmp_path: Path) -> None:
    """A plain file (no executable bit) is not included in the gate."""
    qa_dir = tmp_path / ".orchestrator" / "qa"
    qa_dir.mkdir(parents=True)
    _make_non_executable_file(qa_dir, "notes.txt", "just a readme")

    scripts = find_qa_scripts(tmp_path, ".orchestrator/qa")
    assert scripts == []


# ---------------------------------------------------------------------------
# Scenario 9 — _build_failure_report renders correctly
# ---------------------------------------------------------------------------


def test_build_failure_report_content() -> None:
    """_build_failure_report includes expected stdout/stderr blocks."""
    results = [
        ScriptResult(
            script_path="/repo/.orchestrator/qa/10_check.sh",
            exit_code=0,
            stdout="all good",
            stderr="",
            duration_s=0.5,
        ),
        ScriptResult(
            script_path="/repo/.orchestrator/qa/20_lint.sh",
            exit_code=1,
            stdout="",
            stderr="error: lint failed",
            duration_s=1.2,
        ),
    ]
    report = _build_failure_report(results)

    assert "Scripted QA FAILED" in report
    assert "/repo/.orchestrator/qa/10_check.sh" in report
    assert "exit=0" in report
    assert "all good" in report
    assert "(empty)" in report  # empty stderr for first script
    assert "/repo/.orchestrator/qa/20_lint.sh" in report
    assert "exit=1" in report
    assert "error: lint failed" in report
    assert "(empty)" in report  # empty stdout for second script
