"""Tests for orchestrator.pre_hooks — pre-flight hook runner (Phase 29).

Uses pytest's tmp_path fixture to create real executable scripts on disk
so the tests exercise the actual subprocess machinery without mocking it.

Because run_pre_hooks resolves its hooks_dir relative to REPO_ROOT, these
tests pass an absolute tmp_path so Python's Path division short-circuits
the REPO_ROOT prefix (Path("/repo") / Path("/tmp/x") == Path("/tmp/x")).
"""

import stat
import sys
from pathlib import Path

import pytest

from orchestrator.git_ops import PreHookError
from orchestrator.pre_hooks import run_pre_hooks
from orchestrator.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_script(directory: Path, name: str, content: str) -> Path:
    """Write *content* to *directory/name*, mark it executable, return path."""
    path = directory / name
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _python_exit(code: int) -> str:
    """Return a shebang script body that exits with *code*."""
    python = sys.executable
    return f"#!{python}\nimport sys\nsys.exit({code})\n"


def _python_print_exit(message: str, code: int) -> str:
    """Return a shebang script that prints *message* to stdout then exits."""
    python = sys.executable
    return f"#!{python}\nimport sys\nprint({message!r})\nsys.exit({code})\n"


# ---------------------------------------------------------------------------
# Scenario 1 — hooks directory does not exist
# ---------------------------------------------------------------------------


def test_no_hooks_directory(tmp_path: Path) -> None:
    """run_pre_hooks is a no-op when the hooks directory is absent."""
    hooks_dir = tmp_path / ".orchestrator" / "pre-hooks"
    # Directory intentionally not created.
    run_pre_hooks(hooks_dir, timeout=10)  # must not raise


# ---------------------------------------------------------------------------
# Scenario 2 — directory exists but contains no executable files
# ---------------------------------------------------------------------------


def test_no_executable_files(tmp_path: Path) -> None:
    """Gate passes trivially when the folder has no executable files."""
    hooks_dir = tmp_path / ".orchestrator" / "pre-hooks"
    hooks_dir.mkdir(parents=True)
    # Write a plain non-executable file.
    readme = hooks_dir / "README.md"
    readme.write_text("not a script")
    current = readme.stat().st_mode
    readme.chmod(current & ~(stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH))

    run_pre_hooks(hooks_dir, timeout=10)  # must not raise


# ---------------------------------------------------------------------------
# Scenario 3 — single passing script (exit 0)
# ---------------------------------------------------------------------------


def test_single_passing_script(tmp_path: Path) -> None:
    """One script that exits 0 does not raise."""
    hooks_dir = tmp_path / ".orchestrator" / "pre-hooks"
    hooks_dir.mkdir(parents=True)
    _make_script(hooks_dir, "10_check.py", _python_exit(0))

    run_pre_hooks(hooks_dir, timeout=10)  # must not raise


# ---------------------------------------------------------------------------
# Scenario 4 — single failing script (exit 1)
# ---------------------------------------------------------------------------


def test_single_failing_script_raises_pre_hook_error(tmp_path: Path) -> None:
    """A script that exits 1 raises PreHookError with stdout captured."""
    hooks_dir = tmp_path / ".orchestrator" / "pre-hooks"
    hooks_dir.mkdir(parents=True)
    _make_script(
        hooks_dir,
        "10_fail.py",
        _python_print_exit("abort: lint failed", 1),
    )

    with pytest.raises(PreHookError) as exc_info:
        run_pre_hooks(hooks_dir, timeout=10)

    err = exc_info.value
    assert err.script == "10_fail.py"
    assert err.returncode == 1
    assert "abort: lint failed" in err.output


# ---------------------------------------------------------------------------
# Scenario 5 — fail-fast: second script not executed after first fails
# ---------------------------------------------------------------------------


def test_fail_fast_stops_after_first_failure(tmp_path: Path) -> None:
    """Only the first (failing) script runs; the second is never executed."""
    hooks_dir = tmp_path / ".orchestrator" / "pre-hooks"
    hooks_dir.mkdir(parents=True)
    marker_file = tmp_path / "second_script_ran.txt"

    _make_script(hooks_dir, "10_fail.py", _python_exit(1))

    python = sys.executable
    _make_script(
        hooks_dir,
        "20_should_not_run.py",
        f"#!{python}\nopen({str(marker_file)!r}, 'w').write('ran')\n",
    )

    with pytest.raises(PreHookError) as exc_info:
        run_pre_hooks(hooks_dir, timeout=10)

    assert exc_info.value.script == "10_fail.py"
    assert not marker_file.exists(), "second script must not have run (fail-fast)"


# ---------------------------------------------------------------------------
# Scenario 6 — sorted order: scripts run in lexicographic order
# ---------------------------------------------------------------------------


def test_scripts_run_in_sorted_order(tmp_path: Path) -> None:
    """Scripts are executed in lexicographic filename order."""
    hooks_dir = tmp_path / ".orchestrator" / "pre-hooks"
    hooks_dir.mkdir(parents=True)
    order_file = tmp_path / "order.txt"

    python = sys.executable
    # Add scripts in non-sorted order.
    for name in ["30_c.py", "10_a.py", "20_b.py"]:
        _make_script(
            hooks_dir,
            name,
            (
                f"#!{python}\n"
                f"with open({str(order_file)!r}, 'a') as f:\n"
                f"    f.write({name!r} + '\\n')\n"
            ),
        )

    run_pre_hooks(hooks_dir, timeout=10)  # all exit 0, must not raise

    names = order_file.read_text().strip().splitlines()
    assert names == ["10_a.py", "20_b.py", "30_c.py"]


# ---------------------------------------------------------------------------
# Scenario 7 — timeout exceeded
# ---------------------------------------------------------------------------


def test_script_timeout_raises_pre_hook_error(tmp_path: Path) -> None:
    """A script that sleeps past the timeout raises PreHookError(returncode=124)."""
    hooks_dir = tmp_path / ".orchestrator" / "pre-hooks"
    hooks_dir.mkdir(parents=True)

    python = sys.executable
    _make_script(
        hooks_dir,
        "slow.py",
        f"#!{python}\nimport time\ntime.sleep(60)\n",
    )

    # Use a 1-second timeout so the test finishes quickly.
    with pytest.raises(PreHookError) as exc_info:
        run_pre_hooks(hooks_dir, timeout=1)

    err = exc_info.value
    assert err.returncode == 124
    assert "timed out" in err.output.lower()


# ---------------------------------------------------------------------------
# Scenario 8 — config defaults parse correctly from a minimal orchestrator.toml
# ---------------------------------------------------------------------------


def test_config_defaults_for_pre_hooks(tmp_path: Path) -> None:
    """[pre_hooks] dir/timeout default correctly from a minimal TOML."""
    toml_path = tmp_path / "orchestrator.toml"
    # Write a minimal TOML that sets an unrelated field, leaving [pre_hooks] absent.
    toml_path.write_text('db_path = ".orchestrator/checkpoints.db"\n')

    cfg = load_config(toml_path)

    assert cfg.pre_hooks.dir == ".orchestrator/pre-hooks"
    assert cfg.pre_hooks.timeout == 30


def test_config_pre_hooks_overridable(tmp_path: Path) -> None:
    """[pre_hooks] dir/timeout can be overridden in orchestrator.toml."""
    toml_path = tmp_path / "orchestrator.toml"
    toml_path.write_text(
        "[pre_hooks]\n"
        'dir = ".custom/hooks"\n'
        "timeout = 60\n"
    )

    cfg = load_config(toml_path)

    assert cfg.pre_hooks.dir == ".custom/hooks"
    assert cfg.pre_hooks.timeout == 60
