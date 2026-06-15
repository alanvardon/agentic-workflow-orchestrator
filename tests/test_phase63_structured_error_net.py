"""Phase 63 — close the structured-error net.

Three runtime failures used to escape the orchestrator's structured-error
handling and reach the MCP client as raw exceptions. This phase routes them
through the existing hierarchy / fail-closed paths:

1. StepError is now a FatalError (so the MCP `except FatalError` handler shapes
   it into a {"status": "fatal", ...} response instead of a bare traceback).
2. The QA verdict factory coerces a non-canonical `result` string to FAIL
   (fail-closed) instead of raising pydantic ValidationError inside qa_task.
3. qa_scripts.run_script catches OSError (bad shebang / lost +x) and converts it
   to a non-zero exit so run_qa_scripts reports a clean failure instead of
   crashing.

See ../.misc_notes/remaining_phases/code_review_2026_06_04/phase_63_close_structured_error_net.md
"""

import stat
import sys
from pathlib import Path

import pytest

from orchestrator.errors import FatalError, OrchestratorError
from orchestrator.steps import StepError
from orchestrator.agents.qa import QaResult, _coerce_verdict
from orchestrator.qa_scripts import run_qa_scripts, run_script


# ---------------------------------------------------------------------------
# Finding #1 — StepError is part of the structured-error hierarchy
# ---------------------------------------------------------------------------


class TestStepErrorHierarchy:
    def test_step_error_is_fatal(self):
        assert issubclass(StepError, FatalError)

    def test_step_error_is_orchestrator_error(self):
        # So the MCP server's `except FatalError` (and any broad
        # `except OrchestratorError`) catches it instead of letting it escape.
        assert issubclass(StepError, OrchestratorError)

    def test_step_error_instance_caught_as_fatal(self):
        try:
            raise StepError("script step 'x' failed")
        except FatalError as exc:  # the MCP handler's clause
            assert "script step" in str(exc)
        else:  # pragma: no cover
            pytest.fail("StepError was not caught as FatalError")


# ---------------------------------------------------------------------------
# Finding #2 — QA verdict coercion is fail-closed
# ---------------------------------------------------------------------------


class TestQaVerdictCoercion:
    @pytest.mark.parametrize("raw", ["PASS", "pass", "  pass  ", "Pass", "PASS\n"])
    def test_canonical_and_casing_pass(self, raw):
        assert _coerce_verdict(raw) == "PASS"

    @pytest.mark.parametrize(
        "raw",
        ["FAIL", "fail", "Passed", "PASS.", "PASS!", "", "ok", "true", None, 0],
    )
    def test_anything_non_canonical_is_fail(self, raw):
        # Fail-closed: a drifted / empty / non-string verdict becomes FAIL, never
        # a crash. (Only an exact case-insensitive 'PASS' passes.)
        assert _coerce_verdict(raw) == "FAIL"

    def test_factory_shape_does_not_raise_on_bad_verdict(self):
        # The real failure mode: building QaResult straight from a bad string
        # used to raise ValidationError inside qa_task. Coercion makes it a FAIL.
        result = QaResult(result=_coerce_verdict("definitely-not-pass"), failures="x")
        assert result.result == "FAIL"


# ---------------------------------------------------------------------------
# Finding #9 — qa_scripts.run_script catches OSError instead of crashing
# ---------------------------------------------------------------------------


def _make_executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


class TestRunScriptOSError:
    def test_bad_shebang_returns_126_not_raise(self, tmp_path: Path):
        # +x set (passes the executable filter) but the interpreter doesn't
        # exist, so subprocess.run raises OSError when it tries to exec.
        script = _make_executable(
            tmp_path / "broken.sh", "#!/nonexistent/interpreter/xyz\necho hi\n"
        )
        result = run_script(script, timeout=10)
        assert result.exit_code == 126
        assert "could not be executed" in result.stderr

    def test_run_qa_scripts_reports_failure_not_crash(self, tmp_path: Path):
        qa_dir = tmp_path / ".orchestrator" / "qa"
        qa_dir.mkdir(parents=True)
        _make_executable(qa_dir / "00-broken.sh", "#!/nonexistent/interp\ntrue\n")
        # Must not raise — the OSError is converted to a non-zero exit and flows
        # through the normal fail-fast / failure_report path.
        outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
        assert outcome.passed is False
        assert outcome.failure_report  # non-empty
        assert outcome.results[0].exit_code == 126

    def test_good_script_still_passes(self, tmp_path: Path):
        # Regression guard: the new except clause doesn't disturb the happy path.
        qa_dir = tmp_path / ".orchestrator" / "qa"
        qa_dir.mkdir(parents=True)
        _make_executable(
            qa_dir / "00-ok.sh", f"#!{sys.executable}\nimport sys; sys.exit(0)\n"
        )
        outcome = run_qa_scripts(tmp_path, ".orchestrator/qa", timeout=10)
        assert outcome.passed is True
