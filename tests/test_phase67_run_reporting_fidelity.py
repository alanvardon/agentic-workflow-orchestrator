"""Phase 67 — run-reporting fidelity.

Two findings + the audit-replay invariant:

#4 The audit log must record work that ACTUALLY ran. Pre-67, audit task events
   were emitted by a body-level `audited()` wrapper that re-ran on every resume,
   re-logging already-completed tasks as if they had executed again. They now
   emit from INSIDE each spine @task (@_audited_task), which only runs on real
   execution — so a replayed task is logged exactly once.

#6 An approved plan that decomposes to zero tasks must fail loud
   (EmptyDecompositionError → status="fatal"), not silently return "no_changes".

See ../.misc_notes/remaining_phases/code_review_2026_06_04/phase_67_run_reporting_fidelity.md
"""

import json
from pathlib import Path

import pytest

from orchestrator.errors import FatalError, OrchestratorError
from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.agents.decompose import DecompositionResult, Task
from orchestrator.workflow import EmptyDecompositionError


# ---------------------------------------------------------------------------
# Hermetic stub harness (mirrors test_phase24_audit_trail). docs_task,
# summarize_task and decompose are auto-stubbed suite-wide by conftest; the
# decomposer default returns a single task.
# ---------------------------------------------------------------------------


def _patch(monkeypatch, tmp_path):
    async def _plan(request, model="claude-sonnet-4-6"):
        return PlanResult(title="title", type="feature", plan_text="plan text")

    # Stub the underlying implementation producer (NOT implementation_task), so the
    # decorated @task still runs and emits its audit events.
    async def _run_impl(plan_text, feedback=None, model="claude-sonnet-4-6"):
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def _qa(plan, model="claude-sonnet-4-6"):
        return QaResult(result="PASS")

    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", lambda: None)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", lambda base_branch="main": None)
    monkeypatch.setattr("orchestrator.workflow.plan", _plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", lambda plan, max_slug_length=50, thread_id="": "feature/test")
    monkeypatch.setattr("orchestrator.workflow._run_implementation_producer", _run_impl)
    monkeypatch.setattr("orchestrator.workflow.qa", _qa)
    monkeypatch.setattr("orchestrator.workflow.commit", lambda branch, title, summary, base_branch=None: "abc123")
    monkeypatch.setattr("orchestrator.workflow.push", lambda branch, base_branch=None, auto_rebase=True: None)
    monkeypatch.setattr(
        "orchestrator.workflow.pr_create",
        lambda branch, title, summary, test_plan, base_branch=None, draft=False, reviewers=None, plan_type=None: "https://github.com/test/pr/1",
    )
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)


def _read_events(tmp_path: Path) -> list[dict]:
    log = tmp_path / ".orchestrator" / "audit.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# #4 — replayed tasks are not double-logged on resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_does_not_double_log_replayed_tasks(monkeypatch, tmp_path):
    """preflight/planning/decompose run on the first invoke (which pauses at plan
    approval); the approve_plan resume REPLAYS them. Each must appear exactly once
    — not twice — in the audit log."""
    _patch(monkeypatch, tmp_path)
    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a feature")  # runs preflight/planning/decompose, then pauses
    result = await approve_plan(pending["thread_id"], "yes")  # resumes → replays those tasks
    assert result["status"] == "succeeded"

    events = _read_events(tmp_path)

    def _count(task_name, event_type):
        return sum(
            1 for e in events
            if e.get("task_name") == task_name and e["event_type"] == event_type
        )

    # These three completed BEFORE the interrupt, so the resume replays them.
    # Exactly one start + one complete each — the pre-67 bug logged two.
    for name in ("preflight", "planning", "decompose"):
        assert _count(name, "task_start") == 1, f"{name} task_start double-logged"
        assert _count(name, "task_complete") == 1, f"{name} task_complete double-logged"

    # Tasks that ran for real only on the resume side are still logged once.
    for name in ("create_branch", "implementation", "qa", "commit", "push", "pr_create"):
        assert _count(name, "task_start") == 1, f"{name} missing/duplicated task_start"
        assert _count(name, "task_complete") == 1, f"{name} missing/duplicated task_complete"


@pytest.mark.asyncio
async def test_no_task_failed_in_successful_run_with_interrupt(monkeypatch, tmp_path):
    """The plan-approval interrupt (a GraphInterrupt) must never be mis-logged as a
    task_failed: interrupts fire in the body, never inside an @audited task."""
    _patch(monkeypatch, tmp_path)
    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a feature")
    result = await approve_plan(pending["thread_id"], "yes")
    assert result["status"] == "succeeded"

    events = _read_events(tmp_path)
    assert not [e for e in events if e["event_type"] == "task_failed"]


# ---------------------------------------------------------------------------
# #6 — empty decomposition fails loud
# ---------------------------------------------------------------------------


def test_empty_decomposition_error_is_fatal():
    assert issubclass(EmptyDecompositionError, FatalError)
    assert issubclass(EmptyDecompositionError, OrchestratorError)


@pytest.mark.asyncio
async def test_empty_decomposition_is_fatal_not_no_changes(monkeypatch, tmp_path):
    """A decomposer returning zero tasks yields status='fatal' (via the MCP
    FatalError handler), NOT a silent 'no_changes'."""
    _patch(monkeypatch, tmp_path)

    async def _empty_decompose(plan_text, model="claude-sonnet-4-6", max_tasks=0, tdd=False):
        return DecompositionResult(tasks=[], usage=None)

    # Overrides conftest's autouse single-task stub (runs after it, so wins).
    monkeypatch.setattr("orchestrator.workflow.decompose", _empty_decompose)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a feature")
    # The empty task list is shown for approval; the guard fires on the approve path.
    assert pending["tasks"] == []
    result = await approve_plan(pending["thread_id"], "yes")

    assert result["status"] == "fatal"
    assert result["status"] != "no_changes"
    # No branch was created — the guard runs before create_branch.
    assert "no tasks" in result["error"]


# ---------------------------------------------------------------------------
# Nit — the heartbeat label tables include the decompose + post-loop stages
# ---------------------------------------------------------------------------


def test_next_stage_tables_refreshed():
    from orchestrator.mcp_progress import _NEXT_STAGE as mcp_table
    from orchestrator.cli import _NEXT_STAGE as cli_table

    for table in (mcp_table, cli_table):
        # The stages that used to fall through to the "next stage" placeholder.
        assert table["planning"] == "decompose"
        assert table["decompose"] == "create_branch"
        assert table["summarize"] == "docs"
        assert table["docs"] == "commit"

    # Both progress sinks must agree (documented single-source-of-truth intent).
    assert mcp_table == cli_table
