"""Phase 16 cancellation tests.

Three slices:
1. Cancellation-store helpers (mark/is/clear) — direct unit tests
   against a tmp_path db, no LangGraph involved.
2. Workflow integration — a marked-cancelled thread returns
   status="cancelled" instead of running planning/impl/QA.
3. MCP tools — cancel_run signals; resume_run refuses cancelled
   threads without force=True and clears the flag with it.
"""

from pathlib import Path

import pytest

from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.cancellation import (
    WorkflowCancelled,
    clear_cancelled,
    is_cancelled,
    mark_cancelled,
    raise_if_cancelled,
)


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


def _dir(tmp_path: Path) -> Path:
    """Marker directory for a tmp test — separate from the
    project's `.orchestrator/cancellations/` so these tests don't
    depend on cwd or config."""
    return tmp_path / "cancellations"


def test_is_cancelled_false_for_unknown_thread(tmp_path):
    assert is_cancelled("run-nope", _dir(tmp_path)) is False


def test_mark_then_is_cancelled(tmp_path):
    d = _dir(tmp_path)
    mark_cancelled("run-abc", d)
    assert is_cancelled("run-abc", d) is True


def test_mark_is_idempotent(tmp_path):
    d = _dir(tmp_path)
    mark_cancelled("run-abc", d)
    mark_cancelled("run-abc", d)
    mark_cancelled("run-abc", d)
    assert is_cancelled("run-abc", d) is True


def test_clear_cancelled(tmp_path):
    d = _dir(tmp_path)
    mark_cancelled("run-abc", d)
    clear_cancelled("run-abc", d)
    assert is_cancelled("run-abc", d) is False


def test_clear_cancelled_no_op_when_not_set(tmp_path):
    # missing_ok=True: clearing a never-marked thread is a no-op.
    clear_cancelled("run-never", _dir(tmp_path))


def test_threads_are_isolated(tmp_path):
    d = _dir(tmp_path)
    mark_cancelled("run-aaa", d)
    assert is_cancelled("run-aaa", d) is True
    assert is_cancelled("run-bbb", d) is False


def test_raise_if_cancelled_no_op_when_clean(tmp_path):
    raise_if_cancelled("run-fresh", _dir(tmp_path))  # does not raise


def test_raise_if_cancelled_raises_when_marked(tmp_path):
    d = _dir(tmp_path)
    mark_cancelled("run-xxx", d)
    with pytest.raises(WorkflowCancelled):
        raise_if_cancelled("run-xxx", d)


def test_invalid_thread_id_rejected(tmp_path):
    """Path-traversal characters must be refused — otherwise a caller
    could write a marker outside the cancellations dir."""
    with pytest.raises(ValueError, match="thread_id"):
        mark_cancelled("../escape", _dir(tmp_path))
    with pytest.raises(ValueError, match="thread_id"):
        is_cancelled("with/slash", _dir(tmp_path))


# ---------------------------------------------------------------------------
# Workflow integration
# ---------------------------------------------------------------------------


class _Stubs:
    """Re-used pattern from test_phase11. Records whether expensive
    tasks ran so we can assert a cancelled run skips them entirely."""

    def __init__(self) -> None:
        self.plan_called = False
        self.implement_called = False
        self.qa_called = False
        self.commit_called = False

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        self.plan_called = True
        return PlanResult(title="title", type="feature", plan_text="plan")

    def create_branch(self, plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        self.implement_called = True
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        self.qa_called = True
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch="main") -> str:
        self.commit_called = True
        return "abc123"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        pass

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        return "https://github.com/test/pr/1"


def _patch(stubs: _Stubs, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)

    # Both the checkpointer and the cancellation store resolve their db
    # paths against find_project_root, which falls back to cwd in a
    # non-git dir. Redirect cwd so every test starts with a fresh store.
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)


@pytest.mark.asyncio
async def test_marked_thread_returns_cancelled_status(monkeypatch, tmp_path):
    """If a thread is marked cancelled before the workflow ever runs,
    the entry-time _check_cancel() fires and the planning/impl/qa
    tasks never execute."""
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import cancel_run, implement_feature

    # Choose a known thread_id so we can mark it before the run starts.
    # implement_feature generates its own thread_id, so the cleanest
    # approach is to mark a thread, then call implement_feature and
    # patch uuid to make it use the same id. Simpler: just call
    # cancel_run for some thread_id, then drive the workflow directly
    # via build_workflow with that id.
    thread_id = "run-cancel-me"
    await cancel_run(thread_id)

    from orchestrator.workflow import build_workflow

    config = {"configurable": {"thread_id": thread_id}}
    async with build_workflow() as workflow:
        result = await workflow.ainvoke("doesn't matter", config=config)

    assert result["status"] == "cancelled"
    assert result["thread_id"] == thread_id
    # The cancel check fires before verify_clean_tree_task even runs,
    # so the planning agent should never have been invoked.
    assert stubs.plan_called is False
    assert stubs.implement_called is False
    assert stubs.qa_called is False
    assert stubs.commit_called is False


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_run_marks_thread(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)

    from orchestrator.mcp_server import cancel_run

    result = await cancel_run("run-xyz")
    assert result["status"] == "cancellation_signalled"
    assert result["thread_id"] == "run-xyz"
    assert is_cancelled("run-xyz") is True


@pytest.mark.asyncio
async def test_resume_run_refuses_cancelled_without_force(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)

    from orchestrator.mcp_server import cancel_run, resume_run

    await cancel_run("run-abc")
    result = await resume_run("run-abc")  # force defaults to False

    assert result["status"] == "refused_cancelled"
    assert result["thread_id"] == "run-abc"
    # Flag is still set — refusal must not silently clear it.
    assert is_cancelled("run-abc") is True


@pytest.mark.asyncio
async def test_resume_run_force_clears_flag(monkeypatch, tmp_path):
    """resume_run(force=True) must clear the cancel flag before
    invoking the workflow, otherwise the very next task boundary check
    would re-raise WorkflowCancelled and we'd loop forever."""
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import cancel_run, resume_run

    # Mark cancelled but never invoke a workflow — resume_run should
    # still clear the flag when force=True is passed. With no prior
    # workflow state in the checkpointer, ainvoke(None) on a non-
    # existent thread is the only side effect; assert the flag is gone
    # afterwards regardless of how that call resolves.
    await cancel_run("run-force")
    assert is_cancelled("run-force") is True

    try:
        await resume_run("run-force", force=True)
    except Exception:
        # ainvoke(None) on a thread with no checkpoint may raise; we
        # only care that the cancel flag got cleared first.
        pass

    assert is_cancelled("run-force") is False
