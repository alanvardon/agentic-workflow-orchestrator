"""Phase 21 error classification tests.

Verifies:
1. Each error class inherits from the correct base (RetriableError,
   UserActionError, FatalError) so isinstance checks work at the MCP layer.
2. The MCP server maps each error class to the correct status string and
   includes the expected fields in the response.
3. UserActionError carries an `action` attribute; DirtyTreeError and
   CommitAndPrError have context-appropriate action messages.
"""

from pathlib import Path

import pytest

from orchestrator.errors import FatalError, OrchestratorError, RetriableError, UserActionError
from orchestrator.git_ops import BranchCreationError, CommitAndPrError, DirtyTreeError, PreHookError
from orchestrator.manifest import ManifestError
from orchestrator.workflow import IncompatibleCheckpointError, IncompatiblePipelineError


# ---------------------------------------------------------------------------
# 1. Error hierarchy assertions
# ---------------------------------------------------------------------------

class TestErrorHierarchy:
    def test_retriable_is_orchestrator_error(self):
        assert issubclass(RetriableError, OrchestratorError)

    def test_user_action_is_orchestrator_error(self):
        assert issubclass(UserActionError, OrchestratorError)

    def test_fatal_is_orchestrator_error(self):
        assert issubclass(FatalError, OrchestratorError)

    def test_branch_creation_is_user_action(self):
        assert issubclass(BranchCreationError, UserActionError)

    def test_dirty_tree_is_user_action(self):
        assert issubclass(DirtyTreeError, UserActionError)

    def test_dirty_tree_is_branch_creation(self):
        assert issubclass(DirtyTreeError, BranchCreationError)

    def test_commit_pr_is_user_action(self):
        assert issubclass(CommitAndPrError, UserActionError)

    def test_pre_hook_is_user_action(self):
        assert issubclass(PreHookError, UserActionError)

    def test_manifest_is_fatal(self):
        assert issubclass(ManifestError, FatalError)

    def test_incompatible_checkpoint_is_fatal(self):
        assert issubclass(IncompatibleCheckpointError, FatalError)

    def test_incompatible_pipeline_is_fatal(self):
        assert issubclass(IncompatiblePipelineError, FatalError)


class TestUserActionErrorAction:
    def test_dirty_tree_action_mentions_stash(self):
        exc = DirtyTreeError("dirty")
        assert "stash" in exc.action.lower() or "commit" in exc.action.lower()

    def test_commit_pr_action_mentions_resume(self):
        exc = CommitAndPrError("push failed")
        assert "resume_run" in exc.action

    def test_branch_creation_action_present(self):
        exc = BranchCreationError("branch exists")
        assert exc.action

    def test_pre_hook_carries_attributes(self):
        exc = PreHookError("lint.sh", "output text", 1)
        assert exc.script == "lint.sh"
        assert exc.output == "output text"
        assert exc.returncode == 1
        assert exc.action

    def test_user_action_default_action_is_message(self):
        exc = UserActionError("do something")
        assert exc.action == "do something"

    def test_user_action_explicit_action(self):
        exc = UserActionError("msg", action="explicit action")
        assert exc.action == "explicit action"


# ---------------------------------------------------------------------------
# 2. MCP server response shapes
# ---------------------------------------------------------------------------

from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult


class _BaseStubs:
    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="title", type="feature", plan_text="plan text")

    def create_branch(self, plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch="main") -> str:
        return "abc123"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        pass

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        return "https://github.com/test/pr/1"


def _patch(stubs, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)


@pytest.mark.asyncio
async def test_mcp_user_action_required_on_dirty_tree(monkeypatch, tmp_path):
    """DirtyTreeError → {"status": "user_action_required", "action": ...}"""
    stubs = _BaseStubs()
    stubs.verify_clean_tree = lambda: (_ for _ in ()).throw(
        DirtyTreeError("dirty tree")
    )
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("test")
    assert result["status"] == "user_action_required"
    assert "error" in result
    assert "action" in result
    assert result["thread_id"]


@pytest.mark.asyncio
async def test_mcp_user_action_required_on_commit_pr_error(monkeypatch, tmp_path):
    """CommitAndPrError → {"status": "user_action_required", "action": ...}"""
    stubs = _BaseStubs()

    def _fail_push(branch, base_branch="main", auto_rebase=True):
        raise CommitAndPrError("push failed: simulated")

    stubs.push = _fail_push
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("test")
    thread_id = pending["thread_id"]
    result = await approve_plan(thread_id, "yes")

    assert result["status"] == "user_action_required"
    assert "push failed" in result["error"]
    assert "action" in result
    assert "resume_run" in result["action"]
    assert result["thread_id"] == thread_id


@pytest.mark.asyncio
async def test_mcp_fatal_error_on_incompatible_checkpoint(monkeypatch, tmp_path):
    """IncompatibleCheckpointError → {"status": "incompatible_checkpoint", ...}

    These FatalError subclasses keep their own structured response so callers
    can surface version numbers.
    """
    _patch(_BaseStubs(), monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature, resume_run

    # Start and immediately get a thread_id.
    pending = await implement_feature("test")
    thread_id = pending["thread_id"]

    # Simulate an incompatible checkpoint on resume.
    monkeypatch.setattr(
        "orchestrator.mcp_server.run_with_progress",
        lambda *a, **k: (_ for _ in ()).throw(
            IncompatibleCheckpointError("1.0.0", "1.1.0")
        ),
    )

    result = await resume_run(thread_id)
    assert result["status"] == "incompatible_checkpoint"
    assert result["stored_version"] == "1.0.0"
    assert result["current_version"] == "1.1.0"


@pytest.mark.asyncio
async def test_mcp_fatal_error_on_generic_fatal(monkeypatch, tmp_path):
    """A bare FatalError → {"status": "fatal", ...}"""
    stubs = _BaseStubs()
    stubs.verify_clean_tree = lambda: (_ for _ in ()).throw(
        FatalError("internal assertion failed")
    )
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("test")
    assert result["status"] == "fatal"
    assert "error" in result
    assert "next" in result
    assert result["thread_id"]


@pytest.mark.asyncio
async def test_mcp_retriable_error(monkeypatch, tmp_path):
    """RetriableError → {"status": "retriable_error", ...}"""
    stubs = _BaseStubs()
    stubs.verify_clean_tree = lambda: (_ for _ in ()).throw(
        RetriableError("upstream 503")
    )
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("test")
    assert result["status"] == "retriable_error"
    assert "error" in result
    assert "next" in result
    assert result["thread_id"]
