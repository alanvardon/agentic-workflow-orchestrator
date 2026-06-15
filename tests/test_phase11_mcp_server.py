"""Phase 11 MCP server tool tests.

Verifies the two tool functions against stubbed workflow internals:
- implement_feature returns awaiting_approval with the plan + thread_id
- approve_plan("yes") completes the workflow and returns the PR URL
- approve_plan(feedback) triggers a re-plan and returns awaiting_approval again

The @mcp.tool() decorator returns the original function, so we can call
implement_feature / approve_plan directly in tests without spinning up
the MCP transport layer.
"""

from pathlib import Path

import pytest

from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult


class _Stubs:
    def __init__(self) -> None:
        self.plan_calls: list[str] = []
        self.commit_called = False

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        self.plan_calls.append(request)
        n = len(self.plan_calls)
        return PlanResult(title=f"title-{n}", type="feature", plan_text=f"plan-{n}")

    def create_branch(self, plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch="main") -> str:
        return "abc123def456"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        pass

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        self.commit_called = True
        return "https://github.com/test/pr/1"


def _patch(stubs: _Stubs, monkeypatch, tmp_path: Path) -> None:
    # Stub the agents/git layer.
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)

    # The MCP tools call build_workflow() with no args, so it uses the
    # default db_path ".orchestrator/checkpoints.db" — relative to cwd.
    # Redirect cwd to tmp_path so the DB lands there and each test starts
    # with a fresh checkpoint store.
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)


@pytest.mark.asyncio
async def test_implement_feature_returns_awaiting_approval(monkeypatch, tmp_path):
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("add a tooltip")

    assert result["status"] == "awaiting_approval"
    assert result["thread_id"].startswith("run-")
    assert result["plan"]["plan_text"] == "plan-1"
    assert result["plan"]["title"] == "title-1"
    assert "next" in result
    # The workflow ran planning but should NOT have committed anything
    # before approval.
    assert stubs.commit_called is False


@pytest.mark.asyncio
async def test_approve_plan_yes_completes_workflow(monkeypatch, tmp_path):
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a tooltip")
    thread_id = pending["thread_id"]

    final = await approve_plan(thread_id, "yes")

    assert final["status"] == "succeeded"
    assert final["pr_url"] == "https://github.com/test/pr/1"
    assert final["branch"] == "feature/test"
    assert stubs.commit_called is True


@pytest.mark.asyncio
async def test_approve_plan_feedback_triggers_replan(monkeypatch, tmp_path):
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a tooltip")
    thread_id = pending["thread_id"]
    assert pending["plan"]["plan_text"] == "plan-1"

    # Feedback reply: should re-plan and surface the revised plan.
    revised = await approve_plan(thread_id, "also make it dismissible")

    assert revised["status"] == "awaiting_approval"
    assert revised["thread_id"] == thread_id
    assert revised["plan"]["plan_text"] == "plan-2"
    assert "Feedback: also make it dismissible" in stubs.plan_calls[1]
    assert stubs.commit_called is False

    # Approve the revised plan: should now complete.
    final = await approve_plan(thread_id, "yes")
    assert final["status"] == "succeeded"
    assert stubs.commit_called is True
