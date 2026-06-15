"""Phase 23 — auto-checkout main at workflow start.

Tests cover:
1. ensure_on_main: already on main → no checkout, no pull.
2. ensure_on_main: on a feature branch → checkout + pull run.
3. ensure_on_main: checkout fails → BranchCreationError.
4. ensure_on_main: pull fails → BranchCreationError.
5. ensure_on_main: custom base_branch respected.
6. MCP integration: workflow started from non-main branch succeeds.
7. MCP integration: dirty-tree check still fires before ensure_on_main.
"""

import subprocess
from pathlib import Path

import pytest

from orchestrator.errors import UserActionError
from orchestrator.git_ops import BranchCreationError, ensure_on_main


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


# ---------------------------------------------------------------------------
# Unit tests for ensure_on_main()
# ---------------------------------------------------------------------------

def test_ensure_on_main_noop_when_already_on_main(monkeypatch):
    """Already on main → no git checkout or pull is called."""
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        if "rev-parse --abbrev-ref HEAD" in " ".join(args):
            return _completed(stdout="main\n")
        raise AssertionError(f"unexpected: {' '.join(args)!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    ensure_on_main("main")

    assert not any("checkout" in " ".join(c) for c in calls)
    assert not any("pull" in " ".join(c) for c in calls)


def test_ensure_on_main_switches_and_pulls_when_on_feature_branch(monkeypatch):
    """On feature branch → checkout main then pull."""
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/something\n")
        if "git checkout main" in joined:
            return _completed()
        if "git pull" in joined:
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    ensure_on_main("main")

    assert any("checkout" in " ".join(c) and "main" in " ".join(c) for c in calls)
    assert any("pull" in " ".join(c) for c in calls)


def test_ensure_on_main_raises_on_checkout_failure(monkeypatch):
    """If git checkout fails → BranchCreationError (UserActionError subclass)."""
    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/something\n")
        if "git checkout main" in joined:
            raise subprocess.CalledProcessError(1, args, stderr="pathspec 'main' did not match")
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)

    with pytest.raises(BranchCreationError, match="cannot switch to main"):
        ensure_on_main("main")


def test_ensure_on_main_raises_on_pull_failure(monkeypatch):
    """If git pull fails → BranchCreationError."""
    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/something\n")
        if "git checkout main" in joined:
            return _completed()
        if "git pull" in joined:
            raise subprocess.CalledProcessError(1, args, stderr="network unreachable")
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)

    with pytest.raises(BranchCreationError, match="cannot switch to main"):
        ensure_on_main("main")


def test_ensure_on_main_respects_custom_base_branch(monkeypatch):
    """base_branch='develop' → checks out develop, not main."""
    checked_out = {"branch": None}

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/something\n")
        if "git checkout" in joined:
            for part in args:
                if part not in ("git", "checkout"):
                    checked_out["branch"] = part
            return _completed()
        if "git pull" in joined:
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    ensure_on_main("develop")

    assert checked_out["branch"] == "develop"


# ---------------------------------------------------------------------------
# MCP integration tests
# ---------------------------------------------------------------------------

from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.git_ops import DirtyTreeError


class _BaseStubs:
    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="title", type="feature", plan_text="plan text")

    def create_branch(self, plan, max_slug_length=50, thread_id="") -> str:
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


def _patch(stubs, monkeypatch, tmp_path):
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
async def test_mcp_workflow_calls_ensure_on_main(monkeypatch, tmp_path):
    """ensure_on_main is called during the pre-flight task before any LLM work."""
    received = {}

    class _TrackingStubs(_BaseStubs):
        def ensure_on_main(self, base_branch: str = "main") -> None:
            received["called"] = True
            received["base_branch"] = base_branch

    _patch(_TrackingStubs(), monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("test")
    result = await approve_plan(pending["thread_id"], "yes")

    assert result["status"] == "succeeded"
    assert received.get("called") is True
    assert received.get("base_branch") == "main"


@pytest.mark.asyncio
async def test_mcp_dirty_tree_fires_before_ensure_on_main(monkeypatch, tmp_path):
    """DirtyTreeError from verify_clean_tree fires before ensure_on_main."""
    ensure_on_main_called = {"called": False}

    class _DirtyStubs(_BaseStubs):
        def verify_clean_tree(self) -> None:
            raise DirtyTreeError("dirty tree")

        def ensure_on_main(self, base_branch: str = "main") -> None:
            ensure_on_main_called["called"] = True

    _patch(_DirtyStubs(), monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("test")
    assert result["status"] == "user_action_required"
    assert ensure_on_main_called["called"] is False
