"""Phase 22 — conflict handling for moved main.

Tests cover:
1. push() with main not moved — no rebase, normal push.
2. push() with main moved and auto_rebase=True — rebase runs, push succeeds.
3. push() with main moved, rebase conflicts — aborts and raises UserActionError.
4. push() with main moved and auto_rebase=False — raises UserActionError immediately.
5. fetch failure → CommitAndPrError.
6. GitConfig default and toml parsing.
7. MCP server response shape when rebase conflicts during workflow.
"""

import json
import subprocess
from pathlib import Path

import pytest

from orchestrator.errors import UserActionError
from orchestrator.git_ops import CommitAndPrError, push


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _failing(args, stderr: str = "error") -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(1, args, stderr=stderr)


class _FakeRun:
    """Configurable fake for orchestrator.git_ops._run.

    Tracks all calls and lets each test declare exactly which commands
    are expected and what they return.
    """

    def __init__(self, responses: dict[str, str], failures: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._responses = responses
        self._failures = failures or set()

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(args)
        joined = " ".join(args)
        for key in self._failures:
            if key in joined:
                raise subprocess.CalledProcessError(1, args, stderr=f"simulated: {key}")
        for key, value in self._responses.items():
            if key in joined:
                return _completed(stdout=value)
        raise AssertionError(f"unexpected git/gh call: {joined!r}")


# ---------------------------------------------------------------------------
# Unit tests for push()
# ---------------------------------------------------------------------------

def test_push_main_not_moved(monkeypatch):
    """When origin/main has not moved (behind=0), no rebase runs."""
    fake = _FakeRun({
        "rev-parse --abbrev-ref HEAD": "feature/test\n",
        "git fetch origin": "",
        "rev-list": "0\n",
        "git push -u origin feature/test": "",
    })
    monkeypatch.setattr("orchestrator.git_ops._run", fake)
    push("feature/test")

    assert any("git push -u origin feature/test" in " ".join(c) for c in fake.calls)
    assert not any("rebase" in " ".join(c) for c in fake.calls)


def test_push_auto_rebase_when_main_moved(monkeypatch):
    """When origin/main moved, auto_rebase=True triggers a rebase and uses
    --force-with-lease on the push (Phase 65: rebase rewrites history)."""
    call_count = {"rev-list": 0}

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            call_count["rev-list"] += 1
            return _completed(stdout="2\n")  # 2 new commits on main
        if "git rebase origin/main" in joined:
            return _completed()
        if "git push" in joined and "feature/test" in joined:
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    push("feature/test", auto_rebase=True)  # should not raise


def test_push_rebase_conflict_aborts_and_raises(monkeypatch):
    """When auto-rebase conflicts, abort is called and UserActionError is raised."""
    abort_called = {"called": False}

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            return _completed(stdout="1\n")
        if "git rebase origin/main" in joined and "--abort" not in joined:
            raise subprocess.CalledProcessError(1, args, stderr="CONFLICT (content)")
        if "git rebase --abort" in joined:
            abort_called["called"] = True
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)

    with pytest.raises(UserActionError, match="rebase conflicted"):
        push("feature/test", auto_rebase=True)

    assert abort_called["called"], "git rebase --abort was not called on conflict"


def test_push_no_auto_rebase_raises_when_main_moved(monkeypatch):
    """auto_rebase=False → raise UserActionError immediately on moved main."""
    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            return _completed(stdout="3\n")  # main moved
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)

    with pytest.raises(UserActionError, match="rebase"):
        push("feature/test", auto_rebase=False)


def test_push_fetch_failure_raises_commit_pr_error(monkeypatch):
    """A fetch failure surfaces as CommitAndPrError, not UserActionError."""
    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "git fetch origin" in joined:
            raise subprocess.CalledProcessError(1, args, stderr="network unreachable")
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)

    with pytest.raises(CommitAndPrError, match="fetch failed"):
        push("feature/test")


def test_push_respects_custom_base_branch(monkeypatch):
    """base_branch='develop' → checks origin/develop, not origin/main."""
    checked_remote = {"value": None}

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            # Capture which remote ref was checked.
            for part in args:
                if part.startswith("HEAD.."):
                    checked_remote["value"] = part
            return _completed(stdout="0\n")
        if "git push -u origin feature/test" in joined:
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    push("feature/test", base_branch="develop")

    assert checked_remote["value"] == "HEAD..origin/develop"


# ---------------------------------------------------------------------------
# Config: GitConfig defaults and toml parsing
# ---------------------------------------------------------------------------

def test_git_config_defaults():
    from orchestrator.config import OrchestratorConfig
    cfg = OrchestratorConfig()
    assert cfg.git.auto_rebase is True


def test_git_config_toml_false(tmp_path):
    """[git] auto_rebase = false is respected."""
    toml = tmp_path / "orchestrator.toml"
    toml.write_text("[git]\nauto_rebase = false\n")
    from orchestrator.config import load_config
    cfg = load_config(toml)
    assert cfg.git.auto_rebase is False


# ---------------------------------------------------------------------------
# MCP server: rebase conflict surfaces as user_action_required
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
async def test_mcp_rebase_conflict_returns_user_action(monkeypatch, tmp_path):
    """Rebase conflict during push → user_action_required with an action field."""
    stubs = _BaseStubs()

    def _conflict_push(branch, base_branch="main", auto_rebase=True):
        raise UserActionError(
            "origin/main moved and rebase conflicted; resolve manually then resume_run.",
            action="Run: git rebase origin/main on branch 'feature/test', resolve conflicts, then call resume_run.",
        )

    stubs.push = _conflict_push
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("test")
    thread_id = pending["thread_id"]
    result = await approve_plan(thread_id, "yes")

    assert result["status"] == "user_action_required"
    assert "rebase conflicted" in result["error"]
    assert "resume_run" in result["action"]
    assert result["thread_id"] == thread_id


@pytest.mark.asyncio
async def test_mcp_push_receives_auto_rebase_true_by_default(monkeypatch, tmp_path):
    """push() receives auto_rebase=True (the default) from the workflow config."""
    received = {}

    class _TrackingStubs(_BaseStubs):
        def push(self, branch, base_branch="main", auto_rebase=True):
            received["auto_rebase"] = auto_rebase

    _patch(_TrackingStubs(), monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("test")
    result = await approve_plan(pending["thread_id"], "yes")

    assert result["status"] == "succeeded"
    assert received.get("auto_rebase") is True
