"""Phase 46d — empty-diff resilience.

When the build (impl ⇄ qa) passes but the producer made no changes, there is
nothing to ship. Rather than committing an empty branch and opening a no-op PR
(or failing at commit with "no changes to commit"), the workflow returns a clean
status="no_changes" and skips summarize / docs / commit / push / pr.

Two layers:
1. git_ops.working_tree_has_changes — the dirty/ahead probe, against a real
   throwaway git repo.
2. workflow integration — a build that leaves the tree clean returns
   status="no_changes" with no commit/push/pr; a build with changes proceeds.
"""

import subprocess
import uuid

import pytest
from langgraph.types import Command

from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.manifest import StepResult


# --------------------------- working_tree_has_changes ------------------------


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_working_tree_has_changes_against_real_repo(tmp_path, monkeypatch):
    from orchestrator import git_ops

    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "Tester")
    (tmp_path / "f.txt").write_text("hello\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")

    # _run() runs git in the module-level REPO_ROOT; point it at the temp repo.
    monkeypatch.setattr(git_ops, "REPO_ROOT", tmp_path)

    # Clean tree, no commits ahead of base → nothing to ship.
    assert git_ops.working_tree_has_changes("main") is False

    # An uncommitted edit → has changes.
    (tmp_path / "f.txt").write_text("changed\n")
    assert git_ops.working_tree_has_changes("main") is True

    # Committed on a feature branch (clean tree, but ahead of base) → has changes
    # (the resume-after-commit case).
    _git(tmp_path, "checkout", "-b", "feature/x")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "work")
    assert git_ops.working_tree_has_changes("main") is True


# --------------------------- workflow integration ---------------------------


class _Stubs:
    def __init__(self) -> None:
        self.commit_called = False
        self.push_called = False
        self.pr_called = False

    async def plan(self, request, model="claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="t", type="feature", plan_text="p")

    def create_branch(self, plan, max_slug_length=50, thread_id="") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch="main") -> str:
        self.commit_called = True
        return "abc123"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        self.push_called = True

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        self.pr_called = True
        return "https://github.com/test/pr/1"

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass


def _patch(stubs, monkeypatch, *, has_changes):
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)
    # The empty-diff probe is the unit under test for the branch decision; stub
    # it so the integration test doesn't depend on the real working tree.
    monkeypatch.setattr(
        "orchestrator.workflow.working_tree_has_changes", lambda *a, **k: has_changes
    )


async def _run(stubs, monkeypatch, tmp_path):
    from orchestrator.workflow import build_workflow

    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    async with build_workflow(db_path=str(tmp_path / "ckpt.db")) as workflow:
        result = await workflow.ainvoke("req", config=config)  # plan approval
        result = await workflow.ainvoke(Command(resume="yes"), config=config)
    return result


@pytest.mark.asyncio
async def test_empty_build_returns_no_changes(monkeypatch, tmp_path):
    # Build passes QA but leaves the tree clean → status="no_changes", and the
    # commit/push/pr steps never run (no empty commit, no no-op PR).
    stubs = _Stubs()
    _patch(stubs, monkeypatch, has_changes=False)

    result = await _run(stubs, monkeypatch, tmp_path)

    assert result["status"] == "no_changes"
    assert result["branch"] == "feature/test"
    assert result["qa"]["result"] == "PASS"  # QA still ran and passed
    assert "pr_url" not in result
    assert stubs.commit_called is False
    assert stubs.push_called is False
    assert stubs.pr_called is False


@pytest.mark.asyncio
async def test_build_with_changes_proceeds_to_pr(monkeypatch, tmp_path):
    # The complement: when the build DID change the tree, the spine commits,
    # pushes, and opens the PR exactly as before.
    stubs = _Stubs()
    _patch(stubs, monkeypatch, has_changes=True)

    result = await _run(stubs, monkeypatch, tmp_path)

    assert result["status"] == "succeeded"
    assert result["pr_url"] == "https://github.com/test/pr/1"
    assert stubs.commit_called is True
    assert stubs.push_called is True
    assert stubs.pr_called is True
