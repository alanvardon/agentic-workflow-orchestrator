"""Phase 15 resumability tests.

Two slices of coverage:

1. Idempotency of `commit`, `push`, `pr_create` — each function called
   twice should be effectively a no-op the second time. Tested via
   subprocess-call mocking so we can observe what git/gh commands run.

2. `resume_run` picks up after a stalled task — we make `push` fail
   once, then succeed; the resumed run should NOT re-commit (cached)
   and SHOULD re-attempt push (not cached because it raised).
"""

import json
import subprocess
from pathlib import Path

import pytest

from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.git_ops import (
    CommitAndPrError,
    commit,
    pr_create,
    push,
)


# ---------- Idempotency unit tests (no LangGraph involved) ----------


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def _fake_run_factory(responses: dict[str, str]):
    """Build a fake _run that returns canned stdout per command keyword.

    Match the first key whose substring appears in the joined args.
    Unknown commands raise — forces the test to declare everything.
    """
    def fake_run(args):
        joined = " ".join(args)
        for key, value in responses.items():
            if key in joined:
                return _completed(stdout=value)
        raise AssertionError(f"unexpected git/gh call: {joined!r}")
    return fake_run


def test_commit_returns_existing_sha_when_tree_clean_and_ahead(monkeypatch):
    """If a previous attempt committed, the next call returns its SHA
    without re-running `git add` or `git commit`."""
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "status --porcelain" in joined:
            return _completed(stdout="")  # clean tree
        if "rev-list main..HEAD --count" in joined:
            return _completed(stdout="1\n")  # one commit ahead
        if "rev-parse HEAD" in joined:
            return _completed(stdout="abc123def456\n")
        raise AssertionError(f"unexpected call: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    sha = commit("feature/test", "Test", "Test summary")

    assert sha == "abc123def456"
    # The critical assertion: no `git add` and no `git commit` calls.
    assert not any("add" in " ".join(c) for c in calls)
    assert not any("commit" in " ".join(c) and "-m" in " ".join(c) for c in calls)


def test_commit_raises_when_clean_tree_and_not_ahead(monkeypatch):
    """Genuinely nothing to commit — clean tree, no ahead commits."""
    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "status --porcelain" in joined:
            return _completed(stdout="")
        if "rev-list" in joined:
            return _completed(stdout="0\n")
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)

    with pytest.raises(CommitAndPrError, match="no changes to commit"):
        commit("feature/test", "Test", "summary")


def test_commit_commits_normally_when_tree_dirty(monkeypatch):
    """Dirty tree → normal stage + commit path runs."""
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "status --porcelain" in joined:
            return _completed(stdout="M file.py\n")
        if "rev-list" in joined:
            return _completed(stdout="0\n")
        if joined.startswith("git add"):
            return _completed()
        if joined.startswith("git commit"):
            return _completed()
        if joined == "git rev-parse HEAD":
            return _completed(stdout="newsha123\n")
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    sha = commit("feature/test", "Test", "summary")

    assert sha == "newsha123"
    assert any("git add" in " ".join(c) for c in calls)
    assert any("git commit" in " ".join(c) for c in calls)


def test_pr_create_returns_existing_url_when_pr_exists(monkeypatch):
    """`gh pr view` returns an existing PR — we return its URL without
    calling `gh pr create`."""
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "gh pr view" in joined:
            return _completed(stdout=json.dumps({"state": "OPEN", "url": "https://github.com/x/y/pull/42"}))
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    url = pr_create("feature/test", "Test", "summary", "test plan")

    assert url == "https://github.com/x/y/pull/42"
    assert not any("gh pr create" in " ".join(c) for c in calls)


def test_pr_create_opens_new_pr_when_none_exists(monkeypatch):
    """`gh pr view` exits non-zero → we proceed to `gh pr create`."""
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "gh pr view" in joined:
            raise subprocess.CalledProcessError(1, args, stderr="no pull requests found")
        if "gh pr create" in joined:
            return _completed(stdout="https://github.com/x/y/pull/99\n")
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    url = pr_create("feature/test", "Test", "summary", "test plan")

    assert url == "https://github.com/x/y/pull/99"


def test_push_passes_through_to_git(monkeypatch):
    """No idempotency check needed — `git push` itself is a no-op when
    up to date. Just verify we call it with the right args."""
    calls: list[list[str]] = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed(stdout="feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            return _completed(stdout="0\n")  # main has not moved
        if "git push -u origin feature/test" in joined:
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    push("feature/test")

    assert any("git push -u origin feature/test" in " ".join(c) for c in calls)


# ---------- resume_run integration test ----------


class _ResumeStubs:
    """Stubs that let us simulate a push failure on the first call and
    success on the resume call. Tracks which steps actually ran."""

    def __init__(self, fail_push_once: bool) -> None:
        self.fail_push_once = fail_push_once
        self.commit_call_count = 0
        self.push_call_count = 0
        self.pr_create_call_count = 0

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="t", type="feature", plan_text="p")

    def create_branch(self, plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch="main") -> str:
        self.commit_call_count += 1
        return "abc123"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        self.push_call_count += 1
        if self.fail_push_once and self.push_call_count == 1:
            raise CommitAndPrError("push failed: simulated network blip")

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        self.pr_create_call_count += 1
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
async def test_resume_run_skips_commit_after_push_failure(monkeypatch, tmp_path):
    """The canonical Phase 15 win: push fails, resume_run picks up
    without re-committing.

    Step 1: implement_feature → awaiting_approval (commit not yet called).
    Step 2: approve_plan("yes") → commit runs, push raises → CommitAndPrError.
    Step 3: resume_run(thread_id) with push fixed → commit returns
            CACHED SHA (call count stays at 1), push runs again
            (call count is 2), pr_create runs.
    """
    stubs = _ResumeStubs(fail_push_once=True)
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature, resume_run

    pending = await implement_feature("test")
    thread_id = pending["thread_id"]
    assert stubs.commit_call_count == 0  # commit not called yet — workflow paused at interrupt

    # Approve. The workflow runs commit → push (FAILS) → structured error.
    failed = await approve_plan(thread_id, "yes")
    assert failed["status"] == "user_action_required"
    assert "simulated network blip" in failed["error"]
    assert failed["thread_id"] == thread_id

    assert stubs.commit_call_count == 1
    assert stubs.push_call_count == 1
    assert stubs.pr_create_call_count == 0  # never reached

    # User "fixes" the issue, then resumes. commit is now cached → not
    # called again. push retries → succeeds. pr_create runs.
    final = await resume_run(thread_id)

    assert final["status"] == "succeeded"
    assert final["thread_id"] == thread_id
    assert final["pr_url"] == "https://github.com/test/pr/1"
    assert stubs.commit_call_count == 1  # STILL 1 — cached, did not re-run
    assert stubs.push_call_count == 2    # ran again (was not cached because it raised)
    assert stubs.pr_create_call_count == 1


@pytest.mark.asyncio
async def test_resume_run_returns_thread_id_in_final_response(monkeypatch, tmp_path):
    """thread_id must be in the response of every MCP tool, on every
    code path — surfacing requirement of Phase 15."""
    stubs = _ResumeStubs(fail_push_once=False)
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("test")
    final = await approve_plan(pending["thread_id"], "yes")

    # The thread_id should survive into the final succeeded response.
    assert final["thread_id"] == pending["thread_id"]
