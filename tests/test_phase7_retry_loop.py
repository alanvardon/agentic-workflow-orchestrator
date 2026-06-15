"""Phase 7 retry-loop control-flow tests.

Stub the agent/git functions so the workflow runs against scripted
outcomes — no LLM calls, no git operations, no token cost. Asserts the
retry flow, feedback injection on retries (Phase 42: the failing gate's
detail is threaded into the next producer call, replacing the old
implement/"fix" mode switch), and the final status branch (succeeded vs
failed).

Phase 42: the impl→QA loop runs on the generic retry engine
(retry_block.run_retry_block). The implementation step is now a generic
producer (a StepResult-returning @task, `implementation_task`); QA stays
hard-baked (`qa()` → QaResult, adapted to a gate verdict). So we patch
`implementation_task` directly and record the feedback each call received.

Run with:
    pytest tests/test_phase7_retry_loop.py -v
"""

import uuid
from pathlib import Path

import pytest

from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.manifest import StepResult


class _Stubs:
    """Recorder + scripted-response object for the patched functions.

    The workflow resolves `plan`, `implementation_task`, `qa`,
    `create_branch`, `commit`, `push`, `pr_create` from
    orchestrator.workflow's module globals at call time, so monkey-patching
    those names on that module is sufficient — no need to touch the agent
    modules.
    """

    def __init__(self, qa_verdicts: list[QaResult]) -> None:
        self.qa_verdicts = qa_verdicts
        # Phase 42: each entry is the `feedback` the producer received on that
        # attempt — None on attempt 1, the prior failing gate's detail after.
        self.impl_calls: list[str | None] = []
        self.qa_call_count = 0
        self.commit_called = False

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="t", type="feature", plan_text="p")

    def create_branch(self, plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
        return "feature/test"

    async def implementation_task(
        self,
        plan_text: str,
        feedback: str | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> StepResult:
        self.impl_calls.append(feedback)
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan: PlanResult, model: str = "claude-sonnet-4-6") -> QaResult:
        verdict = self.qa_verdicts[self.qa_call_count]
        self.qa_call_count += 1
        return verdict

    def commit(self, branch: str, title: str, summary: str, base_branch: str = "main") -> str:
        return "abc123def456"

    def push(self, branch: str, base_branch: str = "main", auto_rebase: bool = True) -> None:
        pass

    def pr_create(
        self,
        branch: str,
        title: str,
        summary: str,
        test_plan: str,
        base_branch: str = "main",
        draft: bool = False,
        reviewers: list | None = None,
        labels: list | None = None,
    ) -> str:
        # Phase 15: pr_create is the "did we reach the end" signal —
        # this is the final task in the workflow's success path.
        self.commit_called = True
        return "https://github.com/test/pr/1"

    def verify_clean_tree(self) -> None:
        # No-op: tests don't depend on the real working tree state.
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass


async def _run(stubs: _Stubs, monkeypatch, tmp_path: Path) -> dict:
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr(
        "orchestrator.workflow.create_branch", stubs.create_branch
    )
    monkeypatch.setattr(
        "orchestrator.workflow.implementation_task", stubs.implementation_task
    )
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.setattr(
        "orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree
    )
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)

    from orchestrator.workflow import build_workflow
    from langgraph.types import Command
    from tests.conftest import task_build_config

    # Fresh thread_id per test so the checkpointer doesn't replay a
    # prior run's state. Fresh DB per test (tmp_path) for the same reason.
    # Phase 56: the impl⇄QA loop is the per-task station; pin its exhaustion to
    # "abort" so these retry-mechanics tests keep the pre-56 fail-on-exhaustion
    # behaviour (the new default is pause-and-ask, covered by test_phase52/56).
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    oc = task_build_config(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        # Phase 8 added a plan-approval interrupt before the retry loop.
        # Auto-approve it so Phase 7 tests stay focused on retry-loop logic.
        result = await workflow.ainvoke("test request", config=config)
        if "__interrupt__" in result:
            result = await workflow.ainvoke(Command(resume="yes"), config=config)
        return result


@pytest.mark.asyncio
async def test_pass_on_first_attempt(monkeypatch, tmp_path):
    stubs = _Stubs([QaResult(result="PASS")])
    result = await _run(stubs, monkeypatch, tmp_path)

    assert result["status"] == "succeeded"
    assert result["pr_url"] == "https://github.com/test/pr/1"
    assert result["branch"] == "feature/test"
    assert stubs.impl_calls == [None]  # ran once, no feedback on attempt 1
    assert stubs.qa_call_count == 1
    assert stubs.commit_called is True


@pytest.mark.asyncio
async def test_fail_then_pass_on_second_attempt(monkeypatch, tmp_path):
    stubs = _Stubs(
        [
            QaResult(result="FAIL", failures="missing reset wiring"),
            QaResult(result="PASS"),
        ]
    )
    result = await _run(stubs, monkeypatch, tmp_path)

    assert result["status"] == "succeeded"
    # Attempt 1: no feedback. Retry carries the prior failing gate's detail.
    assert stubs.impl_calls == [None, "missing reset wiring"]
    assert stubs.qa_call_count == 2
    assert stubs.commit_called is True


@pytest.mark.asyncio
async def test_fail_fail_then_pass_on_third_attempt(monkeypatch, tmp_path):
    stubs = _Stubs(
        [
            QaResult(result="FAIL", failures="fail 1"),
            QaResult(result="FAIL", failures="fail 2"),
            QaResult(result="PASS"),
        ]
    )
    result = await _run(stubs, monkeypatch, tmp_path)

    assert result["status"] == "succeeded"
    assert stubs.impl_calls == [None, "fail 1", "fail 2"]
    assert stubs.qa_call_count == 3
    assert stubs.commit_called is True


@pytest.mark.asyncio
async def test_all_three_attempts_fail(monkeypatch, tmp_path):
    stubs = _Stubs(
        [
            QaResult(result="FAIL", failures="fail 1"),
            QaResult(result="FAIL", failures="fail 2"),
            QaResult(result="FAIL", failures="fail 3"),
        ]
    )
    result = await _run(stubs, monkeypatch, tmp_path)

    # Budget exhausted without a pass → on_exhausted="abort" → failed branch
    # returns without ever calling commit/push/pr_create. A broken PR is worse
    # than no PR. qa_failures carries the last failing gate's detail.
    assert result["status"] == "failed"
    assert result["qa_failures"] == "fail 3"
    assert result["branch"] == "feature/test"
    assert "pr_url" not in result
    assert stubs.impl_calls == [None, "fail 1", "fail 2"]
    assert stubs.qa_call_count == 3
    assert stubs.commit_called is False
