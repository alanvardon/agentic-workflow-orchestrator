"""Phase 20 workflow-version gate tests.

Covers:
- schema_version defaults on the three result models.
- A resume whose stored WORKFLOW_VERSION matches the live one proceeds.
- A resume whose stored version differs (simulating a code change while
  the run was paused at the plan-approval interrupt) is refused with a
  clear IncompatibleCheckpointError naming both versions.

Same stub pattern as the other workflow tests — no LLM, no git. The
version gate's storage is record_version_task, which is NOT stubbed: it
returns the module-level WORKFLOW_VERSION and is checkpointed on the first
run, then replayed (cached) on resume. The mismatch test monkeypatches the
live constant after the first run to simulate a version bump.
"""

import uuid

import pytest
from langgraph.types import Command

from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.agents.summarize import SummaryResult
from orchestrator.manifest import StepResult
from orchestrator.workflow import IncompatibleCheckpointError


class _Stubs:
    async def plan(self, request, model="claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="t", type="feature", plan_text="p")

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

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass


def _patch(stubs: _Stubs, monkeypatch) -> None:
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)


def test_schema_version_defaults():
    assert PlanResult(title="t", type="feature", plan_text="p").schema_version == 1
    # Phase 42: ImplementationResult is gone; SummaryResult carries the
    # commit/PR summary + test_plan and keeps the same versioned shape.
    assert SummaryResult(summary="s", test_plan="tp").schema_version == 1
    assert QaResult(result="PASS").schema_version == 1


@pytest.mark.asyncio
async def test_resume_same_version_completes(monkeypatch, tmp_path):
    _patch(_Stubs(), monkeypatch)
    from orchestrator.workflow import build_workflow

    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    async with build_workflow(db_path=str(tmp_path / "ckpt.db")) as workflow:
        result = await workflow.ainvoke("req", config=config)
        assert "__interrupt__" in result
        # No version change — resume proceeds to completion.
        result = await workflow.ainvoke(Command(resume="yes"), config=config)

    assert result["status"] == "succeeded"


@pytest.mark.asyncio
async def test_resume_version_mismatch_refuses(monkeypatch, tmp_path):
    _patch(_Stubs(), monkeypatch)
    from orchestrator.workflow import build_workflow

    import orchestrator.workflow as wf
    original_version = wf.WORKFLOW_VERSION

    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    async with build_workflow(db_path=str(tmp_path / "ckpt.db")) as workflow:
        result = await workflow.ainvoke("req", config=config)
        assert "__interrupt__" in result

        # Simulate an incompatible code change landing while the run is
        # paused at the plan-approval interrupt: bump the live version.
        # record_version_task is cached at the original version from the
        # first run. (Version-agnostic so it survives future bumps.)
        monkeypatch.setattr("orchestrator.workflow.WORKFLOW_VERSION", "9.9.9")

        with pytest.raises(IncompatibleCheckpointError) as ei:
            await workflow.ainvoke(Command(resume="yes"), config=config)

    assert ei.value.stored_version == original_version
    assert ei.value.current_version == "9.9.9"
