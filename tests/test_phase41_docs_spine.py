"""Phase 41 — docs agent as a permanent spine task.

Covers:
- the docs prompt loads from orchestrator/prompts/ via load_prompt (Phase 42:
  moved out of the package agents/ dir to sit with the other built-in prompts).
- the docs agent is strictly documentation-only — never instructed to edit source
  (including the workflow that orchestrates it).
- docs_task runs exactly once on a happy-path workflow, with its model resolved
  from [workflow.docs] and its usage recorded under "docs" in the final result.

The autouse _stub_docs_task fixture (conftest) stubs docs_task for the rest of
the suite; the integration test here overrides it with its own recording stub.
LLM-free / git-free via the usual workflow stubs.
"""

import uuid

import pytest
from langgraph.types import Command

import orchestrator.workflow as wf
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.config import OrchestratorConfig
from orchestrator.manifest import StepResult
from orchestrator.prompt_loader import load_prompt
from orchestrator.usage import TaskUsage


# --------------------------- prompt loader (unit) ---------------------------


def test_docs_prompt_loads_from_prompts_dir():
    # The docs prompt now lives in orchestrator/prompts/ and loads via the same
    # load_prompt() loader as planning/implementation/qa (frontmatter-free).
    prompt = load_prompt("docs")
    assert prompt  # non-empty
    assert not prompt.lstrip().startswith("---")  # no YAML frontmatter
    assert "documentation agent" in prompt.lower()


def test_docs_prompt_is_documentation_only():
    # The docs agent must never be instructed to edit source — not even the
    # workflow that orchestrates it. Its prompt is strictly documentation-only,
    # and the version-bump addendum (which would have edited workflow.py) is gone.
    prompt = load_prompt("docs")
    assert "only documentation" in prompt.lower()
    assert "WORKFLOW_VERSION" not in prompt
    assert not hasattr(wf, "_WORKFLOW_VERSION_ADDENDUM")


# --------------------------- workflow integration ---------------------------


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

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, plan_type=None) -> str:
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


@pytest.mark.asyncio
async def test_docs_task_runs_once_and_usage_recorded(monkeypatch, tmp_path):
    _patch(_Stubs(), monkeypatch)

    # Override the autouse stub with one that records calls and reports usage.
    calls: list[tuple[str, str]] = []

    async def recording_docs(plan_text, model="claude-haiku-4-5-20251001"):
        calls.append((plan_text, model))
        return StepResult(
            step_id="docs", kind="ai_agent", ok=True, detail="updated README.md",
            usage=TaskUsage(model=model, input_tokens=10, output_tokens=5),
        )

    monkeypatch.setattr("orchestrator.workflow.docs_task", recording_docs)

    from orchestrator.workflow import build_workflow

    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    async with build_workflow(
        db_path=str(tmp_path / "ckpt.db"), config=OrchestratorConfig()
    ) as workflow:
        result = await workflow.ainvoke("req", config=config)
        assert "__interrupt__" in result  # plan approval
        result = await workflow.ainvoke(Command(resume="yes"), config=config)

    assert result["status"] == "succeeded"
    # docs ran exactly once, on the passing code, with the model from [workflow.docs].
    assert len(calls) == 1
    assert calls[0][1] == "claude-haiku-4-5-20251001"
    # docs usage is recorded under its own "docs" key in the final aggregate.
    assert "docs" in result["usage"]["by_task"]
    assert result["usage"]["by_task"]["docs"]["input_tokens"] == 10
