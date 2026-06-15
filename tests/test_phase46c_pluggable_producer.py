"""Phase 46c / 56 / 68b — pluggable producer + gate on the per-task station.

The impl⇄QA loop is the built-in `task-build` station; its produce/gate reference
parts by prefixed id. The standard builtin:implementation⇄builtin:qa build is
covered by test_phase56. These tests cover the degrees of freedom:

1. A task-build with SWAPPED producer + gate ([defs.*] ai_agents) runs end-to-end
   on the generic engine — the built-in implementation/QA agents are never touched.
2. A task-build whose gate is a [defs.*] SCRIPT (instead of builtin:qa) runs that
   script as the gate; the built-in QA agent never runs, the built-in producer does.
"""

import uuid

import pytest
from langgraph.types import Command

from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.config import OrchestratorConfig
from orchestrator.manifest import StepResult
from orchestrator.pipeline import build_pipeline


def _task_build_with(produce, gate, defs) -> OrchestratorConfig:
    """An OrchestratorConfig whose task-build station uses the given produce/gate
    refs, with the referenced [defs.*] parts defined."""
    pipeline = build_pipeline({
        "flow": "plan >> decompose >> task-build >> docs >> summarize",
        "stage": {"builtin": {"task-build": {"produce": produce, "gate": gate}}},
        "defs": defs,
    })
    return OrchestratorConfig(pipeline=pipeline)


# --------------------------- end-to-end: swapped ids -------------------------


class _SpineStubs:
    """Happy-path spine stubs. implementation_task / qa raise if called, so a
    test can prove a swapped build never touches the built-in agents."""

    def __init__(self) -> None:
        self.builtin_impl_calls = 0
        self.builtin_qa_calls = 0

    async def plan(self, request, model="claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="t", type="feature", plan_text="p")

    def create_branch(self, plan, max_slug_length=50, thread_id="") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        self.builtin_impl_calls += 1
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        self.builtin_qa_calls += 1
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


def _patch_spine(stubs, monkeypatch):
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)


async def _run(monkeypatch, tmp_path, *, oc, fake_ai=None, fake_script=None):
    stubs = _SpineStubs()
    _patch_spine(stubs, monkeypatch)
    if fake_ai is not None:
        monkeypatch.setattr("orchestrator.workflow.execute_ai_agent", fake_ai)
    if fake_script is not None:
        monkeypatch.setattr("orchestrator.workflow.execute_script", fake_script)

    from orchestrator.workflow import build_workflow

    config = {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        result = await workflow.ainvoke("req", config=config)  # plan approval
        result = await workflow.ainvoke(Command(resume="yes"), config=config)
    return result, stubs


@pytest.mark.asyncio
async def test_swapped_producer_and_gate_run_end_to_end(monkeypatch, tmp_path):
    # The per-task station swaps BOTH ids: produce=["defs:my-coder"],
    # gate=["defs:my-qa"], each an ai_agent def. The station runs them on the
    # generic engine; the built-in implementation/QA agents are never called, and
    # the success dict carries qa=None (no built-in QA verdict).
    oc = _task_build_with(
        produce=["defs:my-coder"], gate=["defs:my-qa"],
        defs={
            "my-coder": {"type": "ai_agent", "path": ".orchestrator/agents/coder.md"},
            "my-qa": {"type": "ai_agent", "path": ".orchestrator/agents/qa.md"},
        },
    )

    agent_calls: list[tuple[str, bool]] = []

    async def fake_ai(step, project_root, plan_text, *, feedback=None, as_gate=False):
        agent_calls.append((step.id, as_gate))
        if as_gate:
            return StepResult(step_id=step.id, kind="ai_agent", ok=True, passed=True, detail="")
        return StepResult(step_id=step.id, kind="ai_agent", ok=True, detail="coded")

    result, stubs = await _run(monkeypatch, tmp_path, oc=oc, fake_ai=fake_ai)

    assert result["status"] == "succeeded"
    assert ("my-coder", False) in agent_calls  # producer ran
    assert ("my-qa", True) in agent_calls  # gate ran as a gate
    assert stubs.builtin_impl_calls == 0  # built-in producer untouched
    assert stubs.builtin_qa_calls == 0  # built-in QA untouched
    assert result["qa"] is None  # no built-in QA verdict to report


@pytest.mark.asyncio
async def test_defs_script_gate_replaces_builtin_qa(monkeypatch, tmp_path):
    # The station keeps the built-in producer but gates on a [defs.*] SCRIPT instead
    # of builtin:qa — the built-in QA agent never runs, the built-in producer does.
    oc = _task_build_with(
        produce=["builtin:implementation"], gate=["defs:qa-script"],
        defs={"qa-script": {"type": "script", "path": "qa.sh"}},
    )

    script_gate_calls = 0

    async def fake_script(step, repo_root, *, as_gate=False):
        nonlocal script_gate_calls
        if as_gate:
            script_gate_calls += 1
            return StepResult(step_id=step.id, kind="script", ok=True, passed=True, detail="")
        return StepResult(step_id=step.id, kind="script", ok=True)

    result, stubs = await _run(monkeypatch, tmp_path, oc=oc, fake_script=fake_script)

    assert result["status"] == "succeeded"
    assert script_gate_calls == 1  # the user's qa.sh gate ran
    assert stubs.builtin_qa_calls == 0  # built-in QA agent never ran
    assert stubs.builtin_impl_calls == 1  # built-in producer still ran
    assert result["qa"] is None  # the built-in QA verdict holder was never set
