"""Phase 55 — plan decomposition tests.

Phase 55 adds a `decompose` step that runs after planning and turns the plan into
an ordered task list, surfaced in the plan-approval payload. It is EXECUTION-INERT:
the list is produced, reviewed, and checkpointed, but nothing consumes it (the
monolithic build still does the work — Phase 56 adds the per-task loop).

Covers: the Task/DecompositionResult schema, the `max_tasks` guidance helper, the
serde round-trip (resume safety), the task list surfaced at plan approval, the
re-decompose-on-feedback path, and the inertness guarantee (one build, unchanged
result dict). Workflow tests use the same stub pattern as test_phase8_interrupt.py
— no LLM or git calls.
"""

import uuid

import pytest
from langgraph.types import Command, Interrupt

from orchestrator.agents.decompose import (
    DecompositionResult,
    Task,
    _build_user_message,
)
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.manifest import StepResult


# --------------------------------------------------------------------------
# Schema + pure helpers (no workflow, no SDK)
# --------------------------------------------------------------------------

def test_task_schema_fields():
    # The contract Phase 56 depends on: exactly these fields, nothing the
    # plan-only decomposer can't produce (no `files`, no `depends_on`).
    # Phase 81: `testable` added (per-task TDD gating; optional, default True).
    assert set(Task.model_fields) == {
        "id", "title", "description", "acceptance_criteria", "testable",
    }
    # Phase 72b: acceptance_criteria is now REQUIRED (the test-author's spec).
    assert Task.model_fields["acceptance_criteria"].is_required()
    # Phase 81: testable defaults True so tdd-off / pre-81 tasks are unchanged.
    assert Task.model_fields["testable"].default is True
    # Phase 82b: feature-level `complexity` added (sets the task budget; optional,
    # defaults "moderate" → additive/resume-safe, no schema_version bump).
    assert set(DecompositionResult.model_fields) == {
        "complexity", "tasks", "schema_version", "usage",
    }
    assert DecompositionResult.model_fields["complexity"].default == "moderate"


def test_user_message_includes_cap_only_when_positive():
    plan = "Add a dark mode toggle."
    capped = _build_user_message(plan, max_tasks=3)
    assert plan in capped
    assert "at most 3" in capped

    uncapped = _build_user_message(plan, max_tasks=0)
    assert plan in uncapped
    assert "at most" not in uncapped


def test_decomposition_result_serde_roundtrip():
    # Resume safety: the checkpointer must be able to (de)serialize the result.
    from orchestrator.workflow import _ALLOWED_MSGPACK_MODULES, _CUSTOM_SERDE

    assert ("orchestrator.agents.decompose", "DecompositionResult") in _ALLOWED_MSGPACK_MODULES

    result = DecompositionResult(
        tasks=[
            Task(id="a", title="A", description="do a", acceptance_criteria="checked"),
            Task(id="b", title="B", description="do b", acceptance_criteria="checked too"),
        ]
    )
    restored = _CUSTOM_SERDE.loads_typed(_CUSTOM_SERDE.dumps_typed(result))
    assert isinstance(restored, DecompositionResult)
    assert restored == result


# --------------------------------------------------------------------------
# Workflow wiring (stubbed agents/git, driven via interrupts)
# --------------------------------------------------------------------------

class _Stubs:
    def __init__(self) -> None:
        self.plan_calls: list[str] = []
        self.decompose_calls: list[str] = []  # plan_text passed to the decomposer
        self.impl_calls: list[str | None] = []

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        self.plan_calls.append(request)
        n = len(self.plan_calls)
        return PlanResult(title=f"title-{n}", type="feature", plan_text=f"plan-{n}")

    async def decompose(
        self, plan_text: str, model: str = "claude-sonnet-4-6", max_tasks: int = 0,
        tdd: bool = False,
    ) -> DecompositionResult:
        self.decompose_calls.append(plan_text)
        # Tag the task ids with the plan_text so a test can tell which plan was
        # decomposed (e.g. "plan-2" → tasks for the revised plan).
        return DecompositionResult(
            tasks=[
                Task(id=f"{plan_text}-t1", title="First", description="step one",
                     acceptance_criteria="one done"),
                Task(id=f"{plan_text}-t2", title="Second", description="step two",
                     acceptance_criteria="two done"),
            ]
        )

    def create_branch(self, plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        self.impl_calls.append(feedback)
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan: PlanResult, model: str = "claude-sonnet-4-6") -> QaResult:
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch="main") -> str:
        return "abc123def456"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        pass

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        return "https://github.com/test/pr/1"

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass


def _patch(stubs: _Stubs, monkeypatch) -> None:
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.decompose", stubs.decompose)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)


def _fresh_config() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}


def _interrupt_value(result: dict) -> dict:
    assert "__interrupt__" in result, f"Expected interrupt, got: {result}"
    interrupts: list[Interrupt] = result["__interrupt__"]
    assert interrupts[0].value["kind"] == "plan_approval"
    return interrupts[0].value


@pytest.mark.asyncio
async def test_task_list_surfaced_at_plan_approval(monkeypatch, tmp_path):
    stubs = _Stubs()
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    config = _fresh_config()
    async with build_workflow(db_path=str(tmp_path / "ckpt.db")) as workflow:
        result = await workflow.ainvoke("add a tooltip", config=config)

    val = _interrupt_value(result)
    # The decomposed list rides alongside the plan in the approval payload.
    assert val["plan"]["plan_text"] == "plan-1"
    assert [t["id"] for t in val["tasks"]] == ["plan-1-t1", "plan-1-t2"]
    assert [t["title"] for t in val["tasks"]] == ["First", "Second"]
    # Decomposed exactly once for the initial plan.
    assert stubs.decompose_calls == ["plan-1"]


@pytest.mark.asyncio
async def test_feedback_redecomposes_the_revised_plan(monkeypatch, tmp_path):
    stubs = _Stubs()
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    config = _fresh_config()
    async with build_workflow(db_path=str(tmp_path / "ckpt.db")) as workflow:
        result = await workflow.ainvoke("add a tooltip", config=config)
        assert [t["id"] for t in _interrupt_value(result)["tasks"]] == ["plan-1-t1", "plan-1-t2"]

        # Feedback → re-plan → the task list must reflect the NEW plan (plan-2),
        # not the stale one.
        result = await workflow.ainvoke(Command(resume="make it dismissible"), config=config)
        val = _interrupt_value(result)
        assert val["plan"]["plan_text"] == "plan-2"
        assert [t["id"] for t in val["tasks"]] == ["plan-2-t1", "plan-2-t2"]

        result = await workflow.ainvoke(Command(resume="yes"), config=config)

    assert result["status"] == "succeeded"
    # Decomposed once per plan version (initial + after feedback).
    assert stubs.decompose_calls == ["plan-1", "plan-2"]

# NOTE: Phase 55's "execution-inert" guarantee was intentionally superseded by
# Phase 56 — the decomposed list now drives the per-task execution station. That
# behaviour (a plan's tasks each running a build) is covered by
# test_phase56_per_task_loop.py.
