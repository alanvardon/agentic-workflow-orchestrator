"""Phase 37 — fully_autonomous runtime behaviour (full-workflow integration).

Drives build_workflow with stubbed agents/git (reusing test_phase56's harness) to
prove the autonomous contract:

  - no plan-approval pause: the run completes on the FIRST ainvoke even though
    planning.human_in_loop defaults true (via the config field AND the env var).
  - approval_gate steps auto-proceed (and the bypass is audited).
  - build retries are unbounded: the loop runs past retry.max until a gate passes.
  - the safety ceiling stops an otherwise-infinite never-passing loop by cost and
    by time, returning status="cancelled", reason="autonomous_ceiling".
  - the caps are autonomous-only: a non-autonomous run ignores them.
"""

import types
import itertools
import json
import uuid

import pytest

from orchestrator.agents.qa import QaResult
from orchestrator.config import ENV_FULLY_AUTONOMOUS, OrchestratorConfig, apply_overrides
from orchestrator.manifest import ApprovalGateStep, StepResult, WorkflowManifest
from orchestrator.usage import TaskUsage

from tests.conftest import task_build_config
from tests.test_phase56_per_task_loop import _Stubs, _patch, _cfg


def _auto(cfg: OrchestratorConfig | None = None, **update) -> OrchestratorConfig:
    """An OrchestratorConfig with fully_autonomous on (+ optional overrides)."""
    cfg = cfg or OrchestratorConfig()
    return cfg.model_copy(update={"fully_autonomous": True, **update})


# ---------------------------------------------------------------------------
# no human pauses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autonomous_run_skips_plan_approval(monkeypatch, tmp_path):
    # planning.human_in_loop is true by default; autonomous mode bypasses it, so
    # the whole run completes on the FIRST ainvoke with no awaiting_approval pause.
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_auto()) as wf:
        result = await wf.ainvoke("req", config=_cfg())

    assert result["status"] == "succeeded"
    assert "__interrupt__" not in result
    assert stubs.pr_created is True


@pytest.mark.asyncio
async def test_autonomous_via_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_FULLY_AUTONOMOUS, "true")
    cfg = apply_overrides(OrchestratorConfig())  # resolves the env var
    assert cfg.fully_autonomous is True

    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as wf:
        result = await wf.ainvoke("req", config=_cfg())

    assert result["status"] == "succeeded"
    assert "__interrupt__" not in result


# NOTE (Phase 68b): test_approval_gate_auto_proceeds_and_is_audited was removed —
# [[steps.work]] approval_gate steps no longer exist (v2 has no approval_gate stage
# type). Autonomous suppression of the surviving human gates (plan / branch / pr) is
# covered by test_autonomous_run_skips_plan_approval above.


# ---------------------------------------------------------------------------
# unbounded retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_are_unbounded_loops_past_max(monkeypatch, tmp_path):
    # retry.max = 1 would normally allow a single attempt; autonomous mode loops
    # until a gate passes. QA fails twice then passes on attempt 3.
    stubs = _Stubs(
        n_tasks=1,
        qa_verdicts=[
            QaResult(result="FAIL", failures="nope 1"),
            QaResult(result="FAIL", failures="nope 2"),
            QaResult(result="PASS"),
        ],
    )
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    cfg = _auto(task_build_config(max_retries=1))
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as wf:
        result = await wf.ainvoke("req", config=_cfg())

    assert result["status"] == "succeeded"
    assert stubs.qa_calls == 3            # looped well past retry.max = 1
    assert len(stubs.impl_plans) == 3     # producer re-ran each attempt


# ---------------------------------------------------------------------------
# safety ceiling
# ---------------------------------------------------------------------------


class _NeverPassStubs(_Stubs):
    """A build whose gate never passes; each producer attempt reports a fixed cost
    so a cost ceiling can be crossed deterministically (reported_cost_usd bypasses
    the price tables, so the test doesn't depend on litellm being installed)."""

    def __init__(self, cost_per_attempt: float = 0.0) -> None:
        super().__init__(n_tasks=1)
        self._cost = cost_per_attempt

    async def impl_producer(self, plan_text, feedback=None, model="claude-sonnet-4-6") -> StepResult:
        self.impl_plans.append(plan_text)
        self.impl_feedback.append(feedback)
        usage = TaskUsage(model="m", input_tokens=0, output_tokens=0,
                          reported_cost_usd=self._cost) if self._cost else None
        return StepResult(step_id="implementation", kind="ai_agent", ok=True, usage=usage)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        self.qa_plans.append(plan.plan_text)
        self.qa_calls += 1
        return QaResult(result="FAIL", failures="never passes")


@pytest.mark.asyncio
async def test_cost_ceiling_stops_never_passing_loop(monkeypatch, tmp_path):
    stubs = _NeverPassStubs(cost_per_attempt=0.60)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    # Cap 1.0 USD; each attempt adds 0.60 → crosses after the 2nd attempt.
    cfg = _auto(autonomous_max_cost_usd=1.0)
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as wf:
        result = await wf.ainvoke("req", config=_cfg())

    assert result["status"] == "cancelled"
    assert result["reason"] == "autonomous_ceiling"
    assert stubs.pr_created is False
    assert len(stubs.impl_plans) >= 2     # ran, but did not loop forever


@pytest.mark.asyncio
async def test_time_ceiling_stops_never_passing_loop(monkeypatch, tmp_path):
    stubs = _NeverPassStubs()
    _patch(stubs, monkeypatch)
    # Deterministic clock: only workflow's own monotonic() calls consume this
    # counter (we replace the module ref in the workflow namespace, not globally),
    # so the elapsed budget is crossed after a fixed number of cancel checks.
    clock = itertools.count(0, 1.0)
    monkeypatch.setattr(
        "orchestrator.workflow.time",
        types.SimpleNamespace(monotonic=lambda: next(clock)),
    )
    from orchestrator.workflow import build_workflow

    cfg = _auto(autonomous_max_seconds=5)
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as wf:
        result = await wf.ainvoke("req", config=_cfg())

    assert result["status"] == "cancelled"
    assert result["reason"] == "autonomous_ceiling"
    assert stubs.pr_created is False


@pytest.mark.asyncio
async def test_caps_ignored_when_not_autonomous(monkeypatch, tmp_path):
    # Same low cost cap, but fully_autonomous OFF: the cap is not enforced, so the
    # build exhausts its (bounded) budget and the run fails normally — not cancelled.
    # Not autonomous → planning.human_in_loop still pauses, so we approve first.
    from langgraph.types import Command

    stubs = _NeverPassStubs(cost_per_attempt=10.0)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    cfg = task_build_config(on_exhausted="abort", max_retries=1).model_copy(
        update={"autonomous_max_cost_usd": 0.01}  # fully_autonomous stays False
    )
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as wf:
        await wf.ainvoke("req", config=(c := _cfg()))            # plan_approval pause
        result = await wf.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "failed"      # exhausted, not cancelled by a cap
    assert result["failed_task_id"] == "task:t1"
