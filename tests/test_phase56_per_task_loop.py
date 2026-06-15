"""Phase 56 — per-task execution loop (the decomposition payoff).

The single impl⇄QA build is replaced by a per-task station (Option B): the frozen
task list from the decomposer (Phase 55) is run one produce⇄gate build per task via
the same engine, followed by an optional whole-diff final_qa. n=1 reproduces the
pre-56 single-build behaviour.

Covers: n=1 back-compat, n>1 fan-out (one build per task, in order), per-task QA
seeing this-task context, partial failure (abort + failed_task_id, later tasks not
run), the optional final_qa, and resume replay (completed tasks aren't re-run).
Same LLM/git-free stub pattern as test_phase7/8/42_spine_gates: patch
_run_implementation_producer (so the real @task wrapper still checkpoints/replays).
"""

import uuid

import pytest
from langgraph.types import Command, Interrupt

from orchestrator.agents.decompose import DecompositionResult, Task
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.manifest import StepResult

from tests.conftest import task_build_config


class _Stubs:
    def __init__(self, n_tasks: int = 1, qa_verdicts: list[QaResult] | None = None) -> None:
        self.tasks = [
            Task(id=f"t{i}", title=f"Task {i}", description=f"do step {i}",
                 acceptance_criteria=f"step {i} works")
            for i in range(1, n_tasks + 1)
        ]
        # plan_text each producer / QA call received, and producer feedback.
        self.impl_plans: list[str] = []
        self.impl_feedback: list[str | None] = []
        self.qa_plans: list[str] = []
        self._qa_verdicts = qa_verdicts  # None → always PASS
        self.qa_calls = 0
        self.pr_created = False

    async def plan(self, request, model="claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="t", type="feature", plan_text="OVERALL-PLAN")

    async def decompose(self, plan_text, model="claude-sonnet-4-6", max_tasks=0, tdd=False) -> DecompositionResult:
        return DecompositionResult(tasks=self.tasks)

    async def impl_producer(self, plan_text, feedback=None, model="claude-sonnet-4-6") -> StepResult:
        self.impl_plans.append(plan_text)
        self.impl_feedback.append(feedback)
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        self.qa_plans.append(plan.plan_text)
        if self._qa_verdicts is None:
            verdict = QaResult(result="PASS")
        else:
            verdict = self._qa_verdicts[self.qa_calls]
        self.qa_calls += 1
        return verdict

    def create_branch(self, plan, max_slug_length=50, thread_id="") -> str:
        return "feature/test"

    def commit(self, branch, title, summary, base_branch="main") -> str:
        return "abc123"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        pass

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        self.pr_created = True
        return "https://github.com/test/pr/1"

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass


def _patch(stubs: _Stubs, monkeypatch) -> None:
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.decompose", stubs.decompose)
    # Patch the INNER producer so the real implementation_task @task still
    # checkpoints/replays across resumes (a faked @task would re-run each time).
    monkeypatch.setattr("orchestrator.workflow._run_implementation_producer", stubs.impl_producer)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)


def _cfg() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}


@pytest.mark.asyncio
async def test_single_task_back_compat(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    oc = task_build_config(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        result = await workflow.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert result["pr_url"] == "https://github.com/test/pr/1"
    # One task → one build: implementation + QA each ran once.
    assert len(stubs.impl_plans) == 1
    assert stubs.qa_calls == 1


@pytest.mark.asyncio
async def test_multiple_tasks_each_run_one_build_in_order(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=3)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    oc = task_build_config(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        result = await workflow.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    # One build per task, in list order — each producer saw its own task's context.
    assert len(stubs.impl_plans) == 3
    assert "Task 1" in stubs.impl_plans[0]
    assert "Task 2" in stubs.impl_plans[1]
    assert "Task 3" in stubs.impl_plans[2]
    assert stubs.qa_calls == 3
    assert stubs.pr_created is True


@pytest.mark.asyncio
async def test_per_task_qa_judges_this_task(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=2)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    oc = task_build_config(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        await workflow.ainvoke(Command(resume="yes"), config=c)

    # QA is scoped to each task (overall plan for context + the task to evaluate).
    assert "OVERALL-PLAN" in stubs.qa_plans[0]
    assert "Evaluate ONLY this task: Task 1" in stubs.qa_plans[0]
    assert "Evaluate ONLY this task: Task 2" in stubs.qa_plans[1]


@pytest.mark.asyncio
async def test_task_failure_aborts_with_failed_task_id(monkeypatch, tmp_path):
    # Task 1 never passes QA → budget exhausts under on_exhausted="abort" → the run
    # fails, names the failing task, opens no PR, and never reaches task 2.
    stubs = _Stubs(
        n_tasks=2,
        qa_verdicts=[QaResult(result="FAIL", failures=f"nope {i}") for i in range(1, 4)],
    )
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    oc = task_build_config(on_exhausted="abort", max_retries=3)
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        result = await workflow.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "failed"
    assert result["failed_task_id"] == "task:t1"
    assert result["qa_failures"] == "nope 3"
    assert "pr_url" not in result
    assert stubs.pr_created is False
    # All 3 attempts were task 1; task 2 never ran.
    assert all("Task 1" in p for p in stubs.impl_plans)
    assert not any("Task 2" in p for p in stubs.impl_plans)


@pytest.mark.asyncio
async def test_whole_diff_qa_stage_runs_once_and_can_fail(monkeypatch, tmp_path):
    # The v2 way to do the old "final_qa": run task-build UNGATED (no per-task QA),
    # then a whole-diff `qa` STAGE after the loop. A FAIL there aborts the run with
    # failed_task_id="qa".
    from orchestrator.config import OrchestratorConfig
    from orchestrator.pipeline import build_pipeline

    stubs = _Stubs(n_tasks=2, qa_verdicts=[QaResult(result="FAIL", failures="cross-task bug")])
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    pipeline = build_pipeline({
        "flow": "plan >> decompose >> task-build >> docs >> summarize >> qa",
        "stage": {"builtin": {
            "task-build": {"produce": ["builtin:implementation"], "gate": [], "ungated": True},
            "qa": {"type": "ai_agent"},
        }},
    })
    oc = OrchestratorConfig(pipeline=pipeline)
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        result = await workflow.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "failed"
    assert result["failed_task_id"] == "qa"
    # Both tasks' producers ran (ungated, no per-task QA), then ONE whole-diff QA fired.
    assert len(stubs.impl_plans) == 2
    assert stubs.qa_calls == 1
    # The whole-diff QA judged the WHOLE plan (not a single task).
    assert "Evaluate ONLY this task" not in stubs.qa_plans[0]


@pytest.mark.asyncio
async def test_resume_does_not_rerun_completed_tasks(monkeypatch, tmp_path):
    # With after_producer pauses, drive 2 tasks across several resumes. Each task's
    # producer must run EXACTLY ONCE — completed tasks replay from the checkpoint
    # rather than re-running when the body re-executes on each resume.
    stubs = _Stubs(n_tasks=2)
    _patch(stubs, monkeypatch)
    from orchestrator.workflow import build_workflow

    oc = task_build_config(human_in_loop={"after_producer": True}, on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))            # plan_approval
        r = await workflow.ainvoke(Command(resume="yes"), config=c)    # task1 producer → pause
        assert r["__interrupt__"][0].value["kind"] == "build_producer_pause"
        assert len(stubs.impl_plans) == 1
        r = await workflow.ainvoke(Command(resume="yes"), config=c)    # task1 QA → task2 producer → pause
        assert r["__interrupt__"][0].value["kind"] == "build_producer_pause"
        assert len(stubs.impl_plans) == 2  # task1 NOT re-run
        result = await workflow.ainvoke(Command(resume="yes"), config=c)  # task2 QA → done

    assert result["status"] == "succeeded"
    assert len(stubs.impl_plans) == 2  # exactly one producer call per task
    assert "Task 1" in stubs.impl_plans[0]
    assert "Task 2" in stubs.impl_plans[1]
