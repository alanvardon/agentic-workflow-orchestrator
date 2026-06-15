"""Phase 72 — separated test-author station (TDD red-green core).

When config.tdd is on, the per-task station authors tests with a DIFFERENT agent
than the implementer, ONCE before the implement loop, confirms the green→red
transition, and freezes the tests with a diff-gate so the implementer can't edit
them. A task the author judges untestable (or a born-green / no-script-gate
situation) gracefully falls back to the classic implement→qa path — TDD never
wedges a run.

Two layers of tests:
  - Unit tests of the red-green helpers (_run_test_author, _hash_test_paths,
    _make_diff_gate) and the config guards — fast, no workflow.
  - Full-workflow tests that patch _run_test_author (so the real test_author_task
    @task still checkpoints/replays) and _hash_test_paths (so the diff-gate's
    verdict is deterministic), exercising the real _run_task_loop wiring.
"""

import uuid

import pytest
from langgraph.types import Command

from orchestrator.agents.decompose import DecompositionResult, Task
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.agents.test_author import TestAuthorResult
from orchestrator.config import OrchestratorConfig
from orchestrator.manifest import StepResult

from tests.conftest import task_build_config


# --------------------------------------------------------------------------- #
# Config guards (decision A + test_paths requirement)
# --------------------------------------------------------------------------- #


def test_tdd_with_fully_autonomous_is_allowed():
    # Phase 76 re-enabled the combo with machinery (hard red-confirm + bounded
    # re-author), so it no longer errors at load. (Phase 72 forbade it.)
    cfg = OrchestratorConfig(tdd=True, fully_autonomous=True, test_paths=["**/*.test.js"])
    assert cfg.tdd is True and cfg.fully_autonomous is True


def test_tdd_requires_test_paths():
    # Without test_paths the diff-gate has nothing to freeze → write-separation is
    # empty, so tdd-on demands it.
    with pytest.raises(ValueError, match="test_paths"):
        OrchestratorConfig(tdd=True)


def test_tdd_off_is_the_unchanged_default():
    cfg = OrchestratorConfig()
    assert cfg.tdd is False
    assert cfg.test_paths == []


# --------------------------------------------------------------------------- #
# Unit: the freeze hash + the diff-gate
# --------------------------------------------------------------------------- #


def test_hash_test_paths_detects_edit_add_delete(tmp_path):
    from orchestrator.workflow import _hash_test_paths

    (tmp_path / "a.test.js").write_text("test one")
    (tmp_path / "b.test.js").write_text("test two")
    root = str(tmp_path)
    baseline = _hash_test_paths(["**/*.test.js"], root)

    # Edit → changes.
    (tmp_path / "a.test.js").write_text("test one EDITED")
    assert _hash_test_paths(["**/*.test.js"], root) != baseline
    # Revert → identical again.
    (tmp_path / "a.test.js").write_text("test one")
    assert _hash_test_paths(["**/*.test.js"], root) == baseline
    # Add a new matching file → changes (set membership).
    (tmp_path / "c.test.js").write_text("test three")
    assert _hash_test_paths(["**/*.test.js"], root) != baseline
    # Delete back to baseline → identical.
    (tmp_path / "c.test.js").unlink()
    assert _hash_test_paths(["**/*.test.js"], root) == baseline


@pytest.mark.asyncio
async def test_diff_gate_passes_when_unchanged_fails_when_tampered(tmp_path):
    from orchestrator.workflow import _hash_test_paths, _make_diff_gate

    (tmp_path / "a.test.js").write_text("frozen")
    root = str(tmp_path)
    snapshot = _hash_test_paths(["**/*.test.js"], root)
    gate = _make_diff_gate(["**/*.test.js"], snapshot, root)

    passed = await gate("builtin:diff-gate")
    assert passed.passed is True

    (tmp_path / "a.test.js").write_text("tampered with")
    failed = await gate("builtin:diff-gate")
    assert failed.passed is False
    assert "frozen" in failed.detail.lower()


# --------------------------------------------------------------------------- #
# Unit: _run_test_author green→red transition + graceful fallbacks
# --------------------------------------------------------------------------- #


def _patch_scripts(monkeypatch, results):
    """Patch _script_gate_steps (non-empty) + _run_script_gates to yield the given
    tuples in order. `results` is a list consumed per call.

    Each entry is (green, failing_output) or (green, failing_output, full_output);
    a 2-tuple is padded so full_output mirrors the failing output (Phase 77b made
    _run_script_gates return the extra full-run element)."""
    from orchestrator import workflow as wf

    monkeypatch.setattr(wf, "_script_gate_steps", lambda cfg, refs: ["<fake-step>"])
    it = iter(results)

    async def _gates(steps, root):
        r = next(it)
        return r if len(r) == 3 else (r[0], r[1], r[1])

    monkeypatch.setattr(wf, "_run_script_gates", _gates)


@pytest.mark.asyncio
async def test_run_test_author_green_to_red_returns_snapshot(monkeypatch):
    from orchestrator import workflow as wf

    _patch_scripts(monkeypatch, [(True, ""), (False, "FAILED: 1 test")])

    async def _author(plan, model, system_prompt=None, allowed_tools=None, disallowed_tools=None, feedback=None):
        return TestAuthorResult(testable=True, summary="covers behaviour X")

    monkeypatch.setattr(wf, "author_tests", _author)
    monkeypatch.setattr(wf, "_hash_test_paths", lambda paths, root: "SNAP")

    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is True
    assert res.snapshot == "SNAP"
    assert "FAILED" in res.red_output


@pytest.mark.asyncio
async def test_run_test_author_born_green_falls_back(monkeypatch):
    # Green before AND green after → the authored test never failed → no proof.
    from orchestrator import workflow as wf

    _patch_scripts(monkeypatch, [(True, ""), (True, "")])

    async def _author(plan, model, system_prompt=None, allowed_tools=None, disallowed_tools=None, feedback=None):
        return TestAuthorResult(testable=True, summary="claims testable")

    monkeypatch.setattr(wf, "author_tests", _author)

    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is False
    assert "born-green" in res.summary


@pytest.mark.asyncio
async def test_run_test_author_baseline_red_falls_back_without_authoring(monkeypatch):
    # Suite not green before → can't prove a transition → don't even author.
    from orchestrator import workflow as wf

    _patch_scripts(monkeypatch, [(False, "pre-existing failure")])
    authored = {"called": False}

    async def _author(plan, model, system_prompt=None, allowed_tools=None, disallowed_tools=None, feedback=None):
        authored["called"] = True
        return TestAuthorResult(testable=True)

    monkeypatch.setattr(wf, "author_tests", _author)

    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is False
    assert authored["called"] is False


@pytest.mark.asyncio
async def test_run_test_author_untestable_passthrough(monkeypatch):
    from orchestrator import workflow as wf

    _patch_scripts(monkeypatch, [(True, "")])

    async def _author(plan, model, system_prompt=None, allowed_tools=None, disallowed_tools=None, feedback=None):
        return TestAuthorResult(testable=False, summary="DOM-only, no harness")

    monkeypatch.setattr(wf, "author_tests", _author)

    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is False
    assert res.summary == "DOM-only, no harness"


@pytest.mark.asyncio
async def test_run_test_author_no_script_gate_falls_back(monkeypatch):
    from orchestrator import workflow as wf

    monkeypatch.setattr(wf, "_script_gate_steps", lambda cfg, refs: [])

    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["builtin:qa"])
    assert res.testable is False
    assert "no deterministic test gate" in res.summary


# --------------------------------------------------------------------------- #
# Full-workflow integration (patch _run_test_author + _hash_test_paths)
# --------------------------------------------------------------------------- #


class _Stubs:
    def __init__(self, n_tasks=1, ta_results=None, qa_verdicts=None):
        self.tasks = [
            Task(id=f"t{i}", title=f"Task {i}", description=f"do step {i}",
                 acceptance_criteria=f"criterion {i}")
            for i in range(1, n_tasks + 1)
        ]
        self.ta_calls: list[str] = []         # plan_text each _run_test_author saw
        self.ta_feedback: list[str | None] = []  # feedback each call saw (re-author)
        self._ta_results = ta_results          # per-task TestAuthorResult, or None → all testable
        self.impl_plans: list[str] = []
        self.qa_calls = 0
        self._qa_verdicts = qa_verdicts
        self.pr_created = False

    async def plan(self, request, model="claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="t", type="feature", plan_text="OVERALL-PLAN")

    async def decompose(self, plan_text, model="claude-sonnet-4-6", max_tasks=0, tdd=False) -> DecompositionResult:
        return DecompositionResult(tasks=self.tasks)

    async def run_test_author(self, plan_text, model, test_paths, gate_refs, feedback=None) -> TestAuthorResult:
        idx = len(self.ta_calls)
        self.ta_calls.append(plan_text)
        self.ta_feedback.append(feedback)
        if self._ta_results is None:
            return TestAuthorResult(testable=True, summary="ok", snapshot="SNAP", red_output="boom")
        return self._ta_results[idx]

    async def impl_producer(self, plan_text, feedback=None, model="claude-sonnet-4-6") -> StepResult:
        self.impl_plans.append(plan_text)
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        verdict = QaResult(result="PASS") if self._qa_verdicts is None else self._qa_verdicts[self.qa_calls]
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


def _patch(stubs: _Stubs, monkeypatch, *, hash_value: str = "SNAP") -> None:
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.decompose", stubs.decompose)
    # Patch the INNER test-author fn so the real test_author_task @task still
    # checkpoints/replays across resumes (a faked @task would re-run each time).
    monkeypatch.setattr("orchestrator.workflow._run_test_author", stubs.run_test_author)
    monkeypatch.setattr("orchestrator.workflow._run_implementation_producer", stubs.impl_producer)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    # Deterministic diff-gate verdict: current hash == snapshot ("SNAP") → pass.
    monkeypatch.setattr("orchestrator.workflow._hash_test_paths", lambda paths, root: hash_value)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)


def _tdd_cfg(*, red_review=False, coverage_critic=False, **kw):
    # red_review / coverage_critic default False so these authoring/diff-gate/resume
    # tests keep their single plan-approval resume and don't invoke the critic agent;
    # Phase 72b/74's own tests opt in.
    return task_build_config(**kw).model_copy(
        update={
            "tdd": True, "test_paths": ["**/*.test.js"],
            "tdd_red_review": red_review, "tdd_coverage_critic": coverage_critic,
        }
    )


def _cfg() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}


@pytest.mark.asyncio
async def test_testable_task_authors_once_and_ships(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")  # matches the stub's snapshot → diff-gate passes
    from orchestrator.workflow import build_workflow

    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        result = await workflow.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert result["pr_url"] == "https://github.com/test/pr/1"
    assert len(stubs.ta_calls) == 1          # test-author ran once for the task
    assert len(stubs.impl_plans) == 1        # implementer ran once (diff-gate + qa passed)
    assert stubs.qa_calls == 1


@pytest.mark.asyncio
async def test_diff_gate_fails_when_tests_tampered(monkeypatch, tmp_path):
    # Current hash never matches the snapshot → the diff-gate fails every attempt,
    # before QA, and the build exhausts → clean failed status carrying the gate's
    # "frozen" feedback. The implementer "touched a test"; the run never ships.
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="TAMPERED")  # != "SNAP"
    from orchestrator.workflow import build_workflow

    oc = _tdd_cfg(on_exhausted="abort", max_retries=2)
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        result = await workflow.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "failed"
    assert result["failed_task_id"] == "task:t1"
    assert "frozen" in (result["qa_failures"] or "").lower()
    assert stubs.pr_created is False
    # Diff-gate is ordered FIRST → it fails before QA each attempt, so QA never runs.
    assert stubs.qa_calls == 0


@pytest.mark.asyncio
async def test_untestable_task_uses_classic_path(monkeypatch, tmp_path):
    # testable=False → no diff-gate added; the classic implement→qa build runs and
    # ships even though the (irrelevant) hash would mismatch.
    stubs = _Stubs(n_tasks=1, ta_results=[TestAuthorResult(testable=False, summary="not unit-testable")])
    _patch(stubs, monkeypatch, hash_value="WOULD-MISMATCH")
    from orchestrator.workflow import build_workflow

    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))
        result = await workflow.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 1
    assert stubs.qa_calls == 1               # classic QA ran (no diff-gate gating it)
    assert stubs.pr_created is True


@pytest.mark.asyncio
async def test_resume_does_not_reauthor_tests(monkeypatch, tmp_path):
    # With an after_producer pause, drive the single task across resumes: the
    # test-author and implementer each run EXACTLY ONCE — on resume the body
    # re-executes but the completed @tasks replay from the checkpoint.
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    oc = _tdd_cfg(human_in_loop={"after_producer": True}, on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as workflow:
        await workflow.ainvoke("req", config=(c := _cfg()))             # plan_approval
        r = await workflow.ainvoke(Command(resume="yes"), config=c)     # author + impl → pause
        assert r["__interrupt__"][0].value["kind"] == "build_producer_pause"
        assert len(stubs.ta_calls) == 1
        assert len(stubs.impl_plans) == 1
        result = await workflow.ainvoke(Command(resume="yes"), config=c)  # diff-gate + qa → done

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 1          # NOT re-authored on resume
    assert len(stubs.impl_plans) == 1
