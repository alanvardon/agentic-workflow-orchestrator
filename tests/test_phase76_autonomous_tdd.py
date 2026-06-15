"""Phase 76 — autonomous-mode TDD (re-enable the guard).

`tdd = true` + `fully_autonomous = true` is no longer refused at load (Phase 72's
decision A). With no human red-review/re-author guard, the per-task station runs a
distinct `_run_autonomous_tdd_task` that stands in two pieces of machinery:

  - RED-CONFIRM IS A HARD GATE — a born-green / non-green-baseline / no-script-gate
    verdict (degrade_kind ∈ the red-confirm failures) ABORTS the task instead of
    silently degrading; a genuinely-untestable verdict still degrades to classic +
    a manual check.
  - BOUNDED RE-AUTHOR — the implement build runs bounded with on_exhausted="abort";
    an exhausted build auto re-authors the (possibly wrong) tests, capped at
    tdd_autonomous_reauthor_max, then aborts rather than looping to the safety
    ceiling.

Reuses the Phase 72 stub harness (patch `_run_test_author` / `_run_implementation_
producer` / `_hash_test_paths`, so the real @tasks still checkpoint/replay).
"""

import pytest
from langgraph.types import Command

from orchestrator import workflow as wf
from orchestrator.agents.test_author import TestAuthorResult
from orchestrator.agents.qa import QaResult
from orchestrator.config import OrchestratorConfig, load_config

from tests.conftest import task_build_config
from tests.test_phase72_test_author import _Stubs, _patch, _patch_scripts, _cfg


# --------------------------------------------------------------------------- #
# config: the dial + the relaxed guard
# --------------------------------------------------------------------------- #


def test_reauthor_cap_defaults_to_two():
    assert OrchestratorConfig().tdd_autonomous_reauthor_max == 2


def test_tdd_with_fully_autonomous_is_allowed():
    # Phase 72 forbade the combo (decision A); Phase 76 re-enables it.
    cfg = OrchestratorConfig(tdd=True, fully_autonomous=True, test_paths=["**/*.test.js"])
    assert cfg.tdd is True and cfg.fully_autonomous is True


def test_load_config_round_trips_reauthor_cap(tmp_path):
    toml = tmp_path / "orchestrator.toml"
    toml.write_text(
        'tdd = true\nfully_autonomous = true\ntest_paths = ["**/*.test.js"]\n'
        "tdd_autonomous_reauthor_max = 5\n",
        encoding="utf-8",
    )
    c = load_config(toml)
    assert c.fully_autonomous is True
    assert c.tdd_autonomous_reauthor_max == 5


# --------------------------------------------------------------------------- #
# unit: _run_test_author tags each testable=False verdict with a degrade_kind
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_degrade_kind_no_gate(monkeypatch):
    monkeypatch.setattr(wf, "_script_gate_steps", lambda cfg, refs: [])
    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["builtin:qa"])
    assert res.testable is False
    assert res.degrade_kind == wf._DEGRADE_NO_GATE


@pytest.mark.asyncio
async def test_degrade_kind_not_green_baseline(monkeypatch):
    _patch_scripts(monkeypatch, [(False, "pre-existing failure")])
    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is False
    assert res.degrade_kind == wf._DEGRADE_NOT_GREEN


@pytest.mark.asyncio
async def test_degrade_kind_untestable(monkeypatch):
    _patch_scripts(monkeypatch, [(True, "")])

    async def _author(plan, model, system_prompt=None, allowed_tools=None,
                      disallowed_tools=None, feedback=None):
        return TestAuthorResult(testable=False, summary="DOM-only, no harness")

    monkeypatch.setattr(wf, "author_tests", _author)
    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is False
    assert res.degrade_kind == wf._DEGRADE_UNTESTABLE
    assert res.summary == "DOM-only, no harness"  # the author's reason is preserved


@pytest.mark.asyncio
async def test_degrade_kind_born_green(monkeypatch):
    _patch_scripts(monkeypatch, [(True, ""), (True, "")])  # green before AND after

    async def _author(plan, model, system_prompt=None, allowed_tools=None,
                      disallowed_tools=None, feedback=None):
        return TestAuthorResult(testable=True, summary="claims testable")

    monkeypatch.setattr(wf, "author_tests", _author)
    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is False
    assert res.degrade_kind == wf._DEGRADE_BORN_GREEN


@pytest.mark.asyncio
async def test_testable_has_no_degrade_kind(monkeypatch):
    _patch_scripts(monkeypatch, [(True, ""), (False, "RED: 1 failing")])

    async def _author(plan, model, system_prompt=None, allowed_tools=None,
                      disallowed_tools=None, feedback=None):
        return TestAuthorResult(testable=True, summary="covers X")

    monkeypatch.setattr(wf, "author_tests", _author)
    monkeypatch.setattr(wf, "_hash_test_paths", lambda paths, root: "SNAP")
    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.testable is True
    assert res.degrade_kind is None


def test_born_green_is_a_red_confirm_failure_but_untestable_is_not():
    assert wf._DEGRADE_BORN_GREEN in wf._RED_CONFIRM_FAILURES
    assert wf._DEGRADE_NO_GATE in wf._RED_CONFIRM_FAILURES
    assert wf._DEGRADE_NOT_GREEN in wf._RED_CONFIRM_FAILURES
    assert wf._DEGRADE_UNTESTABLE not in wf._RED_CONFIRM_FAILURES


# --------------------------------------------------------------------------- #
# integration: the autonomous-TDD station
# --------------------------------------------------------------------------- #


def _auto_tdd_cfg(*, reauthor_max=2, **kw) -> OrchestratorConfig:
    # tdd + fully_autonomous on; coverage_critic off so these tests exercise the
    # red-confirm/re-author machinery, not the critic agent. (model_copy bypasses
    # validation, but the combo is allowed anyway.)
    return task_build_config(**kw).model_copy(update={
        "tdd": True, "test_paths": ["**/*.test.js"],
        "fully_autonomous": True, "tdd_coverage_critic": False,
        "tdd_autonomous_reauthor_max": reauthor_max,
    })


@pytest.mark.asyncio
async def test_autonomous_tdd_happy_path_ships_without_pauses(monkeypatch, tmp_path):
    # testable task, impl passes first attempt: ONE ainvoke, no human pause at all.
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")  # diff-gate passes (== stub snapshot)
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_auto_tdd_cfg()) as w:
        result = await w.ainvoke("req", config=_cfg())

    assert result["status"] == "succeeded"
    assert "__interrupt__" not in result          # autonomous: no red-review pause
    assert result["pr_url"] == "https://github.com/test/pr/1"
    assert len(stubs.ta_calls) == 1               # authored once
    assert len(stubs.impl_plans) == 1             # implemented once (diff-gate + qa passed)
    assert stubs.qa_calls == 1


@pytest.mark.asyncio
async def test_autonomous_born_green_hard_aborts(monkeypatch, tmp_path):
    # A red-confirm failure (born-green) with no human to eyeball it → HARD abort,
    # not the silent classic degrade the supervised path would take.
    stubs = _Stubs(n_tasks=1, ta_results=[
        TestAuthorResult(testable=False, degrade_kind=wf._DEGRADE_BORN_GREEN,
                         summary="authored tests did not fail (born-green)"),
    ])
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_auto_tdd_cfg()) as w:
        result = await w.ainvoke("req", config=_cfg())

    assert result["status"] == "failed"
    assert result["failed_task_id"] == "task:t1"
    assert "red-confirm failed (born_green)" in (result["qa_failures"] or "")
    assert len(stubs.impl_plans) == 0             # never implemented
    assert stubs.pr_created is False


@pytest.mark.asyncio
async def test_autonomous_untestable_degrades_with_manual_check(monkeypatch, tmp_path):
    # A genuinely-untestable verdict is NOT a red-confirm failure: degrade to the
    # classic implement→qa path (ships) and surface a manual check in the result.
    stubs = _Stubs(n_tasks=1, ta_results=[
        TestAuthorResult(testable=False, degrade_kind=wf._DEGRADE_UNTESTABLE,
                         summary="DOM-only, no harness"),
    ])
    _patch(stubs, monkeypatch, hash_value="WOULD-MISMATCH")  # proves no diff-gate gates it
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_auto_tdd_cfg()) as w:
        result = await w.ainvoke("req", config=_cfg())

    assert result["status"] == "succeeded"
    assert stubs.qa_calls == 1                    # classic QA ran (no diff-gate)
    assert stubs.pr_created is True
    assert result["manual_checks"] == [{
        "task_id": "t1",
        "title": "Task 1",
        "acceptance_criteria": "criterion 1",
        "reason": "DOM-only, no harness",
    }]


@pytest.mark.asyncio
async def test_autonomous_bounded_reauthor_then_fails(monkeypatch, tmp_path):
    # The implementation can never pass the frozen tests (QA always FAILs). With
    # reauthor_max=2 and max_retries=1: author + 2 re-authors (3 builds), then the
    # task aborts with a clear re-author reason — NOT an infinite loop to the ceiling.
    stubs = _Stubs(n_tasks=1, qa_verdicts=[QaResult(result="FAIL", failures=f"nope {i}") for i in range(3)])
    _patch(stubs, monkeypatch, hash_value="SNAP")  # diff-gate passes → QA is what fails
    from orchestrator.workflow import build_workflow

    cfg = _auto_tdd_cfg(reauthor_max=2, max_retries=1)
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as w:
        result = await w.ainvoke("req", config=_cfg())

    assert result["status"] == "failed"
    assert result["failed_task_id"] == "task:t1"
    assert "re-author" in (result["qa_failures"] or "")
    assert len(stubs.ta_calls) == 3               # 1 author + 2 re-authors (the cap)
    assert stubs.ta_feedback[0] is None
    assert all("could not make these tests pass" in (f or "") for f in stubs.ta_feedback[1:])
    assert stubs.qa_calls == 3                     # one bounded attempt per round
    assert stubs.pr_created is False


@pytest.mark.asyncio
async def test_autonomous_reauthor_then_succeeds(monkeypatch, tmp_path):
    # Round 0's build fails (QA FAIL) → re-author → round 1's build passes → ships.
    stubs = _Stubs(n_tasks=1, qa_verdicts=[QaResult(result="FAIL", failures="nope"), QaResult(result="PASS")])
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    cfg = _auto_tdd_cfg(reauthor_max=2, max_retries=1)
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as w:
        result = await w.ainvoke("req", config=_cfg())

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 2               # authored, then re-authored once
    assert stubs.ta_feedback[0] is None
    assert "could not make these tests pass" in stubs.ta_feedback[1]
    assert stubs.qa_calls == 2
    assert stubs.pr_created is True
