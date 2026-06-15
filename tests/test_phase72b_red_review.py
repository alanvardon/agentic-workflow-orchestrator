"""Phase 72b — supervised red-review pause + re-author escape.

When tdd is on, after the test-author writes failing tests and the green→red
transition is confirmed, the run PAUSES (default-on) for a human to review the RED
suite before implementation:
  - 'yes'        → implement against the tests;
  - feedback     → re-author the tests with that guidance (the escape Phase 74 reuses);
  - 'abort'      → stop the run (clean status="failed").
Suppressed when tdd_red_review = false (or fully_autonomous). On re-author the prior
red tests are already in the tree, so the green-before precondition is skipped.
"""

import uuid

import pytest
from langgraph.types import Command

from orchestrator import workflow as wf
from orchestrator.agents.test_author import TestAuthorResult, _build_user_message
from orchestrator.config import OrchestratorConfig, load_config
from tests.conftest import task_build_config
from tests.test_phase72_test_author import _Stubs, _patch, _patch_scripts, _cfg


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_tdd_red_review_defaults_on():
    assert OrchestratorConfig().tdd_red_review is True


def test_load_config_round_trips_tdd_red_review(tmp_path):
    toml = tmp_path / "orchestrator.toml"
    toml.write_text(
        'tdd = true\ntest_paths = ["**/*.test.js"]\ntdd_red_review = false\n',
        encoding="utf-8",
    )
    assert load_config(toml).tdd_red_review is False


# --------------------------------------------------------------------------- #
# unit: feedback in the author message + approve-word classification
# --------------------------------------------------------------------------- #


def test_build_user_message_appends_feedback():
    out = _build_user_message("PLAN", "add an empty-input case")
    assert "PLAN" in out
    assert "Re-author feedback" in out
    assert "add an empty-input case" in out


def test_build_user_message_no_feedback_is_plain():
    assert "Re-author feedback" not in _build_user_message("PLAN")


@pytest.mark.parametrize("word,expected", [
    ("yes", True), ("YES", True), ("approve", True), (" ok ", True),
    ("add more cases", False), ("abort", False), (42, False),
])
def test_is_red_review_approve(word, expected):
    assert wf._is_red_review_approve(word) is expected


# --------------------------------------------------------------------------- #
# unit: re-author skips the green-before precondition
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reauthor_skips_green_before(monkeypatch):
    # ONE script result: the red-AFTER check. If green-before still ran it would
    # consume this (False) → "suite not green before authoring" fallback. Skipping
    # it on re-author means the single result is the red-after confirmation.
    _patch_scripts(monkeypatch, [(False, "RED: 1 failing")])

    async def _author(plan, model, system_prompt=None, allowed_tools=None,
                      disallowed_tools=None, feedback=None):
        assert feedback == "tighten the assertion"
        return TestAuthorResult(testable=True, summary="revised")

    monkeypatch.setattr(wf, "author_tests", _author)
    monkeypatch.setattr(wf, "_hash_test_paths", lambda paths, root: "SNAP2")

    res = await wf._run_test_author(
        "plan", "model", ["**/*.test.js"], ["defs:tests"], feedback="tighten the assertion"
    )
    assert res.testable is True
    assert res.snapshot == "SNAP2"
    assert "RED" in res.red_output


# --------------------------------------------------------------------------- #
# integration: the pause + the three resume verbs
# --------------------------------------------------------------------------- #


def _review_cfg(**kw):
    # coverage_critic off: these tests exercise the red-review pause, not the critic.
    return task_build_config(**kw).model_copy(
        update={"tdd": True, "test_paths": ["**/*.test.js"],
                "tdd_red_review": True, "tdd_coverage_critic": False}
    )


@pytest.mark.asyncio
async def test_red_review_pause_then_approve_ships(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_review_cfg(on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))         # pause: plan_approval
        r2 = await w.ainvoke(Command(resume="yes"), config=c)  # proceed → pause: red_review
        iv = r2["__interrupt__"][0].value
        assert iv["kind"] == "red_review"
        assert iv["red_output"] == "boom"
        assert iv["task_id"] == "t1"
        r3 = await w.ainvoke(Command(resume="yes"), config=c)  # implement against tests → ship

    assert r3["status"] == "succeeded"
    assert len(stubs.ta_calls) == 1          # authored once, no re-author
    assert len(stubs.impl_plans) == 1


@pytest.mark.asyncio
async def test_red_review_abort_fails_the_run(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_review_cfg(on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        await w.ainvoke(Command(resume="yes"), config=c)       # → pause: red_review
        r3 = await w.ainvoke(Command(resume="abort"), config=c)  # reject the tests

    assert r3["status"] == "failed"
    assert r3["failed_task_id"] == "task:t1"
    assert len(stubs.impl_plans) == 0        # never implemented


@pytest.mark.asyncio
async def test_red_review_feedback_reauthors(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_review_cfg(on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        await w.ainvoke(Command(resume="yes"), config=c)                  # → pause: red_review (#1)
        r3 = await w.ainvoke(Command(resume="cover the empty case"), config=c)  # re-author → pause (#2)
        assert r3["__interrupt__"][0].value["kind"] == "red_review"
        r4 = await w.ainvoke(Command(resume="yes"), config=c)             # accept → ship

    assert r4["status"] == "succeeded"
    assert len(stubs.ta_calls) == 2                       # authored, then re-authored
    assert stubs.ta_feedback == [None, "cover the empty case"]
    assert len(stubs.impl_plans) == 1


@pytest.mark.asyncio
async def test_red_review_off_does_not_pause(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    cfg = _review_cfg(on_exhausted="abort").model_copy(update={"tdd_red_review": False})
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        r2 = await w.ainvoke(Command(resume="yes"), config=c)  # one resume → ships, no red_review

    assert r2["status"] == "succeeded"
    assert len(stubs.ta_calls) == 1
