"""Phase 78b — TDD-aware decompose.

Under TDD a separate test-author writes each task's FAILING tests before the task
is implemented. So a standalone "write the unit tests" task is redundant and
harmful: it runs after the code exists, can't go red, and degrades (the Phase 75
dogfood hit exactly this and had to be re-planned by hand). Phase 78b makes the
decomposer aware of the division of labour: when `config.tdd`, a note is composed
into its user message telling it not to emit any test-writing task — every task is
behaviour-only with acceptance criteria the test-author turns into tests.

It is a prompt-level instruction threaded from `config.tdd`, NOT a testability
judgement — the decomposer still has no repo access (the Phase 73 trap). With
`tdd` off, the user message and behaviour are byte-for-byte unchanged.
"""

import pytest

from orchestrator.agents.decompose import (
    DecompositionResult,
    Task,
    _TDD_DECOMPOSE_NOTE,
    _build_user_message,
    decompose,
)
from orchestrator.config import OrchestratorConfig


# --------------------------------------------------------------------------- #
# unit: the note is gated on tdd, and composes with the existing max_tasks cap
# --------------------------------------------------------------------------- #


def test_user_message_includes_tdd_note_only_when_tdd():
    plan = "Add a cash-to-close line to the summary."
    on = _build_user_message(plan, max_tasks=0, tdd=True)
    off = _build_user_message(plan, max_tasks=0, tdd=False)

    assert plan in on and plan in off
    assert _TDD_DECOMPOSE_NOTE.strip() in on
    # The substance: test-author owns tests, no standalone test task.
    assert "test-author" in on and "standalone" in on
    # Off: classic behaviour, no note, no mention of the test-author.
    assert _TDD_DECOMPOSE_NOTE.strip() not in off
    assert "test-author" not in off


def test_user_message_defaults_to_no_tdd_note():
    # tdd defaults False, so classic (non-TDD) runs are unchanged.
    assert _TDD_DECOMPOSE_NOTE.strip() not in _build_user_message("plan", max_tasks=0)


def test_tdd_note_composes_with_max_tasks_cap():
    msg = _build_user_message("plan", max_tasks=3, tdd=True)
    assert "at most 3" in msg
    assert _TDD_DECOMPOSE_NOTE.strip() in msg


# --------------------------------------------------------------------------- #
# agent: decompose() forwards tdd into the user message it sends
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_decompose_forwards_tdd_into_user_message(monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return DecompositionResult(
            tasks=[Task(id="t", title="T", description="d", acceptance_criteria="a")]
        )

    monkeypatch.setattr(
        "orchestrator.agents.decompose.run_structured_completion", fake_run
    )

    await decompose("plan text", "model", tdd=True)
    assert _TDD_DECOMPOSE_NOTE.strip() in captured["user_message"]

    captured.clear()
    await decompose("plan text", "model", tdd=False)
    assert _TDD_DECOMPOSE_NOTE.strip() not in captured["user_message"]


# --------------------------------------------------------------------------- #
# workflow: config.tdd is threaded all the way to the decomposer
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("tdd_on", [True, False])
async def test_workflow_threads_config_tdd_to_decompose(monkeypatch, tmp_path, tdd_on):
    from tests.test_phase55_decomposition import _patch, _Stubs, _fresh_config, _interrupt_value
    from orchestrator.workflow import build_workflow

    stubs = _Stubs()
    _patch(stubs, monkeypatch)

    seen: dict = {}

    async def capture(plan_text, model="m", max_tasks=0, tdd=False):
        seen["tdd"] = tdd
        return DecompositionResult(
            tasks=[Task(id="t1", title="T", description="d", acceptance_criteria="a")]
        )

    monkeypatch.setattr("orchestrator.workflow.decompose", capture)

    cfg = (
        OrchestratorConfig(tdd=True, test_paths=["**/*.test.js"])
        if tdd_on
        else OrchestratorConfig()
    )
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=cfg) as wf:
        # The run halts at the plan-approval interrupt; decompose has run by then.
        result = await wf.ainvoke("add a feature", config=_fresh_config())

    _interrupt_value(result)  # asserts we stopped at plan_approval
    assert seen["tdd"] is tdd_on
