"""Phase 81 — decompose-time testability gating.

The decomposer marks each Task `testable`. Under TDD a `testable=False` task SKIPS
the test-author + critic entirely and takes the classic implement→qa path (a
synthesized untestable degrade through the Phase 73 manual_checks plumbing), so
markup/CSS/docs tasks don't pay a full author leg to discover they're born-green.

The distinction from a RUNTIME degrade (Phase 73): there the author still runs and
returns testable=False (one author leg paid); here the author is NEVER called for a
task the decomposer pre-marked non-testable — that's the cost saving.

tdd-off behaviour and the runtime untestable/born-green backstops are unchanged.
"""

import pytest
from langgraph.types import Command

from orchestrator.agents.decompose import Task, _build_user_message

from tests.conftest import task_build_config
from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg


# --------------------------------------------------------------------------- #
# decompose-level: the flag default + the note
# --------------------------------------------------------------------------- #


def test_task_testable_defaults_true():
    # Default True keeps tdd-off and pre-81 checkpoints behaving as before.
    t = Task(id="t", title="T", description="d", acceptance_criteria="c")
    assert t.testable is True


def test_task_testable_can_be_false():
    t = Task(id="t", title="T", description="d", acceptance_criteria="c", testable=False)
    assert t.testable is False


def test_tdd_note_asks_for_per_task_testable_flag():
    msg = _build_user_message("PLAN", max_tasks=0, tdd=True)
    assert "testable" in msg.lower()


def test_non_tdd_message_is_unchanged_and_omits_testable_note():
    # 78b invariant: tdd-off decompose message is byte-for-byte the plain plan.
    msg = _build_user_message("PLAN", max_tasks=0, tdd=False)
    assert msg == "## Plan\n\nPLAN"
    assert "testable" not in msg.lower()


# --------------------------------------------------------------------------- #
# station-level: a non-testable task skips the author and degrades to classic
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_testable_task_skips_author_entirely(monkeypatch, tmp_path):
    stubs = _Stubs(n_tasks=2)
    # t1 marked non-testable at decompose (markup), t2 testable (logic).
    stubs.tasks = [
        Task(id="t1", title="markup", description="add card",
             acceptance_criteria="card renders", testable=False),
        Task(id="t2", title="calc", description="sum two numbers",
             acceptance_criteria="2+2=4", testable=True),
    ]
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    async with build_workflow(
        db_path=str(tmp_path / "ckpt.db"), config=_tdd_cfg(on_exhausted="abort")
    ) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    # THE WIN: the test-author ran ONLY for the testable task — never for t1.
    assert len(stubs.ta_calls) == 1
    # Both tasks were still implemented + qa'd (t1 via the classic path).
    assert len(stubs.impl_plans) == 2
    assert stubs.qa_calls == 2
    # The skipped task's acceptance criterion is surfaced as a manual check.
    assert result["manual_checks"] == [{
        "task_id": "t1",
        "title": "markup",
        "acceptance_criteria": "card renders",
        "reason": "non-testable per decompose",
    }]


@pytest.mark.asyncio
async def test_all_testable_still_authors_per_task(monkeypatch, tmp_path):
    # Default tasks are testable=True → the station runs the author for each.
    stubs = _Stubs(n_tasks=2)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    async with build_workflow(
        db_path=str(tmp_path / "ckpt.db"), config=_tdd_cfg(on_exhausted="abort")
    ) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 2
    assert "manual_checks" not in result


@pytest.mark.asyncio
async def test_tdd_off_ignores_the_flag(monkeypatch, tmp_path):
    # With tdd OFF the flag is inert: classic path, no author, no manual_checks.
    stubs = _Stubs(n_tasks=1)
    stubs.tasks = [
        Task(id="t1", title="x", description="d", acceptance_criteria="c", testable=False),
    ]
    _patch(stubs, monkeypatch, hash_value="SNAP")
    from orchestrator.workflow import build_workflow

    async with build_workflow(
        db_path=str(tmp_path / "ckpt.db"), config=task_build_config(on_exhausted="abort")
    ) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 0        # tdd off → never authors
    assert len(stubs.impl_plans) == 1
    assert "manual_checks" not in result
