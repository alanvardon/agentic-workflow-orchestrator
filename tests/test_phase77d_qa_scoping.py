"""Phase 77d — QA scoping + test-plan under TDD (fourth/final slice of Phase 77).

Two boundary fixes from the Phase 75 dogfood review:

  1. ``qa.md`` is the QA agent's OWN record. ``QaResult`` gained an optional
     ``review`` field (what the agent reviewed/ran, each check ✓/✗); ``write_qa``
     renders it as a ``## Checks performed`` section above any ``## Failures``,
     replacing the bare one-line ``# QA Result: PASS``. The TDD red-green results
     never bleed in — they live in the per-task ``test-author/`` + ``impl/`` folders.

  2. The implementer's ``test-plan.md`` is suppressed when ``config.tdd`` is on (the
     executed evidence supersedes it), and STRICTLY only then — with TDD off the
     manual checklist is still written, unchanged.

Layered like the other Phase 77 tests: fast, LLM-free unit tests of the two artifact
writers + the model field, plus full-workflow tests exercising the real wiring (the
shared ``_Stubs`` / ``_patch`` harness, patching the inner agents so the @tasks still
replay).
"""

import uuid

import pytest
from langgraph.types import Command

import orchestrator.run_artifacts as ra
from orchestrator.agents.qa import QaResult
from orchestrator.agents.summarize import SummaryResult


def _runs(monkeypatch, tmp_path):
    """Point the artifact writer at an isolated runs dir (keeps the repo clean)."""
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")


# --------------------------------------------------------------------------- #
# unit: QaResult.review is a pure-additive optional field
# --------------------------------------------------------------------------- #


def test_qa_result_review_defaults_none():
    # Optional + defaulted → a pure-additive checkpoint change (no schema_version
    # bump), and the scripted-gate FAIL path that returns before the agent runs
    # simply leaves it None.
    assert QaResult(result="PASS").review is None
    assert QaResult(result="PASS").schema_version == 1
    assert QaResult(result="PASS", review="checked X ✓").review == "checked X ✓"


# --------------------------------------------------------------------------- #
# unit: write_qa — QA-agent-only, now with the review section
# --------------------------------------------------------------------------- #


def test_write_qa_renders_review_section(monkeypatch, tmp_path):
    _runs(monkeypatch, tmp_path)
    ra.write_qa(
        "tid",
        QaResult(result="PASS", review="- static-checks ✓ PASS\n- plan adherence ✓ PASS"),
    )
    md = (tmp_path / "runs" / "tid" / "qa.md").read_text(encoding="utf-8")
    assert md.startswith("# QA Result: PASS")
    assert "## Checks performed" in md
    assert "static-checks ✓ PASS" in md and "plan adherence ✓ PASS" in md


def test_write_qa_without_review_is_minimal(monkeypatch, tmp_path):
    # The scripted-gate FAIL path returns a QaResult with no review — qa.md must
    # still render (verdict + failures), with no empty "## Checks performed".
    _runs(monkeypatch, tmp_path)
    ra.write_qa("tid", QaResult(result="FAIL", failures="script lint.sh exited 1"))
    md = (tmp_path / "runs" / "tid" / "qa.md").read_text(encoding="utf-8")
    assert md.startswith("# QA Result: FAIL")
    assert "## Checks performed" not in md
    assert "## Failures" in md and "lint.sh exited 1" in md


def test_write_qa_is_qa_agent_only(monkeypatch, tmp_path):
    # The writer renders ONLY what the QA agent emitted (verdict / review / failures).
    # It never reaches into the red-green loop, so the freeze/snapshot/RED-run
    # vocabulary of the test-author + impl evidence can't leak into qa.md.
    _runs(monkeypatch, tmp_path)
    ra.write_qa("tid", QaResult(result="PASS", review="reviewed the diff against the plan"))
    md = (tmp_path / "runs" / "tid" / "qa.md").read_text(encoding="utf-8").lower()
    for leaked in ("snapshot", "freeze", "red run", "attempt-", "baseline"):
        assert leaked not in md


# --------------------------------------------------------------------------- #
# unit: write_summary — test-plan.md gated strictly on tdd
# --------------------------------------------------------------------------- #


def _summary() -> SummaryResult:
    return SummaryResult(summary="did the thing", test_plan="1. click it\n2. see result")


def test_write_summary_under_tdd_omits_test_plan(monkeypatch, tmp_path):
    _runs(monkeypatch, tmp_path)
    ra.write_summary("tid", _summary(), tdd=True)
    d = tmp_path / "runs" / "tid"
    assert (d / "summary.md").read_text(encoding="utf-8") == "did the thing"
    assert not (d / "test-plan.md").exists()  # executed evidence supersedes it


def test_write_summary_without_tdd_writes_test_plan(monkeypatch, tmp_path):
    # Strict gate: TDD off → the manual checklist still earns its place. The default
    # (no tdd kwarg) is the unchanged classic behaviour.
    _runs(monkeypatch, tmp_path)
    ra.write_summary("tid-a", _summary(), tdd=False)
    ra.write_summary("tid-b", _summary())  # default tdd=False
    for tid in ("tid-a", "tid-b"):
        d = tmp_path / "runs" / tid
        assert (d / "summary.md").exists()
        assert "click it" in (d / "test-plan.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# integration: the real workflow wiring
# --------------------------------------------------------------------------- #


def _cfg() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}


@pytest.mark.asyncio
async def test_tdd_run_omits_test_plan_and_writes_qa_review(monkeypatch, tmp_path):
    # Full TDD workflow: test-plan.md is suppressed, and the QA agent's review
    # flows end-to-end into qa.md's "## Checks performed" section.
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg
    from orchestrator.workflow import build_workflow

    _runs(monkeypatch, tmp_path)
    stubs = _Stubs(
        n_tasks=1,
        qa_verdicts=[QaResult(result="PASS", review="ran static-checks ✓ PASS")],
    )
    _patch(stubs, monkeypatch, hash_value="SNAP")
    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    d = ra._run_dir(c["configurable"]["thread_id"])
    assert (d / "summary.md").exists()
    assert not (d / "test-plan.md").exists()        # TDD → suppressed
    qa_md = (d / "qa.md").read_text(encoding="utf-8")
    assert "## Checks performed" in qa_md and "static-checks ✓ PASS" in qa_md


@pytest.mark.asyncio
async def test_non_tdd_run_writes_test_plan(monkeypatch, tmp_path):
    # The strict gate, proven through the real call site: TDD off → test-plan.md is
    # written exactly as before.
    from tests.conftest import task_build_config
    from tests.test_phase72_test_author import _Stubs, _patch
    from orchestrator.workflow import build_workflow

    _runs(monkeypatch, tmp_path)
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    oc = task_build_config(on_exhausted="abort")  # tdd defaults off
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    d = ra._run_dir(c["configurable"]["thread_id"])
    assert (d / "test-plan.md").exists()  # no automated suite → manual checklist kept
