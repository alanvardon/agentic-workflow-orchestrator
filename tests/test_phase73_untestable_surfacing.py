"""Phase 73 — surface untestable / degraded TDD tasks.

The generic distillation of "73": when tdd is on, a task the test-author judges
untestable (or born-green / no-script-gate) degrades to the classic implement→qa
path. That degradation is now made visible:
  - a test-author-<task>.md artifact is written for EVERY TDD task (not only the
    testable ones), recording the verdict + reason;
  - the degraded tasks' acceptance criteria are collected into the run result
    (`manual_checks`) and a `manual-checks.md` artifact — the spec a human must
    verify by hand. No project specifics; tdd off / all-testable runs are unchanged.
"""

import uuid

import pytest
from langgraph.types import Command

import orchestrator.run_artifacts as ra
from orchestrator.agents.decompose import Task
from orchestrator.agents.test_author import TestAuthorResult


# --------------------------------------------------------------------------- #
# unit: write_manual_checks
# --------------------------------------------------------------------------- #


def test_write_manual_checks_lists_degraded_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")
    ra.write_manual_checks("tid", [
        {"task_id": "t2", "title": "DOM toggle", "acceptance_criteria": "clicking #x flips theme",
         "reason": "DOM-only, no harness"},
    ])
    f = tmp_path / "runs" / "tid" / "manual-checks.md"
    assert f.exists()
    text = f.read_text(encoding="utf-8")
    assert "DOM toggle" in text
    assert "t2" in text
    assert "clicking #x flips theme" in text
    assert "DOM-only, no harness" in text


def test_write_manual_checks_empty_is_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")
    ra.write_manual_checks("tid", [])
    assert not (tmp_path / "runs" / "tid" / "manual-checks.md").exists()


def test_write_test_author_folder_records_untestable_verdict(monkeypatch, tmp_path):
    # An untestable / degraded task still gets a test-author/ folder (Phase 73
    # surfacing preserved), but only summary.md — no copied tests / RED run / hash.
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(ra, "find_project_root", lambda: tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    task = Task(id="t2", title="DOM toggle", description="d", acceptance_criteria="c")
    ra.write_test_author_folder(
        "tid", 2, task, TestAuthorResult(testable=False, summary="no harness"),
        ["**/*.test.js"],
    )
    folder = tmp_path / "runs" / "tid" / "task-02-t2" / "test-author"
    text = (folder / "summary.md").read_text(encoding="utf-8")
    assert "**Testable:** False" in text
    assert "no harness" in text
    # No tests / RED run / hash for a degraded task.
    assert not (folder / "results-test-run.md").exists()
    assert not (folder / "test-snapshot-hash.md").exists()


# --------------------------------------------------------------------------- #
# integration: degraded task surfaces in the result + artifacts; clean otherwise
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_degraded_task_surfaces_in_result(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(
        n_tasks=1,
        ta_results=[TestAuthorResult(testable=False, summary="DOM-only, no harness")],
    )
    _patch(stubs, monkeypatch, hash_value="SNAP")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_tdd_cfg(on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.impl_plans) == 1            # ran the classic path
    assert result["manual_checks"] == [{
        "task_id": "t1",
        "title": "Task 1",
        "acceptance_criteria": "criterion 1",
        "reason": "DOM-only, no harness",
    }]


@pytest.mark.asyncio
async def test_all_testable_run_has_no_manual_checks_key(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(n_tasks=1)  # default → testable=True
    _patch(stubs, monkeypatch, hash_value="SNAP")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=_tdd_cfg(on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    # Omitted entirely when nothing degraded — non-TDD / all-testable shape unchanged.
    assert "manual_checks" not in result
