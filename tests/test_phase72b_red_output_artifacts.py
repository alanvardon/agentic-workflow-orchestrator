"""Phase 72b — RED output → implementer (first attempt) + red→green artifact.

When a TDD task is testable, the test-author has already written failing tests.
Two follow-ons, both consuming what TestAuthorResult already captured:
  - the failing (RED) output is injected into the implementer's FIRST-attempt plan
    so it sees the exact spec it must turn green (before any gate feeds it back);
  - the red→green record is persisted to the run's artifact folder.

The test-author itself always sees the CLEAN task plan (so its @task cache key is
stable across resumes); only the implementer's plan gains the RED section.
"""

import uuid

import pytest
from langgraph.types import Command

import orchestrator.run_artifacts as ra
from orchestrator import workflow
from orchestrator.agents.decompose import Task
from orchestrator.agents.test_author import TestAuthorResult


# --------------------------------------------------------------------------- #
# unit: _compose_red_green
# --------------------------------------------------------------------------- #


def test_compose_red_green_appends_failing_output():
    out = workflow._compose_red_green("PLAN BODY", "AssertionError: expected 3")
    assert "PLAN BODY" in out
    assert "Failing tests to make pass (RED)" in out
    assert "AssertionError: expected 3" in out
    assert "must not edit them" in out


def test_compose_red_green_empty_output_is_unchanged():
    assert workflow._compose_red_green("PLAN BODY", "") == "PLAN BODY"


# --------------------------------------------------------------------------- #
# unit: write_test_author_folder artifact (Phase 77b)
# --------------------------------------------------------------------------- #


def _task(task_id="t1", title="Task one"):
    return Task(id=task_id, title=title, description="do it", acceptance_criteria="works")


def test_write_test_author_folder_writes_red_green_record(monkeypatch, tmp_path):
    # Override conftest's autouse _isolate_runs_dir to point at tmp_path, and the
    # project root the copier globs (no test files there → empty copy, summary only).
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(ra, "find_project_root", lambda: tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    ta = TestAuthorResult(
        testable=True, summary="covers behaviour X",
        snapshot="HASH123", red_output="1 test failing", full_run="FULL RED LOG",
    )
    ra.write_test_author_folder("tid", 1, _task(), ta, ["**/*.test.js"])

    folder = tmp_path / "runs" / "tid" / "task-01-t1" / "test-author"
    summary = (folder / "summary.md").read_text(encoding="utf-8")
    assert "t1" in summary and "covers behaviour X" in summary
    assert (folder / "test-snapshot-hash.md").read_text(encoding="utf-8").find("HASH123") != -1
    # The RED-run evidence is the COMPLETE run (full_run), not the failures summary.
    assert "FULL RED LOG" in (folder / "results-test-run.md").read_text(encoding="utf-8")


def test_write_test_author_folder_sanitises_task_id(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")
    monkeypatch.setattr(ra, "find_project_root", lambda: tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    ra.write_test_author_folder(
        "tid", 2, _task("feat/weird id", "Weird"), TestAuthorResult(testable=True),
        ["**/*.test.js"],
    )
    folders = list((tmp_path / "runs" / "tid").glob("task-*"))
    assert len(folders) == 1
    # no path separators / spaces leaked into the folder name
    assert "/" not in folders[0].name and " " not in folders[0].name
    assert folders[0].name.startswith("task-02-")


# --------------------------------------------------------------------------- #
# integration: the RED output reaches the implementer (and not when untestable)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_red_output_reaches_implementer_first_attempt(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    # _Stubs.run_test_author returns red_output="boom" for a testable task.
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as wf:
        await wf.ainvoke("req", config=(c := _cfg()))
        result = await wf.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.impl_plans) == 1
    assert "boom" in stubs.impl_plans[0]
    assert "Failing tests to make pass (RED)" in stubs.impl_plans[0]


@pytest.mark.asyncio
async def test_untestable_task_gets_no_red_section(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(
        n_tasks=1,
        ta_results=[TestAuthorResult(testable=False, summary="DOM-only, no harness")],
    )
    _patch(stubs, monkeypatch, hash_value="SNAP")
    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as wf:
        await wf.ainvoke("req", config=(c := _cfg()))
        result = await wf.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.impl_plans) == 1
    assert "Failing tests to make pass (RED)" not in stubs.impl_plans[0]
