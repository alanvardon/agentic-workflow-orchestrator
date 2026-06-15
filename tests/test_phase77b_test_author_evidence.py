"""Phase 77b — test-author evidence folder (second slice of the Phase 77 layer).

The flat ``test-author-<id>.md`` is replaced by a per-task ``task-NN-<id>/test-author/``
folder, written ONCE after the authoring process converges. For a TESTABLE task it
is the proof the red-green discipline happened:

  - the FINAL accepted ``test_paths`` file(s), copied verbatim (any language/layout);
  - ``results-test-run.md`` — the COMPLETE RED run (Phase 77a ``full_output``), proof
    the suite actually ran red, not just the failing delta;
  - ``test-snapshot-hash.md`` — the deterministic ``_hash_test_paths`` baseline the
    diff-gate freezes;
  - ``summary.md`` — testable verdict, coverage-critic verdict(s), # re-author rounds,
    red-review outcome.

EVERY TDD task still gets a folder (Phase 73 surfacing): an untestable / degraded
task gets ``summary.md`` only. The verbatim copies land in ``.orchestrator/``, so the
freeze hash + copier must exclude that workspace (``iter_test_files``) or ``**`` would
sweep them back in and break the diff-gate — the landmine this slice fixes.

Layered like the other TDD tests: fast LLM-free unit tests of the artifact writer +
the exclusion helper, plus full-workflow tests that exercise the real ``_run_task_loop``
wiring (patching the inner ``_run_test_author`` so the ``@task`` still replays).
"""

import uuid

import pytest
from langgraph.types import Command

import orchestrator.run_artifacts as ra
from orchestrator import workflow as wf
from orchestrator.agents.coverage_critic import CoverageCriticResult
from orchestrator.agents.decompose import Task
from orchestrator.agents.test_author import TestAuthorResult
from orchestrator.manifest import ScriptStep
from orchestrator.paths import iter_test_files


def _task(task_id="t1", title="Task one"):
    return Task(id=task_id, title=title, description="do it", acceptance_criteria="works")


def _proj(monkeypatch, tmp_path):
    """Point the artifact writer at an isolated runs dir + a controlled project root
    the copier globs, and return the project root."""
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(ra, "find_project_root", lambda: root)
    return root


# --------------------------------------------------------------------------- #
# unit: iter_test_files excludes the orchestrator workspace (the landmine)
# --------------------------------------------------------------------------- #


def test_iter_test_files_excludes_orchestrator_dir(tmp_path):
    # A real project test file + an evidence COPY of it under .orchestrator/.
    (tmp_path / "calc.test.js").write_text("real")
    copy = tmp_path / ".orchestrator" / "runs" / "x" / "test-author" / "calc.test.js"
    copy.parent.mkdir(parents=True)
    copy.write_text("evidence copy")

    matched = iter_test_files(["**/*.test.js"], tmp_path)
    rels = [str(p.relative_to(tmp_path)) for p in matched]
    assert rels == ["calc.test.js"]  # the copy under .orchestrator/ is NOT swept in


def test_hash_test_paths_unperturbed_by_evidence_copies(tmp_path):
    # The freeze hash must not change when 77b writes a verbatim copy into the
    # orchestrator workspace — otherwise the diff-gate would fail spuriously.
    (tmp_path / "calc.test.js").write_text("frozen tests")
    root = str(tmp_path)
    baseline = wf._hash_test_paths(["**/*.test.js"], root)

    copy = tmp_path / ".orchestrator" / "runs" / "tid" / "test-author" / "calc.test.js"
    copy.parent.mkdir(parents=True)
    copy.write_text("frozen tests")  # identical name + content under .orchestrator/

    assert wf._hash_test_paths(["**/*.test.js"], root) == baseline


# --------------------------------------------------------------------------- #
# unit: _run_script_gates returns the COMPLETE run alongside the failures summary
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_script_gates_returns_full_output(tmp_path):
    ok = tmp_path / "ok.sh"
    ok.write_text('#!/bin/sh\necho "PASS-marker"\nexit 0\n')
    ok.chmod(0o755)
    bad = tmp_path / "bad.sh"
    bad.write_text('#!/bin/sh\necho "FAIL-marker"\nexit 1\n')
    bad.chmod(0o755)

    green, failing, full = await wf._run_script_gates(
        [ScriptStep(id="ok", path="ok.sh"), ScriptStep(id="bad", path="bad.sh")],
        str(tmp_path),
    )
    assert green is False
    # failing summary carries only the failing gate; full carries EVERY gate.
    assert "FAIL-marker" in failing and "PASS-marker" not in failing
    assert "PASS-marker" in full and "FAIL-marker" in full


@pytest.mark.asyncio
async def test_run_test_author_captures_full_run_distinct_from_red_output(monkeypatch):
    from tests.test_phase72_test_author import _patch_scripts

    # (green_before), then (green_after=False, failures-summary, COMPLETE-run).
    _patch_scripts(monkeypatch, [
        (True, "", ""),
        (False, "FAILED: 1 test", "test_a PASS\ntest_b FAIL: boom"),
    ])

    async def _author(plan, model, system_prompt=None, allowed_tools=None,
                      disallowed_tools=None, feedback=None):
        return TestAuthorResult(testable=True, summary="covers X")

    monkeypatch.setattr(wf, "author_tests", _author)
    monkeypatch.setattr(wf, "_hash_test_paths", lambda paths, root: "SNAP")

    res = await wf._run_test_author("plan", "model", ["**/*.test.js"], ["defs:tests"])
    assert res.red_output == "FAILED: 1 test"          # failures-only summary
    assert res.full_run == "test_a PASS\ntest_b FAIL: boom"  # the COMPLETE run
    assert "test_a PASS" not in res.red_output         # distinct fields


# --------------------------------------------------------------------------- #
# unit: write_test_author_folder
# --------------------------------------------------------------------------- #


def test_testable_folder_has_all_evidence_and_copies_verbatim(monkeypatch, tmp_path):
    root = _proj(monkeypatch, tmp_path)
    # Two test files, one nested, to prove relative paths are preserved.
    (root / "calc.test.js").write_text("CALC TESTS")
    (root / "sub").mkdir()
    (root / "sub" / "more.test.js").write_text("MORE TESTS")

    ta = TestAuthorResult(
        testable=True, summary="covers behaviour X", snapshot="HASH123",
        red_output="1 failing", full_run="COMPLETE RED LOG (all tests)",
    )
    ra.write_test_author_folder("tid", 1, _task(), ta, ["**/*.test.js"])

    d = tmp_path / "runs" / "tid" / "task-01-t1" / "test-author"
    # Verbatim copies, real names + relative layout preserved.
    assert (d / "calc.test.js").read_text() == "CALC TESTS"
    assert (d / "sub" / "more.test.js").read_text() == "MORE TESTS"
    # results-test-run.md is the COMPLETE run (full_run), not the failures summary.
    rr = (d / "results-test-run.md").read_text(encoding="utf-8")
    assert "COMPLETE RED LOG (all tests)" in rr
    assert "1 failing" not in rr
    assert "HASH123" in (d / "test-snapshot-hash.md").read_text(encoding="utf-8")
    summary = (d / "summary.md").read_text(encoding="utf-8")
    assert "**Testable:** True" in summary and "covers behaviour X" in summary
    assert "calc.test.js" in summary and "sub/more.test.js" in summary


def test_results_run_falls_back_to_red_output_when_no_full_run(monkeypatch, tmp_path):
    _proj(monkeypatch, tmp_path)
    ta = TestAuthorResult(testable=True, snapshot="H", red_output="just the failures")
    ra.write_test_author_folder("tid", 1, _task(), ta, ["**/*.test.js"])
    rr = (tmp_path / "runs" / "tid" / "task-01-t1" / "test-author"
          / "results-test-run.md").read_text(encoding="utf-8")
    assert "just the failures" in rr


def test_untestable_task_gets_summary_only(monkeypatch, tmp_path):
    root = _proj(monkeypatch, tmp_path)
    (root / "calc.test.js").write_text("tests")  # present, but must NOT be copied
    ta = TestAuthorResult(testable=False, degrade_kind="untestable", summary="DOM-only")
    ra.write_test_author_folder("tid", 3, _task("t3"), ta, ["**/*.test.js"])

    d = tmp_path / "runs" / "tid" / "task-03-t3" / "test-author"
    summary = (d / "summary.md").read_text(encoding="utf-8")
    assert "**Testable:** False" in summary
    assert "untestable" in summary and "DOM-only" in summary
    assert not (d / "results-test-run.md").exists()
    assert not (d / "test-snapshot-hash.md").exists()
    assert not (d / "calc.test.js").exists()


def test_summary_records_critic_and_red_review_rounds(monkeypatch, tmp_path):
    _proj(monkeypatch, tmp_path)
    ta = TestAuthorResult(testable=True, snapshot="H", full_run="run")
    rounds_info = {
        "critic_verdicts": [
            {"meaningful": False, "feedback": "assert the toggled value"},
            {"meaningful": True, "feedback": "good"},
        ],
        "critic_rounds": 1,
        "red_review": "approved",
    }
    ra.write_test_author_folder("tid", 1, _task(), ta, ["**/*.test.js"], rounds_info)

    summary = (tmp_path / "runs" / "tid" / "task-01-t1" / "test-author"
               / "summary.md").read_text(encoding="utf-8")
    assert "Coverage critic" in summary
    assert "weak" in summary and "assert the toggled value" in summary
    assert "meaningful" in summary
    assert "Re-author rounds (critic):** 1" in summary
    assert "Red-review:** approved" in summary


# --------------------------------------------------------------------------- #
# integration: the real _run_task_loop wiring writes the folder
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_testable_task_writes_evidence_folder(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    # _Stubs.run_test_author → testable, snapshot="SNAP", red_output="boom".
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    thread_id = c["configurable"]["thread_id"]
    d = ra._run_dir(thread_id) / "task-01-t1" / "test-author"
    assert (d / "summary.md").read_text(encoding="utf-8").find("**Testable:** True") != -1
    assert "boom" in (d / "results-test-run.md").read_text(encoding="utf-8")
    assert "SNAP" in (d / "test-snapshot-hash.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_evidence_folder_records_critic_reauthor(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from tests.test_phase74_coverage_critic import _patch_critic
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    # critic rejects once (re-author), then accepts → recorded in summary.md.
    _patch_critic(monkeypatch, [
        CoverageCriticResult(meaningful=False, feedback="pin the observable value"),
        CoverageCriticResult(meaningful=True, feedback="ok"),
    ])
    oc = _tdd_cfg(coverage_critic=True, on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 2  # authored, re-authored after the critic
    thread_id = c["configurable"]["thread_id"]
    summary = (ra._run_dir(thread_id) / "task-01-t1" / "test-author"
               / "summary.md").read_text(encoding="utf-8")
    assert "Coverage critic" in summary
    assert "pin the observable value" in summary
    assert "Re-author rounds (critic):** 1" in summary
