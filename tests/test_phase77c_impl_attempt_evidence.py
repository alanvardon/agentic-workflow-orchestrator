"""Phase 77c — impl attempt evidence (third slice of the Phase 77 layer).

The FIRST consumer of BOTH 77a seams (`StepResult.full_output` + the
`run_retry_block` `on_attempt` hook). For each implement attempt of a TESTABLE TDD
task the station writes ``task-NN-<id>/impl/attempt-N/``:

  - ``test-results.md`` — the COMPLETE test run after this attempt (every gate that
    produced a captured ``full_output``), proof the WHOLE suite ran every attempt,
    not just the failing delta; including the GREEN one;
  - ``snapshot-hash.md`` — the ``test_paths`` re-hash this attempt vs the frozen 77b
    baseline → explicit MATCH ✓ / MISMATCH ✗ (the freeze proof; a MISMATCH is the
    evidence of why the diff-gate failed the attempt).

The synthetic diff-gate and the LLM ``builtin:qa`` gate carry no ``full_output``, so
they fall out of ``test-results.md`` and what remains is the actual suite run. The
impl/ folder is cleared on ``attempt <= 1`` so a fresh build — notably each
autonomous re-author round (Phase 76), which re-freezes a NEW suite and restarts the
count — never inherits stale higher-numbered folders from a replaced suite.

Layered like the other TDD tests: fast LLM-free unit tests of the artifact writer +
the recorder builder, plus a full-workflow test exercising the real ``_run_task_loop``
wiring (patching the inner ``_run_test_author`` so the ``@task`` still replays).
"""

import uuid

import pytest
from langgraph.types import Command

import orchestrator.run_artifacts as ra
from orchestrator import workflow as wf
from orchestrator.agents.decompose import Task
from orchestrator.agents.test_author import TestAuthorResult
from orchestrator.manifest import StepResult


def _task(task_id="t1", title="Task one"):
    return Task(id=task_id, title=title, description="do it", acceptance_criteria="works")


def _gate(step_id, *, full_output="", passed=True, detail=""):
    return StepResult(
        step_id=step_id, kind="script", ok=True, passed=passed,
        detail=detail, full_output=full_output,
    )


def _runs(monkeypatch, tmp_path):
    """Point the artifact writer at an isolated runs dir."""
    monkeypatch.setattr(ra, "_runs_dir", lambda: tmp_path / "runs")


# --------------------------------------------------------------------------- #
# unit: write_impl_attempt — the per-attempt evidence files
# --------------------------------------------------------------------------- #


def test_green_attempt_records_full_run_and_match(monkeypatch, tmp_path):
    _runs(monkeypatch, tmp_path)
    gate_results = [
        _gate("diff-gate", passed=True),                                  # freeze check
        _gate("tests", full_output="test_a PASS\ntest_b PASS", passed=True),  # the suite
        _gate("qa", passed=True, detail="QA PASS"),                       # LLM verdict
    ]
    ra.write_impl_attempt(
        "tid", 1, _task(), 1, True, gate_results,
        baseline="SNAP", current_hash="SNAP",
    )

    d = tmp_path / "runs" / "tid" / "task-01-t1" / "impl" / "attempt-1"
    tr = (d / "test-results.md").read_text(encoding="utf-8")
    # The COMPLETE suite run is present, labelled by the gate id…
    assert "test_a PASS" in tr and "test_b PASS" in tr
    assert "### tests" in tr
    assert "GREEN" in tr
    # …and the no-full_output gates (synthetic freeze + LLM qa) fall out.
    assert "### diff-gate" not in tr
    assert "QA PASS" not in tr and "### qa" not in tr

    sh = (d / "snapshot-hash.md").read_text(encoding="utf-8")
    assert "**Result:** MATCH ✓" in sh  # the verdict line (prose mentions both words)
    assert "SNAP" in sh


def test_failing_attempt_records_red_run(monkeypatch, tmp_path):
    _runs(monkeypatch, tmp_path)
    gate_results = [
        _gate("diff-gate", passed=True),
        _gate("tests", full_output="test_a PASS\ntest_b FAIL: boom", passed=False),
    ]
    ra.write_impl_attempt(
        "tid", 1, _task(), 2, False, gate_results,
        baseline="SNAP", current_hash="SNAP",
    )

    d = tmp_path / "runs" / "tid" / "task-01-t1" / "impl" / "attempt-2"
    tr = (d / "test-results.md").read_text(encoding="utf-8")
    assert "RED" in tr
    # The whole run is captured (both the passing and failing test), not just the delta.
    assert "test_a PASS" in tr and "test_b FAIL: boom" in tr
    # The freeze still held this attempt (impl changed, tests didn't).
    assert "**Result:** MATCH ✓" in (d / "snapshot-hash.md").read_text(encoding="utf-8")


def test_freeze_mismatch_is_recorded_with_no_run(monkeypatch, tmp_path):
    _runs(monkeypatch, tmp_path)
    # The diff-gate is ordered FIRST; on a tamper it fails before the suite runs, so
    # gate_results carries only the (full_output-less) freeze check.
    gate_results = [_gate("diff-gate", passed=False, detail="tests were modified")]
    ra.write_impl_attempt(
        "tid", 1, _task(), 1, False, gate_results,
        baseline="SNAP", current_hash="TAMPERED",
    )

    d = tmp_path / "runs" / "tid" / "task-01-t1" / "impl" / "attempt-1"
    sh = (d / "snapshot-hash.md").read_text(encoding="utf-8")
    assert "**Result:** MISMATCH ✗" in sh
    assert "SNAP" in sh and "TAMPERED" in sh
    # No suite ran → test-results.md says so rather than omitting the file.
    tr = (d / "test-results.md").read_text(encoding="utf-8")
    assert "No test run was captured" in tr


def test_multiple_attempts_accumulate(monkeypatch, tmp_path):
    _runs(monkeypatch, tmp_path)
    for n, passed in [(1, False), (2, False), (3, True)]:
        ra.write_impl_attempt(
            "tid", 1, _task(), n, passed,
            [_gate("tests", full_output=f"run {n}", passed=passed)],
            baseline="SNAP", current_hash="SNAP",
        )
    impl = tmp_path / "runs" / "tid" / "task-01-t1" / "impl"
    assert sorted(p.name for p in impl.iterdir()) == ["attempt-1", "attempt-2", "attempt-3"]
    assert "run 3" in (impl / "attempt-3" / "test-results.md").read_text(encoding="utf-8")


def test_attempt_one_clears_a_prior_build(monkeypatch, tmp_path):
    # A prior (autonomous re-author) round left attempt-1..3 against a now-replaced
    # suite. The next round restarts at attempt 1 → the impl/ folder is cleared, so
    # only the surviving round's attempts remain (no stale attempt-2/3).
    _runs(monkeypatch, tmp_path)
    for n in (1, 2, 3):
        ra.write_impl_attempt(
            "tid", 1, _task(), n, False,
            [_gate("tests", full_output=f"stale {n}", passed=False)],
            baseline="OLD", current_hash="OLD",
        )
    # New round, new frozen baseline, passes on its first attempt.
    ra.write_impl_attempt(
        "tid", 1, _task(), 1, True,
        [_gate("tests", full_output="fresh run", passed=True)],
        baseline="NEW", current_hash="NEW",
    )

    impl = tmp_path / "runs" / "tid" / "task-01-t1" / "impl"
    assert sorted(p.name for p in impl.iterdir()) == ["attempt-1"]  # 2 & 3 gone
    a1 = (impl / "attempt-1" / "test-results.md").read_text(encoding="utf-8")
    assert "fresh run" in a1 and "stale" not in a1


def test_budget_extension_does_not_clear(monkeypatch, tmp_path):
    # A growable budget keeps the SAME build counting (attempt never resets to 1), so
    # later attempts must NOT wipe the earlier ones of the same build.
    _runs(monkeypatch, tmp_path)
    for n in (1, 2, 3):  # one continuous build that got extended
        ra.write_impl_attempt(
            "tid", 1, _task(), n, n == 3,
            [_gate("tests", full_output=f"run {n}", passed=n == 3)],
            baseline="SNAP", current_hash="SNAP",
        )
    impl = tmp_path / "runs" / "tid" / "task-01-t1" / "impl"
    assert sorted(p.name for p in impl.iterdir()) == ["attempt-1", "attempt-2", "attempt-3"]


# --------------------------------------------------------------------------- #
# unit: _impl_attempt_recorder — the on_attempt callback builder
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recorder_rehashes_and_writes(monkeypatch):
    calls = {}

    def _capture(thread_id, task_index, task, attempt, passed, gate_results, *, baseline, current_hash):
        calls.update(
            thread_id=thread_id, task_index=task_index, attempt=attempt,
            passed=passed, baseline=baseline, current_hash=current_hash,
        )

    monkeypatch.setattr(wf, "write_impl_attempt", _capture)
    # The recorder re-hashes the live tree with the SAME pure fn the diff-gate uses.
    monkeypatch.setattr(wf, "_hash_test_paths", lambda paths, root: "CURRENT")

    on_attempt = wf._impl_attempt_recorder(
        "tid", 2, _task("t2"), ["**/*.test.js"], "BASE", "/repo",
    )
    await on_attempt(3, True, [_gate("tests", full_output="x")])

    assert calls["thread_id"] == "tid" and calls["task_index"] == 2
    assert calls["attempt"] == 3 and calls["passed"] is True
    assert calls["baseline"] == "BASE"          # the frozen 77b snapshot
    assert calls["current_hash"] == "CURRENT"   # re-hashed this attempt


# --------------------------------------------------------------------------- #
# integration: the real _run_task_loop wiring writes impl/attempt-N/
# --------------------------------------------------------------------------- #


def _cfg() -> dict:
    return {"configurable": {"thread_id": f"test-{uuid.uuid4().hex[:8]}"}}


@pytest.mark.asyncio
async def test_testable_task_writes_impl_attempt_folder(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg
    from orchestrator.workflow import build_workflow

    # testable, snapshot="SNAP"; the patched _hash_test_paths also returns "SNAP", so
    # the diff-gate passes and the per-attempt re-hash records a MATCH.
    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    thread_id = c["configurable"]["thread_id"]
    d = ra._run_dir(thread_id) / "task-01-t1" / "impl" / "attempt-1"
    assert d.is_dir()  # the GREEN attempt produced a folder (77a hook fires on the pass)
    assert "GREEN" in (d / "test-results.md").read_text(encoding="utf-8")
    sh = (d / "snapshot-hash.md").read_text(encoding="utf-8")
    assert "**Result:** MATCH ✓" in sh and "SNAP" in sh


@pytest.mark.asyncio
async def test_untestable_task_writes_no_impl_folder(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg
    from orchestrator.workflow import build_workflow

    # testable=False → the classic implement→qa path runs; no diff-gate, no frozen
    # baseline, so on_attempt is None and impl/ is never written (Phase 77c scope).
    stubs = _Stubs(
        n_tasks=1,
        ta_results=[TestAuthorResult(testable=False, summary="not unit-testable")],
    )
    _patch(stubs, monkeypatch, hash_value="WOULD-MISMATCH")
    oc = _tdd_cfg(on_exhausted="abort")
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"), config=oc) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    thread_id = c["configurable"]["thread_id"]
    assert not (ra._run_dir(thread_id) / "task-01-t1" / "impl").exists()
