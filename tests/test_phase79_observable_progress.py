"""Phase 79 — observable progress in chat (background + poll).

The orchestrator already STREAMS progress (run_with_progress heartbeats),
but Claude Code chat doesn't render MCP progress notifications, so the long
post-approval leg looks dead for 5+ minutes. Phase 79 turns progress into
ordinary, persistent chat text by NOT blocking: the long-leg tools can run
the workflow as a tracked background task and return immediately, and a new
`run_status` tool reports where the run stands so the chat can poll + print.

These tests drive the new server-side machinery with fakes (like Phase 19),
so nothing runs a real 5-minute workflow:

  - `background=True` on the long tools returns {status:"started"} at once and
    registers a tracked task (held so the event loop doesn't GC it).
  - `run_status` reports `running` (stage/elapsed read from the audit-log tail,
    NOT aget_state — the DB write-lock is held by the live run), passes through
    a terminal result, surfaces an `awaiting_approval` pause, caches the final
    result for repeat polls, and surfaces an unexpected exception instead of
    vanishing.
  - With no live task, run_status falls back to the checkpoint snapshot.
  - The synchronous (background=False) path is unchanged.
  - cancel_run still works while a background task is live.
"""

import asyncio
import json
from pathlib import Path

import pytest

from orchestrator import mcp_server


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with an empty background-run registry."""
    mcp_server._BG_RUNS.clear()
    yield
    mcp_server._BG_RUNS.clear()


def _no_side_effect_run_log(monkeypatch):
    """append_run resolves its path at import against the REAL project root;
    stub it out so tool-level tests don't append to the repo's runs.jsonl."""
    monkeypatch.setattr(mcp_server, "append_run", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Backgrounding: the long tools return immediately and track the task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_returns_started_immediately(monkeypatch, tmp_path):
    """implement_feature(background=True) returns {status:"started", thread_id}
    without waiting for the workflow, and registers the running task."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()
    _no_side_effect_run_log(monkeypatch)

    gate = asyncio.Event()

    async def fake_run_workflow(thread_id, input_data, **kwargs):
        await gate.wait()
        return {"status": "succeeded", "thread_id": thread_id, "pr_url": "http://pr/1"}

    monkeypatch.setattr(mcp_server, "_run_workflow", fake_run_workflow)

    res = await mcp_server.implement_feature("do a thing", background=True)

    assert res["status"] == "started"
    tid = res["thread_id"]
    assert tid in mcp_server._BG_RUNS
    assert not mcp_server._BG_RUNS[tid].task.done()

    # While the task is pending, run_status reports running.
    status = await mcp_server.run_status(tid)
    assert status["status"] == "running"
    assert status["thread_id"] == tid

    # Let it finish; the final result becomes retrievable via run_status.
    gate.set()
    await mcp_server._BG_RUNS[tid].task
    done = await mcp_server.run_status(tid)
    assert done["status"] == "succeeded"
    assert done["pr_url"] == "http://pr/1"


@pytest.mark.asyncio
async def test_all_three_long_tools_accept_background(monkeypatch, tmp_path):
    """approve_plan and resume_run also background and return 'started'."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()
    _no_side_effect_run_log(monkeypatch)

    async def fake_run_workflow(thread_id, input_data, **kwargs):
        return {"status": "succeeded", "thread_id": thread_id}

    monkeypatch.setattr(mcp_server, "_run_workflow", fake_run_workflow)

    r1 = await mcp_server.approve_plan("run-aaa", "yes", background=True)
    assert r1["status"] == "started"
    assert r1["thread_id"] == "run-aaa"
    await mcp_server._BG_RUNS["run-aaa"].task

    r2 = await mcp_server.resume_run("run-bbb", background=True)
    assert r2["status"] == "started"
    await mcp_server._BG_RUNS["run-bbb"].task


@pytest.mark.asyncio
async def test_synchronous_path_unchanged(monkeypatch, tmp_path):
    """background defaults to False: the tool awaits _run_workflow and returns
    its result directly, registering nothing."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()
    _no_side_effect_run_log(monkeypatch)

    async def fake_run_workflow(thread_id, input_data, **kwargs):
        return {"status": "succeeded", "thread_id": thread_id, "pr_url": "http://pr/2"}

    monkeypatch.setattr(mcp_server, "_run_workflow", fake_run_workflow)

    res = await mcp_server.implement_feature("do a thing")  # background defaults False
    assert res["status"] == "succeeded"
    assert res["pr_url"] == "http://pr/2"
    assert mcp_server._BG_RUNS == {}


# ---------------------------------------------------------------------------
# run_status: live task
# ---------------------------------------------------------------------------


def _write_audit(thread_id_events: list[tuple[str, str, str]]) -> None:
    """Write .orchestrator/audit.log JSONL from (thread_id, event_type,
    task_name) tuples, in order."""
    log = Path(".orchestrator/audit.log")
    log.parent.mkdir(exist_ok=True)
    lines = [
        json.dumps({
            "thread_id": tid,
            "event_type": ev,
            "task_name": task,
            "timestamp": f"2026-06-09T00:00:0{i}+00:00",
        })
        for i, (tid, ev, task) in enumerate(thread_id_events)
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_run_status_running_reads_audit_tail(monkeypatch, tmp_path):
    """While the background task is live, run_status reports the current stage
    from the audit-log tail — it must NOT touch aget_state (the live run holds
    the checkpoint DB write-lock)."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()

    # Events for OUR thread plus an unrelated thread that must be ignored.
    _write_audit([
        ("run-other", "task_start", "planning"),
        ("run-x", "task_complete", "planning"),
        ("run-x", "task_start", "implementation"),
    ])

    # Guard: a live-task run_status must never open the checkpoint DB.
    async def _boom(thread_id):
        raise AssertionError("run_status touched the checkpoint DB on a live run")

    monkeypatch.setattr(mcp_server, "_fetch_existing_state", _boom)

    gate = asyncio.Event()

    async def _pending():
        await gate.wait()
        return {"status": "succeeded", "thread_id": "run-x"}

    mcp_server._BG_RUNS["run-x"] = mcp_server._BgRun(
        asyncio.create_task(_pending()), "req"
    )

    status = await mcp_server.run_status("run-x")
    assert status["status"] == "running"
    assert status["stage"] == "implementation"
    assert status["last_event"]["event_type"] == "task_start"
    assert status["last_event"]["task_name"] == "implementation"
    assert isinstance(status["elapsed_seconds"], (int, float))

    gate.set()
    await mcp_server._BG_RUNS["run-x"].task


@pytest.mark.asyncio
async def test_run_status_passes_through_awaiting_approval(monkeypatch, tmp_path):
    """A backgrounded run that pauses at an interrupt finishes its task with an
    awaiting_approval-shaped dict; run_status must surface that (so the chat
    knows to call approve_plan) rather than reporting it as 'running' forever."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()

    async def _interrupted():
        return {
            "status": "awaiting_approval",
            "thread_id": "run-x",
            "kind": "red_review",
            "plan": None,
            "red_output": "1 failing test",
        }

    mcp_server._BG_RUNS["run-x"] = mcp_server._BgRun(
        asyncio.create_task(_interrupted()), "req"
    )
    await mcp_server._BG_RUNS["run-x"].task

    status = await mcp_server.run_status("run-x")
    assert status["status"] == "awaiting_approval"
    assert status["kind"] == "red_review"


@pytest.mark.asyncio
async def test_run_status_caches_final_result_for_repeat_polls(monkeypatch, tmp_path):
    """Once finished, repeated run_status polls keep returning the terminal
    result (the 'don't lose the final result' landmine) without falling back
    to the snapshot."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()

    async def _boom(thread_id):
        raise AssertionError("fell back to snapshot for a finished tracked run")

    monkeypatch.setattr(mcp_server, "_fetch_existing_state", _boom)

    async def _done():
        return {"status": "succeeded", "thread_id": "run-x", "pr_url": "http://pr/9"}

    mcp_server._BG_RUNS["run-x"] = mcp_server._BgRun(
        asyncio.create_task(_done()), "req"
    )
    await mcp_server._BG_RUNS["run-x"].task

    for _ in range(3):
        status = await mcp_server.run_status("run-x")
        assert status["status"] == "succeeded"
        assert status["pr_url"] == "http://pr/9"


@pytest.mark.asyncio
async def test_run_status_surfaces_unexpected_exception(monkeypatch, tmp_path):
    """_run_workflow shapes known orchestrator errors into dicts itself, so a
    raised exception from a background task is unexpected — run_status surfaces
    it as a structured failure rather than letting it vanish."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()

    async def _explode():
        raise RuntimeError("kaboom")

    mcp_server._BG_RUNS["run-x"] = mcp_server._BgRun(
        asyncio.create_task(_explode()), "req"
    )
    # Let the task finish (and consume the exception) before polling.
    with pytest.raises(RuntimeError):
        await mcp_server._BG_RUNS["run-x"].task

    status = await mcp_server.run_status("run-x")
    assert status["status"] == "fatal"
    assert "kaboom" in status["error"]
    assert status["thread_id"] == "run-x"


# ---------------------------------------------------------------------------
# run_status: no live task → checkpoint snapshot fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_status_no_live_task_falls_back_to_snapshot(monkeypatch, tmp_path):
    """With nothing in the registry (e.g. server restarted, or an idempotency
    replay), run_status reads the checkpoint snapshot — the DB is free then."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()

    sentinel = {"status": "in_progress", "thread_id": "run-ghost", "replayed": True}

    async def fake_fetch(thread_id):
        assert thread_id == "run-ghost"
        return sentinel

    monkeypatch.setattr(mcp_server, "_fetch_existing_state", fake_fetch)

    status = await mcp_server.run_status("run-ghost")
    assert status is sentinel


# ---------------------------------------------------------------------------
# Audit-log tail helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_audit_event_none_when_no_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()
    assert mcp_server._last_audit_event("run-x") is None


@pytest.mark.asyncio
async def test_last_audit_event_filters_by_thread(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()
    _write_audit([
        ("run-x", "task_start", "planning"),
        ("run-y", "task_start", "implementation"),  # different thread, ignored
        ("run-x", "task_complete", "planning"),
    ])
    ev = mcp_server._last_audit_event("run-x")
    assert ev["event_type"] == "task_complete"
    assert ev["task_name"] == "planning"
    # An unknown thread yields None even when the log has other threads.
    assert mcp_server._last_audit_event("run-none") is None


# ---------------------------------------------------------------------------
# cancel_run interplay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_run_works_with_live_background_task(monkeypatch, tmp_path):
    """cancel_run writes a filesystem marker independent of the registry, so it
    works while a background task is live; run_status keeps functioning."""
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir()

    gate = asyncio.Event()

    async def _pending():
        await gate.wait()
        return {"status": "cancelled", "thread_id": "run-c", "reason": "user_cancel"}

    mcp_server._BG_RUNS["run-c"] = mcp_server._BgRun(
        asyncio.create_task(_pending()), "req"
    )

    res = await mcp_server.cancel_run("run-c")
    assert res["status"] == "cancellation_signalled"

    # run_status still reports running until the task settles.
    mid = await mcp_server.run_status("run-c")
    assert mid["status"] == "running"

    gate.set()
    await mcp_server._BG_RUNS["run-c"].task
    final = await mcp_server.run_status("run-c")
    assert final["status"] == "cancelled"
    assert final["reason"] == "user_cancel"
