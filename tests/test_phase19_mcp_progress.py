"""Phase 19 — workflow → MCP progress streaming.

Tests `orchestrator.mcp_progress.run_with_progress` in isolation by
driving a fake workflow object that yields canned astream events.
This decouples the progress logic from LangGraph + the real
checkpointer, so we can assert exactly which messages reach the MCP
sink without running a 5-minute implementation_task.

Coverage:
  - Per-task events translate to ctx.report_progress calls with the
    expected message shape.
  - Workflow-final and __interrupt__ events do NOT trigger progress
    reports (those carry the result, not status).
  - The qa event chooses the "commit" vs "implementation (retry)"
    label based on the QaResult it carries.
  - ctx=None is a valid no-op path (used in tests and when the MCP
    client doesn't support notifications).
  - Sink exceptions do NOT propagate — report_progress is advisory.
"""

import asyncio
from typing import Any

import pytest

from orchestrator.agents.qa import QaResult
from orchestrator.errors import FatalError
from orchestrator.mcp_progress import run_with_progress


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeContext:
    """Records every report_progress call for assertion."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def report_progress(
        self,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        self.calls.append({
            "progress": progress,
            "total": total,
            "message": message,
        })


class _RaisingContext(_FakeContext):
    """report_progress always raises — used to assert the loop is
    resilient to a broken/closed MCP transport."""

    async def report_progress(self, progress, total=None, message=None):
        raise RuntimeError("transport closed")


class _FakeWorkflow:
    """Minimal stand-in for a LangGraph entrypoint. Yields a fixed
    list of astream events when astream() is called."""

    def __init__(self, events: list[dict]) -> None:
        self.events = events

    def astream(
        self,
        input_data: Any,
        config: dict,
        stream_mode: str,
    ):
        async def _gen():
            for event in self.events:
                yield event
        return _gen()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emits_progress_per_task(monkeypatch):
    """One report_progress call per completed task, none for the
    final workflow event."""
    # Disable the heartbeat — it's not what this test is checking.
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "9999")

    events = [
        {"verify_clean_tree_task": None},
        {"planning_task": "fake-plan"},
        {"create_branch_task": "feature/x"},
        {"workflow": {"status": "succeeded", "branch": "feature/x"}},
    ]
    ctx = _FakeContext()
    workflow = _FakeWorkflow(events)
    config = {"configurable": {"thread_id": "run-xyz"}}

    result = await run_with_progress(workflow, "request", config, ctx)

    assert result == {"status": "succeeded", "branch": "feature/x"}
    # 3 task events → 3 progress calls. The workflow event does not
    # generate one (it carries the result, not a status update).
    assert len(ctx.calls) == 3
    assert "done: verify_clean_tree" in ctx.calls[0]["message"]
    assert "done: planning" in ctx.calls[1]["message"]
    assert "done: create_branch" in ctx.calls[2]["message"]
    # progress counter increments monotonically.
    assert [c["progress"] for c in ctx.calls] == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_decompose_event_appends_task_titles(monkeypatch):
    """Phase 82: after decompose, the progress message lists the task titles
    (comma-separated) so the waiting client sees how the plan was split."""
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "9999")
    from orchestrator.agents.decompose import DecompositionResult, Task

    decomp = DecompositionResult(
        tasks=[
            Task(id="add-toggle", title="Add toggle markup", description="d",
                 acceptance_criteria="a"),
            Task(id="update-styles", title="Update styles", description="d",
                 acceptance_criteria="a"),
            Task(id="update-tests", title="Update tests", description="d",
                 acceptance_criteria="a"),
        ],
        usage=None,
    )
    events = [
        {"decompose_task": decomp},
        {"workflow": {"status": "succeeded"}},
    ]
    ctx = _FakeContext()

    await run_with_progress(
        _FakeWorkflow(events), "req", {"configurable": {"thread_id": "x"}}, ctx
    )

    msg = ctx.calls[0]["message"]
    assert msg.startswith("done: decompose")
    assert "→ 3 tasks: Add toggle markup, Update styles, Update tests" in msg


@pytest.mark.asyncio
async def test_interrupt_event_is_returned_not_reported(monkeypatch):
    """When the workflow pauses for plan approval, the __interrupt__
    event becomes the return value — it must not emit a progress
    notification."""
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "9999")

    interrupt_payload = [
        type("FakeInterrupt", (), {"value": {"plan": {"plan_text": "p"}}})()
    ]
    events = [
        {"verify_clean_tree_task": None},
        {"planning_task": "fake-plan"},
        {"__interrupt__": interrupt_payload},
    ]
    ctx = _FakeContext()
    workflow = _FakeWorkflow(events)

    result = await run_with_progress(workflow, "req", {"configurable": {"thread_id": "x"}}, ctx)

    assert "__interrupt__" in result
    # Only the two task events emit progress, not the interrupt.
    assert len(ctx.calls) == 2


@pytest.mark.asyncio
async def test_qa_pass_predicts_commit_next(monkeypatch):
    """After a QA PASS, the heartbeat label should describe the
    commit task as next — not 'next stage' or 'qa'."""
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "0.05")  # fire heartbeat

    qa_pass = QaResult(result="PASS")
    events = [
        {"qa_task": qa_pass},
        # Long sleep simulated by yielding control — gives the
        # heartbeat task a chance to fire at least once with the
        # post-QA label.
        {"__sleep__": 0.15},
        {"workflow": {"status": "succeeded"}},
    ]

    # Custom workflow that interprets the sleep marker.
    class _SleepyWorkflow:
        def astream(self, input_data, config, stream_mode):
            async def _gen():
                for event in events:
                    if "__sleep__" in event:
                        await asyncio.sleep(event["__sleep__"])
                        continue
                    yield event
            return _gen()

    ctx = _FakeContext()
    await run_with_progress(
        _SleepyWorkflow(), "req", {"configurable": {"thread_id": "x"}}, ctx
    )

    # First call is "done: qa". Subsequent heartbeat call(s) should
    # carry the predicted-next label "running commit".
    messages = [c["message"] for c in ctx.calls]
    assert messages[0].startswith("done: qa")
    assert any("running commit" in m for m in messages[1:]), messages


@pytest.mark.asyncio
async def test_qa_fail_predicts_implementation_retry(monkeypatch):
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "0.05")

    qa_fail = QaResult(result="FAIL", failures="something broke")
    events = [
        {"qa_task": qa_fail},
        {"__sleep__": 0.15},
        {"workflow": {"status": "failed"}},
    ]

    class _SleepyWorkflow:
        def astream(self, input_data, config, stream_mode):
            async def _gen():
                for event in events:
                    if "__sleep__" in event:
                        await asyncio.sleep(event["__sleep__"])
                        continue
                    yield event
            return _gen()

    ctx = _FakeContext()
    await run_with_progress(
        _SleepyWorkflow(), "req", {"configurable": {"thread_id": "x"}}, ctx
    )

    messages = [c["message"] for c in ctx.calls]
    assert messages[0].startswith("done: qa")
    assert any("implementation (retry)" in m for m in messages[1:]), messages


# ---------------------------------------------------------------------------
# No-context / advisory-failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctx_none_is_silent_noop(monkeypatch):
    """When ctx is None (CLI, tests, MCP client that doesn't support
    notifications), the loop must complete normally without raising."""
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "9999")

    events = [
        {"verify_clean_tree_task": None},
        {"planning_task": "fake-plan"},
        {"workflow": {"status": "succeeded"}},
    ]
    workflow = _FakeWorkflow(events)

    result = await run_with_progress(
        workflow, "req", {"configurable": {"thread_id": "x"}}, ctx=None
    )
    assert result == {"status": "succeeded"}


@pytest.mark.asyncio
async def test_report_progress_failure_does_not_propagate(monkeypatch):
    """A failing ctx.report_progress must not take down the workflow.
    Progress is advisory per the MCP spec."""
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "9999")

    events = [
        {"verify_clean_tree_task": None},
        {"workflow": {"status": "succeeded"}},
    ]
    workflow = _FakeWorkflow(events)
    ctx = _RaisingContext()

    # This MUST NOT raise even though report_progress always raises.
    result = await run_with_progress(
        workflow, "req", {"configurable": {"thread_id": "x"}}, ctx
    )
    assert result == {"status": "succeeded"}


@pytest.mark.asyncio
async def test_stream_without_final_result_raises(monkeypatch):
    """If astream completes without ever emitting a workflow or
    __interrupt__ event, that's an upstream bug — surface it as a
    RuntimeError rather than returning a misleading None."""
    monkeypatch.setenv("HEARTBEAT_INTERVAL", "9999")

    events = [
        {"verify_clean_tree_task": None},
        # No workflow / __interrupt__ event.
    ]
    workflow = _FakeWorkflow(events)
    ctx = _FakeContext()

    with pytest.raises(FatalError, match="without a final result"):
        await run_with_progress(
            workflow, "req", {"configurable": {"thread_id": "x"}}, ctx
        )
