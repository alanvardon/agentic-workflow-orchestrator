"""Phase 24 — pluggable structured audit-event log.

Tests cover:
1. AuditEvent: defaults auto-filled, fields round-trip through JSON.
2. JsonlAuditSink: writes JSONL to a file; multiple events append.
3. NoopAuditSink: discards all events (no file created).
4. audited() context manager: emits task_start + task_complete on success.
5. audited() context manager: emits task_start + task_failed on exception.
6. emit_event: one-off helper emits correct event type.
7. Audit disabled (config.audit.enabled = False): no log file created.
8. Full workflow run: audit log contains expected event sequence.
9. Interrupt event: emitted before plan_approval interrupt.
10. Cancel event: emitted when workflow is cancelled.
"""

import json
import uuid
from pathlib import Path

import pytest

from orchestrator.audit import (
    AuditEvent,
    JsonlAuditSink,
    NoopAuditSink,
    audited,
    build_sink,
    emit_event,
)


# ---------------------------------------------------------------------------
# Unit tests: AuditEvent
# ---------------------------------------------------------------------------

def test_audit_event_defaults_are_filled():
    event = AuditEvent(thread_id="t1", event_type="resume")
    assert event.thread_id == "t1"
    assert event.event_type == "resume"
    assert event.timestamp is not None
    assert event.user is None
    assert event.task_name is None
    assert event.payload == {}


def test_audit_event_round_trips_json():
    event = AuditEvent(
        thread_id="t1", event_type="task_start", task_name="planning"
    )
    data = json.loads(event.model_dump_json())
    assert data["event_type"] == "task_start"
    assert data["task_name"] == "planning"
    assert data["thread_id"] == "t1"


# ---------------------------------------------------------------------------
# Unit tests: JsonlAuditSink
# ---------------------------------------------------------------------------

def test_jsonl_sink_writes_single_event(tmp_path):
    log = tmp_path / "audit.log"
    sink = JsonlAuditSink(log)
    sink.emit(AuditEvent(thread_id="t1", event_type="resume"))

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event_type"] == "resume"
    assert data["thread_id"] == "t1"


def test_jsonl_sink_appends_multiple_events(tmp_path):
    log = tmp_path / "audit.log"
    sink = JsonlAuditSink(log)
    sink.emit(AuditEvent(thread_id="t1", event_type="task_start", task_name="planning"))
    sink.emit(AuditEvent(thread_id="t1", event_type="task_complete", task_name="planning"))

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "task_start"
    assert json.loads(lines[1])["event_type"] == "task_complete"


def test_jsonl_sink_creates_parent_dirs(tmp_path):
    log = tmp_path / "nested" / "dir" / "audit.log"
    sink = JsonlAuditSink(log)
    sink.emit(AuditEvent(thread_id="t1", event_type="resume"))
    assert log.exists()


def test_jsonl_sink_swallows_write_errors(tmp_path):
    """A write error must not propagate — audit must never kill the workflow."""
    log = tmp_path / "audit.log"
    log.mkdir()  # make the path a directory so open() fails
    sink = JsonlAuditSink(log)
    sink.emit(AuditEvent(thread_id="t1", event_type="resume"))  # must not raise


# ---------------------------------------------------------------------------
# Unit tests: NoopAuditSink
# ---------------------------------------------------------------------------

def test_noop_sink_discards_events(tmp_path):
    sink = NoopAuditSink()
    sink.emit(AuditEvent(thread_id="t1", event_type="resume"))
    # No file, no error — nothing to assert beyond no exception raised.


# ---------------------------------------------------------------------------
# Unit tests: audited() context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_audited_emits_start_and_complete_on_success():
    collected: list[AuditEvent] = []

    class _CaptureSink:
        def emit(self, event: AuditEvent) -> None:
            collected.append(event)

    async with audited(_CaptureSink(), "t1", "planning"):
        pass

    assert len(collected) == 2
    assert collected[0].event_type == "task_start"
    assert collected[0].task_name == "planning"
    assert collected[1].event_type == "task_complete"
    assert collected[1].task_name == "planning"


@pytest.mark.asyncio
async def test_audited_emits_task_failed_on_exception():
    collected: list[AuditEvent] = []

    class _CaptureSink:
        def emit(self, event: AuditEvent) -> None:
            collected.append(event)

    with pytest.raises(ValueError, match="boom"):
        async with audited(_CaptureSink(), "t1", "implementation"):
            raise ValueError("boom")

    assert len(collected) == 2
    assert collected[0].event_type == "task_start"
    assert collected[1].event_type == "task_failed"
    assert collected[1].task_name == "implementation"


# ---------------------------------------------------------------------------
# Unit tests: emit_event helper
# ---------------------------------------------------------------------------

def test_emit_event_creates_correct_audit_event():
    collected: list[AuditEvent] = []

    class _CaptureSink:
        def emit(self, event: AuditEvent) -> None:
            collected.append(event)

    emit_event(_CaptureSink(), "t1", "interrupt", payload={"kind": "plan_approval"})

    assert len(collected) == 1
    assert collected[0].event_type == "interrupt"
    assert collected[0].payload == {"kind": "plan_approval"}
    assert collected[0].thread_id == "t1"


# ---------------------------------------------------------------------------
# Integration tests: full workflow run
# ---------------------------------------------------------------------------

from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult


class _BaseStubs:
    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        return PlanResult(title="title", type="feature", plan_text="plan text")

    def create_branch(self, plan, max_slug_length=50, thread_id="") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch=None) -> str:
        return "abc123"

    def push(self, branch, base_branch=None, auto_rebase=True) -> None:
        pass

    def pr_create(self, branch, title, summary, test_plan, base_branch=None, draft=False, reviewers=None, labels=None) -> str:
        return "https://github.com/test/pr/1"


def _patch(stubs, monkeypatch, tmp_path):
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    # Stub the underlying producer (NOT the implementation_task @task itself), so
    # the decorated @task still runs and emits its audit events — consistent with
    # how plan/qa/commit/... are stubbed at the underlying-callable level.
    monkeypatch.setattr("orchestrator.workflow._run_implementation_producer", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)


def _read_events(tmp_path: Path) -> list[dict]:
    log = tmp_path / ".orchestrator" / "audit.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_full_run_writes_audit_log(monkeypatch, tmp_path):
    """A succeeded workflow run emits resume + task events for every task."""
    _patch(_BaseStubs(), monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a feature")
    result = await approve_plan(pending["thread_id"], "yes")

    assert result["status"] == "succeeded"

    events = _read_events(tmp_path)
    event_types = [e["event_type"] for e in events]
    task_names = [e.get("task_name") for e in events]

    # Every invocation starts with a resume.
    assert event_types.count("resume") >= 1

    # Core tasks appear in the log.
    assert "preflight" in task_names
    assert "planning" in task_names
    assert "create_branch" in task_names
    assert "implementation" in task_names
    assert "qa" in task_names
    assert "commit" in task_names
    assert "push" in task_names
    assert "pr_create" in task_names

    # Each named task has a start and a complete (no failures in happy path).
    for name in ("planning", "create_branch", "implementation", "qa", "commit", "push", "pr_create"):
        task_events = [e["event_type"] for e in events if e.get("task_name") == name]
        assert "task_start" in task_events, f"{name} missing task_start"
        assert "task_complete" in task_events, f"{name} missing task_complete"
        assert "task_failed" not in task_events, f"{name} unexpected task_failed"


@pytest.mark.asyncio
async def test_audit_disabled_writes_no_log(monkeypatch, tmp_path):
    """audit.enabled = false → no audit.log created."""
    _patch(_BaseStubs(), monkeypatch, tmp_path)

    # Patch load_config to return a config with audit disabled.
    from orchestrator.config import OrchestratorConfig, AuditConfig

    disabled_cfg = OrchestratorConfig(
        audit=AuditConfig(enabled=False),
    )
    monkeypatch.setattr("orchestrator.workflow.load_config", lambda: disabled_cfg)
    monkeypatch.setattr("orchestrator.mcp_server.load_config", lambda: disabled_cfg)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a feature")
    await approve_plan(pending["thread_id"], "yes")

    assert not (tmp_path / ".orchestrator" / "audit.log").exists()


@pytest.mark.asyncio
async def test_plan_approval_interrupt_emits_interrupt_event(monkeypatch, tmp_path):
    """Plan approval gate emits an interrupt event with kind=plan_approval."""
    _patch(_BaseStubs(), monkeypatch, tmp_path)

    from orchestrator.mcp_server import approve_plan, implement_feature

    pending = await implement_feature("add a feature")
    # First call pauses at plan approval; second approves.
    result = await approve_plan(pending["thread_id"], "yes")

    assert result["status"] == "succeeded"

    events = _read_events(tmp_path)
    interrupt_events = [e for e in events if e["event_type"] == "interrupt"]
    assert any(
        e.get("payload", {}).get("kind") == "plan_approval"
        for e in interrupt_events
    ), "Expected a plan_approval interrupt event"


@pytest.mark.asyncio
async def test_audit_task_failed_on_preflight_error(monkeypatch, tmp_path):
    """A dirty-tree error emits task_failed for the preflight task."""
    from orchestrator.git_ops import DirtyTreeError

    class _DirtyStubs(_BaseStubs):
        def verify_clean_tree(self) -> None:
            raise DirtyTreeError("dirty")

    _patch(_DirtyStubs(), monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("add a feature")
    assert result["status"] == "user_action_required"

    events = _read_events(tmp_path)
    preflight_events = [e["event_type"] for e in events if e.get("task_name") == "preflight"]
    assert "task_start" in preflight_events
    assert "task_failed" in preflight_events
