"""Phase 80 — capture & surface run failures (error + usage logging).

When run `run-e1f4aa23` died on an Anthropic billing_error, the operator saw only
`"...error result: success" / status: fatal`: the real cause was lost at three
layers and no token/cost was recorded. Phase 80 builds one failure funnel feeding
three sinks (audit payload, error.md artifact, run_status) plus persistent usage.

These tests use fakes — no real billing error, no real subprocess:

  80a  transcript feeder: a fixture jsonl with an `isApiErrorMessage` record →
       a structured cause; session-id resolution + newest-by-mtime fallback. The
       runner funnels a raised ClaudeSDKError into a FatalError carrying that cause.
  80a  Sink A: the audit task_failed payload carries error_type/message/cause.
  80a  Sink B: write_error lays down error.md with cause + traceback + failed task.
  80b  _fatal_error surfaces the cause and reshapes a billing cause → a resumable
       "billing" status pointing at resume_run.
  80c  run_usage_rollup breaks tokens down by category; the run-END rollup lands in
       runs.jsonl and a per-task `usage` audit event fires from _finalize.
"""

import json
import os

import pytest
from claude_agent_sdk import ProcessError, SystemMessage

from orchestrator import run_log, transcript
from orchestrator.agents import runner
from orchestrator.audit import AuditEvent, failure_payload
from orchestrator.errors import FatalError
from orchestrator.run_artifacts import write_error
from orchestrator.usage import TaskUsage, run_usage_rollup


# A canonical CLI-transcript API-error record (the shape the feeder must parse).
def _api_error_line(text="API Error: 400 Your credit balance is too low", status=400):
    return json.dumps({
        "type": "assistant",
        "isApiErrorMessage": True,
        "apiErrorStatus": status,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    })


# ---------------------------------------------------------------------------
# 80a — transcript feeder
# ---------------------------------------------------------------------------


def test_feeder_reads_named_session_cause(tmp_path, monkeypatch):
    """read_api_error_cause(session_id) reads that exact session's transcript and
    pulls the last isApiErrorMessage record into {error, api_status, text}."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "sess-123.jsonl").write_text(
        json.dumps({"type": "assistant", "message": {"content": "hi"}}) + "\n"
        + _api_error_line() + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(transcript, "_project_dir", lambda cwd: proj)

    cause = transcript.read_api_error_cause("sess-123", tmp_path)
    assert cause is not None
    assert cause["api_status"] == 400
    assert cause["error"] == "billing_error"
    assert "credit balance is too low" in cause["text"].lower()


def test_feeder_takes_last_api_error(tmp_path, monkeypatch):
    """When several API-error records exist, the LAST one wins (the fatal one)."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "s.jsonl").write_text(
        _api_error_line("API Error: 429 overloaded", 429) + "\n"
        + _api_error_line("API Error: 400 Your credit balance is too low", 400) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(transcript, "_project_dir", lambda cwd: proj)

    cause = transcript.read_api_error_cause("s", tmp_path)
    assert cause["api_status"] == 400
    assert "credit balance" in cause["text"].lower()


def test_feeder_falls_back_to_newest_when_no_session(tmp_path, monkeypatch):
    """With no session id, the newest *.jsonl by mtime is read (how the cause was
    found by hand on the real run)."""
    proj = tmp_path / "proj"
    proj.mkdir()
    old = proj / "old.jsonl"
    new = proj / "new.jsonl"
    old.write_text(_api_error_line("API Error: 500 old", 500) + "\n", encoding="utf-8")
    new.write_text(_api_error_line("API Error: 400 newest wins", 400) + "\n", encoding="utf-8")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    monkeypatch.setattr(transcript, "_project_dir", lambda cwd: proj)

    cause = transcript.read_api_error_cause(None, tmp_path)
    assert cause["api_status"] == 400
    assert "newest wins" in cause["text"]


def test_feeder_missing_transcript_returns_none(tmp_path, monkeypatch):
    """A missing project dir / session file returns None and never raises."""
    monkeypatch.setattr(transcript, "_project_dir", lambda cwd: tmp_path / "nope")
    assert transcript.read_api_error_cause("sess", tmp_path) is None
    assert transcript.read_api_error_cause(None, tmp_path) is None


def test_is_billing_cause():
    """Credit/billing markers (in cause text, error type, or raw message) classify
    as billing; anything else does not."""
    assert transcript.is_billing_cause({"text": "Your credit balance is too low"})
    assert transcript.is_billing_cause({"error": "billing_error", "text": ""})
    assert transcript.is_billing_cause(None, "billing_error 400")
    assert not transcript.is_billing_cause({"text": "rate limit exceeded"})
    assert not transcript.is_billing_cause(None, "agent did not call emit_qa_result")


# ---------------------------------------------------------------------------
# 80a — runner funnels a raised SDK error into a FatalError carrying the cause
# ---------------------------------------------------------------------------


async def _drive_runner(monkeypatch, *, fake_query, fake_cause):
    """Run run_structured_agent with a patched query/feeder; return the captured
    session id the feeder was called with, plus the raised exception."""
    seen = {}

    def fake_read(session_id, cwd):
        seen["session_id"] = session_id
        return fake_cause

    monkeypatch.setattr(runner, "query", fake_query)
    monkeypatch.setattr(runner, "read_api_error_cause", fake_read)

    with pytest.raises(FatalError) as ei:
        await runner.run_structured_agent(
            system_prompt="s", user_message="u", model="claude-sonnet-4-6",
            allowed_tools=[], disallowed_tools=[], cwd=tmp_cwd(),
            emit_tool_name="emit_qa_result", emit_tool_description="d",
            emit_tool_fields={"result": str},
            result_factory=lambda c, u: c,
        )
    return seen, ei.value


def tmp_cwd():
    from pathlib import Path
    return Path(".")


@pytest.mark.asyncio
async def test_runner_attaches_cause_from_session(monkeypatch):
    """A ClaudeSDKError raised mid-stream is funnelled into a FatalError whose
    message carries the real error text and whose .cause is the structured cause —
    using the session id captured from the FIRST message that carried one."""
    async def fake_query(*, prompt, options):
        yield SystemMessage(subtype="init", data={"session_id": "sess-xyz"})
        raise ProcessError("Claude Code returned an error result: success", exit_code=1)

    cause = {"error": "billing_error", "api_status": 400, "text": "Credit balance is too low"}
    seen, exc = await _drive_runner(monkeypatch, fake_query=fake_query, fake_cause=cause)

    assert seen["session_id"] == "sess-xyz"      # captured from the init SystemMessage
    assert exc.cause == cause
    assert "Credit balance is too low" in str(exc)  # real cause travels in the message


@pytest.mark.asyncio
async def test_runner_handles_raise_before_any_session(monkeypatch):
    """When the SDK raises before any message carries a session id, the feeder is
    invoked with None (newest-transcript fallback) and the FatalError still carries
    whatever cause was found."""
    async def fake_query(*, prompt, options):
        raise ProcessError("boom", exit_code=1)
        yield  # pragma: no cover - unreachable, makes this an async generator

    cause = {"error": "billing_error", "api_status": 400, "text": "Credit balance is too low"}
    seen, exc = await _drive_runner(monkeypatch, fake_query=fake_query, fake_cause=cause)

    assert seen["session_id"] is None
    assert exc.cause == cause


# ---------------------------------------------------------------------------
# 80a — Sink A: the audit task_failed payload
# ---------------------------------------------------------------------------


def test_failure_payload_carries_cause():
    exc = FatalError("agent run failed: Credit balance is too low")
    exc.cause = {"error": "billing_error", "api_status": 400, "text": "Credit balance is too low"}
    payload = failure_payload(exc)
    assert payload["error_type"] == "FatalError"
    assert "Credit balance" in payload["message"]
    assert payload["cause"]["error"] == "billing_error"


def test_failure_payload_without_cause_is_none():
    payload = failure_payload(ValueError("nope"))
    assert payload == {"error_type": "ValueError", "message": "nope", "cause": None}


@pytest.mark.asyncio
async def test_audited_emits_failure_payload():
    """The audited() context manager emits task_failed with the rich payload."""
    from orchestrator.audit import audited

    events: list[AuditEvent] = []

    class _Sink:
        def emit(self, e):
            events.append(e)

    exc = FatalError("billing_error: Credit balance is too low")
    exc.cause = {"error": "billing_error", "api_status": 400, "text": "low"}
    with pytest.raises(FatalError):
        async with audited(_Sink(), "t1", "qa"):
            raise exc

    failed = [e for e in events if e.event_type == "task_failed"]
    assert len(failed) == 1
    assert failed[0].payload["cause"]["error"] == "billing_error"
    assert "Credit balance" in failed[0].payload["message"]


# ---------------------------------------------------------------------------
# 80a — Sink B: error.md run artifact
# ---------------------------------------------------------------------------


def test_write_error_artifact(tmp_path, monkeypatch):
    """write_error lays down error.md with the failed task, the structured cause,
    and a full traceback."""
    monkeypatch.setattr("orchestrator.run_artifacts._runs_dir", lambda: tmp_path)
    try:
        raise FatalError("agent run failed: Credit balance is too low")
    except FatalError as e:
        e.cause = {"error": "billing_error", "api_status": 400, "text": "Credit balance is too low"}
        exc = e

    write_error("run-abc", exc, failed_task="qa")

    md = (tmp_path / "run-abc" / "error.md").read_text(encoding="utf-8")
    assert "**Failed task:** qa" in md
    assert "billing_error" in md
    assert "Credit balance is too low" in md
    assert "Traceback" in md and "FatalError" in md


def test_write_error_never_raises(monkeypatch):
    """A failure writing error.md is swallowed — it must never mask the original
    error."""
    def boom():
        raise RuntimeError("disk gone")
    monkeypatch.setattr("orchestrator.run_artifacts._run_dir", lambda t: boom())
    # Should not raise.
    write_error("run-x", FatalError("orig"), failed_task="qa")


# ---------------------------------------------------------------------------
# 80b — run_status surfaces the cause + reclassifies billing
# ---------------------------------------------------------------------------


def test_fatal_error_billing_is_resumable():
    """A billing cause reshapes fatal → a resumable 'billing' status pointing at
    resume_run, not 'start a fresh implement_feature'."""
    from orchestrator.mcp_server import _fatal_error

    exc = FatalError("agent run failed: Credit balance is too low")
    exc.cause = {"error": "billing_error", "api_status": 400, "text": "Credit balance is too low"}
    resp = _fatal_error("run-1", exc)

    assert resp["status"] == "billing"
    assert resp["cause"]["error"] == "billing_error"
    assert "resume_run" in resp["next"]
    assert "console.anthropic.com" in resp["next"]
    # Steers AWAY from a fresh run — the checkpointed run only re-runs the failed leg.
    assert "Do NOT start a fresh implement_feature" in resp["next"]


def test_fatal_error_nonbilling_keeps_fatal_and_surfaces_cause():
    """A non-billing FatalError stays fatal but still surfaces its (possibly None)
    cause."""
    from orchestrator.mcp_server import _fatal_error

    resp = _fatal_error("run-2", FatalError("invalid orchestrator.toml"))
    assert resp["status"] == "fatal"
    assert resp["cause"] is None
    assert "fresh implement_feature" in resp["next"]


# ---------------------------------------------------------------------------
# 80c — persistent usage & cost
# ---------------------------------------------------------------------------


def test_run_usage_rollup_breaks_tokens_by_category():
    """The rollup sums all four categories per task + a grand total, with cost from
    the price table (cache_read priced far below input — the ~10× landmine)."""
    by_task = {
        "qa": [TaskUsage(model="claude-sonnet-4-6", input_tokens=100,
                         output_tokens=50, cache_read_tokens=900, cache_creation_tokens=10)],
        "implementation": [TaskUsage(model="claude-sonnet-4-6", input_tokens=200,
                                     output_tokens=80, cache_read_tokens=0, cache_creation_tokens=0)],
        "summarize": [],  # empty lists are skipped
    }
    rollup = run_usage_rollup(by_task)

    assert set(rollup["by_task"]) == {"qa", "implementation"}
    assert rollup["by_task"]["qa"]["cache_read_tokens"] == 900
    assert rollup["by_task"]["qa"]["models"] == ["claude-sonnet-4-6"]
    total = rollup["total"]
    assert total["input_tokens"] == 300
    assert total["output_tokens"] == 130
    assert total["cache_read_tokens"] == 900
    assert total["cost_usd"] is not None and total["cost_usd"] > 0


def test_run_usage_rollup_empty():
    assert run_usage_rollup({"qa": [], "implementation": []}) == {}


def test_append_usage_rollup_writes_run_end_line(tmp_path, monkeypatch):
    """append_usage_rollup appends a run_end record with per-category tokens, the
    computed cost, and the terminal status."""
    log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_log, "_LOG_PATH", log)

    rollup = run_usage_rollup({
        "qa": [TaskUsage(model="claude-sonnet-4-6", input_tokens=10,
                         output_tokens=5, cache_read_tokens=1, cache_creation_tokens=2)]
    })
    run_log.append_usage_rollup("run-7", rollup, status="succeeded")

    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["event"] == "run_end"
    assert rec["thread_id"] == "run-7"
    assert rec["status"] == "succeeded"
    assert rec["tokens"]["cache_read_tokens"] == 1
    assert "cost_usd" not in rec["tokens"]  # cost lifted to top level
    assert rec["cost_usd"] is not None


def test_finalize_emits_usage_event_and_rollup(monkeypatch):
    """_finalize fires a per-task `usage` audit event and appends the runs.jsonl
    rollup, tagged with the terminal status."""
    from orchestrator import workflow

    events: list = []
    appended = {}

    class _Sink:
        def emit(self, e):
            events.append(e)

    monkeypatch.setattr(workflow, "_build_task_audit_sink", lambda: _Sink())
    monkeypatch.setattr(workflow, "append_usage_rollup",
                        lambda thread, rollup, *, status: appended.update(
                            thread=thread, status=status, rollup=rollup))

    usage_by_task = {
        "qa": [TaskUsage(model="claude-sonnet-4-6", input_tokens=10, output_tokens=5)],
    }
    result = workflow._finalize(usage_by_task, "run-9", status="succeeded", branch="b")

    assert result["status"] == "succeeded"
    usage_events = [e for e in events if e.event_type == "usage"]
    assert [e.task_name for e in usage_events] == ["qa"]
    assert appended["thread"] == "run-9"
    assert appended["status"] == "succeeded"
