"""Pluggable structured audit-event log.

Emits JSONL records at task boundaries for compliance/security audit trails,
independent of LangSmith (which is for debugging, not audit).

Default sink: .orchestrator/audit.log (JSONL, one event per line).
Swap the sink for tests or production integrations via build_sink() or by
passing a custom AuditSink anywhere a sink is accepted.

Scrubbing policy: by default only metadata is logged (task names, timing,
status). Set audit.include_content = true in orchestrator.toml to opt in
to content logging (plan text, request strings, etc.) — and ensure you have
a scrubbing policy in place first.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# The event vocabulary. "usage" (Phase 80c) records per-task token/cost at run end.
EventType = Literal[
    "task_start", "task_complete", "task_failed",
    "interrupt", "resume", "cancel", "auto_approved", "usage",
]


class AuditEvent(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    thread_id: str
    user: str | None = None
    event_type: EventType
    task_name: str | None = None
    payload: dict = Field(default_factory=dict)


def failure_payload(exc: BaseException) -> dict:
    """The `task_failed` audit payload (Phase 80a).

    Captures the error type, the message, and any structured `cause` the runner's
    transcript feeder attached (e.g. an Anthropic billing_error). Previously the
    payload was `{}` — zero error text — so the real cause was lost at the audit
    layer. Shared by the `audited()` context manager and the workflow's
    `_audited_task` decorator so every task's failure is recorded identically.
    """
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "cause": getattr(exc, "cause", None),
    }


@runtime_checkable
class AuditSink(Protocol):
    def emit(self, event: AuditEvent) -> None: ...


class JsonlAuditSink:
    """Writes audit events as JSONL to a file.

    Errors are logged at WARNING level and swallowed so a disk problem
    never takes down the workflow — same best-effort contract as run_artifacts.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def emit(self, event: AuditEvent) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception:
            logger.warning("audit emit failed writing to %s", self._path, exc_info=True)


class NoopAuditSink:
    """Discards all events — used when audit is disabled or in tests."""

    def emit(self, event: AuditEvent) -> None:
        pass


def build_sink(log_path: str) -> AuditSink:
    """Return a JsonlAuditSink writing to log_path."""
    return JsonlAuditSink(Path(log_path))


def emit_event(
    sink: AuditSink,
    thread_id: str,
    event_type: EventType,
    *,
    task_name: str | None = None,
    user: str | None = None,
    payload: dict | None = None,
) -> None:
    """Emit a single audit event through the sink."""
    sink.emit(AuditEvent(
        thread_id=thread_id,
        user=user,
        event_type=event_type,
        task_name=task_name,
        payload=payload or {},
    ))


@asynccontextmanager
async def audited(
    sink: AuditSink,
    thread_id: str,
    task_name: str,
    *,
    user: str | None = None,
) -> AsyncIterator[None]:
    """Emit task_start before and task_complete/task_failed after an awaited call.

    Usage:
        async with audited(sink, thread_id, "planning"):
            result = await planning_task(request, model)
    """
    emit_event(sink, thread_id, "task_start", task_name=task_name, user=user)
    try:
        yield
        emit_event(sink, thread_id, "task_complete", task_name=task_name, user=user)
    except Exception as exc:
        emit_event(
            sink, thread_id, "task_failed", task_name=task_name, user=user,
            payload=failure_payload(exc),
        )
        raise
