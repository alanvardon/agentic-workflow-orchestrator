"""CLI transcript feeder — recover the real API-error cause the SDK discarded.

When the Claude Agent SDK's subprocess exits non-zero, `query.py` replaces the
`ProcessError` (which held stderr) with the result message's `subtype` string,
and in a hard failure (e.g. an Anthropic `billing_error`) it *raises* before any
ResultMessage is yielded — so by the time the orchestrator catches the error,
the real cause is gone from the exception object entirely.

The one place it survives is the CLI transcript JSONL the subprocess writes to
`~/.claude/projects/<project_key>/<session_id>.jsonl`. Each API error is recorded
there as a record flagged `isApiErrorMessage`. This module reads the tail of that
transcript and pulls the last such record into a small structured cause:

    {"error": "billing_error", "api_status": 400, "text": "Credit balance is too low"}

so the runner can attach it to the FatalError it raises — making the real message
travel through every downstream sink (audit log, error.md artifact, run_status).

Everything here is best-effort: a missing/unreadable transcript returns None and
NEVER raises, so error capture can never mask the original failure.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from claude_agent_sdk import project_key_for_directory

logger = logging.getLogger(__name__)

# How many trailing characters of the error text we keep — enough to carry the
# real message (e.g. "Your credit balance is too low ...") without bloating the
# payload that lands in the audit log / error.md / run_status.
_MAX_TEXT = 2000

# Substrings that mark a credit/billing failure, used by is_billing_cause for the
# Phase 80b reclassification (fatal → resumable "billing"). Kept specific so a
# generic error mentioning neither phrase stays classified as fatal.
_BILLING_MARKERS = ("credit balance", "billing_error", "billing")

# An embedded Anthropic error type inside the message text, e.g.
# {"type":"error","error":{"type":"billing_error","message":"..."}}.
_ERROR_TYPE_RE = re.compile(r'"type"\s*:\s*"(\w*error)"', re.IGNORECASE)
# A leading "API Error: 400 ..." status code in the message text.
_API_STATUS_RE = re.compile(r"API Error:\s*(\d{3})\b")


def _project_dir(cwd: Path) -> Path:
    """The CLI transcript directory for an agent run with this cwd.

    Uses the SDK's own `project_key_for_directory` so the slug matches exactly how
    the CLI names the directory (realpath + NFC normalization), rather than a naive
    '/'→'-' replacement that diverges on decomposed-Unicode filesystems.
    """
    return Path.home() / ".claude" / "projects" / project_key_for_directory(str(cwd))


def _transcript_path(session_id: str | None, cwd: Path) -> Path | None:
    """Resolve the transcript JSONL to read, or None.

    With a captured `session_id`, read exactly that session's file (the run we
    failed in) — never guess if it's missing. With no session id (the SDK raised
    before any message carried one), fall back to the newest `*.jsonl` by mtime in
    the project dir — how the cause was found by hand.
    """
    d = _project_dir(cwd)
    if not d.is_dir():
        return None
    if session_id:
        p = d / f"{session_id}.jsonl"
        return p if p.is_file() else None
    candidates = sorted(
        d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def _extract_text(rec: dict) -> str:
    """Pull the human-readable error text out of a transcript record.

    Prefers the assistant message's text content blocks (where the CLI puts the
    API error string); falls back to top-level string fields, then to the raw
    record. Always returns a (capped) string."""
    msg = rec.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content:
            return content[:_MAX_TEXT]
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined[:_MAX_TEXT]
    for key in ("text", "error", "content"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            return v[:_MAX_TEXT]
    return json.dumps(rec)[:_MAX_TEXT]


def _cause_from_record(rec: dict) -> dict:
    """Shape one `isApiErrorMessage` record into {error, api_status, text}."""
    text = _extract_text(rec)
    status = rec.get("apiErrorStatus")
    if not isinstance(status, int):
        m = _API_STATUS_RE.search(text)
        status = int(m.group(1)) if m else None
    error = None
    m = _ERROR_TYPE_RE.search(text)
    if m:
        error = m.group(1)
    elif "credit balance" in text.lower():
        error = "billing_error"
    return {"error": error, "api_status": status, "text": text}


def read_api_error_cause(session_id: str | None, cwd: Path) -> dict | None:
    """Read the failed run's transcript tail and return the last API-error cause.

    Returns {error, api_status, text} for the last `isApiErrorMessage` record, or
    None if there is no transcript / no such record. Best-effort: any failure is
    logged at WARNING and swallowed so this can never mask the original error.
    """
    try:
        path = _transcript_path(session_id, cwd)
        if path is None:
            return None
        last: dict | None = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("isApiErrorMessage"):
                    last = rec
        if last is None:
            return None
        return _cause_from_record(last)
    except Exception:  # never let error capture mask the original failure
        logger.warning(
            "transcript feeder failed reading cause for session %s", session_id,
            exc_info=True,
        )
        return None


def is_billing_cause(cause: dict | None, message: str = "") -> bool:
    """True if the structured cause (or the raw message) is a credit/billing error.

    Drives the Phase 80b reclassification of a fatal agent failure into a
    resumable "billing" status. Checks both the transcript cause and the exception
    message so it still fires when only one carries the marker.
    """
    hay = " ".join(
        s
        for s in (
            message,
            (cause or {}).get("text"),
            (cause or {}).get("error"),
        )
        if s
    ).lower()
    return any(marker in hay for marker in _BILLING_MARKERS)
