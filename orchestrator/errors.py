"""Structured error hierarchy for the orchestrator.

Three classes cover every failure the orchestrator can produce:

  RetriableError    — transient (network blip, 5xx, 429). The caller
                      can call resume_run immediately without any manual
                      intervention. Do NOT auto-retry inside the workflow.

  UserActionError   — human intervention required (dirty tree, gh auth,
                      branch conflict). Carries an `action` attribute
                      that tells the user exactly what to do.

  FatalError        — config errors, schema mismatches, internal bugs.
                      Should never be retried; the only recovery is to
                      fix the root cause and start a fresh run.

All three inherit from OrchestratorError so callers can catch the base
if they want a single broad net, or the specific subclass for targeted
handling.
"""


class OrchestratorError(Exception):
    """Base class for all structured orchestrator errors.

    `cause` (Phase 80) optionally carries a structured failure cause —
    `{error, api_status, text}` recovered from the CLI transcript when the SDK
    collapsed the real error (e.g. an Anthropic billing_error) into a useless
    subtype string. None unless the runner's transcript feeder attached one;
    every downstream sink (audit payload, error.md, run_status) reads it via
    `getattr(exc, "cause", None)`.
    """

    cause: dict | None = None


class RetriableError(OrchestratorError):
    """Transient failure — the caller can resume_run immediately.

    Examples: network blip during push, upstream 5xx, rate-limit 429.
    The MCP server returns {"status": "retriable_error", ...} so client
    tooling can decide whether to retry automatically. The workflow itself
    never retries these — surfacing them is the right call; auto-retry
    inside the workflow hides real failures.
    """


class UserActionError(OrchestratorError):
    """Human intervention required before the run can proceed.

    Examples: dirty working tree, gh not authenticated, branch conflict.
    Carries an `action` attribute — a plain-English description of what
    the user needs to do. The MCP server returns
    {"status": "user_action_required", "action": ..., ...}.
    """

    def __init__(self, message: str, action: str = "") -> None:
        super().__init__(message)
        self.action = action or str(message)


class FatalError(OrchestratorError):
    """Non-retriable error — fix the root cause and start a fresh run.

    Examples: invalid orchestrator.toml, incompatible checkpoint schema,
    internal assertion failures. The MCP server returns
    {"status": "fatal", ...}.
    """
