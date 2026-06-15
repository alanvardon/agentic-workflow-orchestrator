"""Runtime execution of injected steps.

Plain async functions, one per executable step type. workflow.py wraps each
in a @task so they inherit checkpointing, tracing, and cancel/usage handling
at the @task boundary — the user's step never touches that plumbing.

- execute_script: run an executable; non-zero exit raises StepError.
- execute_ai_agent: run a markdown-defined agent (<step.agent>, a project-root-
  relative path with the full filename, as the system prompt) via the Claude
  Agent SDK, same loop shape as the planning/implementation/qa agents.

approval_gate steps have no runner here — they're a pause (interrupt()) handled
inline in workflow.run_seam, since interrupt() must run in the entrypoint
body, not inside a @task.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from orchestrator.agents.runner import run_structured_agent
from orchestrator.errors import FatalError
from orchestrator.manifest import AiAgentStep, ScriptStep, StepResult
from orchestrator.prompt_loader import load_agent_prompt
from orchestrator.retry_block import feedback_section


class StepError(FatalError):
    """Raised when an injected step fails (non-zero script exit, timeout, or
    a missing agent file).

    A FatalError (not a bare RuntimeError) so it rides the structured-error
    hierarchy: the workflow body lets it propagate, and the MCP server's
    `except FatalError` handler shapes it into a {"status": "fatal",
    "thread_id": ...} response with recovery guidance — instead of a raw
    traceback escaping `ainvoke`. A mis-wired step (bad script, agent that
    never emits) is a config problem to fix, not something to retry, which is
    exactly FatalError's contract."""


def _logger(step_id: str) -> logging.Logger:
    # Child logger per step so injected-step output is attributable without
    # the user adding any logging of their own.
    return logging.getLogger(f"orchestrator.steps.{step_id}")


def _run_script_sync(
    step: ScriptStep, repo_root: Path, *, as_gate: bool = False
) -> StepResult:
    log = _logger(step.id)
    script = repo_root / step.path
    log.info("running script step %r: %s", step.id, step.path)
    try:
        proc = subprocess.run(
            [str(script)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=step.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise StepError(
            f"script step {step.id!r} timed out after {step.timeout}s"
        ) from exc
    except OSError as exc:
        raise StepError(
            f"script step {step.id!r} could not be executed ({step.path}): {exc}"
        ) from exc

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.stdout:
        log.info("[%s] stdout:\n%s", step.id, out)
    if proc.stderr:
        log.info("[%s] stderr:\n%s", step.id, err)

    # The complete run, both streams, untruncated — the evidence layer persists
    # this verbatim. `detail` below is only a short human summary derived from it.
    full_output = "\n".join(part for part in (out, err) if part)

    if proc.returncode != 0:
        report = err or out or "(no output)"
        if as_gate:
            # As a retry-block gate, a non-zero exit is a FAIL verdict (not an
            # abort). Its output becomes the feedback the engine injects into the
            # next producer attempt.
            return StepResult(
                step_id=step.id, kind="script", ok=True, passed=False,
                detail=report, full_output=full_output,
            )
        # As a plain step, the script's own output is the abort reason (pre-hooks
        # behaviour): a non-zero exit fails the whole workflow.
        raise StepError(
            f"script step {step.id!r} failed (exit {proc.returncode}):\n{report}"
        )

    # Keep a short tail of stdout as the human-readable detail. A gate that
    # exits 0 is a PASS verdict; a plain step leaves `passed` unset (None).
    detail = out[-500:] if out else "ok"
    return StepResult(
        step_id=step.id,
        kind="script",
        ok=True,
        passed=True if as_gate else None,
        detail=detail,
        full_output=full_output,
    )


async def execute_script(
    step: ScriptStep, repo_root: Path, *, as_gate: bool = False
) -> StepResult:
    """Run a script step off the event loop (subprocess.run is blocking).

    `as_gate`: when True the step is a retry-block gate — a non-zero exit returns
    `passed=False` with the output as feedback instead of raising.
    """
    return await asyncio.to_thread(_run_script_sync, step, repo_root, as_gate=as_gate)


def _coerce_passed(raw: object) -> bool:
    """Interpret the gate agent's emitted `passed` as a bool.

    The SDK normally delivers a JSON boolean as a Python bool; coerce common
    string spellings defensively (anything else is treated as a FAIL — the
    fail-closed posture)."""
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("true", "1", "yes")


async def execute_ai_agent(
    step: AiAgentStep,
    project_root: Path,
    plan_text: str,
    *,
    feedback: str | None = None,
    as_gate: bool = False,
) -> StepResult:
    """Run a markdown-defined agent against the current working tree.

    The agent loop lives in run_structured_agent; this function resolves the
    agent's markdown prompt, runs the loop, and shapes the result into a
    StepResult. The agent gets the plan in the user message and runs
    `git diff HEAD` itself to see the changes (like the qa agent).

    Retry-block roles:
    - `feedback` (producer): on a retry, the failing gate's detail is appended
      to the user message under a standard heading, so the producer can target
      its fixes — the same feedback-injection the built-in implementation
      producer uses.
    - `as_gate`: the agent is a gate. Its emit tool gains a required `passed`
      bool (the verdict) plus `detail` (the feedback). The default gate tools
      omit Edit/Write but include Bash (for `git diff` etc.) — see the note on
      AiAgentStep.allowed_tools: this is NOT strictly read-only.
    """
    log = _logger(step.id)
    # The prompt body comes from the shared loader (one "file on disk → body"
    # path for built-in and generic agents); a missing file becomes a StepError
    # so this module keeps its single failure type.
    try:
        system_prompt = load_agent_prompt(project_root, step.agent)
    except FileNotFoundError as exc:
        raise StepError(f"agent file not found at {step.agent}") from exc

    parts = ["## Plan", "", plan_text]
    if feedback:
        parts += ["", feedback_section(feedback)]
    user_message = "\n".join(parts)

    if as_gate:
        # A gate judges and reports. The default omits Edit/Write but includes
        # Bash so it can run `git diff HEAD` etc. — Bash can still mutate, so this
        # is NOT strictly read-only; a gate that must not write should set
        # allowed_tools=["Read", "Grep"].
        default_tools = ["Read", "Bash", "Grep"]
        emit_tool_description = (
            "Emit the gate verdict. Call exactly once when the check is "
            "complete: `passed` is true if the check passes, false otherwise; "
            "`detail` is the failure report / feedback (used to guide a retry "
            "on failure, empty on pass). After calling, stop."
        )
        emit_tool_fields = {"passed": bool, "detail": str}
        result_factory = lambda captured, usage: StepResult(
            step_id=step.id,
            kind="ai_agent",
            ok=True,
            passed=_coerce_passed(captured.get("passed")),
            detail=captured.get("detail", "") or "",
            usage=usage,
        )
    else:
        default_tools = ["Read", "Edit", "Write", "Bash", "Grep"]
        emit_tool_description = (
            "Emit the final result of this step. Call exactly once when "
            "done, with a one-line `summary` of what you did. After "
            "calling, stop."
        )
        emit_tool_fields = {"summary": str}
        result_factory = lambda captured, usage: StepResult(
            step_id=step.id,
            kind="ai_agent",
            ok=True,
            detail=captured.get("summary", "") or "",
            usage=usage,
        )

    # A step may override the role-default tools; None inherits it.
    allowed_tools = step.allowed_tools if step.allowed_tools is not None else default_tools

    log.info(
        "running ai_agent step %r (agent=%s, as_gate=%s)", step.id, step.agent, as_gate
    )
    # The shared runner raises FatalError on a missing emit; re-wrap it as
    # StepError so this module keeps its single failure type. (Either error
    # aborts the workflow, but StepError is the documented step contract.) The
    # fail-closed guard means a gate that never emits aborts — it never silently
    # passes.
    try:
        return await run_structured_agent(
            system_prompt=system_prompt,
            user_message=user_message,
            model=step.model,
            allowed_tools=allowed_tools,
            disallowed_tools=step.disallowed_tools,
            cwd=project_root,
            timeout=step.timeout,
            emit_tool_name="emit_step_result",
            emit_tool_description=emit_tool_description,
            emit_tool_fields=emit_tool_fields,
            result_factory=result_factory,
        )
    except FatalError as exc:
        # The runner may have attached a structured cause (Phase 80 — e.g. a
        # billing_error pulled from the transcript). Carry it across the re-wrap so
        # the real cause still reaches the audit payload / error.md / run_status,
        # and reuse the runner's richer message when it already explains the cause.
        err = StepError(
            str(exc) if getattr(exc, "cause", None)
            else f"ai_agent step {step.id!r} did not call emit_step_result"
        )
        err.cause = getattr(exc, "cause", None)
        raise err from exc
