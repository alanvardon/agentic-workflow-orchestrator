"""MCP server exposing the orchestrator to Claude Code.

Tools:
  - implement_feature: start a workflow; returns a plan for approval
  - approve_plan:      resume an awaiting workflow with the user's response
  - resume_run:        recover a workflow that failed mid-task
  - cancel_run:        signal cancellation between task boundaries
  - run_status:        report where a (backgrounded) run stands

The flow is conversational by design. The workflow pauses at plan approval, the
user reviews via Claude Code chat, then approves or revises. Each tool call:
  - opens the AsyncSqliteSaver
  - drives one ainvoke (or resume) call
  - closes the DB and returns to Claude Code

State lives in .orchestrator/checkpoints.db between calls, keyed by the
thread_id that implement_feature generates and approve_plan replays.

Observable progress (Phase 79): the post-approval leg runs 5+ minutes, during
which Claude Code chat shows NOTHING — it doesn't render MCP progress
notifications (they're advisory per spec). So the long-leg tools accept
`background=True`: they kick the workflow off as a tracked asyncio task
(_BG_RUNS) and return {"status": "started", thread_id} at once, and the chat
polls `run_status(thread_id)` to print persistent text updates. While a run is
live it holds the AsyncSqliteSaver write-lock, so run_status reads progress
from the filesystem audit-log tail (NOT aget_state) and the final result from
the in-memory registry; it only reads the checkpoint snapshot when no task is
live (e.g. after a server restart), when the DB is free.

Testing without Claude Code (recommended first step):
  npx @modelcontextprotocol/inspector \\
    /abs/path/to/orchestrator/.venv/bin/python \\
    -m orchestrator.mcp_server

CRITICAL: always invoke via the full venv python path, not the "python"
shim. MCP servers are subprocesses; pyenv/uv auto-activation doesn't
apply to them.
"""

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command
from mcp.server.fastmcp import Context, FastMCP

from orchestrator.cancellation import (
    clear_cancelled,
    is_cancelled,
    mark_cancelled,
)
from orchestrator.config import apply_overrides, load_config
from orchestrator.errors import FatalError, RetriableError, UserActionError
from orchestrator.idempotency import reserve as idempotency_reserve
from orchestrator.mcp_progress import run_with_progress
from orchestrator.paths import find_project_root
from orchestrator.run_artifacts import write_error
from orchestrator.run_log import append_run
from orchestrator.transcript import is_billing_cause
from orchestrator.workflow import (
    build_workflow,
    IncompatibleCheckpointError,
    IncompatiblePipelineError,
)


# AsyncSqliteSaver creates the .db file on demand but not its parent.
# Run once at import (server startup) rather than per-tool-call.
(find_project_root() / ".orchestrator").mkdir(exist_ok=True)

mcp = FastMCP("orchestrator")


# --- Plan-approval hints --------------------------------------------------
# Shared by the synchronous and background paths so the chat gets the same
# "what to do next" text whether the interrupt is returned directly or read
# back via run_status.
_IMPLEMENT_APPROVAL_HINT = (
    "Show the plan_text to the user. Ask whether they approve or want changes. "
    "Then call approve_plan with this same thread_id and their response "
    "('yes' to proceed, or feedback to revise)."
)
_APPROVE_PLAN_REVISED_HINT = (
    "Plan was revised based on the feedback. Show the new plan_text to the "
    "user and call approve_plan again with their next response."
)
_RESUME_APPROVAL_HINT = (
    "Workflow paused for plan approval. Show the plan to the user and call "
    "approve_plan with their response."
)


# --- Background-run registry (Phase 79) -----------------------------------
# A backgrounded workflow runs as an asyncio task on the FastMCP event loop.
# asyncio only keeps a weak reference to tasks, so we hold a STRONG ref here —
# otherwise a 5-minute run could be garbage-collected between tool calls. The
# entry also carries the start time (for elapsed) so run_status can report it
# without re-deriving from the checkpoint.
class _BgRun:
    def __init__(self, task: "asyncio.Task", request: str | None) -> None:
        self.task = task
        self.request = request
        self.started_at = time.monotonic()


_BG_RUNS: dict[str, _BgRun] = {}


def _last_audit_event(thread_id: str) -> dict | None:
    """Return the most recent audit event for thread_id, or None.

    Tails the JSONL audit log (.orchestrator/audit.log). This is a plain
    filesystem read with NO checkpoint-DB access, so it's safe to call while a
    background run holds the AsyncSqliteSaver write-lock — the lock-contention
    landmine the cancellation store documents. Shape:
    {event_type, task_name, timestamp}.
    """
    cfg = load_config()
    log_path = Path(cfg.audit.log_path)
    if not log_path.is_absolute():
        log_path = find_project_root() / log_path
    if not log_path.exists():
        return None
    last: dict | None = None
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("thread_id") == thread_id:
                    last = rec
    except OSError:
        return None
    if last is None:
        return None
    return {
        "event_type": last.get("event_type"),
        "task_name": last.get("task_name"),
        "timestamp": last.get("timestamp"),
    }


def _write_run_error(thread_id: str, exc: BaseException) -> None:
    """Persist a run-terminating failure to .orchestrator/runs/<thread>/error.md
    (Phase 80a, Sink B).

    The failed task name comes from the audit-log tail — the last task_failed event
    for this thread, emitted by the spine's @_audited_task wrapper just before the
    exception propagated here. Best-effort; write_error swallows its own errors so
    this never masks the original failure.
    """
    last = _last_audit_event(thread_id)
    write_error(thread_id, exc, failed_task=(last or {}).get("task_name"))


def _incompatible_checkpoint(
    thread_id: str, exc: IncompatibleCheckpointError
) -> dict:
    """Shape an IncompatibleCheckpointError into a structured response.

    The workflow body refuses to resume a checkpoint created by a different
    WORKFLOW_VERSION. Surface both versions so the chat can explain why and
    offer "start a fresh implement_feature" as the next step.
    """
    return {
        "status": "incompatible_checkpoint",
        "thread_id": thread_id,
        "stored_version": exc.stored_version,
        "current_version": exc.current_version,
        "next": (
            f"This run was created with workflow v{exc.stored_version}, but the "
            f"current code is v{exc.current_version}. In-flight runs can't be "
            "resumed across an incompatible version change — start a fresh "
            "implement_feature for this work."
        ),
    }


def _incompatible_pipeline(
    thread_id: str, exc: IncompatiblePipelineError
) -> dict:
    """Shape an IncompatiblePipelineError into a structured response.

    The v2 pipeline in orchestrator.toml changed since the run started.
    Surface both hashes so the chat can explain why a resume was refused.
    """
    return {
        "status": "incompatible_pipeline",
        "thread_id": thread_id,
        "stored_hash": exc.stored_hash,
        "current_hash": exc.current_hash,
        "next": (
            "The pipeline in orchestrator.toml (flow / stages / referenced "
            "parts) changed since this run started, so it can't be resumed. "
            "Revert the change to resume, or start a fresh implement_feature."
        ),
    }



def _user_action_required(thread_id: str, exc: UserActionError) -> dict:
    """Shape a UserActionError into a structured response.

    The `action` field tells the user exactly what to do before calling
    resume_run. Specific subclasses (CommitAndPrError, DirtyTreeError, etc.)
    are caught by this same handler since they all inherit UserActionError.
    """
    return {
        "status": "user_action_required",
        "thread_id": thread_id,
        "error": str(exc),
        "action": exc.action,
    }


def _retriable_error(thread_id: str, exc: RetriableError) -> dict:
    """Shape a RetriableError into a structured response.

    Transient failures: call resume_run immediately without any manual
    intervention.
    """
    return {
        "status": "retriable_error",
        "thread_id": thread_id,
        "error": str(exc),
        "next": "Transient failure. Call resume_run(thread_id) to retry immediately.",
    }


def _fatal_error(thread_id: str, exc: FatalError) -> dict:
    """Shape a FatalError into a structured response.

    Surfaces the structured `cause` (Phase 80a) the runner's transcript feeder
    attached, so the real reason (e.g. an Anthropic billing_error the SDK collapsed
    to "...error result: success") reaches the caller instead of a useless string.

    Minimal classification (Phase 80b): a credit/billing cause is NOT really fatal —
    the run is fully checkpointed, so topping up and calling resume_run re-runs only
    the failed leg. Reshape it to a clear, RESUMABLE "billing" status pointing at
    resume_run, never "start a fresh implement_feature".

    The specific subclasses IncompatibleCheckpointError and IncompatiblePipelineError
    still have their own handlers above so they can surface extra structured fields.
    """
    cause = getattr(exc, "cause", None)
    if is_billing_cause(cause, str(exc)):
        return {
            "status": "billing",
            "thread_id": thread_id,
            "error": str(exc),
            "cause": cause,
            "next": (
                "Your Anthropic credit balance is too low (billing_error). Top up "
                "at https://console.anthropic.com/settings/billing, then call "
                "resume_run(thread_id) — the run is checkpointed, so only the "
                "failed leg re-runs. Do NOT start a fresh implement_feature."
            ),
        }
    return {
        "status": "fatal",
        "thread_id": thread_id,
        "error": str(exc),
        "cause": cause,
        "next": (
            "This is a non-retriable error. Fix the root cause and start a "
            "fresh implement_feature run."
        ),
    }


def _awaiting_approval(thread_id: str, result: dict, hint: str) -> dict:
    """Shape an interrupt result into the awaiting_approval response.

    The interrupt's value dict is set by orchestrator/workflow.py at the
    interrupt() call site. Plan-approval interrupts carry "plan"; other
    gates (branch/impl/pr approvals, approval_gate steps) carry only
    "kind" and "ask", so `plan` is read defensively.
    """
    interrupt_val = result["__interrupt__"][0].value
    kind = interrupt_val.get("kind")
    # approval_gate steps have their own resume contract: any reply proceeds, the
    # abort words stop the run. Override the (plan-centric) caller hint so the chat
    # surfaces that instead.
    if kind == "step_approval_gate":
        hint = (
            "A pluggable approval_gate step is asking for a decision (see `ask`). "
            "Call approve_plan with this thread_id and the user's reply: "
            "'abort' (or 'no'/'stop') stops the run cleanly; any other reply "
            "proceeds past the gate."
        )
    elif kind == "red_review":
        # TDD red-green (Phase 72b): the test-author wrote failing tests for this
        # task; the human reviews the RED suite before implementation. Show the
        # reviewer `red_output` (the failing output) and `summary`, then resume.
        hint = (
            "TDD red-review: a separate test-author wrote FAILING tests for task "
            f"{interrupt_val.get('task_id')!r} (see `red_output`/`summary`). Call "
            "approve_plan with this thread_id and the user's reply: 'yes' to "
            "implement against these tests, 'abort' (or 'no'/'stop') to stop the "
            "run, or feedback text to re-author the tests."
        )
    return {
        "status": "awaiting_approval",
        "thread_id": thread_id,
        "plan": interrupt_val.get("plan"),
        # plan-approval interrupts also carry the decomposed task list so the chat
        # can show plan + tasks together for review. None for other gates.
        "tasks": interrupt_val.get("tasks"),
        "kind": kind,
        "ask": interrupt_val.get("ask"),
        # red_review interrupts carry the task's failing-test output + author's
        # one-line summary so the chat can show the RED suite for review. None else.
        "task_id": interrupt_val.get("task_id"),
        "red_output": interrupt_val.get("red_output"),
        "summary": interrupt_val.get("summary"),
        "next": hint,
    }


async def _fetch_existing_state(thread_id: str) -> dict:
    """Return the current state of an existing thread.

    Used when an idempotency_key collides — instead of starting a new
    workflow, peek at the checkpointer for the run already claimed by
    that key and return a response in the same shape the caller would
    get from a fresh implement_feature / approve_plan / resume_run.

    The `replayed: True` marker tells the caller that this is not a
    new run — useful for logging and for avoiding showing the plan
    to the user a second time on a double-click.
    """
    config = {"configurable": {"thread_id": thread_id}}
    async with build_workflow() as workflow:
        snapshot = await workflow.aget_state(config)

    # If the run is paused at an interrupt, its pending tasks carry
    # the interrupt values. Surface the plan exactly like a fresh
    # awaiting_approval would.
    interrupts = []
    for task in snapshot.tasks:
        interrupts.extend(task.interrupts)
    if interrupts:
        interrupt_val = interrupts[0].value
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "kind": interrupt_val.get("kind"),
            "plan": interrupt_val.get("plan"),
            "tasks": interrupt_val.get("tasks"),  # decomposed task list
            # red_review (Phase 72b) carries the failing-test output for review.
            "task_id": interrupt_val.get("task_id"),
            "red_output": interrupt_val.get("red_output"),
            "summary": interrupt_val.get("summary"),
            "next": (
                "Replayed from idempotency key. Show the plan (or, for a "
                "red_review pause, the red_output) to the user and call "
                "approve_plan with this thread_id."
            ),
            "replayed": True,
        }

    # Workflow completed (status: succeeded / failed / cancelled).
    # snapshot.values is the entrypoint's return dict.
    if isinstance(snapshot.values, dict) and "status" in snapshot.values:
        result = dict(snapshot.values)
        result["thread_id"] = thread_id
        result["replayed"] = True
        return result

    # Reservation exists but no completed state and no interrupt — the
    # run is mid-task. The caller should poll or call resume_run after
    # whatever they're waiting on has settled.
    return {
        "status": "in_progress",
        "thread_id": thread_id,
        "next": (
            "A run is already underway for this idempotency key. Poll "
            "again later, or call resume_run if you believe the run has "
            "stalled."
        ),
        "replayed": True,
    }


async def _run_workflow(
    thread_id: str,
    input_data,
    *,
    orch_config=None,
    approval_hint: str,
    ctx: Context | None = None,
) -> dict:
    """Drive one workflow leg and shape the outcome into a response dict.

    Shared by the synchronous tool path and the background runner. Opens the
    checkpointer, streams via run_with_progress (so MCP-aware clients still get
    notifications), and converts the result into exactly one of:
      - an awaiting_approval dict (the run paused at an interrupt),
      - a structured error dict (the orchestrator error families), or
      - the workflow's final {status, ...} dict with thread_id injected.

    `orch_config` is the resolved OrchestratorConfig for a fresh start
    (implement_feature applies per-invocation overrides); pass None for resumes
    so build_workflow loads config itself. `input_data` is the request string,
    Command(resume=...), or None — same as run_with_progress.
    """
    lg_config = {"configurable": {"thread_id": thread_id}}
    try:
        async with build_workflow(config=orch_config) as workflow:
            result = await run_with_progress(workflow, input_data, lg_config, ctx)
    except (IncompatibleCheckpointError, IncompatiblePipelineError) as exc:
        if isinstance(exc, IncompatibleCheckpointError):
            return _incompatible_checkpoint(thread_id, exc)
        return _incompatible_pipeline(thread_id, exc)
    except UserActionError as exc:
        return _user_action_required(thread_id, exc)
    except RetriableError as exc:
        _write_run_error(thread_id, exc)
        return _retriable_error(thread_id, exc)
    except FatalError as exc:
        _write_run_error(thread_id, exc)
        return _fatal_error(thread_id, exc)
    except Exception as exc:
        # Truly unexpected (the orchestrator-error families above are the known
        # exits). Capture it to error.md before it escapes so a backgrounded
        # failure isn't lost, then re-raise — run_status's background path shapes
        # the raised exception into a fatal response.
        _write_run_error(thread_id, exc)
        raise
    if "__interrupt__" in result:
        return _awaiting_approval(thread_id, result, approval_hint)
    result["thread_id"] = thread_id
    return result


def _start_background(
    thread_id: str,
    input_data,
    *,
    orch_config=None,
    approval_hint: str,
    request_text: str | None,
) -> dict:
    """Kick off _run_workflow as a tracked background task; return at once.

    The task runs on the FastMCP event loop concurrently with later tool calls
    (run_status, cancel_run). We pass ctx=None: the Context that started the run
    belongs to THIS tool call and is invalid once it returns, and the chat sees
    progress via run_status polling rather than notifications anyway.
    """
    async def _runner() -> dict:
        return await _run_workflow(
            thread_id, input_data, orch_config=orch_config,
            approval_hint=approval_hint, ctx=None,
        )

    _BG_RUNS[thread_id] = _BgRun(asyncio.create_task(_runner()), request_text)
    return {
        "status": "started",
        "thread_id": thread_id,
        "next": (
            "The run is executing in the background. Poll run_status(thread_id) "
            "about every 20s and show the user each update. run_status returns "
            "'running' (with stage + elapsed) while it works, 'awaiting_approval' "
            "if it pauses for a gate (then call approve_plan), or the final "
            "'succeeded'/'failed'/'no_changes'/'cancelled' result when done."
        ),
    }


@mcp.tool()
async def implement_feature(
    request: str,
    approve_plan: bool | None = None,
    base_branch: str | None = None,
    idempotency_key: str | None = None,
    background: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Start a feature, fix, or refactor implementation workflow.

    Use this when the user asks to implement, change, or fix something in
    the target repo. Example user intents:
      - "add a tooltip showing what LTV means"
      - "fix the rounding bug in the X calculation"
      - "refactor the modal close handlers"

    The workflow ALWAYS pauses for plan approval before writing any code
    (unless `approve_plan=False` is passed, or `ORCHESTRATOR_APPROVE_PLAN`
    is set in the environment). When it pauses, this tool returns
    {"status": "awaiting_approval", ...} containing the plan and a
    thread_id. You MUST then:
      1. Show the plan's `plan_text` to the user.
      2. Ask whether they approve, or what they want changed.
      3. Call `approve_plan` with the same thread_id and their response.

    Do NOT call this tool again to "retry" — that starts a fresh workflow
    with a new thread_id, losing the user's review context. To revise an
    in-flight plan, send feedback via `approve_plan` instead.

    Idempotency: when `idempotency_key` is provided, the first call with
    that key claims it and runs the workflow as normal. A second call with
    the SAME key (e.g. double-click, retry,
    CI re-run) returns the existing thread's current state — same
    `thread_id`, no new run, with `replayed: True` added so the caller
    can tell. Keys must match `[A-Za-z0-9._-]+` and are ≤128 chars.

    Args:
        request: Natural-language description of what to implement.
        approve_plan: Per-invocation override for the plan-approval gate.
            None = use `orchestrator.toml` / env var. False = skip the
            approval pause and run straight through to PR. True = require
            approval regardless of config.
        base_branch: Per-invocation override for the PR base branch.
        idempotency_key: Optional caller-supplied key. Reusing a key
            returns the existing run instead of starting a new one.
        background: When True, kick the run off as a tracked task and
            return {"status": "started", thread_id} immediately instead
            of blocking; then poll run_status(thread_id) to report
            progress in chat. The normal approval flow keeps this False
            here (planning is quick — you want the plan back to show the
            user) and passes background=True to approve_plan, which is
            where the 5+ min leg lives. Use background=True here only with
            approve_plan=False, which runs straight through that long leg.

    Returns:
        Awaiting-approval dict (or, if approve_plan was overridden to
        False, the final succeeded/failed dict). With background=True, a
        {"status": "started", thread_id} dict — poll run_status. When an
        idempotency key replays an existing run, the dict carries
        `replayed: True`.
    """
    # The idempotency claim happens BEFORE any other side effect (run_log entry,
    # workflow build). If the key collides, return the original run's state without
    # writing anything new.
    if idempotency_key is not None:
        candidate_thread_id = f"run-{uuid4().hex[:8]}"
        existing = idempotency_reserve(idempotency_key, candidate_thread_id)
        if existing is not None:
            return await _fetch_existing_state(existing)
        thread_id = candidate_thread_id
    else:
        thread_id = f"run-{uuid4().hex[:8]}"

    append_run(thread_id, request, source="mcp", idempotency_key=idempotency_key)
    effective_config = apply_overrides(
        load_config(),
        approve_plan=approve_plan,
        base_branch=base_branch,
    )
    # background=True: kick the workflow off as a tracked task and return at
    # once so the chat polls run_status (see module docstring). Otherwise run
    # synchronously, streaming progress notifications to MCP-aware clients via
    # ctx (injected by FastMCP; None outside the MCP transport).
    if background:
        return _start_background(
            thread_id, request, orch_config=effective_config,
            approval_hint=_IMPLEMENT_APPROVAL_HINT, request_text=request,
        )
    return await _run_workflow(
        thread_id, request, orch_config=effective_config,
        approval_hint=_IMPLEMENT_APPROVAL_HINT, ctx=ctx,
    )


@mcp.tool()
async def resume_run(
    thread_id: str,
    force: bool = False,
    background: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Resume a workflow that failed mid-task without restarting it.

    Use this when a previous `implement_feature` or `approve_plan` call
    returned an error (e.g. push failed, gh pr create failed). commit/push/PR
    are three independent @tasks, so the successful steps are cached in the
    checkpointer — only the failed task (and anything downstream) re-runs.

    Use AFTER fixing the underlying issue. Examples:
      - push failed → authenticate gh, restore network, then resume_run
      - gh pr create failed (no remote, no auth) → fix auth, then resume_run
      - commit failed mid-workflow → investigate, may need manual cleanup
        before resume_run

    Do NOT use this to resume a plan-approval interrupt — that's what
    `approve_plan` is for. resume_run is specifically for recovering
    from a task failure.

    If the thread has been marked cancelled via `cancel_run`, resume_run
    refuses with {"status": "refused_cancelled", ...} unless `force=True`
    is passed (which clears the cancel flag first).

    Args:
        thread_id: The thread_id from the prior failed response.
        force: When the thread is cancelled, set True to clear the
            cancel flag and resume anyway. Default False refuses.
        background: When True, run the recovery leg as a tracked task and
            return {"status": "started", thread_id} immediately; poll
            run_status(thread_id) for progress. Default False blocks.

    Returns:
        Same shape as approve_plan: another awaiting_approval (rare,
        only if the workflow re-entered the planning loop somehow), a
        succeeded dict with pr_url, or a failed dict. If the thread is
        cancelled and `force` is False, a refused_cancelled dict.
    """
    if is_cancelled(thread_id):
        if not force:
            return {
                "status": "refused_cancelled",
                "thread_id": thread_id,
                "next": (
                    "This run was cancelled via cancel_run. Call resume_run "
                    "again with force=True to clear the cancel flag and resume."
                ),
            }
        clear_cancelled(thread_id)

    # Passing None as the input is the functional API's resume incantation: it
    # continues the paused/failed workflow from its last checkpoint instead of
    # starting fresh. background=True runs the recovery leg as a tracked task
    # and reports via run_status (see module docstring); otherwise synchronous.
    if background:
        return _start_background(
            thread_id, None, approval_hint=_RESUME_APPROVAL_HINT,
            request_text=None,
        )
    return await _run_workflow(
        thread_id, None, approval_hint=_RESUME_APPROVAL_HINT, ctx=ctx,
    )


@mcp.tool()
async def cancel_run(thread_id: str) -> dict:
    """Signal cancellation for a running or paused workflow.

    The workflow checks the cancel flag between @task boundaries; the
    task currently executing (if any) completes before cancellation
    takes effect. LLM tokens already spent on the in-flight task are
    NOT refunded — "cancel" means "stop after the current task," not
    "abort instantly."

    Use this when:
      - A run is stuck in a plan-approval loop you no longer want to drive
      - An implementation_task is mid-flight and the request was wrong
      - A QA-retry loop is wasting attempts on something unfixable

    The cancel flag is persisted in `.orchestrator/checkpoints.db`. To
    resume a cancelled thread later, call `resume_run(thread_id,
    force=True)` — that clears the flag and re-enters the workflow.

    Args:
        thread_id: The thread_id of the run to cancel.

    Returns:
        {"status": "cancellation_signalled", "thread_id": thread_id}
    """
    mark_cancelled(thread_id)
    return {
        "status": "cancellation_signalled",
        "thread_id": thread_id,
        "next": (
            "The workflow will exit at the next task boundary. To resume "
            "the run later, call resume_run with force=True."
        ),
    }


@mcp.tool()
async def approve_plan(
    thread_id: str,
    response: str,
    background: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Resume an awaiting workflow with the user's response to the plan.

    Call this ONLY after `implement_feature` (or a prior `approve_plan`)
    returned {"status": "awaiting_approval", "thread_id": ..., "plan": ...}
    and the user has responded to the plan.

    response should be:
      - "yes" → approve the current plan. The workflow proceeds through
        branch creation, implementation (5+ min), QA, and PR. On success
        returns {"status": "succeeded", "pr_url": ..., ...}. On QA failure
        after 3 attempts, returns {"status": "failed", "qa_failures": ...}.
        If QA passes but the build made no changes, returns
        {"status": "no_changes", "branch": ...} — no commit, no PR.
      - Anything else → treated as feedback. The planner regenerates the
        plan incorporating the feedback, and returns ANOTHER
        "awaiting_approval" response. Loop until the user says "yes".

    Loop pattern:
      result = await implement_feature(request)
      while result["status"] == "awaiting_approval":
          # show result["plan"] to user, get their response
          result = await approve_plan(result["thread_id"], response)
      # result["status"] is now "succeeded" or "failed"

    In the Claude Code chat, the "yes" approval starts a 5+ min leg that would
    otherwise block with no visible activity. Prefer background=True for it,
    then poll run_status(thread_id) ~every 20s and print each update:
      result = await approve_plan(thread_id, "yes", background=True)
      while result["status"] in ("started", "running"):
          # wait ~20s, then:
          result = await run_status(thread_id)
      # awaiting_approval (call approve_plan again) or a terminal status

    Args:
        thread_id: From the most recent awaiting_approval response.
        response: "yes" to proceed, or feedback text to revise the plan.
        background: When True, run the post-approval leg as a tracked task
            and return {"status": "started", thread_id} immediately; poll
            run_status(thread_id) for progress. Recommended for "yes" in
            the interactive chat. Default False blocks until done.

    Returns:
        On revision: same awaiting_approval shape (loop again).
        On success: {"status": "succeeded", "branch": str, "pr_url": str, ...}
        On QA exhaustion: {"status": "failed", "qa_failures": str, ...}
        On an empty build: {"status": "no_changes", "branch": str, ...}
    """
    # "yes" on approval kicks off the longest section of the workflow (planning
    # → impl → QA → commit → push → PR), so this is the call that benefits most
    # from background=True: return at once and let the chat poll run_status (see
    # module docstring). A revision reply re-enters the planning loop and pauses
    # again quickly, so backgrounding it is harmless either way. _run_workflow
    # injects thread_id into the final dict so the chat can still call resume_run
    # if something fails downstream.
    if background:
        return _start_background(
            thread_id, Command(resume=response),
            approval_hint=_APPROVE_PLAN_REVISED_HINT, request_text=None,
        )
    return await _run_workflow(
        thread_id, Command(resume=response),
        approval_hint=_APPROVE_PLAN_REVISED_HINT, ctx=ctx,
    )


@mcp.tool()
async def run_status(thread_id: str) -> dict:
    """Report where a backgrounded run stands. Poll this after a tool returned
    {"status": "started", thread_id}.

    Drive the poll loop from chat: call run_status about every 20s and print
    each update to the user, so the long run shows visible, persistent activity
    instead of dead air. Stop when the status is no longer "running".

    Possible statuses:
      - "running": still working. `stage` is the latest task from the audit log,
        `elapsed_seconds` is wall-clock since the run started, `last_event` is
        {event_type, task_name, timestamp}. Wait ~20s and poll again.
      - "awaiting_approval": the run paused at a gate (plan / red_review /
        approval_gate). Show the plan or red_output to the user and call
        approve_plan(thread_id, response) — optionally background=True again for
        the leg that follows.
      - terminal: "succeeded" (with pr_url), "no_changes", "failed"
        (with qa_failures), "cancelled", or "aborted" — the run is done.
      - an error shape ("user_action_required" / "retriable_error" /
        "incompatible_checkpoint" / "incompatible_pipeline" / "fatal"): handle
        as documented for that status, then resume_run / start fresh as advised.

    Implementation note: while a run is live it holds the checkpoint DB
    write-lock, so this reads progress from the filesystem audit-log tail and
    the final result from the in-memory task — it touches the checkpoint
    snapshot only when no task is tracked (e.g. after a server restart).

    Args:
        thread_id: The thread_id from the "started" response.

    Returns:
        A status dict as described above.
    """
    run = _BG_RUNS.get(thread_id)
    if run is not None:
        if not run.task.done():
            # The live run holds the AsyncSqliteSaver write-lock — read progress
            # from the audit-log tail, NOT aget_state, to avoid "database is
            # locked" (the cancellation store documents this contention).
            last = _last_audit_event(thread_id)
            return {
                "status": "running",
                "thread_id": thread_id,
                "elapsed_seconds": round(time.monotonic() - run.started_at, 1),
                "stage": (last or {}).get("task_name"),
                "last_event": last,
                "next": "Still working. Poll run_status again in ~20s.",
            }
        # Finished: the task's return value is the terminal response dict
        # (_run_workflow already shaped awaiting_approval / errors / final
        # status, all carrying thread_id). Kept in the registry so repeat polls
        # keep returning it — the "don't lose the final result" landmine.
        try:
            return run.task.result()
        except Exception as exc:  # pragma: no cover - defensive
            # _run_workflow shapes the known orchestrator error families into
            # dicts itself, so a raised exception is unexpected. Surface it
            # rather than letting the backgrounded failure vanish.
            return {
                "status": "fatal",
                "thread_id": thread_id,
                "error": str(exc),
                "next": (
                    "The background run failed unexpectedly. Inspect the logs, "
                    "fix the root cause, and start a fresh implement_feature."
                ),
            }
    # No tracked task — e.g. the server restarted, or an idempotency replay.
    # Nothing holds the DB lock now, so reading the checkpoint snapshot is safe.
    return await _fetch_existing_state(thread_id)


if __name__ == "__main__":
    mcp.run()
