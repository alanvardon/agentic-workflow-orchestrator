"""Workflow → MCP progress streaming.

Wraps `workflow.astream(stream_mode="updates")` and routes per-task
completion events (and heartbeat ticks during long-running tasks) to
the MCP client via `Context.report_progress`.

This is the MCP-side counterpart to `cli._run_with_progress` — same
event-loop shape, same _NEXT_STAGE label table, just a different
sink. The CLI prints to stderr; this module sends MCP progress
notifications. Either one converts the workflow's astream events
into "what's happening now" updates the human can see.

Why this exists at all: `approve_plan("yes")` previously blocked
the MCP tool call for 5+ minutes (planning → implementation → QA
→ commit → push → pr_create) with zero feedback in Claude Code chat.
With progress streaming, the client sees `done: planning (3s)` →
`running implementation (45s elapsed)` → `done: implementation
(4m 12s)` etc.

Notification failures are best-effort: report_progress is advisory
(the MCP spec lets the client ignore it), and a flaky write must
not kill the workflow. Exceptions inside the progress sink are
swallowed.
"""

import asyncio
import os
import time
from typing import Any

from mcp.server.fastmcp import Context

from orchestrator.errors import FatalError


# Keep in sync with the @entrypoint function name in workflow.py.
# The final workflow result arrives as a stream event keyed by this
# name; everything else keyed by `<taskname>_task` is a per-task
# completion event.
_ENTRYPOINT_NAME = "workflow"


# "task that just completed" → "task that's now running." Lets the
# heartbeat label describe the IN-PROGRESS task rather than what just
# finished. qa is ambiguous (PASS → commit; FAIL → another impl
# attempt) and is disambiguated inline by reading the QaResult.
# Keep in sync with the workflow body in orchestrator/workflow.py
# AND with cli.py's _NEXT_STAGE — single source of truth would be
# nice but worth refactoring only if a third caller appears.
_NEXT_STAGE = {
    None: "verify_clean_tree",
    "verify_clean_tree": "planning",
    "planning": "decompose",
    "decompose": "create_branch",
    "create_branch": "implementation",
    "implementation": "qa",
    "summarize": "docs",
    "docs": "commit",
    "commit": "push",
    "push": "pr_create",
    "pr_create": "finishing",
}


def _format_elapsed(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


async def run_with_progress(
    workflow: Any,
    input_data: Any,
    config: dict,
    ctx: Context | None,
) -> dict:
    """Drive the workflow's astream and emit MCP progress events.

    Args:
        workflow: the entrypoint object from `build_workflow()`.
        input_data: same as you'd pass to `workflow.ainvoke` — the
            initial request string, or `Command(resume=...)`, or None
            for a resume.
        config: LangGraph runtime config (thread_id, etc.).
        ctx: the MCP `Context` injected by FastMCP; pass None to run
            silently (used in tests and when the MCP client doesn't
            support progress notifications).

    Returns:
        The workflow result dict — either `{"__interrupt__": [...]}`
        when the workflow paused for approval, or the entrypoint's
        final `{"status": ..., ...}` return value.
    """
    heartbeat_interval = float(os.environ.get("HEARTBEAT_INTERVAL", "15"))
    last_event_time = time.monotonic()
    final_result: dict | None = None
    current_label = f"running {_NEXT_STAGE[None]}"
    progress_count = 0
    stop = asyncio.Event()

    async def _report(message: str) -> None:
        if ctx is None:
            return
        try:
            await ctx.report_progress(
                progress=float(progress_count),
                total=None,
                message=message,
            )
        except Exception:
            # Notifications are advisory per the MCP spec. A sink
            # failure (closed transport, slow client) must not take
            # down a 5-minute workflow.
            pass

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=heartbeat_interval)
                return
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - last_event_time
                await _report(
                    f"{current_label} ({_format_elapsed(elapsed)} elapsed)"
                )

    hb = asyncio.create_task(heartbeat())
    try:
        async for event in workflow.astream(
            input_data, config=config, stream_mode="updates"
        ):
            for key, value in event.items():
                now = time.monotonic()
                task_elapsed = now - last_event_time
                last_event_time = now

                if key == "__interrupt__":
                    final_result = {"__interrupt__": value}
                elif key == _ENTRYPOINT_NAME:
                    final_result = value
                else:
                    progress_count += 1
                    # Strip the _task suffix for readable output —
                    # "implementation" beats "implementation_task".
                    name = key.removesuffix("_task")
                    message = f"done: {name} ({_format_elapsed(task_elapsed)})"
                    # After decompose, append the task titles so the waiting
                    # client can see whether the plan was interpreted sensibly.
                    # getattr guard: the loop sees every task key, not just
                    # decompose. Comma separator — this is a plain notification
                    # string, not terminal-rendered like the CLI's bullet form.
                    if name == "decompose":
                        tasks = getattr(value, "tasks", None)
                        if tasks:
                            titles = ", ".join(t.title for t in tasks)
                            message += f" → {len(tasks)} tasks: {titles}"
                    await _report(message)
                    # Predict the next stage so the heartbeat label is
                    # accurate. qa branches: PASS → commit, FAIL →
                    # another implementation attempt.
                    if name == "qa":
                        next_stage = (
                            "commit"
                            if getattr(value, "result", None) == "PASS"
                            else "implementation (retry)"
                        )
                    else:
                        next_stage = _NEXT_STAGE.get(name, "next stage")
                    current_label = f"running {next_stage}"
    finally:
        stop.set()
        await hb

    if final_result is None:
        raise FatalError("astream completed without a final result")
    return final_result
