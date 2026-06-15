"""Debug CLI for the orchestrator.

Not the production interface — that's the MCP server. This
exists so you can run the workflow end-to-end from a single shell command
and confirm "is the orchestrator itself broken?" without Claude Code +
MCP indirection layers in the picture.

Run:
    python -m orchestrator.cli "add a console.log to App.recalc"
    # or, via the installed script:
    implement-feature add a console.log to App.recalc

Env vars:
    ORCHESTRATOR_DEBUG=1   show full traceback on failure
    HEARTBEAT_INTERVAL=15  seconds between progress pings (default 15)
"""

import argparse
import asyncio
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

from orchestrator.config import apply_overrides, load_config
from orchestrator.errors import FatalError, OrchestratorError, RetriableError, UserActionError
from orchestrator.git_ops import BranchCreationError, CommitAndPrError, PreHookError
from orchestrator.manifest import ManifestError
from orchestrator.run_log import append_run
from orchestrator.workflow import build_workflow


# Known error types get a friendly message; everything else falls through
# to the generic handler. Add new known errors here as the workflow grows.
_KNOWN_ERRORS: tuple[type[Exception], ...] = (
    BranchCreationError,
    CommitAndPrError,
    PreHookError,
    ManifestError,
    OrchestratorError,
)

_RULE = "=" * 60


def _format_elapsed(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


# The entrypoint function in orchestrator.workflow is named `workflow`,
# so its final-result stream event arrives keyed by that name. If you
# rename the @entrypoint function, update this constant.
_ENTRYPOINT_NAME = "workflow"

# Maps "task that just completed" -> "task that's now running", so the
# heartbeat can show what's actually happening rather than what's behind
# us. The `qa` case is ambiguous (PASS goes to commit, FAIL loops back
# to implementation) — handled inline by reading the QaResult.
# Keep in sync with the workflow body in orchestrator/workflow.py.
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


async def _run_with_progress(workflow, input_data, config) -> dict:
    """Stream the workflow with per-task progress markers and a heartbeat.

    Emits:
      - ``done: <task> (Xs)`` after each @task completes, with the time
        that task itself took (not cumulative)
      - ``... running <task> (Ys elapsed)`` every HEARTBEAT_INTERVAL
        seconds while a task is still running — catches long-running
        stages (implementation_task is 5+ min) that otherwise look hung.
        Label is predicted from the workflow's known task order, so it
        tells you what's CURRENTLY running, not what just finished.

    Returns the workflow result dict: either ``{"__interrupt__": [...]}``
    if the workflow paused for plan approval, or the entrypoint's return
    value (the final ``{"status": ..., ...}`` dict).
    """
    heartbeat_interval = float(os.environ.get("HEARTBEAT_INTERVAL", "15"))
    last_event_time = time.monotonic()
    final_result: dict | None = None
    current_label = f"running {_NEXT_STAGE[None]}"
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=heartbeat_interval)
                return
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - last_event_time
                print(
                    f"  ... {current_label} ({_format_elapsed(elapsed)} elapsed)",
                    file=sys.stderr,
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
                    # Strip the _task suffix for readable output —
                    # "implementation" reads better than "implementation_task".
                    name = key.removesuffix("_task")
                    print(
                        f"  done: {name} ({_format_elapsed(task_elapsed)})",
                        file=sys.stderr,
                    )
                    # After decompose, show the task titles the decomposer
                    # produced — the one piece of info that tells the waiting
                    # user whether the plan was interpreted sensibly. getattr
                    # guard: the loop sees every task key, not just decompose.
                    if name == "decompose":
                        tasks = getattr(value, "tasks", None)
                        if tasks:
                            titles = " · ".join(t.title for t in tasks)
                            print(
                                f"    → {len(tasks)} tasks: {titles}",
                                file=sys.stderr,
                            )
                    # Predict what's running now so the heartbeat label
                    # is accurate. qa is ambiguous (PASS → commit, FAIL
                    # → another implementation attempt) — disambiguate
                    # by reading the QaResult.
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


def _print_success(result: dict, thread_id: str) -> None:
    """Format a successful workflow result with the PR URL prominent."""
    print()
    print(_RULE)
    print("Workflow complete")
    print(_RULE)
    print(f"  Branch:    {result['branch']}")
    print(f"  PR:        {result['pr_url']}")
    print(f"  thread_id: {thread_id}")
    print(_RULE)
    _print_usage_banner(result)


def _print_qa_failure(result: dict, thread_id: str) -> None:
    """Format a QA-exhausted workflow result (3 attempts, all FAIL)."""
    print()
    print(_RULE)
    print("Workflow failed after 3 implementation attempts")
    print(_RULE)
    print(f"  Branch:    {result['branch']}")
    print(f"  thread_id: {thread_id}")
    print("  Last QA failures:")
    for line in (result.get("qa_failures") or "").splitlines():
        print(f"    {line}")
    print(_RULE)
    print(
        "The branch and attempted diffs are in your repo; review and decide "
        "what to do with them."
    )
    _print_usage_banner(result)


def _fmt_cost(cost: float | None) -> str:
    if cost is None:
        return "?"
    return f"${cost:.3f}"


def _print_usage_banner(result: dict) -> None:
    usage = result.get("usage")
    if not usage or "by_task" not in usage:
        return

    _THIN = "-" * 60
    print()
    print(_RULE)
    print("Token usage")
    print(_RULE)
    by_task = usage["by_task"]
    for task_name, data in by_task.items():
        cost_str = _fmt_cost(data.get("cost_usd"))
        print(
            f"  {task_name:<18} {data['input_tokens']:>8,} in  /"
            f"  {data['output_tokens']:>6,} out  ({cost_str})"
        )
    total = usage["total"]
    print(_THIN)
    cost_str = _fmt_cost(total.get("cost_usd"))
    print(
        f"  {'TOTAL':<18} {total['input_tokens']:>8,} in  /"
        f"  {total['output_tokens']:>6,} out  ({cost_str})"
    )
    print(_RULE)


def _report_failure(thread_id: str, exc: Exception) -> None:
    """Print a human-readable error and exit non-zero.

    Banner style diverges per error class:
      UserActionError  — tells the user exactly what to do, then how to resume
      RetriableError   — transient; resume_run can be called immediately
      FatalError       — non-retriable; start a fresh run
      everything else  — unexpected; suggest ORCHESTRATOR_DEBUG=1

    ORCHESTRATOR_DEBUG=1 always enables the full traceback.
    """
    if os.environ.get("ORCHESTRATOR_DEBUG"):
        traceback.print_exc()
        print()

    if isinstance(exc, PreHookError):
        print(f"\n{_RULE}", file=sys.stderr)
        print("Pre-hook aborted the workflow", file=sys.stderr)
        print(_RULE, file=sys.stderr)
        print(f"  Hook:   {exc.script!r} (exit {exc.returncode})", file=sys.stderr)
        print(f"  Output: {exc.output}", file=sys.stderr)
        print(f"  Action: {exc.action}", file=sys.stderr)
        print(f"  thread_id: {thread_id}", file=sys.stderr)
    elif isinstance(exc, UserActionError):
        print(f"\n{_RULE}", file=sys.stderr)
        print(f"Workflow paused — action required  [{type(exc).__name__}]", file=sys.stderr)
        print(_RULE, file=sys.stderr)
        print(f"  Error:  {exc}", file=sys.stderr)
        print(f"  Action: {exc.action}", file=sys.stderr)
        print(f"  thread_id: {thread_id}", file=sys.stderr)
        print("Once resolved, call resume_run or re-run with the same thread_id.", file=sys.stderr)
    elif isinstance(exc, RetriableError):
        print(f"\n{_RULE}", file=sys.stderr)
        print("Transient error — safe to retry  [RetriableError]", file=sys.stderr)
        print(_RULE, file=sys.stderr)
        print(f"  Error:  {exc}", file=sys.stderr)
        print(f"  thread_id: {thread_id}", file=sys.stderr)
        print("Call resume_run immediately — no manual action required.", file=sys.stderr)
    elif isinstance(exc, FatalError):
        print(f"\n{_RULE}", file=sys.stderr)
        print("Fatal error — start a fresh run  [FatalError]", file=sys.stderr)
        print(_RULE, file=sys.stderr)
        print(f"  Error:  {exc}", file=sys.stderr)
        print(f"  thread_id: {thread_id}", file=sys.stderr)
        print("This error cannot be retried. Fix the root cause and start fresh.", file=sys.stderr)
    else:
        print(
            f"\nWorkflow failed with unexpected error "
            f"({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        print(f"  thread_id: {thread_id}", file=sys.stderr)
        print(
            "  Run with ORCHESTRATOR_DEBUG=1 to see the full traceback.",
            file=sys.stderr,
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI args. Flags must precede the request text.

    All override flags default to None — that's the signal to
    apply_overrides() to fall back to the env var, then the config file.
    """
    parser = argparse.ArgumentParser(
        prog="implement-feature",
        description="Run the orchestrator end-to-end against a request.",
    )
    parser.add_argument(
        "--no-approve-plan",
        dest="approve_plan",
        action="store_false",
        default=None,
        help="Skip the plan-approval pause for this run.",
    )
    parser.add_argument(
        "--base-branch",
        type=str,
        default=None,
        help="Override the PR base branch for this run.",
    )
    parser.add_argument(
        "request",
        nargs=argparse.REMAINDER,
        help="The feature/fix/refactor request (joined with spaces).",
    )
    return parser.parse_args(argv)


async def run() -> None:
    args = _parse_args(sys.argv[1:])
    request = " ".join(args.request) if args.request else "add a dark mode toggle"
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    Path(".orchestrator").mkdir(exist_ok=True)
    append_run(thread_id, request, source="cli")

    effective_config = apply_overrides(
        load_config(),
        approve_plan=args.approve_plan,
        base_branch=args.base_branch,
    )

    print(f"thread_id: {thread_id}")
    print(f"request:   {request}")

    try:
        async with build_workflow(config=effective_config) as workflow:
            result = await _run_with_progress(workflow, request, config)

            # Plan-approval interrupt loop. Each user reply either approves
            # ("yes") or triggers a re-plan with feedback.
            while "__interrupt__" in result:
                interrupt_val = result["__interrupt__"][0].value
                plan = interrupt_val.get("plan")
                if plan is not None:
                    # Plan-approval interrupt: show the full plan text.
                    print(f"\n--- Plan for approval (thread_id: {thread_id}) ---")
                    print(plan["plan_text"])
                else:
                    # A non-plan gate (branch/impl/pr approval, or an
                    # approval_gate step): just show the prompt.
                    kind = interrupt_val.get("kind", "approval")
                    print(f"\n--- {kind} (thread_id: {thread_id}) ---")
                print("\n" + interrupt_val.get("ask", "Proceed? Reply 'yes'."))
                response = input("> ").strip()
                result = await _run_with_progress(
                    workflow, Command(resume=response), config
                )

            # Workflow returned without an interrupt. Branch on status.
            status = result.get("status")
            if status == "succeeded":
                _print_success(result, thread_id)
            elif status == "failed":
                _print_qa_failure(result, thread_id)
                sys.exit(1)
            elif status == "aborted":
                # An approval_gate step was resumed with an abort.
                print()
                print(_RULE)
                print(f"Workflow aborted at approval gate: {result.get('aborted_at')}")
                print(_RULE)
                print(f"  thread_id: {thread_id}")
                print("Nothing was committed (gates run before the commit line).")
                _print_usage_banner(result)
                sys.exit(1)
            elif status == "no_changes":
                # QA passed but the build produced no diff — nothing to commit, so
                # no PR was opened. Not a failure; exit 0.
                print()
                print(_RULE)
                print("Build passed QA but produced no changes — nothing to commit")
                print(_RULE)
                print(f"  Branch:    {result.get('branch')}")
                print(f"  thread_id: {thread_id}")
                print("No commit and no PR were created (the working tree was clean).")
                _print_usage_banner(result)
            else:
                # Unknown shape — dump raw so we at least see something.
                print("\n--- Result ---")
                print(result)
                print(f"\nthread_id: {thread_id}")
    except Exception as exc:
        _report_failure(thread_id, exc)
        sys.exit(1)


def main() -> None:
    """Sync entrypoint for the `implement-feature` console script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
