"""Shared agent-loop runner.

`implementation.py`, `qa.py`, and `steps.execute_ai_agent` all ran the SAME
agent-loop body and differed only in their result schema:

  1. a closure-captured `@tool` for structured output
  2. `create_sdk_mcp_server(...)`
  3. `ClaudeAgentOptions(...)` assembly, pinning the emit tool into allowed_tools
  4. the `async for msg in query(...)` loop, capturing the ResultMessage
  5. a fail-closed guard: a missing emit must raise, never default a verdict
  6. `TaskUsage` extraction from `result_msg.usage`

Items 1–6 are genuine duplication and live here now. The per-agent **result
model** is NOT duplication — it is the contract the workflow depends on, so it
stays typed in each caller, supplied via `result_factory`. Each agent shrinks to
a thin wrapper that passes its prompt, tools, emit-tool schema, and factory.

Fail-closed is centralised: if the agent never calls its emit tool, the runner
raises `FatalError`. (`steps.execute_ai_agent` catches and re-wraps this as its
own `StepError` to preserve that module's contract.)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, TypeVar

from anthropic import AsyncAnthropic
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    create_sdk_mcp_server,
    query,
    tool,
)
from pydantic import BaseModel

from orchestrator.errors import FatalError
from orchestrator.tracing import emit_llm_run
from orchestrator.transcript import read_api_error_cause
from orchestrator.usage import TaskUsage

R = TypeVar("R")
# Result models for run_structured_completion carry a settable `usage` field.
M = TypeVar("M", bound=BaseModel)


def _extract_usage(result_msg: ResultMessage | None, model: str) -> TaskUsage | None:
    """Build a TaskUsage from the SDK's ResultMessage, or None if absent.

    Identical logic previously inlined in all three call sites.
    """
    if result_msg is None or not result_msg.usage:
        return None
    u = result_msg.usage
    return TaskUsage(
        model=model,
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read_tokens=u.get("cache_read_input_tokens", 0),
        cache_creation_tokens=u.get("cache_creation_input_tokens", 0),
        reported_cost_usd=result_msg.total_cost_usd,
    )


def _usage_from_completion(usage, model: str) -> TaskUsage:
    """Build a TaskUsage from a raw Anthropic `response.usage` object.

    The completion path's counterpart to `_extract_usage` (which reads the agent
    SDK's dict-shaped ResultMessage.usage). Here `usage` is the SDK object whose
    cache fields may be absent on older API shapes — hence the getattr guards.
    """
    return TaskUsage(
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


async def run_structured_completion(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    tool_name: str,
    tool_description: str,
    schema: type[BaseModel],
    result_model: type[M],
    max_tokens: int = 4096,
) -> M:
    """Run one forced-tool-use Anthropic completion as structured output.

    The raw-SDK sibling of `run_structured_agent`: instead of an agent loop with
    file tools, this is a single `messages.create` that forces the model to call a
    fake emit tool (`tool_choice`), so the response shape is guaranteed. The tool's
    input is validated into `result_model`, and the call's token usage is attached
    to the result's `.usage` field (every result model here carries one).

    `schema` is the emit tool's input_schema — typically a sub-model that EXCLUDES
    `usage`/`schema_version` (e.g. _PlanSchema), so the model is never asked to fill
    those in; `result_model` is the full type returned (e.g. PlanResult).

    This is NOT mergeable with run_structured_agent — different SDK, different usage
    shape, no tools/cwd. They are deliberate siblings.
    """
    client = AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        tools=[
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": schema.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user_message}],
    )
    tool_use = next(block for block in response.content if block.type == "tool_use")
    result = result_model.model_validate(tool_use.input)
    result.usage = _usage_from_completion(response.usage, model)
    # Surface this call's tokens + cost as an LLM run in the active LangSmith
    # trace (the bare AsyncAnthropic client is not auto-instrumented).
    emit_llm_run(
        result.usage,
        name=tool_name,
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        },
        outputs={"role": "assistant", "content": tool_use.input},
    )
    return result


async def run_structured_agent(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    allowed_tools: list[str],
    disallowed_tools: list[str],
    cwd: Path,
    timeout: int | None = None,
    emit_tool_name: str,
    emit_tool_description: str,
    emit_tool_fields: dict[str, type],
    result_factory: Callable[[dict, TaskUsage | None], R],
) -> R:
    """Run one agent-loop turn-cycle that ends when the agent calls its emit
    tool, then build a typed result from the captured args.

    The emit tool is created in-process and its args are captured via a closure
    into `captured`. The pinned MCP tool name is appended to `allowed_tools` here,
    so callers pass only their own (operator-configurable) tools.

    Fail-closed: if the agent never calls the emit tool, raise FatalError.
    """
    captured: dict = {}

    @tool(emit_tool_name, emit_tool_description, emit_tool_fields)
    async def _emit(args: dict) -> dict:
        captured.update(args)
        return {"content": [{"type": "text", "text": "Captured. You may stop now."}]}

    server = create_sdk_mcp_server(
        name="orchestrator", version="1.0.0", tools=[_emit]
    )
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools + [f"mcp__orchestrator__{emit_tool_name}"],
        disallowed_tools=disallowed_tools,
        mcp_servers={"orchestrator": server},
        cwd=str(cwd),
        permission_mode="acceptEdits",
        model=model,
        setting_sources=["project"],
    )

    result_msg: ResultMessage | None = None
    # Captured from the FIRST streamed message that carries a session id (the init
    # SystemMessage), NOT only the ResultMessage — on a hard failure (e.g. a
    # billing_error) the SDK raises before any ResultMessage is yielded, so the id
    # we need to find the right transcript would otherwise be lost. None until seen;
    # the transcript feeder then falls back to the newest transcript by mtime.
    session_id: str | None = None

    def _capture_session_id(msg) -> None:
        nonlocal session_id
        if session_id is not None:
            return
        data = getattr(msg, "data", None)
        if isinstance(data, dict) and data.get("session_id"):
            session_id = data["session_id"]
            return
        sid = getattr(msg, "session_id", None)
        if sid:
            session_id = sid

    async def _consume() -> None:
        nonlocal result_msg
        async for msg in query(prompt=user_message, options=options):
            _capture_session_id(msg)
            if isinstance(msg, ResultMessage):
                result_msg = msg

    def _fatal_from_sdk(exc: ClaudeSDKError) -> FatalError:
        # The SDK has already discarded the ProcessError stderr (query.py), so the
        # real cause lives only in the CLI transcript. Read its tail and fold the
        # API-error text into the message + attach the structured cause, so it
        # travels to the audit log, error.md, and run_status downstream.
        cause = read_api_error_cause(session_id, cwd)
        detail = f" — {cause['text']}" if cause and cause.get("text") else ""
        err = FatalError(
            f"agent run failed before calling {emit_tool_name}: {exc}{detail}"
        )
        err.cause = cause
        return err

    # Optional wall-clock timeout over the whole agent loop. None = no limit. On
    # expiry asyncio cancels the query;
    # we surface a FatalError so the run aborts with a clear reason. This is a
    # wall-clock bound, NOT the SDK's max_turns (a turn count) — different knob.
    # A ClaudeSDKError (e.g. the subprocess exiting non-zero on a billing failure)
    # is funnelled through the transcript feeder so the real cause survives.
    if timeout is not None:
        try:
            await asyncio.wait_for(_consume(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            err = FatalError(
                f"agent timed out after {timeout}s before calling {emit_tool_name}"
            )
            err.cause = read_api_error_cause(session_id, cwd)
            raise err from exc
        except ClaudeSDKError as exc:
            raise _fatal_from_sdk(exc) from exc
    else:
        try:
            await _consume()
        except ClaudeSDKError as exc:
            raise _fatal_from_sdk(exc) from exc

    if not captured:
        raise FatalError(f"agent did not call {emit_tool_name}")

    usage = _extract_usage(result_msg, model)
    # The implementation agent runs in a Claude Code subprocess (query()), which
    # LangSmith cannot see; emit the cumulative tokens + cost as one LLM run so
    # the trace header rolls up cost for this leg too.
    emit_llm_run(
        usage,
        name=emit_tool_name,
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        },
        outputs={"role": "assistant", "content": captured},
    )
    return result_factory(captured, usage)
