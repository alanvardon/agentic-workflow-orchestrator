"""Emit per-agent LLM runs into the active LangSmith trace.

LangGraph auto-traces the workflow *graph* when ``LANGSMITH_TRACING=true``, but
the actual model calls happen outside LangChain's instrumentation: the
structured agents go through a bare ``AsyncAnthropic`` client
(``run_structured_completion``) and the implementation agent runs in a Claude
Code subprocess (``run_structured_agent`` → ``claude_agent_sdk.query``). Neither
is auto-instrumented, so the trace tree carried no LLM-type child runs — and the
LangSmith header rolls up cost from LLM runs, so it showed nothing.

This module bridges that gap. Every agent call already produces a ``TaskUsage``;
``emit_llm_run`` attaches an ``llm``-type child run under the current LangGraph
node carrying the token counts **and our own computed cost**. The cost is passed
directly as ``total_cost`` in ``usage_metadata`` (LangSmith honours a
client-supplied cost — see ``ExtractedUsageMetadata``), so the header is correct
even for model IDs that are not in LangSmith's server-side price map.

Observability must never break a run: ``emit_llm_run`` swallows its own errors
and no-ops when tracing is off or there is no active trace.
"""

from __future__ import annotations

import logging

from orchestrator.usage import TaskUsage

logger = logging.getLogger(__name__)


def usage_metadata(usage: TaskUsage) -> dict:
    """Build LangSmith-canonical ``usage_metadata`` from a ``TaskUsage``.

    Token convention matches LangSmith's own Anthropic normaliser: cache tokens
    are *additive* — Anthropic's reported ``input_tokens`` excludes cache, so the
    canonical ``input_tokens`` is base + cache_read + cache_creation, with the
    breakdown carried (informationally) in ``input_token_details``.

    Cost, when known, is supplied directly as ``total_cost`` so LangSmith does
    not have to resolve it from a model price map (our future model IDs are not
    in it). ``cost_usd()`` already prefers the SDK's reported cost, then litellm,
    then the hardcoded table — so this stays the single source of truth.
    """
    base_input = usage.input_tokens
    cache_read = usage.cache_read_tokens
    cache_creation = usage.cache_creation_tokens
    adjusted_input = base_input + cache_read + cache_creation

    meta: dict = {
        "input_tokens": adjusted_input,
        "output_tokens": usage.output_tokens,
        "total_tokens": adjusted_input + usage.output_tokens,
    }

    details: dict = {}
    if cache_read:
        details["cache_read"] = cache_read
    if cache_creation:
        details["ephemeral_5m_input_tokens"] = cache_creation
    if details:
        meta["input_token_details"] = details

    cost = usage.cost_usd()
    if cost is not None:
        meta["total_cost"] = cost

    return meta


def emit_llm_run(
    usage: TaskUsage | None,
    *,
    name: str,
    inputs: dict | None = None,
    outputs: dict | None = None,
) -> None:
    """Attach an ``llm``-type child run carrying usage + cost to the live trace.

    No-op when ``usage`` is ``None`` or no LangSmith trace is active — i.e.
    tracing is off, or this runs outside a LangGraph node so there is no parent
    run tree to nest under. Never raises: observability must not be able to fail
    a workflow run.
    """
    if usage is None:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        parent = get_current_run_tree()
        if parent is None:
            return

        child = parent.create_child(
            name=name,
            run_type="llm",
            inputs=inputs or {},
            extra={
                "metadata": {
                    "ls_provider": "anthropic",
                    "ls_model_name": usage.model,
                    "usage_metadata": usage_metadata(usage),
                }
            },
        )
        child.post()
        child.end(outputs=outputs or {})
        child.patch()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("LangSmith usage emit skipped: %s", exc)
