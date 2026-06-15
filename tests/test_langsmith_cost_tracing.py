"""LangSmith cost-in-trace emission (orchestrator/tracing.py).

LangGraph traces the workflow graph, but the model calls happen outside
LangChain's instrumentation (a bare AsyncAnthropic client + a Claude Code
subprocess), so the trace had no LLM runs and the LangSmith header showed no
cost. These tests cover the bridge: building canonical usage_metadata (with our
own cost) and attaching it as an llm child run under the active trace.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from orchestrator.tracing import emit_llm_run, usage_metadata
from orchestrator.usage import TaskUsage


# ── usage_metadata: token math + cost ────────────────────────────────────────


def test_usage_metadata_cache_tokens_are_additive():
    """Anthropic's input_tokens excludes cache; canonical input_tokens folds the
    cache tokens back in, with the breakdown carried in input_token_details."""
    usage = TaskUsage(
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        cache_creation_tokens=100,
    )

    meta = usage_metadata(usage)

    assert meta["input_tokens"] == 1300  # 1000 + 200 + 100
    assert meta["output_tokens"] == 500
    assert meta["total_tokens"] == 1800
    assert meta["input_token_details"] == {
        "cache_read": 200,
        "ephemeral_5m_input_tokens": 100,
    }


def test_usage_metadata_carries_computed_cost():
    usage = TaskUsage(
        model="claude-sonnet-4-6", input_tokens=1000, output_tokens=500
    )

    meta = usage_metadata(usage)

    assert usage.cost_usd() is not None
    assert meta["total_cost"] == pytest.approx(usage.cost_usd())


def test_usage_metadata_prefers_reported_cost():
    usage = TaskUsage(
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
        reported_cost_usd=0.42,
    )

    assert usage_metadata(usage)["total_cost"] == 0.42


def test_usage_metadata_omits_cost_for_unknown_model():
    """No reported cost and a model that neither litellm nor the price table
    knows → no total_cost key (rather than a bogus 0.0)."""
    usage = TaskUsage(
        model="totally-unknown-model-xyz", input_tokens=10, output_tokens=5
    )

    meta = usage_metadata(usage)

    assert usage.cost_usd() is None
    assert "total_cost" not in meta
    # tokens are still recorded
    assert meta["input_tokens"] == 10


def test_usage_metadata_no_cache_omits_details():
    usage = TaskUsage(
        model="claude-sonnet-4-6", input_tokens=1000, output_tokens=500
    )

    assert "input_token_details" not in usage_metadata(usage)


# ── emit_llm_run: attaches an llm child run to the active trace ───────────────


class _FakeChild:
    def __init__(self):
        self.posted = self.patched = False
        self.ended_with = "UNSET"

    def post(self):
        self.posted = True

    def end(self, outputs=None):
        self.ended_with = outputs

    def patch(self):
        self.patched = True


class _FakeParent:
    def __init__(self):
        self.create_kwargs = None
        self.child = None

    def create_child(self, **kwargs):
        self.create_kwargs = kwargs
        self.child = _FakeChild()
        return self.child


def _patch_run_tree(monkeypatch, value):
    monkeypatch.setattr(
        "langsmith.run_helpers.get_current_run_tree", lambda: value
    )


def test_emit_creates_llm_child_with_usage_and_cost(monkeypatch):
    parent = _FakeParent()
    _patch_run_tree(monkeypatch, parent)
    usage = TaskUsage(
        model="claude-sonnet-4-6",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
    )

    emit_llm_run(
        usage,
        name="emit_plan",
        inputs={"messages": [{"role": "user", "content": "hi"}]},
        outputs={"role": "assistant", "content": {"ok": True}},
    )

    kw = parent.create_kwargs
    assert kw["run_type"] == "llm"
    assert kw["name"] == "emit_plan"
    md = kw["extra"]["metadata"]
    assert md["ls_provider"] == "anthropic"
    assert md["ls_model_name"] == "claude-sonnet-4-6"
    assert md["usage_metadata"]["input_tokens"] == 1200
    assert "total_cost" in md["usage_metadata"]
    # full lifecycle so the run reaches LangSmith with its end_time + outputs
    assert parent.child.posted and parent.child.patched
    assert parent.child.ended_with == {"role": "assistant", "content": {"ok": True}}


def test_emit_noop_when_usage_none(monkeypatch):
    """Returns before touching LangSmith — never builds a trace tree for nothing."""
    sentinel = []
    _patch_run_tree(monkeypatch, sentinel.append)  # would record if called
    emit_llm_run(None, name="x")
    assert sentinel == []


def test_emit_noop_without_active_trace(monkeypatch):
    _patch_run_tree(monkeypatch, None)
    # no parent run tree (tracing off / outside a node) → silent no-op
    emit_llm_run(
        TaskUsage(model="claude-sonnet-4-6", input_tokens=1, output_tokens=1),
        name="x",
    )


def test_emit_never_raises(monkeypatch):
    class _Boom:
        def create_child(self, **kwargs):
            raise RuntimeError("boom")

    _patch_run_tree(monkeypatch, _Boom())
    # observability must not be able to fail a run
    emit_llm_run(
        TaskUsage(model="claude-sonnet-4-6", input_tokens=1, output_tokens=1),
        name="x",
    )


# ── wiring: run_structured_completion emits through the real runner ───────────


class _FakeUsage:
    input_tokens = 1000
    output_tokens = 500
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _FakeToolUse:
    type = "tool_use"
    input = {"foo": "bar"}


class _FakeResponse:
    content = [_FakeToolUse()]
    usage = _FakeUsage()


class _FakeMessages:
    async def create(self, **kwargs):
        return _FakeResponse()


class _FakeAnthropic:
    messages = _FakeMessages()


class _Schema(BaseModel):
    foo: str


class _Result(BaseModel):
    foo: str
    usage: TaskUsage | None = None


@pytest.mark.asyncio
async def test_run_structured_completion_emits_llm_run(monkeypatch):
    from orchestrator.agents import runner

    monkeypatch.setattr(runner, "AsyncAnthropic", lambda: _FakeAnthropic())
    parent = _FakeParent()
    _patch_run_tree(monkeypatch, parent)

    result = await runner.run_structured_completion(
        system_prompt="sys",
        user_message="do it",
        model="claude-sonnet-4-6",
        tool_name="emit_plan",
        tool_description="emit",
        schema=_Schema,
        result_model=_Result,
    )

    assert result.foo == "bar"
    # the call produced an llm run carrying this leg's cost
    md = parent.create_kwargs["extra"]["metadata"]
    assert parent.create_kwargs["run_type"] == "llm"
    assert md["ls_model_name"] == "claude-sonnet-4-6"
    assert md["usage_metadata"]["input_tokens"] == 1000
    assert md["usage_metadata"]["total_cost"] == pytest.approx(result.usage.cost_usd())
