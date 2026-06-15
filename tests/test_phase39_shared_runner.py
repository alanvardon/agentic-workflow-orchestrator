"""Phase 39 — shared agent-loop runner tests.

LLM-free. The runner's agent loop normally needs a live model; here we patch
`query` and `create_sdk_mcp_server` in orchestrator.agents.runner so we can:

  - drive the emit tool deterministically (factory wiring), and
  - simulate the agent never emitting (fail-closed).

Plus a pure unit test for usage extraction, and a no-LLM check that QA's scripted
gate short-circuits to FAIL *before* the runner is ever called.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import ResultMessage

from orchestrator.agents import runner as runner_mod
from orchestrator.agents.runner import _extract_usage, run_structured_agent
from orchestrator.errors import FatalError


def _result_msg(usage=None, cost=None) -> ResultMessage:
    """A minimal real ResultMessage — the runner's loop checks isinstance, so a
    genuine instance (not a stand-in) is required."""
    return ResultMessage(
        subtype="success",
        duration_ms=0,
        duration_api_ms=0,
        is_error=False,
        num_turns=1,
        session_id="test-session",
        total_cost_usd=cost,
        usage=usage,
    )


_FAKE_USAGE = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 10,
    "cache_creation_input_tokens": 5,
}


class _DummyServer:
    pass


def _patch_sdk(monkeypatch, *, emit_args):
    """Patch the runner's SDK calls so the loop runs without a live model.

    `create_sdk_mcp_server` is patched to grab the in-process emit tool. `query`
    is patched to an async generator: if `emit_args` is not None it invokes the
    grabbed tool's handler (simulating the model calling emit), then yields one
    ResultMessage; if `emit_args` is None it yields without ever emitting.
    """
    grabbed: dict = {}

    def fake_create_server(name, version, tools):
        grabbed["tool"] = tools[0]   # the SdkMcpTool produced by @tool(...)
        return _DummyServer()

    async def fake_query(prompt, options):
        if emit_args is not None:
            await grabbed["tool"].handler(emit_args)
        yield _result_msg(usage=_FAKE_USAGE, cost=0.0123)

    monkeypatch.setattr(runner_mod, "create_sdk_mcp_server", fake_create_server)
    monkeypatch.setattr(runner_mod, "query", fake_query)


async def _run(**overrides):
    kwargs = dict(
        system_prompt="sys",
        user_message="msg",
        model="claude-sonnet-4-6",
        allowed_tools=["Read"],
        disallowed_tools=[],
        cwd=Path("."),
        emit_tool_name="emit_qa_result",
        emit_tool_description="emit",
        emit_tool_fields={"result": str, "failures": str},
        result_factory=lambda c, u: ("RESULT", c, u),
    )
    kwargs.update(overrides)
    return await run_structured_agent(**kwargs)


# --------------------------- fail-closed ---------------------------


@pytest.mark.asyncio
async def test_run_structured_agent_fail_closed(monkeypatch):
    # Model never calls the emit tool → captured stays empty → FatalError.
    _patch_sdk(monkeypatch, emit_args=None)
    with pytest.raises(FatalError, match="did not call emit_qa_result"):
        await _run()


# --------------------------- factory wiring ---------------------------


@pytest.mark.asyncio
async def test_run_structured_agent_factory_wiring(monkeypatch):
    # The captured args reach result_factory, and usage flows through.
    from orchestrator.agents.qa import QaResult

    _patch_sdk(monkeypatch, emit_args={"result": "FAIL", "failures": "bad diff"})
    result = await _run(
        result_factory=lambda c, u: QaResult(
            result=c["result"], failures=(c.get("failures") or None), usage=u
        ),
    )
    assert isinstance(result, QaResult)
    assert result.result == "FAIL"
    assert result.failures == "bad diff"
    assert result.usage is not None
    assert result.usage.input_tokens == 100
    assert result.usage.reported_cost_usd == 0.0123


@pytest.mark.asyncio
async def test_run_structured_agent_empty_failures_becomes_none(monkeypatch):
    # PASS with an empty failures string → factory maps "" to None (the QA
    # contract preserved across the refactor).
    from orchestrator.agents.qa import QaResult

    _patch_sdk(monkeypatch, emit_args={"result": "PASS", "failures": ""})
    result = await _run(
        result_factory=lambda c, u: QaResult(
            result=c["result"], failures=(c.get("failures") or None), usage=u
        ),
    )
    assert result.result == "PASS"
    assert result.failures is None


# --------------------------- scripted-gate short-circuit ---------------------------


@pytest.mark.asyncio
async def test_qa_scripted_gate_short_circuits_before_runner(monkeypatch):
    # A failing scripted gate must return FAIL without ever invoking the runner.
    from orchestrator.agents import qa as qa_mod
    from orchestrator.agents.planning import PlanResult

    monkeypatch.setattr(
        qa_mod, "load_config",
        lambda: SimpleNamespace(qa=SimpleNamespace(scripts_dir="x", scripts_timeout=60)),
    )
    monkeypatch.setattr(
        qa_mod, "run_qa_scripts",
        lambda **k: SimpleNamespace(passed=False, failure_report="script X failed"),
    )

    called = {"runner": False}

    async def boom(**k):
        called["runner"] = True
        raise AssertionError("runner must not run when the scripted gate fails")

    monkeypatch.setattr(qa_mod, "run_structured_agent", boom)

    result = await qa_mod.qa(
        PlanResult(title="t", type="feature", plan_text="p"), "claude-sonnet-4-6"
    )
    assert result.result == "FAIL"
    assert result.failures == "script X failed"
    assert called["runner"] is False


# --------------------------- usage extraction ---------------------------


def test_extract_usage_maps_all_fields():
    msg = _result_msg(
        usage={
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
        },
        cost=0.5,
    )
    u = _extract_usage(msg, "claude-sonnet-4-6")
    assert u.model == "claude-sonnet-4-6"
    assert u.input_tokens == 1
    assert u.output_tokens == 2
    assert u.cache_read_tokens == 3
    assert u.cache_creation_tokens == 4
    assert u.reported_cost_usd == 0.5


def test_extract_usage_none_when_absent():
    assert _extract_usage(None, "m") is None
    assert _extract_usage(_result_msg(usage=None), "m") is None
