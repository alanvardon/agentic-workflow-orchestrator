"""Phase 46a — ai_agent defs are first-class producers/gates.

AiAgentStep gains optional allowed_tools / disallowed_tools / timeout, threaded
through execute_ai_agent into run_structured_agent. When allowed_tools is unset,
the role default applies (read-only as a gate, write tools as a producer).
"""

import pytest

from orchestrator.manifest import AiAgentStep, load_manifest
from orchestrator.steps import StepError, execute_ai_agent


def _write_agent(tmp_path, name="impl.md", body="You are an agent."):
    d = tmp_path / ".orchestrator" / "agents"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


def _capture_runner(monkeypatch):
    """Stub run_structured_agent to record its kwargs and return a StepResult."""
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return kwargs["result_factory"](
            {"summary": "ok", "passed": True, "detail": ""}, None
        )

    monkeypatch.setattr("orchestrator.steps.run_structured_agent", fake_run)
    return captured


@pytest.mark.asyncio
async def test_custom_tools_and_timeout_threaded(monkeypatch, tmp_path):
    _write_agent(tmp_path)
    captured = _capture_runner(monkeypatch)
    step = AiAgentStep(
        id="x", agent=".orchestrator/agents/impl.md",
        allowed_tools=["Read", "Bash"], disallowed_tools=["Write"], timeout=42,
    )
    await execute_ai_agent(step, tmp_path, "plan")
    assert captured["allowed_tools"] == ["Read", "Bash"]
    assert captured["disallowed_tools"] == ["Write"]
    assert captured["timeout"] == 42


@pytest.mark.asyncio
async def test_missing_agent_file_raises_step_error(tmp_path):
    # The unified loader raises FileNotFoundError; execute_ai_agent re-wraps it
    # as StepError so the step's single failure type is preserved.
    step = AiAgentStep(id="x", agent=".orchestrator/agents/missing.md")
    with pytest.raises(StepError, match="agent file not found"):
        await execute_ai_agent(step, tmp_path, "plan")


@pytest.mark.asyncio
async def test_producer_default_tools_when_unset(monkeypatch, tmp_path):
    _write_agent(tmp_path)
    captured = _capture_runner(monkeypatch)
    step = AiAgentStep(id="x", agent=".orchestrator/agents/impl.md")
    await execute_ai_agent(step, tmp_path, "plan")
    # Producer role default: write tools, no denylist, no timeout.
    assert captured["allowed_tools"] == ["Read", "Edit", "Write", "Bash", "Grep"]
    assert captured["disallowed_tools"] == []
    assert captured["timeout"] is None


@pytest.mark.asyncio
async def test_gate_default_tools_are_read_only(monkeypatch, tmp_path):
    _write_agent(tmp_path)
    captured = _capture_runner(monkeypatch)
    step = AiAgentStep(id="x", agent=".orchestrator/agents/impl.md")
    await execute_ai_agent(step, tmp_path, "plan", as_gate=True)
    # Gate role default stays read-only.
    assert captured["allowed_tools"] == ["Read", "Bash", "Grep"]


def test_manifest_roundtrips_tool_config(tmp_path):
    # An ai_agent step with tool/timeout config parses through load_manifest.
    _write_agent(tmp_path)
    (tmp_path / "orchestrator.toml").write_text(
        """
[[steps.work]]
id = "x"
type = "ai_agent"
agent = ".orchestrator/agents/impl.md"
allowed_tools = ["Read", "Edit"]
disallowed_tools = ["Bash"]
timeout = 99
""",
        encoding="utf-8",
    )
    m = load_manifest(project_root=tmp_path)
    step = m.for_seam("work")[0]
    assert isinstance(step, AiAgentStep)
    assert step.allowed_tools == ["Read", "Edit"]
    assert step.disallowed_tools == ["Bash"]
    assert step.timeout == 99


def test_tool_config_defaults_unset():
    step = AiAgentStep(id="x", agent="d/a.md")
    assert step.allowed_tools is None
    assert step.disallowed_tools == []
    assert step.timeout is None
