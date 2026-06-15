"""Phase 40 — config schema tests (v2 pipeline form, Phase 68b).

Covers v2 config loading (stage/part round-trip, default_model inheritance), the
fail-loud guards (unknown top-level key, v1 [workflow.*]/[steps.*] rejected with a
migration message, a pipeline missing its flow), the auto-derived PR label, the
runner's wall-clock timeout, and the removal of tool_profile.py. All LLM-free.
"""

import asyncio
import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.pipeline import PipelineError


# --------------------------- v2 schema round-trip ---------------------------


def test_part_roundtrip(tmp_path):
    p = tmp_path / "orchestrator.toml"
    p.write_text(
        "flow = "
        '"plan >> decompose >> task-build >> docs >> summarize"\n'
        "[builtin.implementation]\n"
        'allowed_tools = ["Read", "Bash"]\n'
        'disallowed_tools = ["Write"]\n'
    )
    impl = load_config(p).part("builtin:implementation")
    assert impl.allowed_tools == ["Read", "Bash"]
    assert impl.disallowed_tools == ["Write"]


def test_defaults_when_infra_only(tmp_path):
    # An infra-only config (no flow / pipeline tables) keeps the default pipeline.
    p = tmp_path / "orchestrator.toml"
    p.write_text('db_path = ".x/y.db"\n')
    cfg = load_config(p)
    assert cfg.stage("plan").human_in_loop is True
    assert "Edit" in cfg.part("builtin:implementation").allowed_tools
    assert [s.id for s in cfg.pipeline.stages][:3] == ["plan", "decompose", "task-build"]


def test_default_model_inheritance(tmp_path):
    p = tmp_path / "orchestrator.toml"
    p.write_text(
        'default_model = "claude-opus-4-7"\n'
        "flow = "
        '"plan >> decompose >> task-build >> docs >> summarize"\n'
        "[stage.builtin.docs]\n"
        'type = "ai_agent"\n'
        'model = "claude-haiku-4-5"\n'
    )
    cfg = load_config(p)
    # plan has no model → inherits default_model
    assert cfg.resolved_model(cfg.stage("plan").model) == "claude-opus-4-7"
    # docs sets its own → overrides
    assert cfg.resolved_model(cfg.stage("docs").model) == "claude-haiku-4-5"


def test_stage_roundtrip(tmp_path):
    cfg = OrchestratorConfig()
    assert cfg.stage("docs").model == "claude-haiku-4-5-20251001"
    assert cfg.stage("docs").timeout == 120

    p = tmp_path / "orchestrator.toml"
    p.write_text(
        "flow = "
        '"plan >> decompose >> task-build >> docs >> summarize"\n'
        '[stage.builtin.docs]\ntype = "ai_agent"\nmodel = "claude-sonnet-4-6"\ntimeout = 300\n'
    )
    docs = load_config(p).stage("docs")
    assert docs.model == "claude-sonnet-4-6"
    assert docs.timeout == 300


# --------------------------- fail-loud guards ---------------------------


def test_unknown_top_level_key_rejected(tmp_path):
    p = tmp_path / "orchestrator.toml"
    p.write_text('bogus_key = "x"\n')
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_config(p)


def test_unknown_stage_key_rejected(tmp_path):
    p = tmp_path / "orchestrator.toml"
    p.write_text(
        "flow = "
        '"plan >> decompose >> task-build >> docs >> summarize"\n'
        "[stage.builtin.plan]\nnot_a_field = 1\n"
    )
    with pytest.raises(PipelineError):
        load_config(p)


def test_pipeline_missing_flow_rejected(tmp_path):
    # A stage table without a flow line is a malformed pipeline → fail loud.
    p = tmp_path / "orchestrator.toml"
    p.write_text('[stage.builtin.plan]\ntype = "ai_agent"\n')
    with pytest.raises(PipelineError, match="flow"):
        load_config(p)


# --------------------------- v1 config rejected (migration) ---------------------------


def test_v1_workflow_table_rejected(tmp_path):
    # The whole v1 [workflow.*] dialect is rejected with a migration message.
    p = tmp_path / "orchestrator.toml"
    p.write_text("[workflow.qa]\nmax_retries = 7\n")
    with pytest.raises(ValueError, match="v1 orchestrator.toml"):
        load_config(p)


def test_v1_steps_table_rejected(tmp_path):
    p = tmp_path / "orchestrator.toml"
    p.write_text('[[steps.work]]\nid = "x"\ntype = "approval_gate"\n')
    with pytest.raises(ValueError, match="v1 orchestrator.toml"):
        load_config(p)


# --------------------------- auto-derived PR label ---------------------------


def _fake_gh_run(branch, captured):
    """Fake git_ops._run: rev-parse → branch; gh pr view → no PR; else → URL."""
    def run(args):
        captured.append(args)
        if args[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return SimpleNamespace(stdout=f"{branch}\n", stderr="")
        if args[:3] == ["gh", "pr", "view"]:
            raise subprocess.CalledProcessError(1, args)  # no existing PR
        return SimpleNamespace(stdout="https://github.com/o/r/pull/1\n", stderr="")
    return run


def test_pr_create_derives_label_from_plan_type(monkeypatch):
    from orchestrator import git_ops

    cmds: list[list[str]] = []
    monkeypatch.setattr(git_ops, "_run", _fake_gh_run("feature/x", cmds))
    url = git_ops.pr_create(
        "feature/x", "t", "s", "tp", base_branch="main", plan_type="fix"
    )
    assert url == "https://github.com/o/r/pull/1"
    create = next(c for c in cmds if c[:3] == ["gh", "pr", "create"])
    assert "--label" in create
    assert create[create.index("--label") + 1] == "bug"  # fix → bug


def test_pr_create_unknown_type_no_label(monkeypatch):
    from orchestrator import git_ops

    cmds: list[list[str]] = []
    monkeypatch.setattr(git_ops, "_run", _fake_gh_run("feature/x", cmds))
    git_ops.pr_create(
        "feature/x", "t", "s", "tp", base_branch="main", plan_type="mystery"
    )
    create = next(c for c in cmds if c[:3] == ["gh", "pr", "create"])
    assert "--label" not in create


# --------------------------- runner wall-clock timeout ---------------------------


class _DummyServer:
    pass


@pytest.mark.asyncio
async def test_runner_timeout_raises_fatal(monkeypatch):
    from orchestrator.agents import runner as runner_mod
    from orchestrator.errors import FatalError

    async def slow_query(prompt, options):
        await asyncio.sleep(5)
        yield None  # never reached — wait_for cancels first

    monkeypatch.setattr(
        runner_mod, "create_sdk_mcp_server", lambda **k: _DummyServer()
    )
    monkeypatch.setattr(runner_mod, "query", slow_query)

    with pytest.raises(FatalError, match="timed out"):
        await runner_mod.run_structured_agent(
            system_prompt="s",
            user_message="m",
            model="claude-sonnet-4-6",
            allowed_tools=[],
            disallowed_tools=[],
            cwd=Path("."),
            timeout=0.05,
            emit_tool_name="emit_x",
            emit_tool_description="d",
            emit_tool_fields={"summary": str},
            result_factory=lambda c, u: c,
        )


# --------------------------- tool_profile.py removed ---------------------------


def test_tool_profile_module_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("orchestrator.tool_profile")
