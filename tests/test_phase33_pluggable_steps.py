"""Phase 33 pluggable-steps tests.

Three layers, all LLM-free:
- manifest: load/validate/hash + frontmatter stripping (pure).
- execute_script: real tiny scripts (success / non-zero / timeout).
- workflow integration: a approval_gate seam fires mid-run and the workflow
  completes; a mid-run manifest edit refuses the resume.

The ai_agent runner's agent loop needs a live model and is not exercised
here; its non-LLM parts (agent-file resolution, frontmatter strip) are.
"""

import uuid
from pathlib import Path

import pytest
from langgraph.types import Command

from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.manifest import (
    ApprovalGateStep,
    AiAgentStep,
    ManifestError,
    ScriptStep,
    StepResult,
    WorkflowManifest,
    load_manifest,
)
from orchestrator.prompt_loader import load_agent_prompt
from orchestrator.steps import StepError, execute_script



# --------------------------- manifest loader ---------------------------


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_load_valid_manifest(tmp_path):
    _write(tmp_path / ".orchestrator/scripts/lint.sh", "#!/bin/sh\nexit 0\n")
    _write(tmp_path / ".orchestrator/agents/docs.md", "You are a doc agent.")
    _write(
        tmp_path / "orchestrator.toml",
        """
[[steps.work]]
id = "lint"
type = "script"
path = ".orchestrator/scripts/lint.sh"

[[steps.work]]
id = "docs"
type = "ai_agent"
agent = ".orchestrator/agents/docs.md"

[[steps.work]]
id = "gate"
type = "approval_gate"
ask = "ok?"
""",
    )
    m = load_manifest(project_root=tmp_path)
    work = m.for_seam("work")
    assert isinstance(work[0], ScriptStep)
    assert isinstance(work[1], AiAgentStep)
    assert isinstance(work[2], ApprovalGateStep)
    assert work[2].ask == "ok?"


def test_no_steps_table_is_empty(tmp_path):
    _write(tmp_path / "orchestrator.toml", 'default_model = "claude-sonnet-4-6"\n')
    m = load_manifest(project_root=tmp_path)
    assert m.is_empty()


def test_unknown_seam_raises(tmp_path):
    _write(
        tmp_path / "orchestrator.toml",
        '[[steps.after_everything]]\nid="x"\ntype="approval_gate"\n',
    )
    with pytest.raises(ManifestError, match="unknown seam"):
        load_manifest(project_root=tmp_path)


def test_duplicate_id_raises(tmp_path):
    _write(
        tmp_path / "orchestrator.toml",
        """
[[steps.work]]
id = "dup"
type = "approval_gate"

[[steps.work]]
id = "dup"
type = "approval_gate"
""",
    )
    with pytest.raises(ManifestError, match="duplicate step id"):
        load_manifest(project_root=tmp_path)


def test_missing_script_raises(tmp_path):
    _write(
        tmp_path / "orchestrator.toml",
        '[[steps.work]]\nid="lint"\ntype="script"\npath=".orchestrator/scripts/nope.sh"\n',
    )
    with pytest.raises(ManifestError, match="script not found"):
        load_manifest(project_root=tmp_path)


def test_unknown_agent_raises(tmp_path):
    _write(
        tmp_path / "orchestrator.toml",
        '[[steps.work]]\nid="docs"\ntype="ai_agent"\nagent=".orchestrator/agents/ghost.md"\n',
    )
    with pytest.raises(ManifestError, match="agent file not found"):
        load_manifest(project_root=tmp_path)


def test_manifest_hash_changes_with_steps():
    a = WorkflowManifest(steps={"work": [ApprovalGateStep(id="g", ask="a")]})
    b = WorkflowManifest(steps={"work": [ApprovalGateStep(id="g", ask="b")]})
    empty = WorkflowManifest()
    assert a.manifest_hash() != b.manifest_hash()
    assert a.manifest_hash() != empty.manifest_hash()
    # Stable across instances.
    assert a.manifest_hash() == WorkflowManifest(
        steps={"work": [ApprovalGateStep(id="g", ask="a")]}
    ).manifest_hash()


def test_load_agent_prompt_strips_frontmatter(tmp_path):
    # The generic ai_agent loader reads a project-root-relative file and returns
    # the body with any leading `---` frontmatter stripped (shared kernel).
    (tmp_path / "fm.md").write_text("---\nname: docs\nmodel: x\n---\nYou are an agent.\n")
    assert load_agent_prompt(tmp_path, "fm.md") == "You are an agent.\n"
    (tmp_path / "plain.md").write_text("No frontmatter here.")
    assert load_agent_prompt(tmp_path, "plain.md") == "No frontmatter here."


def test_load_agent_prompt_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_agent_prompt(Path("/nonexistent"), "nope.md")


# --------------------------- execute_script ---------------------------


@pytest.mark.asyncio
async def test_execute_script_success(tmp_path):
    script = tmp_path / "ok.sh"
    script.write_text("#!/bin/sh\necho hello\nexit 0\n")
    script.chmod(0o755)
    result = await execute_script(ScriptStep(id="ok", path="ok.sh"), tmp_path)
    assert result.ok
    assert result.kind == "script"
    assert "hello" in result.detail


@pytest.mark.asyncio
async def test_execute_script_nonzero_raises(tmp_path):
    script = tmp_path / "fail.sh"
    script.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    script.chmod(0o755)
    with pytest.raises(StepError, match="exit 3"):
        await execute_script(ScriptStep(id="fail", path="fail.sh"), tmp_path)


@pytest.mark.asyncio
async def test_execute_script_timeout_raises(tmp_path):
    script = tmp_path / "slow.sh"
    script.write_text("#!/bin/sh\nsleep 5\n")
    script.chmod(0o755)
    with pytest.raises(StepError, match="timed out"):
        await execute_script(
            ScriptStep(id="slow", path="slow.sh", timeout=1), tmp_path
        )

# NOTE (Phase 68b): the workflow-integration tests (approval_gate seam fires
# mid-run, work step fires, ai_agent human_in_loop pauses, mid-run manifest edit
# refuses resume) were removed — those v1 seam/[[steps.work]] mechanisms no
# longer exist in the workflow (replaced by the v2 pipeline + per-stage dispatch;
# see test_phase56_per_task_loop / test_phase68_*). The pure manifest-loader and
# execute_script tests above stay green against the (still-present) manifest lib.
