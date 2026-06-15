"""Phase 48 — AiAgentStep's `dir` + `agent` split merged into one `agent` path.

`agent` is now a single project-root-relative path (full filename included);
the separate `dir` field is gone. A stale `dir = ...` in config is rejected at
load time (extra="forbid") rather than being silently ignored.
"""

import pytest

from orchestrator.manifest import (
    AiAgentStep,
    ManifestError,
    _agent_file,
    load_manifest,
)


def _write(p, body="agent prompt"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_agent_file_resolves_single_path(tmp_path):
    step = AiAgentStep(id="x", agent="team/agents/docs.md")
    assert _agent_file(tmp_path, step) == tmp_path / "team/agents/docs.md"


def test_nested_agent_path_loads(tmp_path):
    _write(tmp_path / "team/agents/docs.md")
    (tmp_path / "orchestrator.toml").write_text(
        '[[steps.work]]\n'
        'id = "docs"\n'
        'type = "ai_agent"\n'
        'agent = "team/agents/docs.md"\n',
        encoding="utf-8",
    )
    step = load_manifest(project_root=tmp_path).for_seam("work")[0]
    assert isinstance(step, AiAgentStep)
    assert step.agent == "team/agents/docs.md"


def test_stale_dir_key_rejected(tmp_path):
    # The merged-away `dir` must fail loud, not be silently ignored.
    _write(tmp_path / ".orchestrator/agents/docs.md")
    (tmp_path / "orchestrator.toml").write_text(
        '[[steps.work]]\n'
        'id = "docs"\n'
        'type = "ai_agent"\n'
        'agent = "docs.md"\n'
        'dir = ".orchestrator/agents"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="(?i)extra|dir"):
        load_manifest(project_root=tmp_path)


def test_missing_agent_file_fails_loud(tmp_path):
    (tmp_path / "orchestrator.toml").write_text(
        '[[steps.work]]\n'
        'id = "docs"\n'
        'type = "ai_agent"\n'
        'agent = "team/agents/ghost.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="agent file not found"):
        load_manifest(project_root=tmp_path)
