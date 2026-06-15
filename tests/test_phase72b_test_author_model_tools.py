"""Phase 72b — test-author model + tools override via prompt frontmatter.

The test-author's model and tools come from its prompt file's frontmatter — the
same mechanism every other built-in agent uses — resolved from the convention/
bundled file or the config.test_author_path override. Frontmatter that says
nothing falls back to default_model + the author-role default tools, so the
unset case is byte-for-byte today's behaviour.
"""

import pytest

from orchestrator import workflow
from orchestrator.agents.test_author import (
    TestAuthorResult,
    author_tests,
    _DEFAULT_TOOLS,
)
from orchestrator.config import OrchestratorConfig


def _cfg(**kw) -> OrchestratorConfig:
    return OrchestratorConfig(tdd=True, test_paths=["**/*.test.js"], **kw)


def _write_prompt(tmp_path, body: str) -> None:
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "ta.md").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def test_model_falls_back_to_default_when_frontmatter_silent():
    # The bundled test-author.md carries no frontmatter → default_model.
    cfg = _cfg(default_model="claude-sonnet-4-6")
    assert workflow._test_author_model(cfg) == "claude-sonnet-4-6"


def test_model_comes_from_override_frontmatter(monkeypatch, tmp_path):
    _write_prompt(tmp_path, "---\nmodel: opus\n---\nYou are a test-author agent.")
    monkeypatch.setattr(workflow, "find_project_root", lambda: tmp_path)
    cfg = _cfg(test_author_path="prompts/ta.md")
    # alias 'opus' is normalised to the full id.
    assert workflow._test_author_model(cfg) == "claude-opus-4-8"


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #


def test_tools_none_when_frontmatter_silent():
    assert workflow._test_author_tools(_cfg()) == (None, None)


def test_tools_come_from_override_frontmatter(monkeypatch, tmp_path):
    _write_prompt(tmp_path, "---\ntools: [Read, Bash]\ndisallowed_tools: [Write]\n---\nbody")
    monkeypatch.setattr(workflow, "find_project_root", lambda: tmp_path)
    allowed, disallowed = workflow._test_author_tools(_cfg(test_author_path="prompts/ta.md"))
    assert allowed == ["Read", "Bash"]
    assert disallowed == ["Write"]


# --------------------------------------------------------------------------- #
# a missing override file degrades to defaults (the prompt loader owns the error)
# --------------------------------------------------------------------------- #


def test_missing_override_file_yields_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "find_project_root", lambda: tmp_path)
    cfg = _cfg(default_model="claude-sonnet-4-6", test_author_path="prompts/gone.md")
    assert workflow._test_author_model(cfg) == "claude-sonnet-4-6"
    assert workflow._test_author_tools(cfg) == (None, None)


# --------------------------------------------------------------------------- #
# author_tests threads the resolved tools through to the runner
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_author_tests_uses_role_default_tools_when_none(monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return TestAuthorResult(testable=True, summary="ok")

    monkeypatch.setattr("orchestrator.agents.test_author.run_structured_agent", fake_run)
    await author_tests("plan", "model")
    assert captured["allowed_tools"] == _DEFAULT_TOOLS
    assert captured["disallowed_tools"] == []


@pytest.mark.asyncio
async def test_author_tests_passes_explicit_tools(monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return TestAuthorResult(testable=True, summary="ok")

    monkeypatch.setattr("orchestrator.agents.test_author.run_structured_agent", fake_run)
    await author_tests("plan", "model", None, ["Read"], ["Write"])
    assert captured["allowed_tools"] == ["Read"]
    assert captured["disallowed_tools"] == ["Write"]
