"""Phase 72b — config.test_author_path (arbitrary test-author prompt path).

A real TOML knob (`test_author_path`) points the test-author at an arbitrary
prompt file, instead of being pinned to the convention location
(.orchestrator/prompts/test-author.md → bundled default). Backward-compatible:
unset = today's behaviour.

Three seams are covered:
  - config:        the key round-trips, defaults None, and is optional under tdd.
  - prompt loader: load_prompt(name, path_override=…) loads the body from an
                   arbitrary path AND still appends the name's emit-tool footer.
  - workflow:      _test_author_prompt(config) resolves the override path when
                   set, else falls back to the default; author_tests threads a
                   passed system_prompt through to the agent runner.
"""

import pytest

from orchestrator import workflow
from orchestrator.agents.test_author import TestAuthorResult, author_tests
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.prompt_loader import load_prompt


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_test_author_path_defaults_none():
    assert OrchestratorConfig().test_author_path is None


def test_test_author_path_is_optional_under_tdd():
    # Unlike test_paths, the prompt path is optional even when tdd is on — it
    # falls back to the convention/bundled default.
    cfg = OrchestratorConfig(tdd=True, test_paths=["**/*.test.js"])
    assert cfg.test_author_path is None


def test_load_config_round_trips_test_author_path(tmp_path):
    toml = tmp_path / "orchestrator.toml"
    toml.write_text(
        'tdd = true\n'
        'test_paths = ["**/*.test.js"]\n'
        'test_author_path = "prompts/my-author.md"\n',
        encoding="utf-8",
    )
    cfg = load_config(toml)
    assert cfg.test_author_path == "prompts/my-author.md"


def test_unknown_misspelling_still_rejected(tmp_path):
    # extra="forbid" guards typos like `test_author_paths`.
    toml = tmp_path / "orchestrator.toml"
    toml.write_text('test_author_paths = "x.md"\n', encoding="utf-8")
    with pytest.raises(Exception):
        load_config(toml)


# --------------------------------------------------------------------------- #
# prompt loader: path_override
# --------------------------------------------------------------------------- #


def test_load_prompt_path_override_loads_body_and_appends_footer(tmp_path):
    custom = tmp_path / "custom-author.md"
    custom.write_text("CUSTOM AUTHOR BODY", encoding="utf-8")
    out = load_prompt("test-author", path_override=custom)
    assert "CUSTOM AUTHOR BODY" in out          # body came from the override path
    assert "emit_test_author_result" in out     # footer for the name still applied


def test_load_prompt_path_override_strips_frontmatter(tmp_path):
    # The body is frontmatter-stripped the same way as any prompt file.
    custom = tmp_path / "custom-author.md"
    custom.write_text(
        "---\nmodel: claude-opus-4-7\n---\nBODY AFTER FRONTMATTER",
        encoding="utf-8",
    )
    out = load_prompt("test-author", path_override=custom)
    assert "BODY AFTER FRONTMATTER" in out
    assert "model: claude-opus-4-7" not in out


def test_load_prompt_path_override_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="prompt path override not found"):
        load_prompt("test-author", path_override=tmp_path / "nope.md")


# --------------------------------------------------------------------------- #
# workflow: _test_author_prompt resolution
# --------------------------------------------------------------------------- #


def test_test_author_prompt_uses_override_when_set(monkeypatch, tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "my-author.md").write_text(
        "OVERRIDE AUTHOR BODY", encoding="utf-8"
    )
    monkeypatch.setattr(workflow, "find_project_root", lambda: tmp_path)
    cfg = OrchestratorConfig(
        tdd=True, test_paths=["**/*.test.js"], test_author_path="prompts/my-author.md"
    )
    out = workflow._test_author_prompt(cfg)
    assert "OVERRIDE AUTHOR BODY" in out
    assert "emit_test_author_result" in out


def test_test_author_prompt_falls_back_to_default_when_unset():
    cfg = OrchestratorConfig(tdd=True, test_paths=["**/*.test.js"])
    out = workflow._test_author_prompt(cfg)
    # The bundled default body + its footer.
    assert "test-author agent" in out
    assert "emit_test_author_result" in out


def test_test_author_prompt_missing_override_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "find_project_root", lambda: tmp_path)
    cfg = OrchestratorConfig(
        tdd=True, test_paths=["**/*.test.js"], test_author_path="prompts/gone.md"
    )
    with pytest.raises(FileNotFoundError):
        workflow._test_author_prompt(cfg)


# --------------------------------------------------------------------------- #
# agent: author_tests threads the system_prompt through
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_author_tests_passes_system_prompt_through(monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return TestAuthorResult(testable=True, summary="ok")

    monkeypatch.setattr(
        "orchestrator.agents.test_author.run_structured_agent", fake_run
    )
    await author_tests("plan", "model", "MY CUSTOM PROMPT")
    assert captured["system_prompt"] == "MY CUSTOM PROMPT"


@pytest.mark.asyncio
async def test_author_tests_defaults_to_bundled_prompt(monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return TestAuthorResult(testable=True, summary="ok")

    monkeypatch.setattr(
        "orchestrator.agents.test_author.run_structured_agent", fake_run
    )
    await author_tests("plan", "model")  # no system_prompt → bundled default
    assert "test-author agent" in captured["system_prompt"]
    assert "emit_test_author_result" in captured["system_prompt"]
