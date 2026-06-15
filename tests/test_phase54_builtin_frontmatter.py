"""Phase 54 — a downloaded agent's frontmatter drives the BUILT-IN spine agents.

Phase 53 made frontmatter drive generic [defs.*] ai_agents. This extends the
same plug-and-play rule to the built-ins (plan/implementation/qa/docs/summarize):
drop a prompt into .orchestrator/prompts/<name>.md and its frontmatter
`model`/`tools` become that built-in stage/part's defaults, with an explicit key
in the user's orchestrator.toml still overriding. No frontmatter → today's
defaults (Phase 68b: v2 stage/part form).

The merge lives in config.load_config, so every consumer (which resolves model
via config.resolved_model and reads tools off config.stage(...) / config.part(...))
picks it up.
"""

from pathlib import Path

import pytest

import orchestrator.config as config_mod
import orchestrator.prompt_loader as pl
from orchestrator.agent_frontmatter import AgentFrontmatter
from orchestrator.config import load_config


def _repo(tmp_path: Path, prompts: dict[str, str] | None = None, toml: str | None = None) -> Path:
    """Build a repo with optional .orchestrator/prompts/<name>.md files and an
    optional orchestrator.toml. Returns the toml path for load_config(path=…)."""
    pdir = tmp_path / ".orchestrator" / "prompts"
    pdir.mkdir(parents=True, exist_ok=True)
    for name, body in (prompts or {}).items():
        (pdir / f"{name}.md").write_text(body, encoding="utf-8")
    toml_path = tmp_path / "orchestrator.toml"
    if toml is not None:
        toml_path.write_text(toml, encoding="utf-8")
    return toml_path


@pytest.fixture
def at_root(tmp_path, monkeypatch):
    """Point the prompt-frontmatter lookup at tmp_path."""
    monkeypatch.setattr(pl, "find_project_root", lambda: tmp_path)
    return tmp_path


# --------------------------- load_prompt_frontmatter ---------------------------


def test_override_frontmatter_is_read(at_root):
    _repo(at_root, prompts={"qa": "---\nmodel: opus\ntools: Read, Grep\n---\nBody.\n"})
    fm = pl.load_prompt_frontmatter("qa")
    assert fm.model == "claude-opus-4-8"
    assert fm.allowed_tools == ["Read", "Grep"]


def test_bundled_prompts_have_no_frontmatter(at_root):
    # These shipped prompts stay frontmatter-free, so the merge is a no-op for a
    # repo that doesn't override them. (qa is the deliberate exception — it pins a
    # cheaper model; see test_bundled_qa_pins_sonnet_model.)
    for name in ("planning", "implementation", "docs", "summarize"):
        assert pl.load_prompt_frontmatter(name) == AgentFrontmatter()


def test_bundled_qa_pins_sonnet_model(at_root):
    # qa.md intentionally ships with a `model: claude-sonnet-4-6` frontmatter pin
    # (cost: QA runs on the cheaper Sonnet rather than the Opus default). That pin
    # is the bundled default and a repo's own .orchestrator/prompts/qa.md still
    # overrides it. Only model is set; everything else stays default.
    fm = pl.load_prompt_frontmatter("qa")
    assert fm.model == "claude-sonnet-4-6"
    assert fm == AgentFrontmatter(model="claude-sonnet-4-6")


# --------------------------- merge into built-in config ---------------------------


_FLOW = 'flow = "plan >> decompose >> task-build >> docs >> summarize"\n'


def test_dropped_in_qa_drives_model_and_tools(at_root):
    toml = _repo(
        at_root,
        prompts={"qa": "---\nname: strict\nmodel: opus\ntools: Read, Grep\n---\nQA prompt.\n"},
        toml="",  # empty toml → default pipeline; only frontmatter speaks
    )
    cfg = load_config(path=toml)
    qa = cfg.part("builtin:qa")
    assert qa.allowed_tools == ["Read", "Grep"]
    assert cfg.resolved_model(qa.model) == "claude-opus-4-8"


def test_no_toml_file_still_applies_frontmatter(at_root):
    # A repo with a dropped-in agent but no orchestrator.toml at all.
    _repo(at_root, prompts={"implementation": "---\nmodel: opus\n---\nImpl.\n"})
    cfg = load_config(path=at_root / "orchestrator.toml")  # file does not exist
    impl = cfg.part("builtin:implementation")
    assert cfg.resolved_model(impl.model) == "claude-opus-4-8"
    # tools the frontmatter didn't set keep their default.
    assert impl.allowed_tools == ["Read", "Edit", "Write", "Bash"]


def test_toml_overrides_frontmatter(at_root):
    toml = _repo(
        at_root,
        prompts={"qa": "---\nmodel: opus\ntools: Read, Grep\n---\nQA.\n"},
        toml=_FLOW + '[builtin.qa]\nmodel = "claude-sonnet-4-6"\n',
    )
    cfg = load_config(path=toml)
    qa = cfg.part("builtin:qa")
    assert cfg.resolved_model(qa.model) == "claude-sonnet-4-6"  # TOML wins
    assert qa.allowed_tools == ["Read", "Grep"]                 # tools still frontmatter


def test_no_frontmatter_keeps_defaults(at_root):
    toml = _repo(at_root, prompts={"qa": "Just a QA prompt, no frontmatter.\n"}, toml="")
    cfg = load_config(path=toml)
    qa = cfg.part("builtin:qa")
    assert qa.allowed_tools == ["Read", "Grep", "Bash"]
    assert cfg.resolved_model(qa.model) == cfg.default_model


def test_docs_and_summarize_models_overridable_by_frontmatter(at_root):
    toml = _repo(
        at_root,
        prompts={
            "docs": "---\nmodel: opus\n---\nDocs.\n",
            "summarize": "---\nmodel: sonnet\n---\nSummarize.\n",
        },
        toml="",
    )
    cfg = load_config(path=toml)
    assert cfg.resolved_model(cfg.stage("docs").model) == "claude-opus-4-8"
    assert cfg.resolved_model(cfg.stage("summarize").model) == "claude-sonnet-4-6"


def test_frontmatter_human_in_loop_is_ignored_for_builtins(at_root):
    # human_in_loop is NOT a built-in frontmatter dial — a dropped-in qa agent
    # carrying it must NOT error or change anything (it's simply ignored).
    toml = _repo(
        at_root,
        prompts={"qa": "---\nmodel: opus\nhuman_in_loop: true\n---\nQA.\n"},
        toml="",
    )
    cfg = load_config(path=toml)  # no error
    assert cfg.resolved_model(cfg.part("builtin:qa").model) == "claude-opus-4-8"


def test_human_in_loop_on_a_part_rejected(at_root):
    # human_in_loop is a STAGE dial, not a part field — setting it on [builtin.qa]
    # is fail-loud (PartSpec extra="forbid").
    from orchestrator.pipeline import PipelineError

    toml = _repo(at_root, toml=_FLOW + "[builtin.qa]\nhuman_in_loop = true\n")
    with pytest.raises(PipelineError):
        load_config(path=toml)
