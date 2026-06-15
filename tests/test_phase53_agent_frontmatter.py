"""Phase 53 — agent `.md` frontmatter is honoured, so a file downloaded from
anywhere plugs straight in.

Goal: drop in any agent definition (a Claude Code subagent, a shared prompt)
and have it work without editing — its `model`/`tools` become defaults, unknown
keys (`name`/`description`/`color`/…) are ignored rather than rejected, and the
prompt body is everything after the `---` block. An explicit key on the TOML
entry still overrides the frontmatter (per-project tweak without forking the
prompt).

Two slices:
1. Parser unit tests (`agent_frontmatter`) — the split, the lenient field
   mapping, the Claude Code `tools` string + `model` aliases.
2. Manifest integration — frontmatter resolves into the AiAgentStep at load
   time (so it folds into manifest_hash), TOML overrides win, and a fully
   metadata'd downloaded agent loads clean.
"""

from pathlib import Path

import pytest

from orchestrator.agent_frontmatter import (
    AgentFrontmatter,
    parse_agent_frontmatter,
    split_frontmatter,
)
from orchestrator.manifest import AiAgentStep, load_manifest


# --------------------------- split_frontmatter ---------------------------


def test_split_no_frontmatter_returns_whole_body():
    assert split_frontmatter("just a prompt") == ({}, "just a prompt")


def test_split_extracts_meta_and_body():
    meta, body = split_frontmatter("---\nmodel: sonnet\n---\nThe prompt body.\n")
    assert meta == {"model": "sonnet"}
    assert body == "The prompt body.\n"


def test_split_missing_closing_fence_is_all_body():
    text = "---\nmodel: sonnet\nnever closes\n"
    assert split_frontmatter(text) == ({}, text)


def test_split_malformed_yaml_falls_back_to_body():
    # Unparseable YAML must not raise — the file is kept as the body.
    text = "---\n: : not valid : :\n- broken\n---\nbody\n"
    meta, body = split_frontmatter(text)
    assert meta == {}
    assert body == text  # whole file preserved when the block can't be parsed


def test_split_non_mapping_frontmatter_ignored():
    meta, body = split_frontmatter("---\n- a\n- b\n---\nbody\n")
    assert meta == {}  # a YAML list isn't config
    assert body == "body\n"


# --------------------------- parse_agent_frontmatter ---------------------------


def test_parse_claude_code_subagent_verbatim():
    """A real Claude Code subagent file: comma-string tools, model alias, and
    metadata we don't use — all handled without the user editing anything."""
    text = (
        "---\n"
        "name: code-reviewer\n"
        "description: Reviews code for bugs\n"
        "tools: Read, Grep, Bash\n"
        "model: opus\n"
        "color: blue\n"
        "---\n"
        "You are a meticulous reviewer.\n"
    )
    fm, body = parse_agent_frontmatter(text)
    assert fm.model == "claude-opus-4-8"           # alias → full id
    assert fm.allowed_tools == ["Read", "Grep", "Bash"]  # comma string → list
    assert fm.disallowed_tools is None
    assert body == "You are a meticulous reviewer.\n"   # metadata stripped


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("opus", "claude-opus-4-8"),
        ("sonnet", "claude-sonnet-4-6"),
        ("haiku", "claude-haiku-4-5-20251001"),
        ("Sonnet", "claude-sonnet-4-6"),          # case-insensitive
        ("claude-opus-4-7", "claude-opus-4-7"),    # full id passes through
        ("inherit", None),                          # "no opinion"
        ("", None),
    ],
)
def test_model_alias_resolution(alias, expected):
    fm, _ = parse_agent_frontmatter(f"---\nmodel: {alias}\n---\nbody")
    assert fm.model == expected


def test_tools_as_yaml_list():
    fm, _ = parse_agent_frontmatter("---\ntools: [Read, Edit, Write]\n---\nb")
    assert fm.allowed_tools == ["Read", "Edit", "Write"]


def test_allowed_tools_key_wins_over_tools():
    fm, _ = parse_agent_frontmatter(
        "---\ntools: Read\nallowed_tools: [Read, Edit]\n---\nb"
    )
    assert fm.allowed_tools == ["Read", "Edit"]


def test_disallowed_and_timeout_and_hil():
    fm, _ = parse_agent_frontmatter(
        "---\ndisallowed_tools: Bash\ntimeout: 900\nhuman_in_loop: true\n---\nb"
    )
    assert fm.disallowed_tools == ["Bash"]
    assert fm.timeout == 900
    assert fm.human_in_loop is True


def test_timeout_true_is_not_coerced_to_one():
    fm, _ = parse_agent_frontmatter("---\ntimeout: true\n---\nb")
    assert fm.timeout is None  # a bool is not a valid timeout


def test_empty_frontmatter_is_all_none():
    fm, body = parse_agent_frontmatter("plain prompt, no frontmatter")
    assert fm == AgentFrontmatter()
    assert body == "plain prompt, no frontmatter"


# --------------------------- manifest integration ---------------------------


def _project(tmp_path: Path, agent_body: str, toml: str) -> Path:
    (tmp_path / "agent.md").write_text(agent_body, encoding="utf-8")
    (tmp_path / "orchestrator.toml").write_text(toml, encoding="utf-8")
    return tmp_path


_SEAM_TOML = (
    '[[steps.work]]\n'
    'id = "review"\n'
    'type = "ai_agent"\n'
    'agent = "agent.md"\n'
)


def test_seam_agent_inherits_frontmatter_defaults(tmp_path):
    _project(
        tmp_path,
        "---\nmodel: opus\ntools: Read, Grep\n---\nReview the diff.\n",
        _SEAM_TOML,
    )
    step = load_manifest(project_root=tmp_path).for_seam("work")[0]
    assert isinstance(step, AiAgentStep)
    assert step.model == "claude-opus-4-8"
    assert step.allowed_tools == ["Read", "Grep"]


def test_toml_key_overrides_frontmatter(tmp_path):
    # frontmatter says opus; the TOML entry pins sonnet for this repo. Tools,
    # which the TOML omits, still come from the frontmatter.
    _project(
        tmp_path,
        "---\nmodel: opus\ntools: Read, Grep\n---\nReview.\n",
        _SEAM_TOML + 'model = "claude-sonnet-4-6"\n',
    )
    step = load_manifest(project_root=tmp_path).for_seam("work")[0]
    assert step.model == "claude-sonnet-4-6"      # TOML wins
    assert step.allowed_tools == ["Read", "Grep"]  # frontmatter still supplies tools


def test_unknown_frontmatter_keys_do_not_break_load(tmp_path):
    # The whole point: a downloaded agent with name/description/color just loads.
    _project(
        tmp_path,
        "---\nname: x\ndescription: y\ncolor: red\nmodel: haiku\n---\nPrompt.\n",
        _SEAM_TOML,
    )
    step = load_manifest(project_root=tmp_path).for_seam("work")[0]
    assert step.model == "claude-haiku-4-5-20251001"


def test_no_frontmatter_keeps_toml_and_defaults(tmp_path):
    _project(tmp_path, "Just a prompt, no frontmatter.\n", _SEAM_TOML)
    step = load_manifest(project_root=tmp_path).for_seam("work")[0]
    assert step.model == "claude-sonnet-4-6"  # AiAgentStep default, unchanged
    assert step.allowed_tools is None


def test_def_agent_frontmatter_folds_into_manifest_hash(tmp_path):
    """Resume safety (decision 8): editing the frontmatter changes the resolved
    def, so manifest_hash changes and a mid-run resume refuses."""
    build_toml = (
        '[[steps.work]]\n'
        'id = "loop"\n'
        'type = "build"\n'
        'produce = ["fixer"]\n'
        'ungated = true\n\n'
        '[steps.defs.fixer]\n'
        'type = "ai_agent"\n'
        'agent = "agent.md"\n'
    )
    _project(tmp_path, "---\nmodel: opus\n---\nFix it.\n", build_toml)
    h1 = load_manifest(project_root=tmp_path).manifest_hash()

    (tmp_path / "agent.md").write_text("---\nmodel: sonnet\n---\nFix it.\n", encoding="utf-8")
    h2 = load_manifest(project_root=tmp_path).manifest_hash()
    assert h1 != h2

    # And the resolved def carries the frontmatter model.
    fixer = load_manifest(project_root=tmp_path).defs["fixer"]
    assert fixer.model == "claude-sonnet-4-6"


# --------------------------- built-in prompt overrides ---------------------------


def test_prompt_override_strips_frontmatter(tmp_path, monkeypatch):
    """A built-in prompt override (.orchestrator/prompts/<name>.md) may also be a
    downloaded file — its frontmatter must not leak into the loaded prompt."""
    import orchestrator.prompt_loader as pl

    prompts = tmp_path / ".orchestrator" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "qa.md").write_text(
        "---\nname: my-qa\nmodel: opus\n---\nYou are the QA reviewer.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pl, "find_project_root", lambda: tmp_path)

    prompt = pl.load_prompt("qa")
    assert not prompt.lstrip().startswith("---")   # frontmatter gone
    assert "You are the QA reviewer." in prompt
    assert "my-qa" not in prompt
