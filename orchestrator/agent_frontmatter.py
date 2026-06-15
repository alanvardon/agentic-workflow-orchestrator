"""Parse YAML frontmatter from an agent `.md` so a file downloaded from anywhere
plugs straight into the orchestrator — no editing required.

The whole point: a user drops in an agent definition (a Claude Code subagent, a
shared team prompt, whatever) and it just works. Its `model` and `tools` are
honoured as the agent's defaults, the prompt body is everything after the `---`
block, and any keys we don't recognise (`name`, `description`, `color`, …) are
ignored rather than rejected.

So this is deliberately LENIENT — the opposite of `extra="forbid"`. A real
downloaded agent carries metadata we have no use for; forcing the user to strip
it to avoid a load error would defeat the plug-and-play goal. We read what we
understand and quietly skip the rest.

Resolution order (applied in `manifest.py`): frontmatter provides the default;
an explicit key on the `[steps.defs.*]` / seam TOML entry overrides it. So the
shared `.md` stays canonical, and a repo can swap the model or tighten tools in
one line of TOML without forking the prompt.
"""

from __future__ import annotations

import re

import yaml
from pydantic import BaseModel

# Bare model aliases (Claude Code subagent style) → current full model IDs.
# A value that isn't an alias is passed through unchanged (assumed a full ID);
# "inherit"/"" means "no opinion" → fall back to the TOML/built-in default.
_MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

# Frontmatter keys we map onto AiAgentStep fields. `tools` is the Claude Code
# spelling for the allow-list; `allowed_tools` (our own name) wins if both exist.
_OPERATIONAL_FIELDS = ("model", "allowed_tools", "disallowed_tools", "timeout", "human_in_loop")


class AgentFrontmatter(BaseModel):
    """The operational dials we lift out of an agent's frontmatter. Every field
    is optional — `None` means "frontmatter said nothing", so the TOML value or
    the field default stands."""

    model: str | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    timeout: int | None = None
    human_in_loop: bool | None = None


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading `---` YAML block off `text`.

    Returns `(meta, body)`. `meta` is `{}` when there is no frontmatter, the
    block is malformed, or the closing `---` fence is missing — in every such
    case the original text is returned as the body, so a file is never lost to a
    parse error (it just isn't treated as having config).
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            raw = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            try:
                meta = yaml.safe_load(raw)
            except yaml.YAMLError:
                return {}, text
            return (meta if isinstance(meta, dict) else {}), body
    return {}, text  # no closing fence → it's all body


def _as_tool_list(value: object) -> list[str] | None:
    """Coerce a tools value to a clean list. Accepts a YAML list
    (`[Read, Edit]`) or the Claude Code comma/space string (`"Read, Edit"`)."""
    if isinstance(value, str):
        parts = [t.strip() for t in re.split(r"[,\s]+", value) if t.strip()]
        return parts or None
    if isinstance(value, (list, tuple)):
        parts = [str(t).strip() for t in value if str(t).strip()]
        return parts or None
    return None


def _normalise_model(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    m = value.strip()
    if not m or m.lower() == "inherit":
        return None
    return _MODEL_ALIASES.get(m.lower(), m)


def parse_agent_frontmatter(text: str) -> tuple[AgentFrontmatter, str]:
    """Parse an agent file into `(frontmatter_config, prompt_body)`.

    Lenient: unknown keys are ignored, and only the operational dials we
    understand are lifted out. `tools` (Claude Code) and `allowed_tools` (ours)
    both feed the allow-list, the latter winning if both are present."""
    meta, body = split_frontmatter(text)
    lower = {str(k).lower(): v for k, v in meta.items()}

    timeout = lower.get("timeout")
    # bool is an int subclass — exclude it so `timeout: true` doesn't become 1.
    timeout = int(timeout) if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) else None

    hil = lower.get("human_in_loop")

    fm = AgentFrontmatter(
        model=_normalise_model(lower.get("model")),
        allowed_tools=_as_tool_list(lower.get("allowed_tools", lower.get("tools"))),
        disallowed_tools=_as_tool_list(lower.get("disallowed_tools")),
        timeout=timeout,
        human_in_loop=hil if isinstance(hil, bool) else None,
    )
    return fm, body
