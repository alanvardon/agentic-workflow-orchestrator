"""Prompt loader.

The single place that turns a prompt `.md` on disk into a system-prompt body.
Two entry points share one kernel (read_prompt_body → frontmatter-stripped body):

- load_prompt(name): a BUILT-IN agent ('planning', 'qa', …), resolved by name.
  Resolution order:
    1. .orchestrator/prompts/{name}.md in the target repo (cwd at runtime)
    2. orchestrator/prompts/{name}.md bundled with this package
  This lets any repo override the default prompts by dropping files into
  .orchestrator/prompts/ without touching the orchestrator package itself.
  A tool-call footer ("When done") is appended after loading for agents that
  require one — it is NOT part of the prompt file, so overrides stay plug-and-
  play: write your custom persona/rules/checklist and the orchestrator handles
  the structured-output wiring automatically.

- load_agent_prompt(project_root, agent): a GENERIC ai_agent step, resolved by
  explicit project-root-relative path. No override search, no footer — just the
  body. Used by steps.execute_ai_agent.

Both strip any leading `---` frontmatter the same way (its model/tools are
honoured separately via load_prompt_frontmatter / manifest frontmatter parsing).
"""

from pathlib import Path

from orchestrator.agent_frontmatter import (
    AgentFrontmatter,
    parse_agent_frontmatter,
    split_frontmatter,
)
from orchestrator.paths import find_project_root

# Bundled defaults live next to this file in orchestrator/prompts/.
_BUNDLED_DIR = Path(__file__).parent / "prompts"


def _resolve_prompt_path(name: str) -> Path | None:
    """The file load_prompt(name) reads: the repo override if present, else the
    bundled default. None if neither exists."""
    override = find_project_root() / ".orchestrator" / "prompts" / f"{name}.md"
    if override.exists():
        return override
    bundled = _BUNDLED_DIR / f"{name}.md"
    return bundled if bundled.exists() else None

# Tool-call footers appended unconditionally after the prompt body.
# These tell the agent how to return its result to the orchestrator.
# They are intentionally generic — no project-specific content.
_IMPLEMENTATION_FOOTER = """\
## When done

1. Confirm every change you made to yourself, organised by the areas the plan touches.

2. If `.claude/skills/static-checks/SKILL.md` exists, run the static checks per that skill. If any check fails, fix the violation and re-run until the script exits 0. Do not proceed until all checks pass.

3. Call the `emit_step_result` tool with:
   - `summary`: a one-line description of what you changed

This call is how the orchestrator captures your output. If you don't call it, the workflow has nothing to record and will fail.

You do NOT produce the PR summary or test plan — those are generated separately, after QA passes, from your diff. Your only structured output is the one-line `summary` above.
"""

_QA_FOOTER = """\
## When done

Call `emit_qa_result` exactly once with:

- `result`: `"PASS"` if every check passed, `"FAIL"` if any failed
- `failures`: empty string when PASS; when FAIL, a markdown report of all failing checks with this structure:
  ```
  # QA failures

  ## <check name>
  <exact description of the problem and its location — file path, line number, code snippet if helpful>

  ## <next failing check>
  ...

  ## Suggested next steps
  <if the fix is obvious, describe it; otherwise omit this section>
  ```

This call is how the orchestrator captures your verdict. If you don't call it, the workflow has nothing to record and will fail. Do not modify any files — your only output is the `emit_qa_result` call.
"""

_TEST_AUTHOR_FOOTER = """\
## When done

Call `emit_test_author_result` exactly once with:

- `testable`: `true` if you wrote one or more FAILING tests that exercise this task's behaviour through its public interface; `false` if the behaviour is not unit-testable or the criteria are too vague to pin down (do NOT write an empty or always-passing test — emit `false` instead).
- `reason`: on `true`, a one-line note on what behaviour the tests cover; on `false`, the one-line UNTESTABLE reason.

Before emitting `true`, make sure you actually ran the suite and saw your new tests FAIL (red). This call is how the orchestrator captures your output. If you don't call it, the workflow has nothing to record and will fail. Write test files only — do not implement the feature, and do not edit source files to make your own tests pass.
"""

_COVERAGE_CRITIC_FOOTER = """\
## When done

Call `emit_coverage_critic_result` exactly once with:

- `meaningful`: `true` if the tests would FAIL when the behaviour is implemented wrongly (they pin down this task's behaviour through its public interface); `false` if one or more are vacuous, tautological, assert only the shape of data, or would pass against a stub implementation.
- `feedback`: on `false`, name which test(s) are weak and what behaviour they should assert instead, so the author can revise them; on `true`, a one-line confirmation.

This call is how the orchestrator captures your verdict. If you don't call it, the workflow has nothing to record and will fail. Do not write or edit any files — your only output is the `emit_coverage_critic_result` call.
"""

_FOOTERS: dict[str, str] = {
    "implementation": _IMPLEMENTATION_FOOTER,
    "qa": _QA_FOOTER,
    "test-author": _TEST_AUTHOR_FOOTER,
    "coverage-critic": _COVERAGE_CRITIC_FOOTER,
}


def read_prompt_body(path: Path) -> str:
    """Read a prompt `.md` and return its body with leading `---` frontmatter
    stripped. The shared kernel behind both load_prompt (built-in, by name) and
    load_agent_prompt (generic, by path) — one place owns "file on disk → body".

    A prompt file may be downloaded from anywhere; stripping here means its
    metadata never leaks into the prompt body. The frontmatter's model/tools are
    honoured separately (load_prompt_frontmatter / manifest frontmatter parsing).
    """
    return split_frontmatter(path.read_text(encoding="utf-8"))[1]


def load_prompt(name: str, path_override: Path | None = None) -> str:
    """Return the full prompt for a built-in agent `name` ('planning', 'qa', …).

    Loads the body from the target repo override or the bundled default,
    then appends the tool-call footer for agents that require one.
    Raises FileNotFoundError if neither source exists (broken package).

    `path_override`: load the body from this exact path instead of the
    override-then-bundled search (used by config.test_author_path to point an
    agent at an arbitrary prompt file). The footer for `name` is still appended,
    so the structured-output contract holds wherever the body came from. Raises
    FileNotFoundError if the given path does not exist.
    """
    if path_override is not None:
        if not path_override.exists():
            raise FileNotFoundError(f"prompt path override not found: {path_override}")
        path = path_override
    else:
        path = _resolve_prompt_path(name)
        if path is None:
            raise FileNotFoundError(f"no prompt found for {name!r} (override or bundled)")
    body = read_prompt_body(path)

    footer = _FOOTERS.get(name, "")
    return body.rstrip() + "\n\n" + footer if footer else body


def load_agent_prompt(project_root: Path, agent: str) -> str:
    """Return a generic ai_agent step's prompt body.

    `agent` is the prompt file's path relative to `project_root`, full filename
    included, so the prompt file is <project_root>/<agent>. Mirrors
    manifest._agent_file so load-time validation and runtime loading resolve the
    same path. Raises FileNotFoundError when the file is absent — the caller
    (steps.execute_ai_agent) translates that into its own StepError.

    Unlike load_prompt this does NOT search the override/bundled locations and
    appends no footer: a generic agent is fully defined by its own file.
    """
    path = project_root / agent
    if not path.exists():
        raise FileNotFoundError(agent)
    return read_prompt_body(path)


def load_prompt_frontmatter(name: str) -> AgentFrontmatter:
    """The frontmatter config (model/tools) of a built-in agent's prompt file.

    Resolves the same override-then-bundled path as load_prompt, so a prompt
    downloaded into .orchestrator/prompts/<name>.md drives the built-in agent's
    model and tools (config.load_config merges this in, with [workflow.<step>]
    overriding). Returns an empty AgentFrontmatter when there's no file or no
    frontmatter — i.e. today's behaviour, defaults untouched."""
    path = _resolve_prompt_path(name)
    if path is None:
        return AgentFrontmatter()
    return parse_agent_frontmatter(path.read_text(encoding="utf-8"))[0]
