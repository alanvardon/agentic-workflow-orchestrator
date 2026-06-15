"""Phase 78a — coverage critic on Haiku.

The coverage critic is a read-only meaningfulness judge (does this test pin down
behaviour, or is it vacuous/tautological/shape-only?). That is a classification
task Haiku handles well, and with TDD on by default it runs on every testable
task. Phase 78a moves it off the default model (Sonnet) onto Haiku by pinning
`model: haiku` in the bundled `coverage-critic.md` frontmatter — the per-token
price drops without trimming the review work.

The lever is the bundled-prompt frontmatter, NOT a code constant: a project can
still point the critic at any model via its own
`.orchestrator/prompts/coverage-critic.md`. These tests lock both: the bundled
default IS Haiku, and the resolver still reads the frontmatter (so an override
wins) rather than hardcoding it.
"""

from orchestrator import workflow as wf
from orchestrator.agent_frontmatter import AgentFrontmatter
from orchestrator.config import OrchestratorConfig
from orchestrator.prompt_loader import load_prompt, load_prompt_frontmatter


def test_bundled_critic_prompt_pins_haiku():
    # The bundled coverage-critic.md frontmatter sets model: haiku, normalised to
    # the full Haiku id by the frontmatter parser.
    assert load_prompt_frontmatter("coverage-critic").model == "claude-haiku-4-5-20251001"


def test_critic_model_honours_frontmatter_over_default():
    # Phase 78a: the frontmatter model is honoured, so the critic runs on Haiku
    # even when the project default is Sonnet.
    c = OrchestratorConfig(default_model="claude-sonnet-4-6")
    assert wf._coverage_critic_model(c) == "claude-haiku-4-5-20251001"


def test_critic_model_is_not_hardcoded(monkeypatch):
    # Overridability: the model comes from the prompt frontmatter, not a code
    # constant. A project that drops in its own coverage-critic.md (here simulated
    # by patching the frontmatter loader) drives the model — Haiku is only the
    # bundled default.
    monkeypatch.setattr(
        wf, "load_prompt_frontmatter",
        lambda name: AgentFrontmatter(model="claude-opus-4-8"),
    )
    c = OrchestratorConfig(default_model="claude-sonnet-4-6")
    assert wf._coverage_critic_model(c) == "claude-opus-4-8"


def test_frontmatter_does_not_leak_into_prompt_body():
    # The model: haiku frontmatter is stripped from the system prompt the critic
    # actually receives — only the persona/rules survive (plus the tool footer).
    body = load_prompt("coverage-critic")
    assert "model: haiku" not in body
    assert body.lstrip().startswith("You are a test-quality critic")
