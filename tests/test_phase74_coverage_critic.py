"""Phase 74 — per-task coverage critic (test-meaningfulness backstop).

When tdd is on, after the test-author writes failing tests a separate read-only
critic judges whether they MEANINGFULLY pin down the task's behaviour. A negative
verdict re-authors the tests with the critic's feedback, bounded by
tdd_critic_max_attempts; still weak after the budget → proceed and record a manual
check (never wedge). Default-on; generic (no project specifics); reuses #111's
re-author escape and #73's manual-check surfacing.
"""

import uuid

import pytest
from langgraph.types import Command

from orchestrator import workflow as wf
from orchestrator.agents.coverage_critic import (
    CoverageCriticResult,
    critique_tests,
    _coerce_bool,
    _DEFAULT_TOOLS,
)
from orchestrator.config import OrchestratorConfig, load_config


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_critic_defaults_on_with_budget():
    c = OrchestratorConfig()
    assert c.tdd_coverage_critic is True
    assert c.tdd_critic_max_attempts == 2


def test_load_config_round_trips_critic_dials(tmp_path):
    toml = tmp_path / "orchestrator.toml"
    toml.write_text(
        'tdd = true\ntest_paths = ["**/*.test.js"]\n'
        'tdd_coverage_critic = false\ntdd_critic_max_attempts = 5\n',
        encoding="utf-8",
    )
    c = load_config(toml)
    assert c.tdd_coverage_critic is False
    assert c.tdd_critic_max_attempts == 5


# --------------------------------------------------------------------------- #
# agent: fail-open coercion + prompt/tools threading
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False), ("false", False), ("no", False), ("0", False),
    ("yes", True), ("meaningful", True), (None, True), (42, True),  # fail-open
])
def test_coerce_bool_is_fail_open(raw, expected):
    assert _coerce_bool(raw) is expected


@pytest.mark.asyncio
async def test_critique_tests_threads_prompt_and_tools(monkeypatch):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return CoverageCriticResult(meaningful=True, feedback="ok")

    monkeypatch.setattr("orchestrator.agents.coverage_critic.run_structured_agent", fake_run)
    await critique_tests("plan", "model")  # defaults
    assert captured["allowed_tools"] == _DEFAULT_TOOLS
    assert "Edit" not in captured["allowed_tools"] and "Write" not in captured["allowed_tools"]

    captured.clear()
    await critique_tests("plan", "model", "CUSTOM", ["Read"], ["Bash"])
    assert captured["system_prompt"] == "CUSTOM"
    assert captured["allowed_tools"] == ["Read"]
    assert captured["disallowed_tools"] == ["Bash"]


# --------------------------------------------------------------------------- #
# workflow helpers: frontmatter-driven model/tools
# (Phase 78a: the bundled critic prompt now pins model: haiku; tools stay silent.)
# --------------------------------------------------------------------------- #


def test_critic_model_resolves_to_bundled_haiku_frontmatter():
    # Phase 78a: the bundled coverage-critic.md sets model: haiku, so the critic
    # resolves to Haiku regardless of default_model (was Sonnet pre-78a, when the
    # prompt was frontmatter-silent). The model resolution path itself is unchanged.
    c = OrchestratorConfig(tdd=True, test_paths=["**/*.test.js"], default_model="claude-sonnet-4-6")
    assert wf._coverage_critic_model(c) == "claude-haiku-4-5-20251001"


def test_critic_tools_none_when_frontmatter_silent():
    assert wf._coverage_critic_tools(OrchestratorConfig()) == (None, None)


# --------------------------------------------------------------------------- #
# integration: critic passes / re-authors / exhausts → manual check
# --------------------------------------------------------------------------- #


def _patch_critic(monkeypatch, verdicts):
    """Patch the inner _run_coverage_critic so the real critic_task @task still
    checkpoints/replays. Returns the list of plan_texts each call saw."""
    it = iter(verdicts)
    calls: list[str] = []

    async def _critic(plan_text, model):
        calls.append(plan_text)
        return next(it)

    monkeypatch.setattr("orchestrator.workflow._run_coverage_critic", _critic)
    return calls


@pytest.mark.asyncio
async def test_critic_pass_ships_without_reauthor(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    critic_calls = _patch_critic(monkeypatch, [CoverageCriticResult(meaningful=True, feedback="ok")])
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"),
                              config=_tdd_cfg(coverage_critic=True, on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 1          # authored once
    assert len(critic_calls) == 1            # critiqued once, passed
    assert "manual_checks" not in result


@pytest.mark.asyncio
async def test_critic_reauthors_then_passes(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    critic_calls = _patch_critic(monkeypatch, [
        CoverageCriticResult(meaningful=False, feedback="assert the actual total"),
        CoverageCriticResult(meaningful=True, feedback="ok"),
    ])
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"),
                              config=_tdd_cfg(coverage_critic=True, on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"
    assert len(stubs.ta_calls) == 2                          # re-authored once
    assert stubs.ta_feedback == [None, "assert the actual total"]  # critic feedback fed back
    assert len(critic_calls) == 2
    assert "manual_checks" not in result


@pytest.mark.asyncio
async def test_critic_exhausts_budget_then_flags_manual_check(monkeypatch, tmp_path):
    from tests.test_phase72_test_author import _Stubs, _patch, _tdd_cfg, _cfg
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(n_tasks=1)
    _patch(stubs, monkeypatch, hash_value="SNAP")
    # max_attempts default 2 → critic fails at rounds 0,1,2 then proceeds.
    critic_calls = _patch_critic(monkeypatch, [
        CoverageCriticResult(meaningful=False, feedback=f"weak {i}") for i in range(3)
    ])
    async with build_workflow(db_path=str(tmp_path / "ckpt.db"),
                              config=_tdd_cfg(coverage_critic=True, on_exhausted="abort")) as w:
        await w.ainvoke("req", config=(c := _cfg()))
        result = await w.ainvoke(Command(resume="yes"), config=c)

    assert result["status"] == "succeeded"            # proceeds, never wedges
    assert len(stubs.ta_calls) == 3                    # author + 2 re-authors
    assert len(critic_calls) == 3
    assert len(stubs.impl_plans) == 1                  # implemented anyway
    assert result["manual_checks"] == [{
        "task_id": "t1",
        "title": "Task 1",
        "acceptance_criteria": "criterion 1",
        "reason": "coverage critic unresolved after 2 re-author(s): weak 2",
    }]
