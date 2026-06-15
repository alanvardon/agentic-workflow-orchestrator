"""Phase 37 — fully_autonomous config + apply_overrides resolution.

Unit-level coverage for the three new knobs (fully_autonomous,
autonomous_max_seconds, autonomous_max_cost_usd): TOML round-trip, env-var
resolution, kwarg-wins-over-env, and that they default off / are non-mutating.
The runtime gate-suppression behaviour lives in test_phase37_autonomous_run.py.
"""

import pytest

from orchestrator.config import (
    ENV_AUTONOMOUS_MAX_COST_USD,
    ENV_AUTONOMOUS_MAX_SECONDS,
    ENV_FULLY_AUTONOMOUS,
    OrchestratorConfig,
    apply_overrides,
    load_config,
)


def _base() -> OrchestratorConfig:
    return OrchestratorConfig()


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_defaults_off():
    cfg = _base()
    assert cfg.fully_autonomous is False
    assert cfg.autonomous_max_seconds == 0
    assert cfg.autonomous_max_cost_usd == 0.0


# ---------------------------------------------------------------------------
# TOML round-trip
# ---------------------------------------------------------------------------


def test_load_config_round_trips_autonomous(tmp_path):
    toml = tmp_path / "orchestrator.toml"
    toml.write_text(
        "fully_autonomous = true\n"
        "autonomous_max_seconds = 900\n"
        "autonomous_max_cost_usd = 2.5\n",
        encoding="utf-8",
    )
    cfg = load_config(toml)
    assert cfg.fully_autonomous is True
    assert cfg.autonomous_max_seconds == 900
    assert cfg.autonomous_max_cost_usd == 2.5


def test_unknown_top_level_key_still_rejected(tmp_path):
    # extra="forbid" guards typos like `full_autonomous`.
    toml = tmp_path / "orchestrator.toml"
    toml.write_text("full_autonomous = true\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(toml)


# ---------------------------------------------------------------------------
# kwarg path
# ---------------------------------------------------------------------------


def test_kwarg_sets_fully_autonomous():
    cfg = apply_overrides(_base(), fully_autonomous=True)
    assert cfg.fully_autonomous is True


def test_kwarg_sets_ceilings():
    cfg = apply_overrides(
        _base(), autonomous_max_seconds=120, autonomous_max_cost_usd=1.25
    )
    assert cfg.autonomous_max_seconds == 120
    assert cfg.autonomous_max_cost_usd == 1.25


# ---------------------------------------------------------------------------
# env var path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_env_var_fully_autonomous(monkeypatch, value, expected):
    monkeypatch.setenv(ENV_FULLY_AUTONOMOUS, value)
    cfg = apply_overrides(_base())
    assert cfg.fully_autonomous is expected


def test_env_var_ceilings(monkeypatch):
    monkeypatch.setenv(ENV_AUTONOMOUS_MAX_SECONDS, "300")
    monkeypatch.setenv(ENV_AUTONOMOUS_MAX_COST_USD, "0.75")
    cfg = apply_overrides(_base())
    assert cfg.autonomous_max_seconds == 300
    assert cfg.autonomous_max_cost_usd == 0.75


def test_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv(ENV_FULLY_AUTONOMOUS, "true")
    cfg = apply_overrides(_base(), fully_autonomous=False)
    assert cfg.fully_autonomous is False


def test_invalid_env_bool_fails_loud(monkeypatch):
    monkeypatch.setenv(ENV_FULLY_AUTONOMOUS, "maybe")
    with pytest.raises(ValueError):
        apply_overrides(_base())


def test_invalid_env_cost_fails_loud(monkeypatch):
    monkeypatch.setenv(ENV_AUTONOMOUS_MAX_COST_USD, "lots")
    with pytest.raises(ValueError):
        apply_overrides(_base())


# ---------------------------------------------------------------------------
# non-mutation / identity
# ---------------------------------------------------------------------------


def test_does_not_mutate_input():
    base = _base()
    apply_overrides(base, fully_autonomous=True, autonomous_max_seconds=99)
    assert base.fully_autonomous is False
    assert base.autonomous_max_seconds == 0


def test_no_override_returns_same_identity():
    base = _base()
    assert apply_overrides(base) is base
