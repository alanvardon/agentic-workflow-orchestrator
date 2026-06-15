"""Phase 68 — v2 pipeline schema + validation tests.

Pure unit tests for orchestrator.pipeline.build_pipeline: parsing the four
namespaces, the validation rules, the manifest hash, and a smoke test that the
real orchestrator.v2.example.toml parses. No workflow, no LLM.
"""

import copy
import tomllib

import pytest

from orchestrator.paths import find_project_root
from orchestrator.pipeline import (
    BUILTIN_PART_TYPES,
    Pipeline,
    PipelineError,
    StageSpec,
    assert_shippable,
    build_pipeline,
)


def _base() -> dict:
    """A minimal valid v2 config dict."""
    return {
        "flow": "plan >> decompose >> task-build >> qa",
        "stage": {
            "builtin": {
                "plan": {"type": "ai_agent", "human_in_loop": True},
                "decompose": {"type": "ai_agent"},
                "task-build": {
                    "produce": ["builtin:implementation"],
                    "gate": ["builtin:qa"],
                },
                "qa": {"type": "ai_agent"},
            }
        },
        "builtin": {"implementation": {}},
        "defs": {},
    }


# ── happy paths ──────────────────────────────────────────────────────────────

def test_minimal_pipeline_builds_in_flow_order():
    p = build_pipeline(_base())
    assert isinstance(p, Pipeline)
    assert [s.id for s in p.stages] == ["plan", "decompose", "task-build", "qa"]


def test_builtin_part_resolves_without_explicit_table():
    # builtin:qa used as a gate with no [builtin.qa] table — still resolves.
    p = build_pipeline(_base())
    tb = p.stage("task-build")
    assert tb.gate == ["builtin:qa"]
    assert "qa" in BUILTIN_PART_TYPES


def test_effective_type_inferred_for_builtins():
    p = build_pipeline(_base())
    assert p.stage("task-build").effective_type == "build"
    assert p.stage("plan").effective_type == "ai_agent"


def test_builtin_stages_in_flow_are_auto_supplied_without_tables():
    # A flow naming built-in stages with NO [stage.builtin.*] tables builds a full
    # default pipeline (override only what you want).
    p = build_pipeline({"flow": "plan >> decompose >> task-build >> docs >> summarize"})
    assert [s.id for s in p.stages] == [
        "plan", "decompose", "task-build", "docs", "summarize",
    ]
    tb = p.stage("task-build")
    assert tb.produce == ["builtin:implementation"]
    assert tb.gate == ["builtin:qa"]
    assert tb.retry.on_exhausted == "approval_gate"


def test_user_stage_with_type_and_path():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> qa >> gitleaks"
    d["stage"]["user"] = {"gitleaks": {"type": "script", "path": ".orchestrator/checks/gitleaks.sh"}}
    p = build_pipeline(d)
    assert p.stage("gitleaks").namespace == "user"


def test_user_stage_placing_a_part_via_uses():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> final-qa"
    del d["stage"]["builtin"]["qa"]
    d["stage"]["user"] = {"final-qa": {"uses": "builtin:qa"}}
    p = build_pipeline(d)
    assert p.stage("final-qa").uses == "builtin:qa"


def test_real_v2_example_parses():
    path = find_project_root() / "orchestrator.example.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    p = build_pipeline(data)
    # sanity: the documented default flow is present and ordered
    ids = [s.id for s in p.stages]
    assert ids[:3] == ["plan", "decompose", "task-build"]
    assert "qa" in ids


# ── manifest hash ────────────────────────────────────────────────────────────

def test_manifest_hash_is_stable():
    assert build_pipeline(_base()).manifest_hash() == build_pipeline(_base()).manifest_hash()


def test_manifest_hash_changes_when_referenced_part_body_changes():
    d1 = _base()
    d1["flow"] = "plan >> decompose >> task-build >> qa"
    d1["stage"]["builtin"]["task-build"]["gate"] = ["defs:lint"]
    d1["defs"] = {"lint": {"type": "script", "path": "a.sh"}}
    d2 = copy.deepcopy(d1)
    d2["defs"]["lint"]["path"] = "b.sh"            # edit the part BODY only
    assert build_pipeline(d1).manifest_hash() != build_pipeline(d2).manifest_hash()


# ── validation failures ──────────────────────────────────────────────────────

def test_missing_flow_rejected():
    d = _base()
    del d["flow"]
    with pytest.raises(PipelineError, match="flow"):
        build_pipeline(d)


def test_unknown_builtin_stage_rejected():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> qa >> frobnicate"
    d["stage"]["builtin"]["frobnicate"] = {"type": "ai_agent"}
    d["flow"] = "plan >> decompose >> task-build >> qa >> frobnicate"
    with pytest.raises(PipelineError, match="not a known built-in stage"):
        build_pipeline(d)


def test_builtin_type_mismatch_rejected():
    d = _base()
    d["stage"]["builtin"]["plan"]["type"] = "script"
    with pytest.raises(PipelineError, match="does not match"):
        build_pipeline(d)


def test_user_stage_missing_path_rejected():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> qa >> mine"
    d["stage"]["user"] = {"mine": {"type": "script"}}
    with pytest.raises(PipelineError, match="`path` is required"):
        build_pipeline(d)


def test_user_stage_missing_type_and_uses_rejected():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> qa >> mine"
    d["stage"]["user"] = {"mine": {"model": "x"}}
    with pytest.raises(PipelineError, match="needs type"):
        build_pipeline(d)


def test_flow_references_undefined_stage_rejected():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> qa >> ghost"
    with pytest.raises(PipelineError, match="no \\[stage"):
        build_pipeline(d)


def test_orphan_stage_rejected():
    d = _base()
    d["stage"]["builtin"]["docs"] = {"type": "ai_agent"}  # defined, not in flow
    with pytest.raises(PipelineError, match="orphan"):
        build_pipeline(d)


def test_flow_referencing_a_rail_rejected():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> qa >> commit"
    with pytest.raises(PipelineError, match="locked git rail"):
        build_pipeline(d)


def test_duplicate_stage_id_across_namespaces_rejected():
    d = _base()
    d["stage"]["user"] = {"qa": {"type": "ai_agent", "path": "x.md"}}
    with pytest.raises(PipelineError, match="both stage namespaces"):
        build_pipeline(d)


def test_unknown_gate_reference_rejected():
    d = _base()
    d["stage"]["builtin"]["task-build"]["gate"] = ["defs:nope"]
    with pytest.raises(PipelineError, match="neither a"):
        build_pipeline(d)


def test_producing_build_without_gate_rejected():
    d = _base()
    d["stage"]["builtin"]["task-build"]["gate"] = []
    with pytest.raises(PipelineError, match="must list at least one `gate`"):
        build_pipeline(d)


def test_ungated_build_allowed():
    d = _base()
    d["stage"]["builtin"]["task-build"]["gate"] = []
    d["stage"]["builtin"]["task-build"]["ungated"] = True
    p = build_pipeline(d)
    assert p.stage("task-build").ungated is True


def test_task_build_without_decompose_rejected():
    d = _base()
    d["flow"] = "plan >> task-build >> qa"
    del d["stage"]["builtin"]["decompose"]
    with pytest.raises(PipelineError, match="requires a `decompose`"):
        build_pipeline(d)


def test_decompose_after_task_build_rejected():
    d = _base()
    d["flow"] = "plan >> task-build >> decompose >> qa"
    with pytest.raises(PipelineError, match="must come before"):
        build_pipeline(d)


def test_unknown_builtin_part_rejected():
    d = _base()
    d["builtin"]["frobnicate"] = {}
    with pytest.raises(PipelineError, match="not a known built-in part"):
        build_pipeline(d)


def test_extra_key_on_stage_rejected():
    d = _base()
    d["stage"]["builtin"]["plan"]["bogus"] = 1
    with pytest.raises(PipelineError):
        build_pipeline(d)


# ── shippability (require a summarize stage) ──────────────────────────────────

def test_assert_shippable_rejects_pipeline_without_summarize():
    # _base() has no summarize stage — build_pipeline (pure validator) accepts it,
    # but assert_shippable (the load-time guard) rejects it.
    p = build_pipeline(_base())
    with pytest.raises(PipelineError, match="summarize"):
        assert_shippable(p)


def test_assert_shippable_passes_with_summarize():
    d = _base()
    d["flow"] = "plan >> decompose >> task-build >> summarize >> qa"
    d["stage"]["builtin"]["summarize"] = {"type": "ai_agent"}
    assert_shippable(build_pipeline(d))  # must not raise
