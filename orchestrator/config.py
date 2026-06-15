"""User-facing config for the orchestrator (v2: declarative pipeline).

Reads orchestrator.toml (the v2 `flow` / `[stage.*]` / `[builtin.*]` / `[defs.*]`
shape — see orchestrator.v2.example.toml) and exposes a typed OrchestratorConfig
carrying a resolved `pipeline` (orchestrator.pipeline.Pipeline) plus the infra
tables. Missing file → the default pipeline. The v1 `[workflow.*]` / `[[steps.work]]`
shape is rejected at load with a migration message (Phase 68).

Usage:
    from orchestrator.config import load_config, OrchestratorConfig
    config = load_config()                    # reads orchestrator.toml if present
    config = load_config(Path("other.toml"))  # explicit path
"""

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orchestrator.paths import find_project_root
from orchestrator.pipeline import (
    BUILTIN_STAGE_TYPES,
    Pipeline,
    StageSpec,
    assert_shippable,
    build_pipeline,
)


# Env var names for the per-invocation overrides.
ENV_APPROVE_PLAN = "ORCHESTRATOR_APPROVE_PLAN"
ENV_BASE_BRANCH = "ORCHESTRATOR_BASE_BRANCH"
ENV_FULLY_AUTONOMOUS = "ORCHESTRATOR_FULLY_AUTONOMOUS"
ENV_AUTONOMOUS_MAX_SECONDS = "ORCHESTRATOR_AUTONOMOUS_MAX_SECONDS"
ENV_AUTONOMOUS_MAX_COST_USD = "ORCHESTRATOR_AUTONOMOUS_MAX_COST_USD"

_TRUE_LITERALS = {"true", "1", "yes", "on"}
_FALSE_LITERALS = {"false", "0", "no", "off"}


def _parse_bool_env(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_LITERALS:
        return True
    if lowered in _FALSE_LITERALS:
        return False
    raise ValueError(
        f"{name}={value!r} is not a valid boolean. Use one of "
        f"{sorted(_TRUE_LITERALS | _FALSE_LITERALS)}."
    )


def _parse_int_env(name: str, value: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not a valid integer.") from exc


def _parse_float_env(name: str, value: str) -> float:
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not a valid number.") from exc


_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_DOCS_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_SUMMARIZE_MODEL = "claude-haiku-4-5-20251001"


# The default pipeline (used when there is no orchestrator.toml). Equivalent to
# the pre-68 spine: plan → decompose → per-task build (impl⇄QA) → docs →
# summarize, with the git rails wrapping it implicitly.
_DEFAULT_PIPELINE_DICT: dict = {
    "flow": "plan >> decompose >> task-build >> docs >> summarize",
    "stage": {
        "builtin": {
            "plan": {"type": "ai_agent", "human_in_loop": True},
            "decompose": {"type": "ai_agent"},
            "task-build": {
                "produce": ["builtin:implementation"],
                "gate": ["builtin:qa"],
                "retry": {"max": 3, "on_exhausted": "approval_gate"},
            },
            "docs": {"type": "ai_agent", "model": _DEFAULT_DOCS_MODEL, "timeout": 120},
            "summarize": {
                "type": "ai_agent",
                "model": _DEFAULT_SUMMARIZE_MODEL,
                "allowed_tools": ["Read", "Bash", "Grep"],
                "timeout": 120,
            },
        }
    },
    "builtin": {
        "implementation": {"allowed_tools": ["Read", "Edit", "Write", "Bash"]},
        "qa": {"allowed_tools": ["Read", "Grep", "Bash"]},
    },
}


def default_pipeline() -> Pipeline:
    return build_pipeline(_DEFAULT_PIPELINE_DICT)


class PreHooksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dir: str = ".orchestrator/pre-hooks"
    timeout: int = 30


class QaConfig(BaseModel):
    """Scripted QA gate: executable checks under scripts_dir that run before the
    QA agent. Orthogonal to the `qa` pipeline stage / `builtin.qa` part."""

    model_config = ConfigDict(extra="forbid")
    scripts_dir: str = ".orchestrator/qa"
    scripts_timeout: int = 60


class GitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    auto_rebase: bool = True


class PrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_branch: str = "main"
    draft: bool = False
    reviewers: list[str] = Field(default_factory=list)
    # Pre-PR pause: pause for a human before commit/push/open-pr (was the v1
    # [workflow.commit].human_in_loop gate). Suppressed under fully_autonomous.
    human_in_loop: bool = False


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    log_path: str = ".orchestrator/audit.log"
    include_content: bool = False


class BranchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_slug_length: int = 50
    # Pre-branch pause: pause for a human before the branch is created (was the v1
    # [workflow.branch].human_in_loop gate). Suppressed under fully_autonomous.
    human_in_loop: bool = False


class OrchestratorConfig(BaseModel):
    # Pipeline is a frozen dataclass (orchestrator.pipeline), not a pydantic model.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    default_model: str = _DEFAULT_MODEL
    db_path: str = ".orchestrator/checkpoints.db"
    fully_autonomous: bool = False
    autonomous_max_seconds: int = 0
    autonomous_max_cost_usd: float = 0.0

    # TDD / red-green station (Phase 72). When tdd is true, the per-task station
    # authors tests with a separate agent BEFORE the implement loop, confirms the
    # green→red transition, and freezes the tests so the implementer can't edit
    # them (the diff-gate). test_paths is the globset (project-root-relative) the
    # diff-gate hashes — without it there is nothing to freeze, so it is required
    # when tdd is on. Both default off → classic implement→qa pipeline unchanged.
    tdd: bool = False
    test_paths: list[str] = Field(default_factory=list)
    # Optional project-root-relative path to the test-author's prompt. When unset,
    # the test-author falls back to the convention (.orchestrator/prompts/
    # test-author.md → bundled default). Lets you point the author at an arbitrary
    # prompt file; the emit-tool footer is appended either way. Only consulted when
    # tdd is on, so it is harmless (a no-op) otherwise.
    test_author_path: str | None = None
    # Supervised red-review pause (Phase 72b). When tdd is on, after the test-author
    # writes failing tests and the green→red transition is confirmed, the run pauses
    # for a human to review the RED tests before implementation: 'yes' implements,
    # feedback re-authors the tests, 'abort' stops the run. Default on (it is the
    # supervised guard); set false to let red-green run unattended. Only consulted
    # when tdd is on and the run is not fully_autonomous.
    tdd_red_review: bool = True
    # Coverage critic (Phase 74). When tdd is on, after the test-author writes
    # failing tests a separate read-only agent judges whether they MEANINGFULLY pin
    # down the task's behaviour (catches tautological/vacuous tests a green→red gate
    # can't). A negative verdict re-authors the tests with the critic's feedback,
    # bounded by tdd_critic_max_attempts; if still weak after the budget the run
    # proceeds and the concern is recorded as a manual check (never wedges). Default
    # on (the meaningfulness backstop); set false to skip it.
    tdd_coverage_critic: bool = True
    # Max critic-driven re-author rounds before proceeding-with-a-manual-check.
    tdd_critic_max_attempts: int = 2
    # Autonomous TDD re-author budget (Phase 76). Only consulted when tdd AND
    # fully_autonomous are both on. With no human red-review/re-author escape, the
    # implement build is run with a BOUNDED budget; if it can't make the frozen
    # tests pass within that budget the tests are AUTOMATICALLY re-authored (they
    # may be wrong / over-constrained), capped at this many rounds. Exhausting the
    # cap fails the task (status="failed") rather than looping a wrong frozen test
    # to the autonomous safety ceiling.
    tdd_autonomous_reauthor_max: int = 2

    pipeline: Pipeline = Field(default_factory=default_pipeline)
    branch: BranchConfig = Field(default_factory=BranchConfig)
    pre_hooks: PreHooksConfig = Field(default_factory=PreHooksConfig)
    qa: QaConfig = Field(default_factory=QaConfig)
    git: GitConfig = Field(default_factory=GitConfig)
    pr: PrConfig = Field(default_factory=PrConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)

    def resolved_model(self, model: str | None) -> str:
        """Return `model` or fall back to default_model when None."""
        return model if model is not None else self.default_model

    def stage(self, stage_id: str) -> StageSpec | None:
        for s in self.pipeline.stages:
            if s.id == stage_id:
                return s
        return None

    def part(self, ref: str):
        """A reusable part ([builtin.*]/[defs.*]) by prefixed ref, or None."""
        return self.pipeline.parts.get(ref)

    @model_validator(mode="after")
    def _validate_tdd(self) -> "OrchestratorConfig":
        """TDD invariants (Phase 72).

        tdd + fully_autonomous is ALLOWED as of Phase 76: with no human present
        the human guards (red-review / re-author) are replaced by machinery — the
        green→red red-confirm becomes a HARD gate (a born-green / non-green-
        baseline / no-script-gate verdict aborts the task instead of silently
        degrading), and the implement build runs with a BOUNDED budget whose
        exhaustion triggers an automatic, capped re-author cycle
        (tdd_autonomous_reauthor_max) rather than looping a wrong frozen test to
        the safety ceiling. (Phase 72 originally forbade the combo — decision A —
        until that machinery existed.)

        test_paths is required when tdd is on — it is the globset the diff-gate
        freezes; with nothing to freeze the write-separation guarantee is empty.
        """
        if self.tdd and not self.test_paths:
            raise ValueError(
                "tdd = true requires test_paths (the globset the diff-gate "
                "freezes, e.g. test_paths = [\"**/*.test.js\"]). Without it there "
                "are no tests to freeze. Set test_paths or disable tdd."
            )
        return self


# v2 stage ids / part ids whose model/tools may be supplied by a prompt file's
# frontmatter (.orchestrator/prompts/<name>.md). Maps the v2 id → the prompt name.
_FRONTMATTER_STAGE_PROMPTS: dict[str, str] = {
    "plan": "planning",
    "decompose": "decompose",
    "docs": "docs",
    "summarize": "summarize",
    "qa": "qa",
}
_FRONTMATTER_PART_PROMPTS: dict[str, str] = {
    "builtin:implementation": "implementation",
    "builtin:qa": "qa",
}
_FRONTMATTER_FIELDS = ("model", "allowed_tools", "disallowed_tools", "timeout")


_V1_MIGRATION = (
    "This looks like a v1 orchestrator.toml ([workflow.*] / [[steps.work]] / "
    "[steps.defs.*]). The config format is now v2 (flow + [stage.*] + [builtin.*] "
    "+ [defs.*]). See orchestrator.example.toml for the v2 shape."
)


def _reject_v1(data: dict) -> None:
    if "workflow" in data or "steps" in data:
        raise ValueError(_V1_MIGRATION)


# Every key allowed at the top level of orchestrator.toml: the scalar dials, the
# pipeline-shaping tables (owned by pipeline.build_pipeline), and the infra/rail
# tables. Anything else is a typo — fail loud at load (the v2 extra="forbid").
_ALLOWED_TOP_LEVEL: frozenset[str] = frozenset({
    "default_model", "db_path", "fully_autonomous",
    "autonomous_max_seconds", "autonomous_max_cost_usd",
    "tdd", "test_paths", "test_author_path", "tdd_red_review",
    "tdd_coverage_critic", "tdd_critic_max_attempts",
    "tdd_autonomous_reauthor_max",
    "flow", "stage", "builtin", "defs",
    "branch", "pre_hooks", "qa", "git", "pr", "audit",
})


def _reject_unknown_top_level(data: dict) -> None:
    unknown = sorted(set(data) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise ValueError(
            f"unknown top-level key(s) in orchestrator.toml: {unknown}. "
            f"Allowed: {sorted(_ALLOWED_TOP_LEVEL)}."
        )


def load_config(path: Path | None = None) -> OrchestratorConfig:
    """Load v2 config from orchestrator.toml; defaults if the file is missing.

    Even with no file, a dropped-in prompt's frontmatter still drives the
    built-ins (see _merge_builtin_frontmatter), so the no-file path is handled as
    an empty config rather than short-circuited.
    """
    if path is None:
        path = find_project_root() / "orchestrator.toml"

    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
    else:
        data = {}
    _reject_v1(data)
    _reject_unknown_top_level(data)

    # An infra-only config (no `flow` and no pipeline tables) keeps the default
    # pipeline — a user tweaking only [git]/[pr]/[pre_hooks]/[audit] needn't
    # restate the whole pipeline. A `flow` line (or stray [stage.*]/[builtin.*]/
    # [defs.*] without one) goes through build_pipeline, which fails loud on a
    # missing flow.
    if "flow" not in data and not any(k in data for k in ("stage", "builtin", "defs")):
        pipeline = default_pipeline()
    else:
        pipeline = build_pipeline(data)
        assert_shippable(pipeline)  # every run ships → require a summarize stage (Q4)
    # Prompt frontmatter fills built-in model/tools the user didn't set in TOML.
    pipeline = _merge_builtin_frontmatter(pipeline, data)

    fields: dict = {"pipeline": pipeline}
    for key in ("default_model", "db_path", "fully_autonomous",
                "autonomous_max_seconds", "autonomous_max_cost_usd",
                "tdd", "test_paths", "test_author_path", "tdd_red_review",
                "tdd_coverage_critic", "tdd_critic_max_attempts",
                "tdd_autonomous_reauthor_max"):
        if key in data:
            fields[key] = data[key]
    if "branch" in data:
        fields["branch"] = BranchConfig.model_validate(data["branch"])
    if "pre_hooks" in data:
        fields["pre_hooks"] = PreHooksConfig.model_validate(data["pre_hooks"])
    if "qa" in data:
        fields["qa"] = QaConfig.model_validate(data["qa"])
    if "git" in data:
        fields["git"] = GitConfig.model_validate(data["git"])
    if "pr" in data:
        fields["pr"] = PrConfig.model_validate(data["pr"])
    if "audit" in data:
        fields["audit"] = AuditConfig.model_validate(data["audit"])
    return OrchestratorConfig(**fields)


def _merge_builtin_frontmatter(pipeline: Pipeline, data: dict) -> Pipeline:
    """Let a built-in stage/part's prompt frontmatter supply its model/tools.

    A prompt dropped into .orchestrator/prompts/<name>.md drives the built-in the
    same way frontmatter drives a [defs.*] agent: frontmatter is the default; an
    explicit key in the user's TOML overrides it. "Explicit" means the user wrote
    the key in orchestrator.toml — a code/built-in default does NOT count — so the
    raw `data` is consulted for the user-set keys, not the resolved value (which
    already carries defaults). A built-in named in `flow` without a table has no
    user keys, so frontmatter drives it fully.
    """
    from orchestrator.prompt_loader import load_prompt_frontmatter

    stage_tables = (data.get("stage") or {}).get("builtin") or {}
    part_tables = data.get("builtin") or {}

    new_stages = []
    changed = False
    for s in pipeline.stages:
        prompt = _FRONTMATTER_STAGE_PROMPTS.get(s.id) if s.namespace == "builtin" else None
        if prompt is None:
            new_stages.append(s)
            continue
        fm = load_prompt_frontmatter(prompt)
        user_keys = set((stage_tables.get(s.id) or {}).keys())
        updates = {}
        for field in _FRONTMATTER_FIELDS:
            if field in user_keys:
                continue  # an explicit TOML value wins
            value = getattr(fm, field, None)
            if value is not None:
                updates[field] = value
        if updates:
            new_stages.append(s.model_copy(update=updates))
            changed = True
        else:
            new_stages.append(s)

    new_parts = dict(pipeline.parts)
    for ref, prompt in _FRONTMATTER_PART_PROMPTS.items():
        p = new_parts.get(ref)
        if p is None or p.namespace != "builtin":
            continue
        fm = load_prompt_frontmatter(prompt)
        pid = ref.split(":", 1)[1]
        user_keys = set((part_tables.get(pid) or {}).keys())
        if "tools" in user_keys:
            user_keys.add("allowed_tools")  # `tools` is the PartSpec alias
        updates = {}
        for field in _FRONTMATTER_FIELDS:
            if field in user_keys:
                continue
            value = getattr(fm, field, None)
            if value is not None:
                updates[field] = value
        if updates:
            new_parts[ref] = p.model_copy(update=updates)
            changed = True

    if not changed:
        return pipeline
    return Pipeline(flow=pipeline.flow, stages=tuple(new_stages), parts=new_parts)


def apply_overrides(
    config: OrchestratorConfig,
    *,
    approve_plan: bool | None = None,
    base_branch: str | None = None,
    fully_autonomous: bool | None = None,
    autonomous_max_seconds: int | None = None,
    autonomous_max_cost_usd: float | None = None,
) -> OrchestratorConfig:
    """Overlay per-invocation overrides (kwarg → env var → unchanged)."""
    if approve_plan is None and (raw := os.environ.get(ENV_APPROVE_PLAN)) is not None:
        approve_plan = _parse_bool_env(ENV_APPROVE_PLAN, raw)
    if base_branch is None and (raw := os.environ.get(ENV_BASE_BRANCH)) is not None:
        base_branch = raw.strip() or None
    if fully_autonomous is None and (raw := os.environ.get(ENV_FULLY_AUTONOMOUS)) is not None:
        fully_autonomous = _parse_bool_env(ENV_FULLY_AUTONOMOUS, raw)
    if autonomous_max_seconds is None and (raw := os.environ.get(ENV_AUTONOMOUS_MAX_SECONDS)) is not None:
        autonomous_max_seconds = _parse_int_env(ENV_AUTONOMOUS_MAX_SECONDS, raw)
    if autonomous_max_cost_usd is None and (raw := os.environ.get(ENV_AUTONOMOUS_MAX_COST_USD)) is not None:
        autonomous_max_cost_usd = _parse_float_env(ENV_AUTONOMOUS_MAX_COST_USD, raw)

    updates: dict = {}
    if approve_plan is not None:
        # approve_plan toggles the plan stage's human_in_loop on the pipeline.
        new_stages = tuple(
            s.model_copy(update={"human_in_loop": approve_plan}) if s.id == "plan" else s
            for s in config.pipeline.stages
        )
        updates["pipeline"] = Pipeline(
            flow=config.pipeline.flow, stages=new_stages, parts=config.pipeline.parts
        )
    if base_branch is not None:
        updates["pr"] = config.pr.model_copy(update={"base_branch": base_branch})
    if fully_autonomous is not None:
        updates["fully_autonomous"] = fully_autonomous
    if autonomous_max_seconds is not None:
        updates["autonomous_max_seconds"] = autonomous_max_seconds
    if autonomous_max_cost_usd is not None:
        updates["autonomous_max_cost_usd"] = autonomous_max_cost_usd

    return config.model_copy(update=updates) if updates else config
