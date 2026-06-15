"""Pluggable-step manifest.

Lets operators declare steps in the workflow by editing orchestrator.toml,
without touching workflow.py. The core spine
(verify_clean_tree → plan → branch → [work] → summarize → docs →
commit → push → pr) stays hard-coded. The one project-specific region is
*work*: an ordered list, `[[steps.work]]`, that runs AFTER the branch is made
and BEFORE summarize. The whole region is pre-commit, so cancel/abort never
leaves a half-shipped state. Steps run in list order — there is no seam
vocabulary and no order inference.

Setup (verify → plan → branch) and shipping (summarize → docs → commit → push →
pr) are fixed spine, tuned via [workflow.*] / [pre_hooks] / [pr] tables, never
listed. The pre-plan use case is covered by [pre_hooks]; plan sign-off by
[workflow.planning].human_in_loop. See _REMOVED_SEAMS for the migration errors.

The impl ⇄ qa loop is itself a `build` step in the work list, declared
explicitly in orchestrator.toml. The built-in ids `implementation` (producer)
and `qa` (gate) resolve to the spine's own agents when not redefined under
[steps.defs.*]; declare them there to swap in your own.

TOML shape (everything optional; no [steps.work] = no declared steps):

    [[steps.work]]
    id    = "docs"
    type  = "ai_agent"
    agent = "team/agents/docs.md"   # project-root-relative path (full filename)
    model = "claude-sonnet-4-6"     # optional — overrides the agent's frontmatter

The agent `.md` is the definition: its YAML frontmatter (a leading `---` block)
supplies the agent's defaults — `model` and `tools` — so a file downloaded from
anywhere (e.g. a Claude Code subagent) plugs straight in without editing.
Frontmatter we don't recognise (`name`, `description`, …) is ignored, the body
is the prompt, and an operational key set on the TOML entry above overrides the
frontmatter for that repo. See agent_frontmatter.parse_agent_frontmatter.

    [[steps.work]]
    id   = "security_gate"
    type = "approval_gate"
    ask  = "QA passed. Approve security posture before commit?"

A `build` step (formerly the `retry` block): run producer(s), then gate(s),
re-running the producers with the failing gate's feedback until a gate passes
or the retry budget is exhausted. produce/gate reference definitions under
[steps.defs.*]:

    [[steps.work]]
    id      = "lint-loop"
    type    = "build"
    produce = ["lint-fix"]           # ids defined in [steps.defs.*]
    gate    = ["lint-check"]         # gate verdict = script exit / agent `passed`
    retry   = { max = 3, on_exhausted = "abort" }   # abort | approval_gate | proceed
    # under on_exhausted="approval_gate" a human may grant more attempts at the
    # exhaustion prompt; cap the total with the optional max_total_attempts.

    [steps.defs.lint-fix]
    type  = "ai_agent"
    agent = "team/agents/lint-fixer.md"

    [steps.defs.lint-check]
    type = "script"
    path = ".orchestrator/scripts/lint.sh"

The gating guarantee: a build step must list at least one `gate` unless it sets
`ungated = true` (then the producer runs once, no gate, no retry).

The loader validates at load time (before any LLM spend): unknown seam
names, duplicate ids, missing script paths, unknown agent references, and —
for build steps — that produce/gate reference a [steps.defs.*] id or a built-in
(`implementation` / `qa`), that no id is both a producer and a gate, and that
`produce` (and `gate`, unless `ungated`) is non-empty. A problem raises
ManifestError with a clear message.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from orchestrator.agent_frontmatter import parse_agent_frontmatter
from orchestrator.errors import FatalError
from orchestrator.paths import find_project_root
from orchestrator.usage import TaskUsage


# The only region where steps are declared. `work` runs after the branch is
# made and before summarize — entirely BEFORE the commit line, so a declared
# step can never create a half-shipped state on cancel. Steps run in list order.
# Setup and shipping are fixed spine, not declared here (see _REMOVED_SEAMS).
SEAMS: tuple[str, ...] = ("work",)

# Seam names that no longer have a home now that the one variable region is the
# `work` list. Kept as a lookup so the loader raises a migration-guiding error
# instead of a bare "unknown seam".
_REMOVED_SEAMS: dict[str, str] = {
    "before_plan": (
        "it ran before planning; use [pre_hooks] (scripts run after the "
        "clean-tree check, before planning) instead."
    ),
    "after_plan": (
        "it ran after plan approval; plan sign-off is now "
        "[workflow.planning].human_in_loop. For work on the code, use "
        "[[steps.work]] (which runs after the branch is made)."
    ),
    "after_branch": (
        "it was the build's region; declare your steps in [[steps.work]] "
        "(which runs in the same place — after the branch, before summarize)."
    ),
    "before_commit": (
        "it ran after summarize; place the step in [[steps.work]] instead — "
        "the whole work list runs before summarize, so its output is now "
        "included in the commit/PR summary's diff."
    ),
    "after_impl": (
        "it fired once per implementation attempt, inside the impl⇄QA loop; "
        "that loop is now a 'build' step in [[steps.work]]. To check each "
        "attempt, add a gate to the build step instead."
    ),
    "after_qa": (
        "it fired once after QA passed; add the step to [[steps.work]] "
        "(ordered after the build step) to run once on the QA-passed tree."
    ),
}

# Built-in producer/gate ids a build step may reference WITHOUT a matching
# [steps.defs.*] entry. They resolve to the spine's own implementation producer /
# QA gate at runtime (workflow._run_build_step). Redefining either id under
# [steps.defs.*] overrides the built-in.
_BUILTIN_PRODUCER_IDS: frozenset[str] = frozenset({"implementation"})
_BUILTIN_GATE_IDS: frozenset[str] = frozenset({"qa"})


class ManifestError(FatalError):
    """Raised at load time for a malformed or invalid step manifest."""


class _BaseStep(BaseModel):
    # `id` is the step's RESUME IDENTITY: a step's @task cache key derives
    # from it, so renaming an id mid-run makes resume re-run that step.
    id: str


class ScriptStep(_BaseStep):
    type: Literal["script"] = "script"
    path: str
    # Seconds before the script is killed. Bounds cancel latency, since
    # cancel is checked between steps, not mid-execution.
    timeout: int = 60


class ApprovalGateStep(_BaseStep):
    type: Literal["approval_gate"] = "approval_gate"
    ask: str = "Proceed?"


class AiAgentStep(_BaseStep):
    # Forbid unknown keys so a typo'd key fails at load time with a clear "extra
    # inputs are not permitted" error, rather than being silently ignored and
    # surfacing later as a confusing "agent file not found".
    model_config = ConfigDict(extra="forbid")
    type: Literal["ai_agent"] = "ai_agent"
    # `agent` is the system prompt's path relative to the project root, full
    # filename INCLUDING the extension (e.g. "team/agents/docs.md") — no .md is
    # appended for you. It's a single per-step path, so agents can live anywhere
    # (there is no global agents_dir).
    agent: str
    # A user-declared ai_agent step resolves its model as TOML `model` → agent-file
    # frontmatter → this last-resort default. It intentionally does NOT fall back to
    # OrchestratorConfig.default_model: load_manifest() parses the TOML independently
    # of the config, so wiring default_model in would couple the two loaders. A user
    # who wants the run-wide default sets it here or in frontmatter. This is the one
    # model literal that stays outside config.py.
    model: str = "claude-sonnet-4-6"
    # Optional tool/timeout config so an ai_agent def is a first-class producer/gate.
    # These may instead be set in the agent file's frontmatter (model/tools), which
    # load_manifest folds in as defaults — a TOML value here overrides it. When
    # `allowed_tools` is None, the role default applies:
    # ["Read", "Bash", "Grep"] as a gate (Bash lets it run `git diff HEAD` etc.),
    # or ["Read", "Edit", "Write", "Bash", "Grep"] as a producer. `timeout` is the
    # agent-loop wall-clock in seconds (None = no limit). NOTE: the gate default
    # is NOT strictly read-only — Bash can still mutate the tree. A gate that must
    # not write should set allowed_tools = ["Read", "Grep"].
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = Field(default_factory=list)
    timeout: int | None = None
    # When true, pause AFTER the agent runs (before the workflow continues) so a
    # human can inspect what it produced. Same reply contract as approval_gate: an
    # abort word ('abort'/'no'/'stop') stops the run; anything else proceeds.
    # For an agent placed directly at a seam the pause fires right after it runs.
    # For a [steps.defs.*] agent used as a retry-block PRODUCER it fires once,
    # after the block succeeds (the gate passed) — not on intermediate failed
    # attempts, and not if the block exhausts its budget. Ignored on a retry-block
    # gate (a verdict-only judge run every attempt).
    human_in_loop: bool = False


class RetryConfig(BaseModel):
    """Retry behaviour for a build step's producer⇄gate loop.

    `max` is the attempt budget (>= 1); `on_exhausted` decides what happens when
    it runs out: abort the run, ask a human, or proceed anyway.

    Under on_exhausted="approval_gate" a human may reply with a count at the
    exhaustion prompt to grant more attempts. `max_total_attempts` is an
    optional hard ceiling on the TOTAL attempts a run may reach that way
    (None = unbounded); an over-large grant is clamped to it.
    """

    model_config = ConfigDict(extra="forbid")
    max: int = Field(default=3, ge=1)
    on_exhausted: Literal["abort", "approval_gate", "proceed"] = "abort"
    max_total_attempts: int | None = Field(default=None, ge=1)


class HumanInLoopConfig(BaseModel):
    """Per-build human pauses, configured where the build is declared, so the
    pauses work for ANY producer/gate (not only the built-in implementation/qa).

    - `after_producer`: pause after the producer(s) run, before the gate(s), on
      every attempt.
    - `on_gate_fail`: pause when a gate fails; reply with an abort word
      ('abort'/'no'/'stop') to stop the run, anything else to retry. Both default
      off.
    """

    model_config = ConfigDict(extra="forbid")
    after_producer: bool = False
    on_gate_fail: bool = False


class BuildStep(_BaseStep):
    """A declarative build step injected at a seam.

    Run producer(s), then gate(s), re-running the producers — with the failing
    gate's feedback injected — until a gate passes or `retry.max` is exhausted.
    Gates run in declared order; the first to fail short-circuits the rest and
    triggers a retry. `produce` and `gate` hold ids that reference [steps.defs.*]
    definitions. The gating guarantee: `gate` must be non-empty unless
    `ungated=true` (then the producer runs once, no gate, no retry).
    """

    type: Literal["build"] = "build"
    produce: list[str]
    gate: list[str] = Field(default_factory=list)
    ungated: bool = False
    retry: RetryConfig = Field(default_factory=RetryConfig)
    human_in_loop: HumanInLoopConfig = Field(default_factory=HumanInLoopConfig)


# Steps that can be INJECTED at a seam (a build step is one of them).
Step = Annotated[
    Union[ScriptStep, ApprovalGateStep, AiAgentStep, BuildStep],
    Field(discriminator="type"),
]
_STEP_ADAPTER: TypeAdapter = TypeAdapter(Step)

# Steps that can be DEFINED under [steps.defs.*] and referenced by a retry block
# as a producer or gate. Only executable mutator/checker steps qualify — a
# approval_gate is a pause, not a producer or gate, so it's excluded; and a retry
# block can't reference another block. Both variants are gate-capable: a script
# gate's verdict is its exit code, and an ai_agent gate is run with a
# `passed`-emitting tool at execution time (see steps.execute_ai_agent).
StepDef = Annotated[
    Union[ScriptStep, AiAgentStep],
    Field(discriminator="type"),
]
_STEPDEF_ADAPTER: TypeAdapter = TypeAdapter(StepDef)


class StepResult(BaseModel):
    """Result of running one injected step.

    A single registered type for ALL injected steps, so the workflow's
    serde allowlist stays closed no matter how many steps users add — they
    never touch _ALLOWED_MSGPACK_MODULES.
    """

    step_id: str
    kind: str
    ok: bool = True
    detail: str = ""
    # The COMPLETE runner log (every line, both streams), kept untruncated so
    # the evidence layer (Phase 77) can persist the full run. `detail` stays the
    # short human summary (a tail / failure report); `full_output` is the whole
    # thing. Empty for steps that produce no captured output (e.g. ai_agent).
    full_output: str = ""
    # Gate verdict. None = this step is not a gate; True/False = a gate's
    # pass/fail. The retry engine fails closed on a None in a gate slot.
    passed: bool | None = None
    usage: TaskUsage | None = None


class WorkflowManifest(BaseModel):
    # seam name → ordered steps injected at that seam.
    steps: dict[str, list[Step]] = Field(default_factory=dict)
    # id → step definition, referenced by retry blocks' produce/gate.
    defs: dict[str, StepDef] = Field(default_factory=dict)

    def for_seam(self, seam: str) -> list[Step]:
        return self.steps.get(seam, [])

    def is_empty(self) -> bool:
        # defs alone (with no seam steps referencing them) execute nothing.
        return not any(self.steps.values())

    def _hashable_step(self, step: Step) -> dict:
        d = step.model_dump()
        if isinstance(step, BuildStep):
            # Resume safety: fold the referenced defs into the block's hashed form
            # so editing a def *body* (not just the block) also
            # refuses the resume. The block dump alone only names def ids, so
            # without this a changed lint.sh / agent would resume silently.
            d["_resolved_defs"] = {
                rid: self.defs[rid].model_dump()
                for rid in (step.produce + step.gate)
                if rid in self.defs
            }
        return d

    def manifest_hash(self) -> str:
        """Stable hash of the resolved manifest.

        Snapshotted into the run's first checkpoint; compared on resume so a
        mid-run orchestrator.toml edit refuses the resume, the same way an
        incompatible WORKFLOW_VERSION does.
        """
        canonical = {
            seam: [self._hashable_step(s) for s in self.steps.get(seam, [])]
            for seam in SEAMS
            if self.steps.get(seam)
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _agent_file(project_root: Path, step: AiAgentStep) -> Path:
    return project_root / step.agent


def _resolve_agent_frontmatter(project_root: Path, step: AiAgentStep) -> AiAgentStep:
    """Fold the agent file's frontmatter into the step as defaults.

    The agent `.md` is the definition: its frontmatter supplies model/tools so a
    downloaded file plugs in without restating them in TOML. An operational key
    set explicitly on the TOML entry still wins (per-project override); a key the
    TOML omitted takes the frontmatter value. Resolving HERE, at load time, means
    `manifest_hash` (which dumps the resolved step) folds the frontmatter in, so
    editing either source correctly refuses a mid-run resume."""
    fm, _body = parse_agent_frontmatter(
        _agent_file(project_root, step).read_text(encoding="utf-8")
    )
    updates = {}
    for field in ("model", "allowed_tools", "disallowed_tools", "timeout", "human_in_loop"):
        if field in step.model_fields_set:
            continue  # an explicit TOML value overrides the frontmatter default
        value = getattr(fm, field)
        if value is not None:
            updates[field] = value
    return step.model_copy(update=updates) if updates else step


def _validate_build_step(step: BuildStep, defs: dict[str, StepDef]) -> None:
    """Validate a build step's references against [steps.defs.*].

    retry.max (>= 1) and retry.on_exhausted are already enforced by the Pydantic
    model; this covers the cross-references and the gating guarantee.
    """
    if not step.produce:
        raise ManifestError(
            f"build step {step.id!r}: `produce` must list at least one "
            f"[steps.defs.*] id."
        )
    if not step.gate and not step.ungated:
        raise ManifestError(
            f"build step {step.id!r}: `gate` must list at least one "
            f"[steps.defs.*] id, or set `ungated = true` to run the producer "
            f"once without a gate."
        )
    both = sorted(set(step.produce) & set(step.gate))
    if both:
        raise ManifestError(
            f"build step {step.id!r}: {both} listed as both producer and gate; "
            f"a step is one or the other."
        )
    # Producer/gate ids must resolve to a [steps.defs.*] entry OR a built-in:
    # `implementation` as a producer, `qa` as a gate. The built-ins let a user
    # build that wants the spine's own agents reference them without a redundant
    # def.
    for rid in step.produce:
        if rid not in defs and rid not in _BUILTIN_PRODUCER_IDS:
            raise ManifestError(
                f"build step {step.id!r}: references unknown producer {rid!r}. "
                f"Define it under [steps.defs.{rid}] (or use the built-in "
                f"{sorted(_BUILTIN_PRODUCER_IDS)})."
            )
    for rid in step.gate:
        if rid not in defs and rid not in _BUILTIN_GATE_IDS:
            raise ManifestError(
                f"build step {step.id!r}: references unknown gate {rid!r}. "
                f"Define it under [steps.defs.{rid}] (or use the built-in "
                f"{sorted(_BUILTIN_GATE_IDS)})."
            )
    # Gate-capability: defs are restricted to script | ai_agent (StepDef
    # excludes approval_gate), and both are gate-capable — a script gate's verdict
    # is its exit code; an ai_agent gate is run with a `passed`-emitting tool.
    # So a referenced gate id is always gate-capable; no further check needed.


def load_manifest(
    config_path: Path | None = None, project_root: Path | None = None
) -> WorkflowManifest:
    """Load and validate the step manifest from orchestrator.toml.

    Returns an empty manifest when the file or its [steps] table is absent.
    Raises ManifestError on any validation problem — unknown seam, duplicate
    id, missing script path, or unknown agent reference.
    """
    if project_root is None:
        project_root = find_project_root()
    if config_path is None:
        config_path = project_root / "orchestrator.toml"

    if not config_path.exists():
        return WorkflowManifest()

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    raw = data.get("steps", {})
    if not raw:
        return WorkflowManifest()
    if not isinstance(raw, dict):
        raise ManifestError(
            "[steps] must be a table of seam arrays, e.g. [[steps.work]]"
        )

    # All step ids (seam steps AND [steps.defs.*]) share one namespace so a
    # retry block can reference a def unambiguously by id.
    seen_ids: dict[str, str] = {}

    # [steps.defs.*] — the producer/gate step definitions referenced by retry
    # blocks. A table keyed by id (NOT a seam array), so pull it out
    # before the seam loop and parse it first, so retry blocks can be validated
    # against it.
    defs_raw = raw.pop("defs", {})
    defs: dict[str, StepDef] = {}
    if defs_raw:
        if not isinstance(defs_raw, dict):
            raise ManifestError(
                "[steps.defs] must be a table of named step definitions, "
                "e.g. [steps.defs.my-gate]."
            )
        for def_id, body in defs_raw.items():
            if not isinstance(body, dict):
                raise ManifestError(
                    f"[steps.defs.{def_id}] must be a table with a `type`."
                )
            try:
                # The id is the table key; inject it so the model is complete.
                definition = _STEPDEF_ADAPTER.validate_python({**body, "id": def_id})
            except ValidationError as exc:
                raise ManifestError(
                    f"invalid step def {def_id!r}: {exc}"
                ) from exc
            if def_id in seen_ids:
                raise ManifestError(
                    f"duplicate step id {def_id!r}; ids must be unique."
                )
            seen_ids[def_id] = "steps.defs"
            if isinstance(definition, ScriptStep):
                if not (project_root / definition.path).exists():
                    raise ManifestError(
                        f"step def {def_id!r}: script not found at "
                        f"{definition.path!r}."
                    )
            elif isinstance(definition, AiAgentStep):
                if not _agent_file(project_root, definition).exists():
                    raise ManifestError(
                        f"step def {def_id!r}: agent file not found at "
                        f"{definition.agent}."
                    )
                definition = _resolve_agent_frontmatter(project_root, definition)
            defs[def_id] = definition

    steps: dict[str, list[Step]] = {}

    for seam, items in raw.items():
        if seam not in SEAMS:
            if seam in _REMOVED_SEAMS:
                raise ManifestError(
                    f"seam {seam!r} was removed in Phase 46: "
                    f"{_REMOVED_SEAMS[seam]} Valid seams: {', '.join(SEAMS)}."
                )
            raise ManifestError(
                f"unknown seam {seam!r}. Valid seams: {', '.join(SEAMS)}."
            )
        if not isinstance(items, list):
            raise ManifestError(
                f"[steps.{seam}] must be an array of tables ([[steps.{seam}]])."
            )

        parsed: list[Step] = []
        for item in items:
            try:
                step = _STEP_ADAPTER.validate_python(item)
            except ValidationError as exc:
                raise ManifestError(
                    f"invalid step in seam {seam!r}: {exc}"
                ) from exc

            if step.id in seen_ids:
                raise ManifestError(
                    f"duplicate step id {step.id!r} (in {seen_ids[step.id]!r} "
                    f"and {seam!r}); ids must be unique."
                )
            seen_ids[step.id] = seam

            if isinstance(step, ScriptStep):
                if not (project_root / step.path).exists():
                    raise ManifestError(
                        f"step {step.id!r}: script not found at {step.path!r}."
                    )
            elif isinstance(step, AiAgentStep):
                if not _agent_file(project_root, step).exists():
                    raise ManifestError(
                        f"step {step.id!r}: agent file not found at "
                        f"{step.agent}."
                    )
                step = _resolve_agent_frontmatter(project_root, step)
            elif isinstance(step, BuildStep):
                _validate_build_step(step, defs)

            parsed.append(step)

        steps[seam] = parsed

    return WorkflowManifest(steps=steps, defs=defs)
