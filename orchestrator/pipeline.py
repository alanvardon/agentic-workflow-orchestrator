"""Phase 68: the v2 declarative pipeline — `stage` / `flow` / `builtin` / `defs`.

Parses the v2 config shape (see orchestrator.v2.example.toml at the project root)
into a resolved, validated, flow-ordered Pipeline.

This is the FOUNDATION layer for Phase 68: it is fully self-contained and
unit-tested, and is deliberately NOT yet wired into the live workflow (that
cutover is the follow-up). The running orchestrator still uses the v1 config in
config.py / manifest.py. Building v2 additively keeps the existing suite green
while the new model is proven in isolation.

Two axes decide where a thing lives (mirrors the example file):

    IN THE FLOW?           BUILT-IN?     TABLE
    ----------------------------------------------------------
    yes (a stage)          built-in      [stage.builtin.<id>]
    yes (a stage)          yours         [stage.user.<id>]
    no  (a reusable part)  built-in      [builtin.<id>]
    no  (a reusable part)  yours         [defs.<id>]

A STAGE's id IS its header's last segment. A reusable PART is referenced from a
stage's produce / gate / uses by a prefixed id: `builtin:<id>` or `defs:<id>`.
Order comes only from `flow`; the git rails are implicit, locked anchors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestrator.errors import FatalError
from orchestrator.flow import FlowGraph, parse as parse_flow
from orchestrator.manifest import HumanInLoopConfig, RetryConfig


class PipelineError(FatalError):
    """A malformed or invalid v2 pipeline config. Fail-loud at load."""


# Built-in STAGES (can appear as [stage.builtin.<id>]) → their fixed, known type.
# `type` on a built-in stage is a VALIDATED label: it must match this.
BUILTIN_STAGE_TYPES: dict[str, str] = {
    "plan": "ai_agent",
    "decompose": "ai_agent",
    "task-build": "build",
    "docs": "ai_agent",
    "summarize": "ai_agent",
    "qa": "ai_agent",
}

# Built-in reusable PARTS (referenced via `builtin:<id>`) → their known type.
BUILTIN_PART_TYPES: dict[str, str] = {
    "implementation": "ai_agent",
    "qa": "ai_agent",
}

# Deterministic git rails: implicit, locked anchors. They may never be used as a
# stage id or appear in `flow` — they auto-wrap the pipeline.
RAILS: tuple[str, ...] = ("verify-clean-tree", "branch", "commit", "push", "open-pr")

_PART_TYPES = ("script", "ai_agent")


class StageSpec(BaseModel):
    """One pipeline stage. `id` is the header's last segment; `namespace` is
    'builtin' or 'user'. Most fields are role-specific and validated in
    build_pipeline (a flat model keeps TOML round-tripping simple and gives
    clearer errors than a discriminated union of near-identical shapes)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    namespace: str  # "builtin" | "user"

    # Common
    type: str | None = None
    model: str | None = None
    path: str | None = None              # optional override on built-ins; required on user
    timeout: int | None = None
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = Field(default_factory=list)
    # bool for single-agent stages (e.g. plan); a HumanInLoopConfig dict for a
    # build stage's per-attempt pauses.
    human_in_loop: bool | HumanInLoopConfig = False
    uses: str | None = None              # place a part as a stage: "builtin:<id>" | "defs:<id>"

    # decompose
    max_tasks: int = 0

    # build / task-build
    produce: list[str] = Field(default_factory=list)
    gate: list[str] = Field(default_factory=list)
    ungated: bool = False
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @property
    def effective_type(self) -> str:
        """The resolved type: built-in stages inherit their known type; user
        stages declare it (or are typed by what they `uses`)."""
        if self.type is not None:
            return self.type
        if self.namespace == "builtin":
            return BUILTIN_STAGE_TYPES.get(self.id, "ai_agent")
        return "ai_agent"


class PartSpec(BaseModel):
    """A reusable producer/gate referenced by produce / gate / uses. Lives under
    [builtin.<id>] (namespace 'builtin') or [defs.<id>] (namespace 'defs')."""

    model_config = ConfigDict(extra="forbid")

    id: str
    namespace: str  # "builtin" | "defs"
    type: str | None = None
    path: str | None = None
    model: str | None = None
    tools: list[str] | None = None        # alias for allowed_tools (TOML ergonomics)
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] = Field(default_factory=list)
    timeout: int | None = None            # agent-loop / script wall-clock seconds


@dataclass(frozen=True)
class Pipeline:
    """A resolved, validated v2 pipeline."""

    flow: FlowGraph
    stages: tuple[StageSpec, ...]          # in flow order
    parts: dict[str, PartSpec]             # keyed "builtin:<id>" / "defs:<id>"

    def stage(self, sid: str) -> StageSpec:
        for s in self.stages:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def manifest_hash(self) -> str:
        """Stable hash of the resolved pipeline (flow order + each stage body +
        the bodies of any parts it references). Mirrors manifest.manifest_hash:
        folding referenced part bodies in means editing a part also changes the
        hash, so a mid-run config edit can be refused on resume."""
        canonical = {
            "flow": [list(g) for g in self.flow.groups],
            "stages": [
                {
                    **s.model_dump(mode="json"),
                    "_resolved_parts": {
                        ref: self.parts[ref].model_dump(mode="json")
                        for ref in (s.produce + s.gate + ([s.uses] if s.uses else []))
                        if ref in self.parts
                    },
                }
                for s in self.stages
            ],
        }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ── parsing + validation ─────────────────────────────────────────────────────

def build_pipeline(data: dict) -> Pipeline:
    """Validate a v2 config dict into a resolved Pipeline, or raise PipelineError.

    `data` is the parsed TOML (tomllib.load). Only the pipeline-shaping tables are
    read here: `flow`, `[stage.*]`, `[builtin.*]`, `[defs.*]`. Infra/rail tables
    (`[branch]`, `[git]`, `[pr]`, `[pre_hooks]`, `[audit]`) are owned elsewhere.
    """
    if "flow" not in data:
        raise PipelineError("missing required `flow` line.")
    flow = parse_flow(data["flow"])

    stages = _parse_stages(data)
    _supply_builtin_stage_defaults(flow, stages)
    parts = _parse_parts(data)

    _validate_flow_coverage(flow, stages)
    _validate_stages(stages, parts, flow)

    ordered = tuple(stages[sid] for sid in flow.ordered_ids())
    return Pipeline(flow=flow, stages=ordered, parts=parts)


def assert_shippable(pipeline: Pipeline) -> None:
    """Reject a pipeline that can never produce a commit/PR summary.

    Every run ships — the commit / push / open-pr git rails are implicit and
    always run — and shipping needs the summary the `summarize` stage produces.
    So a pipeline without a `summarize` stage is fail-loud at load; there is no
    silent fallback (Phase 68b, decision Q4).

    Kept OUT of build_pipeline (which is the pure schema validator) so unit tests
    can still build minimal, non-shipping pipelines. config.load_config calls
    this after build_pipeline, which is the real fail-loud-at-load point.
    """
    if not any(s.id == "summarize" for s in pipeline.stages):
        raise PipelineError(
            "the pipeline has no `summarize` stage, but every run ships (commit / "
            "push / open-pr) and needs the commit/PR summary it produces. Add "
            "`summarize` to the flow and a [stage.builtin.summarize] table — see "
            "orchestrator.v2.example.toml."
        )


def _parse_stages(data: dict) -> dict[str, StageSpec]:
    stage_root = data.get("stage", {})
    if not isinstance(stage_root, dict):
        raise PipelineError("[stage] must be a table of [stage.builtin.*] / [stage.user.*].")
    out: dict[str, StageSpec] = {}
    for namespace in ("builtin", "user"):
        tables = stage_root.get(namespace, {})
        if not isinstance(tables, dict):
            raise PipelineError(f"[stage.{namespace}] must be a table of named stages.")
        for sid, body in tables.items():
            if not isinstance(body, dict):
                raise PipelineError(f"[stage.{namespace}.{sid}] must be a table.")
            if sid in out:
                raise PipelineError(
                    f"stage id {sid!r} is defined in both stage namespaces; ids "
                    "must be unique across stage.builtin and stage.user."
                )
            try:
                out[sid] = StageSpec(id=sid, namespace=namespace, **body)
            except ValidationError as exc:
                raise PipelineError(f"[stage.{namespace}.{sid}]: {exc}") from exc
    return out


# Default bodies for built-in stages auto-supplied when named in `flow` without a
# table. Only `task-build` needs more than an empty body (its produce/gate recipe);
# the rest infer everything (type from BUILTIN_STAGE_TYPES, model from default).
_BUILTIN_STAGE_DEFAULT_BODY: dict[str, dict] = {
    "task-build": {
        "produce": ["builtin:implementation"],
        "gate": ["builtin:qa"],
        "retry": {"max": 3, "on_exhausted": "approval_gate"},
    },
}


def _supply_builtin_stage_defaults(
    flow: FlowGraph, stages: dict[str, StageSpec]
) -> None:
    """A flow id naming a known built-in stage with no [stage.builtin.<id>] table
    gets a default StageSpec, so a config can list a built-in in `flow` and
    override only the stages it cares about (or none — `flow` alone suffices)."""
    for sid in flow.ordered_ids():
        if sid in stages or sid not in BUILTIN_STAGE_TYPES:
            continue
        body = _BUILTIN_STAGE_DEFAULT_BODY.get(sid, {})
        stages[sid] = StageSpec(id=sid, namespace="builtin", **body)


def _parse_parts(data: dict) -> dict[str, PartSpec]:
    out: dict[str, PartSpec] = {}
    for namespace, prefix in (("builtin", "builtin"), ("defs", "defs")):
        tables = data.get(namespace, {})
        if not isinstance(tables, dict):
            raise PipelineError(f"[{namespace}] must be a table of named parts.")
        for pid, body in tables.items():
            if not isinstance(body, dict):
                raise PipelineError(f"[{namespace}.{pid}] must be a table.")
            try:
                out[f"{prefix}:{pid}"] = PartSpec(id=pid, namespace=namespace, **body)
            except ValidationError as exc:
                raise PipelineError(f"[{namespace}.{pid}]: {exc}") from exc
    return out


def _validate_flow_coverage(flow: FlowGraph, stages: dict[str, StageSpec]) -> None:
    flow_ids = flow.ordered_ids()
    for sid in flow_ids:
        if sid in RAILS:
            raise PipelineError(
                f"flow references {sid!r}, which is a locked git rail — rails are "
                "implicit and must not appear in `flow`."
            )
        if sid not in stages:
            raise PipelineError(
                f"flow references stage {sid!r}, but no [stage.builtin.{sid}] or "
                f"[stage.user.{sid}] is defined."
            )
    flow_set = set(flow_ids)
    for sid in stages:
        if sid not in flow_set:
            raise PipelineError(
                f"stage {sid!r} is defined but never appears in `flow` (orphan)."
            )


def _validate_stages(
    stages: dict[str, StageSpec], parts: dict[str, PartSpec], flow: FlowGraph
) -> None:
    for sid, s in stages.items():
        if sid in RAILS:
            raise PipelineError(f"{sid!r} is a reserved git rail and cannot be a stage id.")
        if s.namespace == "builtin":
            _validate_builtin_stage(s)
        else:
            _validate_user_stage(s)
        _validate_refs(s, parts)

    # Built-in parts referenced as gates/producers must be known.
    _validate_part_namespaces(parts)

    # task-build placement: at most one build stage; if the built-in `task-build`
    # is present it must come after `decompose`.
    _validate_build_ordering(stages, flow)


def _validate_builtin_stage(s: StageSpec) -> None:
    if s.id not in BUILTIN_STAGE_TYPES:
        raise PipelineError(
            f"[stage.builtin.{s.id}]: {s.id!r} is not a known built-in stage. "
            f"Known: {sorted(BUILTIN_STAGE_TYPES)}. (For your own stage use "
            "[stage.user.<id>].)"
        )
    known = BUILTIN_STAGE_TYPES[s.id]
    if s.type is not None and s.type != known:
        raise PipelineError(
            f"[stage.builtin.{s.id}]: type={s.type!r} does not match the built-in's "
            f"known type {known!r}. `type` is a validated label; fix or omit it."
        )
    if s.uses is not None:
        raise PipelineError(
            f"[stage.builtin.{s.id}]: `uses` is for placing a part under "
            "[stage.user.<id>]; a built-in stage already names its implementation."
        )


def _validate_user_stage(s: StageSpec) -> None:
    if s.uses is not None:
        # placing a reusable part as a stage; type/path come from the part.
        return
    if s.type == "build":
        # a user-declared build loop; its produce/gate are checked in _validate_refs.
        return
    if s.type not in _PART_TYPES:
        raise PipelineError(
            f"[stage.user.{s.id}]: a user stage needs type = \"script\", "
            f"\"ai_agent\", or \"build\" (or a `uses` pointing at a part); "
            f"got type={s.type!r}."
        )
    if not s.path:
        raise PipelineError(
            f"[stage.user.{s.id}]: `path` is required on a user stage."
        )


def _validate_refs(s: StageSpec, parts: dict[str, PartSpec]) -> None:
    """produce/gate/uses must resolve to a known part or built-in part id."""
    def resolve(ref: str, where: str) -> None:
        if ref in parts:
            return
        if ref.startswith("builtin:") and ref.split(":", 1)[1] in BUILTIN_PART_TYPES:
            return  # a built-in part used without an explicit [builtin.*] override
        raise PipelineError(
            f"stage {s.id!r}: {where} references {ref!r}, which is neither a "
            "defined part ([builtin.*] / [defs.*]) nor a known built-in "
            f"(builtin:{sorted(BUILTIN_PART_TYPES)})."
        )

    for ref in s.produce:
        resolve(ref, "produce")
    for ref in s.gate:
        resolve(ref, "gate")
    if s.uses is not None:
        resolve(s.uses, "uses")

    # A build stage must gate (unless ungated). Only meaningful when it produces.
    if s.produce and not s.gate and not s.ungated:
        raise PipelineError(
            f"stage {s.id!r}: a build stage must list at least one `gate` "
            "(or set ungated = true to run the producer once)."
        )
    if (s.gate or s.produce) and s.effective_type not in ("build",) and s.namespace == "builtin":
        # produce/gate only belong on the build station among built-ins.
        raise PipelineError(
            f"[stage.builtin.{s.id}]: produce/gate are only valid on the "
            "'task-build' station."
        )


def _validate_part_namespaces(parts: dict[str, PartSpec]) -> None:
    for ref, p in parts.items():
        if p.namespace == "builtin":
            if p.id not in BUILTIN_PART_TYPES:
                raise PipelineError(
                    f"[builtin.{p.id}]: {p.id!r} is not a known built-in part. "
                    f"Known: {sorted(BUILTIN_PART_TYPES)}. (For your own part use "
                    "[defs.<id>].)"
                )
            if p.type is not None and p.type != BUILTIN_PART_TYPES[p.id]:
                raise PipelineError(
                    f"[builtin.{p.id}]: type={p.type!r} != known "
                    f"{BUILTIN_PART_TYPES[p.id]!r}."
                )
        else:  # defs
            if p.type not in _PART_TYPES:
                raise PipelineError(
                    f"[defs.{p.id}]: needs type = \"script\" or \"ai_agent\"; "
                    f"got {p.type!r}."
                )
            if not p.path:
                raise PipelineError(f"[defs.{p.id}]: `path` is required.")


def _validate_build_ordering(stages: dict[str, StageSpec], flow: FlowGraph) -> None:
    # The per-task fan-out station is the single built-in `task-build` (one TOML
    # key, so inherently unique). User-declared `build` stages (ordinary single
    # build loops, e.g. a migrated [[steps.work]] lint-loop) may appear freely.
    order = flow.ordered_ids()
    if "task-build" in stages and "decompose" in stages:
        if order.index("decompose") > order.index("task-build"):
            raise PipelineError(
                "`decompose` must come before `task-build` in the flow."
            )
    if "task-build" in stages and "decompose" not in stages:
        raise PipelineError(
            "`task-build` requires a `decompose` stage earlier in the flow."
        )
