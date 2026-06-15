# load_dotenv reads the .env file in the current working directory and sets
# environment variables. AsyncAnthropic() (inside run_structured_completion) with
# no args picks up ANTHROPIC_API_KEY from os.environ; load it before the first call.
from dotenv import load_dotenv
load_dotenv()

import asyncio
import sys
from typing import Literal

from pydantic import BaseModel, Field

from orchestrator.usage import TaskUsage
from orchestrator.agents.runner import run_structured_completion
from orchestrator.prompt_loader import load_prompt


_DECOMPOSE_SYSTEM_PROMPT = load_prompt("decompose")


# The unit of a decomposed plan. This model is the contract the per-task loop
# depends on, so its fields are deliberately limited to (a) what a fresh-context
# producer / a gate / a human reviewer actually need, and (b) what a plan-only
# decomposer can produce reliably. That rules out `files` (a guess without repo
# access — misleading if wrong) and any `depends_on`/DAG (order is list order;
# later tasks see earlier tasks' edits via the working tree).
class Task(BaseModel):
    id: str = Field(
        description="stable, unique kebab-case slug for this task, e.g. 'add-toggle-markup'"
    )
    title: str = Field(description="short, human-scannable name")
    description: str = Field(
        description="what THIS task changes — its slice of the plan, not a "
                    "restatement of the whole plan"
    )
    acceptance_criteria: str = Field(
        description="REQUIRED: one or more concrete, checkable statements of the "
                    "observable behaviour that confirms this task is done — naming "
                    "the input/action and the observed result. It is the spec the "
                    "test-author writes tests against (Phase 72b); state behaviour, "
                    "not implementation steps.",
    )
    # Phase 81: per-task testability, consulted ONLY under config.tdd. True → the
    # task gets the full red-green station; False → the station skips the
    # test-author/critic and runs the classic implement→qa path (so markup/CSS/
    # docs tasks don't pay a full author leg to discover they're born-green).
    # Default True keeps tdd-OFF and pre-81 checkpoints behaving as before.
    testable: bool = Field(
        default=True,
        description="true ONLY if this task's acceptance behaviour is genuinely "
                    "unit-testable deterministic logic (calculation, parsing, "
                    "data/state transforms); false for presentation/copy, markup, "
                    "CSS, documentation, config, or pure file renames/moves, whose "
                    "tests would pass on the first run and prove nothing. When in "
                    "doubt, set true.",
    )


# Schema used as the emit_decomposition tool's input_schema. Excludes
# schema_version/usage so the model is never asked to fill those in — mirrors the
# _PlanSchema / PlanResult split in planning.py.
class _DecompositionSchema(BaseModel):
    # Phase 82b: judged BEFORE the task list (declared first so it anchors the
    # split — the model commits to a size, then fills a budget). Informs task
    # count; it is NOT a mechanical clamp (under-splitting costs more in retries
    # than an extra clean task). Defaulted → additive-optional, resume-safe, no
    # schema_version bump, same as Task.testable (Phase 81).
    complexity: Literal["trivial", "moderate", "complex"] = Field(
        default="moderate",
        description="Overall implementation complexity of the WHOLE feature, judged "
                    "before you split — it sets your task budget. 'trivial' = a "
                    "single file or single behaviour (a copy/CSS tweak, a rename "
                    "sweep, a one-line config change) → emit 1 task. 'moderate' = "
                    "one main behaviour with a little surrounding work → 1-3 tasks. "
                    "'complex' = genuinely several independent behaviours or "
                    "subsystems → split by behaviour, as many as the plan needs. "
                    "Size from the feature, NOT from the plan's paragraph/step count. "
                    "Each task is a fresh agent with its own QA round, so splitting "
                    "is never free — only split when behaviours are truly independent.",
    )
    tasks: list[Task] = Field(
        description="ordered list of tasks; a later task may build on earlier "
                    "ones. Emit a SINGLE task when the change is atomic."
    )


class DecompositionResult(_DecompositionSchema):
    # Schema version. Bump on incompatible shape changes; pure additions of
    # optional fields don't need a bump. Not part of the tool schema.
    # v2 (Phase 72b): Task.acceptance_criteria became required (was optional).
    schema_version: int = 2
    # Populated after the API call returns; not part of the LLM tool schema.
    usage: TaskUsage | None = None


# Phase 78b: under TDD a separate test-author writes each task's failing tests
# BEFORE implementation, so a standalone "write tests" task is redundant and
# harmful — it runs after the code exists, can't go red, and degrades.
# Phase 81: the note ALSO asks the decomposer to mark each task `testable`, so the
# station can skip the test-author/critic on non-unit-testable tasks (markup/CSS/
# docs) instead of paying a full author leg to find they're born-green. This is a
# behaviour judgement from the task description (still no repo access, contra the
# original Phase 73 caution) — biased to `true`, with the runtime untestable/
# born-green escapes as the backstop for mis-tags. Stated in the user message (not
# the static prompt) because both halves are conditional on `config.tdd`.
_TDD_DECOMPOSE_NOTE = (
    "\n\nThis run uses test-driven development: a separate test-author writes the "
    "FAILING tests for each task before it is implemented, so the tests are owned "
    "elsewhere. Do NOT emit any standalone 'write tests' / 'add tests' / 'unit "
    "tests' task. Every task must be behaviour-only, with acceptance criteria the "
    "test-author will turn into tests.\n\nFor EACH task set `testable`: true only "
    "when its acceptance behaviour is genuinely unit-testable deterministic logic "
    "(calculation, parsing, data/state transforms); false for presentation/copy, "
    "markup, CSS, documentation, config, or pure file renames/moves — work whose "
    "test would pass on the first run and prove nothing. When in doubt, set true."
)


def _build_user_message(plan_text: str, max_tasks: int, tdd: bool = False) -> str:
    """The decomposer's user message: the plan, plus optional `max_tasks` guidance
    and, under TDD, the no-standalone-test-task note (Phase 78b).

    Pure (no I/O) so the cap and TDD-note behaviour are unit-testable without the
    SDK. The prompt already asks for the fewest tasks; this is the soft cap. There
    is no hard validation error — an over-split run should surface for review, not
    abort."""
    guidance = (
        f"\n\nProduce at most {max_tasks} task(s); prefer the fewest tasks that "
        "are each independently checkable."
        if max_tasks and max_tasks > 0
        else ""
    )
    tdd_note = _TDD_DECOMPOSE_NOTE if tdd else ""
    return f"## Plan\n\n{plan_text}{guidance}{tdd_note}"


async def decompose(
    plan_text: str, model: str, max_tasks: int = 0, tdd: bool = False
) -> DecompositionResult:
    """Turn an approved plan into an ordered task list.

    Same forced-tool-use structured-output path as planning.plan(), via the shared
    run_structured_completion. Reads ONLY `plan_text` — no repo access (repo-aware
    decomposition is a later phase). `tdd` injects the no-standalone-test-task note
    (Phase 78b) when the run is test-driven.
    """
    return await run_structured_completion(
        system_prompt=_DECOMPOSE_SYSTEM_PROMPT,
        user_message=_build_user_message(plan_text, max_tasks, tdd),
        model=model,
        tool_name="emit_decomposition",
        tool_description="Emit the ordered task breakdown of the approved plan.",
        schema=_DecompositionSchema,
        result_model=DecompositionResult,
    )


# Allow `python -m orchestrator.agents.decompose "the plan text"` from the terminal.
if __name__ == "__main__":
    from orchestrator.config import OrchestratorConfig

    text = " ".join(sys.argv[1:]) or (
        "Add a dark mode toggle to the header and persist the choice in localStorage."
    )
    result = asyncio.run(decompose(text, OrchestratorConfig().default_model))
    print(result.model_dump_json(indent=2))
