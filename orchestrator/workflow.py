import asyncio
import functools
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# Version of the @entrypoint body. Bump on INCOMPATIBLE body changes — reordered/
# removed tasks, new required tasks, changed control flow a half-finished checkpoint
# can't safely resume into. Pure additive changes (a new optional gate, a new
# trailing task legacy checkpoints haven't reached) don't strictly require a bump.
# On resume, the version stored at run creation is compared against this constant; a
# mismatch refuses the resume with a clear error rather than a confusing mid-run
# deserialization failure. Per-version history: ../CHANGELOG.md.
#
# 2.0.0 (Phase 68b): the declarative-pipeline cutover. The body now dispatches
# the post-branch stages from config.pipeline (the v2 `flow`) instead of the
# hard-coded spine + v1 [[steps.work]] seam; the manifest-hash resume gate became
# a pipeline-hash gate. An incompatible bump — old 1.x checkpoints cannot resume.
#
# 2.1.0 (Phase 72): the per-task station gained a run-once `test_author_task`
# (gated on config.tdd) before each task's implement loop — a new checkpointed
# @task in the body's task graph, so a half-finished 2.0.0 checkpoint can't safely
# resume into it. With tdd off (the default) the task is never called and the
# graph is identical to 2.0.0, but the body now CAN emit it, so the bump.
#
# 2.2.0 (Phase 72b): Task.acceptance_criteria became REQUIRED (was optional). The
# DecompositionResult is checkpointed; a pre-2.2.0 checkpoint may carry a task with
# no criteria, which now fails to deserialize — the version gate refuses that
# resume cleanly instead of a confusing mid-run ValidationError.
#
# 2.3.0 (Phase 72b): the supervised red-review pause + re-author escape. The per-
# task station now runs `_author_with_review`, which can emit a new `red_review`
# interrupt after red-confirm and loop `test_author_task` (with feedback/attempt) to
# re-author. New control flow + interrupt the body can reach → a half-finished 2.2.0
# checkpoint can't safely resume into it. Gated on config.tdd_red_review (default on).
#
# 2.4.0 (Phase 74): the per-task coverage critic. _author_with_review now runs a new
# checkpointed `critic_task` on the authored tests and can loop `test_author_task` to
# re-author on a negative verdict. A new @task in the body's graph → a half-finished
# 2.3.0 checkpoint can't safely resume into it. Gated on config.tdd_coverage_critic
# (default on); CoverageCriticResult is on the serde allowlist.
#
# 2.5.0 (Phase 76): autonomous-mode TDD. tdd + fully_autonomous is no longer refused
# at load; instead the per-task station runs a distinct `_run_autonomous_tdd_task`
# when both are on — a HARD red-confirm gate (a born-green / non-green-baseline /
# no-script-gate verdict aborts the task instead of degrading) plus a BOUNDED
# re-author cycle (the implement build runs bounded with on_exhausted="abort"; an
# exhausted build re-authors the tests, capped at tdd_autonomous_reauthor_max, then
# aborts). New control flow + new test_author_task / critic_task invocation patterns
# (per-round `attempt`) the body can now reach → a half-finished 2.4.0 checkpoint
# can't safely resume into it. With tdd or fully_autonomous off the supervised /
# non-TDD per-task graph is unchanged. No serde change (TestAuthorResult.degrade_kind
# is an additive optional field).
#
# 2.6.0 (Phase 77a): evidence foundations. StepResult gains a `full_output` field
# (the complete, untruncated runner log) and run_retry_block gains an `on_attempt`
# per-attempt hook. Both are additive and inert until a consumer is wired (77c), so
# the graph is unchanged — but a checkpointed StepResult shape changed, so a
# half-finished 2.5.0 TDD run can't resume across the upgrade and needs a fresh start.
#
# 2.7.0 (Phase 77b): test-author evidence folder. TestAuthorResult gains a `full_run`
# field (the COMPLETE final RED run, via 77a's full-output capture) — a checkpointed
# shape change on a serde-allowlisted model, so a half-finished 2.6.0 TDD run can't
# resume across the upgrade. The per-task `test-author/` folder (final tests copied
# verbatim + RED run + freeze hash + process summary.md) replaces the flat
# `test-author-<id>.md`. The post-branch task graph is unchanged (the writer is a
# side-effect after the authored verdict, like its predecessor) and TDD-off runs are
# byte-for-byte identical; the bump is purely for the changed checkpointed shape.
#
# 2.7.0 (Phase 77c, unchanged): impl attempt evidence — the FIRST consumer of both
# 77a seams. The TDD station wires run_retry_block's on_attempt hook (via _run_build_step
# / _run_task_build_step) to write task-NN-<id>/impl/attempt-N/ (the COMPLETE run via
# full_output + the freeze MATCH/MISMATCH vs the 77b baseline) for every implement
# attempt of a TESTABLE task — including the GREEN one. No new @task, no checkpointed
# shape change (the recorder is a best-effort side-effect in the entrypoint frame, like
# the 77b writer), so the graph is byte-for-byte unchanged and no bump is needed; the
# classic / non-TDD path passes on_attempt=None and writes nothing.
#
# 2.8.0 (Phase 81): decompose-time testability gating. The decomposer marks each Task
# `testable` (default True, additive optional → no schema_version bump); under TDD a
# `testable=False` task now SKIPS the test-author/critic and takes the classic
# implement→qa path (a synthesized untestable degrade through the Phase 73 manual_checks
# plumbing) — no `test_author_task` / `critic_task` @tasks are created for it. That
# changes the per-task graph vs a pre-81 checkpoint, so a half-finished 2.7.0 TDD run
# can't resume across the upgrade. tdd-OFF runs are functionally unchanged (the flag is
# only consulted under config.tdd).
WORKFLOW_VERSION = "2.8.0"


from orchestrator.errors import FatalError


class IncompatibleCheckpointError(FatalError):
    """Raised on resume when the checkpoint was created by a different
    WORKFLOW_VERSION than the code now attempting to resume it.

    Carries both versions so callers can show a clear message and decide
    whether to abandon the run and start fresh.
    """

    def __init__(self, stored_version: str, current_version: str) -> None:
        self.stored_version = stored_version
        self.current_version = current_version
        super().__init__(
            f"checkpoint was created with workflow v{stored_version}; "
            f"current is v{current_version}. This run cannot be safely "
            f"resumed — start a fresh run."
        )


class IncompatiblePipelineError(FatalError):
    """Raised on resume when the v2 pipeline in orchestrator.toml was edited since
    the run started. The resolved pipeline's hash is snapshotted into the first
    checkpoint; a different hash on resume means the flow / stages / referenced
    part bodies changed underneath the run, so we refuse rather than resume into a
    shifted graph. The companion of the WORKFLOW_VERSION gate — a second hash over
    the pipeline (Phase 68b; was IncompatibleManifestError over the v1 steps).
    """

    def __init__(self, stored_hash: str, current_hash: str) -> None:
        self.stored_hash = stored_hash
        self.current_hash = current_hash
        super().__init__(
            f"pipeline config changed since this run started "
            f"(snapshot {stored_hash}, current {current_hash}). In-flight "
            f"runs can't absorb a pipeline edit — start a fresh run."
        )


class EmptyDecompositionError(FatalError):
    """Raised when the decomposer returns zero tasks for an approved plan.

    An empty task list would make the per-task station a no-op, the tree stay
    clean, and the run return status="no_changes" — indistinguishable from a
    build that legitimately made no edits. That hides what is almost always a
    decomposer (or plan) failure, so we fail loud instead. A FatalError so the
    MCP server shapes it into a {"status": "fatal", ...} response; raised before
    branch creation, so nothing is half-shipped.
    """

    def __init__(self) -> None:
        super().__init__(
            "decomposition produced no tasks for an approved plan. This is a "
            "decomposer or plan failure, not an empty build — fix the plan/"
            "decomposer and start a fresh run."
        )

# LangGraph's Functional API: @entrypoint marks the top-level workflow
# function, @task marks a checkpointable unit of work. Together they let
# you write a workflow as ordinary async Python and get durability,
# tracing, and resume-on-crash semantics for free.
from langgraph.func import entrypoint, task
from langgraph.types import interrupt

# AsyncSqliteSaver writes checkpoint state to a SQLite file on disk — durable
# across process restarts and crashes. The .aio submodule is the async variant;
# the sync variant lives in langgraph.checkpoint.sqlite.
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# JsonPlusSerializer is the default serde the checkpointer uses to encode
# task inputs/outputs into the SQLite blob columns. We override it below
# with an explicit allowlist of custom types — see _CUSTOM_SERDE.
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph.config import get_config

from orchestrator.agents.decompose import decompose, DecompositionResult
from orchestrator.agents.planning import plan, PlanResult
from orchestrator.agents.qa import qa, QaResult
from orchestrator.agents.summarize import summarize, SummaryResult
from orchestrator.agents.test_author import author_tests, TestAuthorResult
from orchestrator.agents.coverage_critic import critique_tests, CoverageCriticResult
from orchestrator.agents.runner import run_structured_agent
from orchestrator.prompt_loader import load_prompt, load_prompt_frontmatter
from orchestrator.agent_frontmatter import AgentFrontmatter, parse_agent_frontmatter
from orchestrator.retry_block import RetryBlock, feedback_section, run_retry_block
from orchestrator.audit import (
    AuditSink, NoopAuditSink, build_sink, emit_event, failure_payload,
)
from orchestrator.cancellation import WorkflowCancelled, raise_if_cancelled
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.manifest import (
    AiAgentStep,
    BuildStep,
    HumanInLoopConfig,
    ScriptStep,
    StepResult,
)
from orchestrator.steps import StepError, execute_ai_agent, execute_script
from orchestrator.usage import TaskUsage, aggregate_usage, run_usage_rollup
from orchestrator.git_ops import (
    commit,
    create_branch,
    ensure_on_main,
    pr_create,
    push,
    verify_clean_tree,
    working_tree_has_changes,
    PreHookError,
)
from orchestrator.paths import find_project_root, iter_test_files
from orchestrator.pre_hooks import run_pre_hooks
from orchestrator.run_log import append_usage_rollup
from orchestrator.run_artifacts import (
    rename_with_branch,
    write_decomposition,
    write_impl_attempt,
    write_manual_checks,
    write_plan,
    write_qa,
    write_summary,
    write_test_author_folder,
    write_usage,
)


# Future LangGraph versions will refuse to deserialize types that aren't
# on this allowlist (the warning today; a hard error tomorrow). Register
# every Pydantic model that flows through a @task so resume keeps working
# across upgrades. Each entry is (module_path, class_name).
_ALLOWED_MSGPACK_MODULES = [
    ("orchestrator.agents.planning", "PlanResult"),
    # The decomposer's task list. `Task` rides inside DecompositionResult, so only
    # the container type needs registering.
    ("orchestrator.agents.decompose", "DecompositionResult"),
    # The summarizer's commit/PR summary + test_plan.
    ("orchestrator.agents.summarize", "SummaryResult"),
    ("orchestrator.agents.qa", "QaResult"),
    # The test-author's per-task verdict (Phase 72). Checkpointed so a resume
    # replays the authored result rather than re-authoring the tests.
    ("orchestrator.agents.test_author", "TestAuthorResult"),
    # The coverage critic's per-task verdict (Phase 74). Checkpointed so a resume
    # replays the verdict rather than re-running the critic.
    ("orchestrator.agents.coverage_critic", "CoverageCriticResult"),
    ("orchestrator.usage", "TaskUsage"),
    # One registered type for ALL injected steps, so the allowlist stays closed
    # however many steps users add.
    ("orchestrator.manifest", "StepResult"),
]

_CUSTOM_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES,
)


# Audit emission for spine @tasks. _audited_task is stacked UNDER @task so each
# task_start/complete/failed fires only on REAL execution — a task that replays
# from the checkpoint on resume short-circuits before the wrapper runs, so it is
# never re-logged. (The pre-67 body-level `audited()` wrapper re-fired on every
# resume.) The sink is rebuilt from config inside the task rather than passed in,
# which would change the task's checkpoint cache key.
def _build_task_audit_sink() -> AuditSink:
    """The audit sink as seen from inside a @task.

    Rebuilt from the current config (cheap — JsonlAuditSink just holds a path)
    rather than passed in as a @task input, which would change the task's
    checkpoint cache key. Mirrors the entrypoint body's own sink construction,
    and reads config via this module's `load_config` so tests that patch
    `orchestrator.workflow.load_config` reach it too.
    """
    cfg = load_config()
    if not cfg.audit.enabled:
        return NoopAuditSink()
    return build_sink(str(find_project_root() / cfg.audit.log_path))


def _audited_task(task_name: str):
    """Emit task_start / task_complete / task_failed from INSIDE a spine @task.

    Stacked UNDER @task (`@task` above, `@_audited_task(...)` below), so a @task
    that REPLAYS from the checkpoint on resume short-circuits before this wrapper
    ever runs — the events fire exactly once, for the attempt that actually
    executed. This replaces the old body-level `audited()` wrapper, which
    re-entered on every resume and re-logged completed tasks as if they had run
    again (a fidelity bug for a compliance log).

    The interrupt invariant still holds: interrupt() is only ever called from the
    entrypoint body, never inside a @task, so this `except Exception` can never
    catch a GraphInterrupt and mis-log it as task_failed.
    """
    def decorate(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            thread_id = get_config()["configurable"]["thread_id"]
            sink = _build_task_audit_sink()
            emit_event(sink, thread_id, "task_start", task_name=task_name)
            try:
                result = await fn(*args, **kwargs)
            except Exception as exc:
                emit_event(
                    sink, thread_id, "task_failed", task_name=task_name,
                    payload=failure_payload(exc),
                )
                raise
            emit_event(sink, thread_id, "task_complete", task_name=task_name)
            return result
        return wrapper
    return decorate


# @task wraps an async function so LangGraph can:
#   - record its inputs and outputs to the checkpointer
#   - skip re-running it on resume if its inputs haven't changed
#   - surface it as a span in the LangSmith trace tree
# @task knows nothing about which checkpointer is in use — that's
# configured on the @entrypoint below.
# The version gate's storage mechanism. This task records the WORKFLOW_VERSION
# current at the moment a run is first created. Because @task results are
# checkpointed and replayed (not recomputed) on resume, calling it at the top of
# the body returns:
#   - on the first run: the live WORKFLOW_VERSION (and persists it)
#   - on every resume:   the CACHED value — the version that created the run
# That cached value is exactly "what version of the workflow created this
# checkpoint", which LangGraph doesn't expose natively. The body compares it
# against the live constant and refuses the resume on mismatch. Adding a new
# @task name is safe for older checkpoints: cache keys are per-function-name, so
# a legacy checkpoint just runs this fresh on resume (no false mismatch).
@task
async def record_version_task() -> str:
    return WORKFLOW_VERSION


# Pipeline snapshot. Mirrors record_version_task EXACTLY — takes no input and
# recomputes the hash itself, so its checkpointed value is unambiguously "the
# pipeline hash at run-creation time" with nothing that could look input-dependent.
# Returns the live hash on the first run (and persists it), the cached
# creation-time hash on every resume; the body refuses the resume if
# orchestrator.toml's pipeline changed mid-run. Hashes via load_config() (not a
# captured config) so a test that patches orchestrator.workflow.load_config — or a
# default-config run with no file — computes the same hash on both sides.
@task
async def record_pipeline_hash_task() -> str:
    return load_config().pipeline.manifest_hash()


# Per-step task factories. Each injected step is wrapped in a @task
# NAMED for its step id (`step:<id>`) — so it appears under its own id in the
# LangSmith trace tree and gets its own checkpoint identity, instead of every
# script (or every ai_agent) step collapsing onto one shared task name.
#
# A fresh wrapper is built per call on purpose. LangGraph derives a task's
# identity from its NAME plus its call position in the entrypoint body — not
# from the function object or its inputs — so a freshly-built, deterministically
# named task replays correctly on resume. (It also sidesteps task()'s mutation
# of func.__name__: each wrapper closes over its own fresh function, so names
# never clobber each other.) Step inputs are primitives, not Pydantic Step
# models, so the serde allowlist needs only StepResult.
#
# `attempt` is carried purely for context (it tags the trace inputs and the
# approval_gate payload); per-attempt distinctness comes from call position, not
# from this value.
def _make_script_task(step_id: str, *, as_gate: bool = False):
    async def run_script_step(
        step_id: str, path: str, timeout: int, repo_root: str, attempt: int = 0
    ) -> StepResult:
        return await execute_script(
            ScriptStep(id=step_id, path=path, timeout=timeout),
            Path(repo_root),
            as_gate=as_gate,
        )

    return task(run_script_step, name=f"step:{step_id}")


def _make_ai_agent_task(step_id: str, *, as_gate: bool = False):
    async def run_ai_agent_step(
        step_id: str,
        agent: str,
        model: str,
        repo_root: str,
        plan_text: str,
        attempt: int = 0,
        feedback: str | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        timeout: int | None = None,
    ) -> StepResult:
        # tools/timeout are threaded through (v2): a [defs.*] / [stage.user.*]
        # ai_agent's own allowed_tools/disallowed_tools/timeout reach the runner.
        return await execute_ai_agent(
            AiAgentStep(
                id=step_id,
                agent=agent,
                model=model,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools or [],
                timeout=timeout,
            ),
            Path(repo_root),
            plan_text,
            feedback=feedback,
            as_gate=as_gate,
        )

    return task(run_ai_agent_step, name=f"step:{step_id}")


class StepGateAborted(RuntimeError):
    """Raised when a human pause is resumed with an abort decision
    ('abort'/'no'/'stop') — a build's gate-fail pause or an ai_agent stage's
    review pause. Propagates out of _dispatch_stage to the entrypoint body, which
    converts it into a clean status="aborted" return. All gates run before the
    commit line, so an abort never leaves a half-shipped state.
    """

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        super().__init__(f"workflow aborted at step {step_id!r}")


class BuildFailed(RuntimeError):
    """A `build` step ran its full retry budget without a passing gate under
    on_exhausted="abort" (or a human declined to keep retrying). Carries the
    failing gate's last feedback so the entrypoint body can return the clean
    status="failed" dict (a QA-exhausted run ends `failed` with `qa_failures`,
    never a raw exception). Build steps are pre-commit, so nothing is half-shipped.
    """

    def __init__(self, step_id: str, attempts: int, last_feedback: str | None) -> None:
        self.step_id = step_id
        self.attempts = attempts
        self.last_feedback = last_feedback
        super().__init__(
            f"build step {step_id!r} did not pass its gate(s) after "
            f"{attempts} attempt(s)"
        )


# Resume values (case-insensitive) that mean "stop the run" at an approval_gate.
# Anything else proceeds — replying to a gate is how you resume past it.
_GATE_ABORT_WORDS = frozenset({"abort", "no", "stop"})


def _is_abort(decision) -> bool:
    """True if a human's resume value at a gate/pause is an abort word.

    The single home for the abort decision check that every interrupt site
    (approval_gate, ai_agent review, build producer/gate pauses, retry review)
    shares. A non-string decision (or any non-abort word) is not an abort —
    the run proceeds."""
    return isinstance(decision, str) and decision.strip().lower() in _GATE_ABORT_WORDS


def _record_usage(usage_by_task: dict, key: str, result) -> None:
    """Append `result`'s token usage under `key`, if it has any.

    The single home for the `if result.usage: usage_by_task[...].append(...)` pair
    that every agent step shares. setdefault covers both the pre-seeded spine keys
    (planning/qa/...) and dynamic step ids; a result with no usage is a no-op."""
    if result.usage:
        usage_by_task.setdefault(key, []).append(result.usage)


class AutonomousCeilingExceeded(WorkflowCancelled):
    """A fully-autonomous run hit its time or cost safety ceiling.

    Subclasses WorkflowCancelled so it stops the run through the SAME between-task
    path as a user cancel — but carries a `reason` the entrypoint surfaces so a
    caller can tell a budget trip from a human `cancel_run`. No cancel marker is
    written (unlike cancel_run), so the thread can still be resumed with a larger
    budget."""

    def __init__(self, thread_id: str, reason: str):
        super().__init__(thread_id)
        self.reason = reason


def _part_to_step(part, config):
    """Bridge a v2 PartSpec ([builtin.*] / [defs.*]) to the manifest step model
    the runners (steps.execute_script / execute_ai_agent) consume.

    Built-in implementation/qa never reach here — the build loop resolves them via
    injected callables (they are the spine's own agents). This handles [defs.*]
    scripts/agents used as a build's producer or gate. A [defs.*] script/agent is
    always typed and pathed (validated at load), so .type / .path are present.
    `tools` is the PartSpec alias for `allowed_tools`.
    """
    if part.type == "script":
        return ScriptStep(
            id=part.id,
            path=part.path,
            timeout=part.timeout if part.timeout is not None else 60,
        )
    allowed = part.allowed_tools if part.allowed_tools is not None else part.tools
    return AiAgentStep(
        id=part.id,
        agent=part.path,
        model=config.resolved_model(part.model),
        allowed_tools=allowed,
        disallowed_tools=part.disallowed_tools,
        timeout=part.timeout,
    )


async def _run_build_step(
    block_step: BuildStep,
    config,
    plan_text: str,
    check_cancel,
    usage_by_task: dict,
    *,
    builtin_producers: dict | None = None,
    builtin_gates: dict | None = None,
    thread_id: str | None = None,
    audit=None,
    autonomous: bool = False,
    on_attempt=None,
) -> None:
    """Run one produce⇄gate build via the generic engine (run_retry_block).

    `block_step` is a synthetic manifest BuildStep carrying PREFIXED produce/gate
    refs ("builtin:<id>" / "defs:<id>"). Each ref resolves in order: an injected
    built-in callable (`builtin_producers`/`builtin_gates`: the spine's own
    implementation producer / QA gate, made task-aware by the caller) → else a
    [builtin.*] / [defs.*] part via config.part, bridged to a manifest step
    (_part_to_step) and run through the SAME @task factories so it inherits
    checkpoint/replay. A gate's verdict is its StepResult.passed (script: exit
    code; ai_agent: the emitted `passed`); on a retry, the failing gate's feedback
    is injected into producer ai_agents.

    Two human pauses come from THIS build's HumanInLoopConfig: `after_producer`
    pauses after producers, before gates, every attempt (kind
    `build_producer_pause`); `on_gate_fail` pauses on a failing gate (kind
    `build_gate_failed`) where an abort word stops the run, anything else retries.
    Under on_exhausted="approval_gate" the exhaustion prompt accepts a count to
    grant more attempts (bounded by retry.max_total_attempts). A non-proceed
    outcome raises BuildFailed → the clean status="failed" return (builds are
    pre-commit, so nothing is half-shipped). interrupt() is reachable because this
    helper runs in the entrypoint frame.

    `on_attempt(attempt, passed, gate_results)` (Phase 77a hook) is an optional
    read-only per-attempt observer threaded straight to run_retry_block: it fires
    once after the gates resolve EVERY attempt, including the GREEN one. The TDD
    station passes it to persist per-attempt evidence (Phase 77c); a None default
    keeps every other build byte-for-byte unchanged.
    """
    repo_root = str(find_project_root())
    builtin_producers = builtin_producers or {}
    builtin_gates = builtin_gates or {}
    hil = block_step.human_in_loop

    async def on_producers_done(attempt: int) -> None:
        # Optional pause after the producer(s), before the gate(s), every attempt
        # — driven by this build's human_in_loop.after_producer. Suppressed in
        # autonomous mode.
        if hil.after_producer and not autonomous:
            if audit is not None and thread_id is not None:
                emit_event(audit, thread_id, "interrupt",
                           payload={"kind": "build_producer_pause", "step_id": block_step.id})
            interrupt({
                "kind": "build_producer_pause",
                "step_id": block_step.id,
                "ask": "Producer complete. Proceed to the gate?",
                "attempt": attempt,
            })

    async def on_gate_failed(attempt: int, feedback: str) -> bool:
        # Always log a gate failure (scripted gates and LLM gates alike) so it's
        # visible in the log; the retry budget lives on the build's retry.max, so
        # the message reports the attempt number without a total.
        logger.error(
            "gate FAIL (attempt %d):\n%s",
            attempt,
            feedback or "(no failure details)",
        )
        # Optional pause on a gate failure — driven by this build's
        # human_in_loop.on_gate_fail. An abort word stops the run; anything else
        # retries. Suppressed in autonomous mode (the loop just retries).
        if hil.on_gate_fail and not autonomous:
            if audit is not None and thread_id is not None:
                emit_event(audit, thread_id, "interrupt",
                           payload={"kind": "build_gate_failed", "step_id": block_step.id})
            decision = interrupt({
                "kind": "build_gate_failed",
                "step_id": block_step.id,
                "failures": feedback,
                "ask": (
                    f"Gate FAIL (attempt {attempt}). "
                    "Retry? Reply 'yes' or 'abort'."
                ),
            })
            if _is_abort(decision):
                return False  # stop now; don't spend another attempt
        return True

    async def run_producer(pid: str, feedback: str | None) -> StepResult:
        if pid in builtin_producers:
            return await builtin_producers[pid](pid, feedback)
        part = config.part(pid)
        if part is None:
            raise StepError(
                f"build step {block_step.id!r}: producer {pid!r} is not a built-in "
                f"and has no [builtin.*] / [defs.*] definition."
            )
        step = _part_to_step(part, config)
        if isinstance(step, ScriptStep):
            result = await _make_script_task(step.id)(
                step.id, step.path, step.timeout, repo_root
            )
        else:  # AiAgentStep — feedback is injected into its user message
            result = await _make_ai_agent_task(step.id)(
                step.id, step.agent, step.model, repo_root, plan_text, 0, feedback,
                step.allowed_tools, step.disallowed_tools, step.timeout,
            )
        _record_usage(usage_by_task, step.id, result)
        return result

    async def run_gate(gid: str) -> StepResult:
        if gid in builtin_gates:
            return await builtin_gates[gid](gid)
        part = config.part(gid)
        if part is None:
            raise StepError(
                f"build step {block_step.id!r}: gate {gid!r} is not a built-in "
                f"and has no [builtin.*] / [defs.*] definition."
            )
        step = _part_to_step(part, config)
        if isinstance(step, ScriptStep):
            result = await _make_script_task(step.id, as_gate=True)(
                step.id, step.path, step.timeout, repo_root
            )
        else:  # AiAgentStep gate — emits a `passed` verdict
            result = await _make_ai_agent_task(step.id, as_gate=True)(
                step.id, step.agent, step.model, repo_root, plan_text, 0, None,
                step.allowed_tools, step.disallowed_tools, step.timeout,
            )
        _record_usage(usage_by_task, step.id, result)
        return result

    block = RetryBlock(
        producers=block_step.produce,
        gates=block_step.gate,
        max_retries=block_step.retry.max,
        on_exhausted=block_step.retry.on_exhausted,
        max_total_attempts=block_step.retry.max_total_attempts,
    )
    result = await run_retry_block(
        block=block,
        run_producer=run_producer,
        run_gate=run_gate,
        check_cancel=check_cancel,
        on_producers_done=on_producers_done,
        on_attempt=on_attempt,   # Phase 77c per-attempt evidence (None elsewhere)
        on_gate_failed=on_gate_failed,
        interrupt_fn=interrupt,  # used only when on_exhausted="approval_gate"
        autonomous=autonomous,   # unbounded budget; loop until a gate passes
    )
    if not result.proceed:
        raise BuildFailed(block_step.id, result.attempts, result.last_feedback)


# ---------------------------------------------------------------------------
# The per-task execution station.
# Loops the FROZEN task list from the decomposer and runs each task as a
# produce⇄gate build via the same engine the rest of the spine uses —
# _run_build_step / run_retry_block — with the [workflow.task_build] recipe.
#
# Two nested loops: the OUTER per-task loop here, wrapping the INNER per-attempt
# retry inside each task's build. The task list is a checkpointed decompose_task
# result, so it replays deterministically on resume; each task's build @tasks
# replay positionally — no separate task-list hash gate is needed (the existing
# checkpoint guards cover it).
# ---------------------------------------------------------------------------


def _compose_task_plan(plan_text: str, task) -> str:
    """The producer's plan text for one task: the overall plan + THIS task's slice.
    The agent reads the working tree itself for cumulative state, so the diff is
    implicit; only the task focus is injected here."""
    parts = [plan_text, "", f"## Current task: {task.title}", "", task.description]
    if task.acceptance_criteria:
        parts += ["", f"Acceptance criteria: {task.acceptance_criteria}"]
    return "\n".join(parts)


def _compose_task_qa(plan_text: str, task) -> str:
    """The QA gate's plan text for one task: judge ONLY this task. The diff may
    include earlier completed tasks, so the note tells QA not to fail the review
    for unrelated prior changes (the whole-diff acceptance is the optional
    final_qa pass)."""
    parts = [plan_text, "", f"## Evaluate ONLY this task: {task.title}", "", task.description]
    if task.acceptance_criteria:
        parts += ["", f"Acceptance criteria: {task.acceptance_criteria}"]
    parts += [
        "",
        "Note: the diff may also include earlier, already-completed tasks. Judge "
        "ONLY whether the task above is correctly implemented; do not fail the "
        "review for unrelated changes from earlier tasks.",
    ]
    return "\n".join(parts)


def _compose_red_green(plan_text: str, red_output: str) -> str:
    """Append the failing-test (RED) output to the implementer's plan (Phase 72b).

    On a TDD task the test-author has already written failing tests; injecting
    their output gives the implementer's FIRST attempt the exact spec it must turn
    green (before any gate has run to feed it back). The tests are frozen by the
    diff-gate, so the note tells the implementer to change implementation only.
    Empty red_output → the plan is returned unchanged."""
    if not red_output:
        return plan_text
    return "\n".join([
        plan_text,
        "",
        "## Failing tests to make pass (RED)",
        "",
        "A separate test-author wrote tests for this task; they currently FAIL and "
        "are FROZEN — you must not edit them. Make them pass by changing the "
        "implementation only. The failing output:",
        "",
        "```",
        red_output.strip(),
        "```",
    ])


def _impl_model(config) -> str:
    """Resolved model for the built-in implementation producer ([builtin.implementation])."""
    part = config.part("builtin:implementation")
    return config.resolved_model(part.model if part else None)


def _qa_model(config) -> str:
    """Resolved model for the built-in QA gate. The per-task gate's config lives
    on [builtin.qa]; fall back to the whole-diff [stage.builtin.qa] stage, then to
    default_model (qa-agent TOOLS resolve the same precedence inside agents.qa)."""
    spec = config.part("builtin:qa") or config.stage("qa")
    return config.resolved_model(spec.model if spec else None)


def _builtin_build_callables(
    config, plan_result, usage_by_task, qa_holder, thread_id, *,
    impl_plan: str | None = None, qa_plan=None,
):
    """The spine's own implementation producer / QA gate as a build's built-in
    callables, keyed by their prefixed refs. The task loop passes per-task
    impl_plan / qa_plan to make them task-aware; a whole-tree user build passes
    neither (the overall plan is used)."""
    impl_text = impl_plan if impl_plan is not None else plan_result.plan_text
    qa_arg = qa_plan if qa_plan is not None else plan_result

    async def _impl(pid: str, feedback: str | None) -> StepResult:
        # Audit task_start/complete is emitted inside implementation_task
        # (via @_audited_task), so it fires only on real execution, not replay.
        result = await implementation_task(impl_text, feedback, _impl_model(config))
        _record_usage(usage_by_task, "implementation", result)
        return result

    async def _qa(gid: str) -> StepResult:
        qa_result = await qa_task(qa_arg, _qa_model(config))
        _record_usage(usage_by_task, "qa", qa_result)
        write_qa(thread_id, qa_result)
        qa_holder["qa"] = qa_result
        return StepResult(
            step_id="qa",
            kind="ai_agent",
            ok=True,
            passed=(qa_result.result == "PASS"),
            detail=qa_result.failures or "",
        )

    return {"builtin:implementation": _impl}, {"builtin:qa": _qa}


def _build_hil(spec) -> HumanInLoopConfig:
    """A build stage's per-attempt human pauses. StageSpec.human_in_loop is
    bool | HumanInLoopConfig; only a HumanInLoopConfig drives a build's pauses."""
    return spec.human_in_loop if isinstance(spec.human_in_loop, HumanInLoopConfig) else HumanInLoopConfig()


# ---------------------------------------------------------------------------
# Phase 72 — TDD / red-green test-author station.
#
# Three layers of separation, all required (see CHANGELOG / phase docs):
#   ROLE  — a distinct test-author (test_author_task) writes the tests; the
#           implementation producer never authors them.
#   WRITE — the implementer mechanically cannot change the tests, enforced by the
#           diff-gate (perms are coarse, not path-scoped), ordered FIRST.
#   ONCE  — tests are authored once per task, BEFORE the retry loop (run_retry_block
#           re-runs all producers each attempt), and replayed (not regenerated) on
#           resume because test_author_task is a checkpointed @task.
#
# All of this is gated on config.tdd; with it off the per-task station is byte-for-
# byte the classic implement→qa loop. Graceful-degrade: a task the author judges
# untestable (or a born-green / no-script-gate situation) returns testable=False
# and runs the classic path — TDD never wedges a run.
# ---------------------------------------------------------------------------


def _test_author_frontmatter(config) -> AgentFrontmatter:
    """The test-author prompt file's frontmatter (model/tools/…).

    Resolved from the same file _test_author_prompt loads: config.test_author_path
    when set, else the convention/bundled default. Same mechanism the other
    built-ins use (load_prompt_frontmatter / _merge_builtin_frontmatter) — the
    prompt file fully defines the agent, persona + model + tools. A missing
    override file yields an empty frontmatter; _test_author_prompt raises the
    clear FileNotFoundError, so model/tools resolution never crashes first."""
    if config.test_author_path:
        path = find_project_root() / config.test_author_path
        if path.exists():
            return parse_agent_frontmatter(path.read_text(encoding="utf-8"))[0]
        return AgentFrontmatter()
    return load_prompt_frontmatter("test-author")


def _test_author_model(config) -> str:
    """Resolved model for the test-author: the prompt frontmatter's `model` if it
    sets one, else default_model. (Frontmatter is the only override surface — the
    test-author is internal, not a [builtin.*] part.)"""
    return config.resolved_model(_test_author_frontmatter(config).model)


def _test_author_tools(config) -> tuple[list[str] | None, list[str] | None]:
    """(allowed, disallowed) tools for the test-author from prompt frontmatter, or
    (None, None) when it says nothing — author_tests then uses its role default
    (Read/Edit/Write/Bash)."""
    fm = _test_author_frontmatter(config)
    return fm.allowed_tools, fm.disallowed_tools


def _test_author_prompt(config) -> str:
    """The test-author's system prompt.

    config.test_author_path (project-root-relative) points the author at an
    arbitrary prompt file; the emit-tool footer is appended by load_prompt either
    way. Unset → the convention/bundled default (.orchestrator/prompts/
    test-author.md → bundled), today's behaviour."""
    if config.test_author_path:
        path = find_project_root() / config.test_author_path
        return load_prompt("test-author", path_override=path)
    return load_prompt("test-author")


# ── Coverage critic (Phase 74) — same frontmatter-driven model/tools mechanism as
# the other built-ins; overridable via .orchestrator/prompts/coverage-critic.md. No
# path config knob (the critic is fully internal). ──────────────────────────────


def _coverage_critic_model(config) -> str:
    """Resolved model for the coverage critic: the prompt frontmatter's model, else
    default_model."""
    return config.resolved_model(load_prompt_frontmatter("coverage-critic").model)


def _coverage_critic_tools(config) -> tuple[list[str] | None, list[str] | None]:
    """(allowed, disallowed) tools from the critic prompt frontmatter, or (None,
    None) → critique_tests uses its read-only role default (Read/Bash/Grep)."""
    fm = load_prompt_frontmatter("coverage-critic")
    return fm.allowed_tools, fm.disallowed_tools


async def _run_coverage_critic(plan_text: str, model: str) -> CoverageCriticResult:
    """Run the coverage critic on a task's authored tests. Factored out of
    critic_task so the @task stays a pure checkpoint boundary (tests patch this)."""
    config = load_config()
    allowed, disallowed = _coverage_critic_tools(config)
    return await critique_tests(
        plan_text, model, load_prompt("coverage-critic"), allowed, disallowed
    )


def _script_gate_steps(config, gate_refs):
    """The ScriptStep for each SCRIPT-type part referenced in a build's gate.

    The deterministic suite the red-confirm runs: LLM gates (e.g. builtin:qa) are
    skipped — only scripts give a cheap, exit-code green/red signal. An empty list
    means there is no deterministic test gate, so the transition can't be proven."""
    steps = []
    for ref in gate_refs:
        part = config.part(ref)
        if part is not None and part.type == "script":
            steps.append(_part_to_step(part, config))
    return steps


async def _run_script_gates(steps, repo_root: str):
    """Run the resolved script gates; return (all_green, failing_output, full_output).

    Each runs as a gate (a non-zero exit is a FAIL verdict, not an abort), so a
    red suite yields its output as feedback rather than raising.

    `failing_output` is the failures-only summary (each failing gate's `detail`) —
    the feedback the implementer's first attempt sees. `full_output` is the
    COMPLETE log of EVERY gate, pass and fail (Phase 77a `StepResult.full_output`):
    the red-green evidence layer (Phase 77b) persists it as proof of what actually
    ran, not just what failed."""
    green = True
    failures = []
    full = []
    for step in steps:
        result = await execute_script(step, Path(repo_root), as_gate=True)
        full.append(f"### {step.id}\n{result.full_output}")
        if not result.passed:
            green = False
            failures.append(f"### {step.id}\n{result.detail}")
    return green, "\n\n".join(failures), "\n\n".join(full)


def _hash_test_paths(test_paths, repo_root: str) -> str:
    """A content+membership hash of every file matching the test_paths globset.

    The diff-gate's frozen baseline: any edit, deletion, or sneaked-in test file
    changes the hash. Patterns are project-root-relative; `**` recurses. The
    orchestrator's own `.orchestrator/` workspace is excluded (via iter_test_files)
    so the Phase 77b evidence copies written there can't perturb the freeze."""
    root = Path(repo_root)
    h = hashlib.sha256()
    for p in iter_test_files(test_paths, root):
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _make_diff_gate(test_paths, snapshot: str, repo_root: str):
    """A gate callable that FAILS if the frozen test files changed since authoring.

    Closes over the per-task snapshot; re-hashes test_paths each attempt and
    asserts byte-identity (content + set membership). Ordered FIRST in the gate
    list, so a 'green' suite is only ever trusted against pristine tests."""
    async def _diff_gate(gid: str) -> StepResult:
        current = _hash_test_paths(test_paths, repo_root)
        ok = current == snapshot
        return StepResult(
            step_id="diff-gate",
            kind="script",
            ok=True,
            passed=ok,
            detail="" if ok else (
                "The test files are frozen after authoring and must not change, "
                "but they were modified during implementation. Revert every edit, "
                "addition, and deletion to the test files; make the failing tests "
                "pass by changing the implementation only."
            ),
        )

    return _diff_gate


def _impl_attempt_recorder(thread_id, task_index, task, test_paths, snapshot, repo_root):
    """Build the Phase 77c per-attempt evidence callback for a TESTABLE TDD task.

    Returns an `on_attempt(attempt, passed, gate_results)` coroutine for
    run_retry_block's 77a hook. It fires once per implement attempt — every
    attempt, including the GREEN one — re-hashes the test_paths globset (the same
    pure `_hash_test_paths` the diff-gate uses, so the freeze verdict is
    deterministic and reads the same tree the gates just judged) and writes
    task-NN-<id>/impl/attempt-N/ (the COMPLETE run + MATCH/MISMATCH vs `snapshot`,
    the 77b baseline). Only the testable path builds this; the classic / untestable
    path passes on_attempt=None and writes no impl/ evidence. A read-only observer
    of best-effort side-effect files, so it replays cleanly on resume."""
    async def on_attempt(attempt: int, passed: bool, gate_results: list) -> None:
        current_hash = _hash_test_paths(test_paths, repo_root)
        write_impl_attempt(
            thread_id, task_index, task, attempt, passed, gate_results,
            baseline=snapshot, current_hash=current_hash,
        )
    return on_attempt


# TestAuthorResult.degrade_kind values (Phase 76). A testable=False verdict is
# either a legitimate "not unit-testable" judgement (graceful classic fallback,
# even when autonomous) or a red-confirm FAILURE — the green→red guarantee could
# not be established. Autonomous TDD hard-aborts the latter (no human to eyeball
# it); supervised TDD degrades both (the human is the guard).
_DEGRADE_UNTESTABLE = "untestable"          # the author judged the behaviour not unit-testable
_DEGRADE_NO_GATE = "no_gate"                # no deterministic script gate to confirm against
_DEGRADE_NOT_GREEN = "not_green_baseline"   # the suite was not green before authoring
_DEGRADE_BORN_GREEN = "born_green"          # the authored tests did not fail → proved nothing

# The red-confirm degrade kinds: the cases autonomous TDD treats as a hard abort
# (everything except the author's own UNTESTABLE judgement).
_RED_CONFIRM_FAILURES = frozenset({_DEGRADE_NO_GATE, _DEGRADE_NOT_GREEN, _DEGRADE_BORN_GREEN})


async def _run_test_author(
    plan_text: str, model: str, test_paths: list[str], gate_refs: list[str],
    feedback: str | None = None,
) -> TestAuthorResult:
    """Author tests for one task and confirm the green→red transition.

    The supervised red-green core:
      1. the deterministic suite must be GREEN before authoring — else we can't
         prove the new tests are what turn it red (classic fallback);
      2. the test-author writes failing tests, or emits UNTESTABLE (classic
         fallback);
      3. the suite must now be RED — a still-green suite is a born-green no-op that
         proves nothing (classic fallback).
    On success it snapshots the test_paths globset (the diff-gate's frozen
    baseline). A testable=False return routes the task to the classic implement→qa
    path; this function never wedges the run. Factored out of test_author_task so
    the @task stays a pure checkpoint boundary (tests patch this inner fn).

    `feedback` set → a RE-AUTHOR (Phase 72b red-review escape): the prior author's
    failing tests are already in the tree, so the green-before precondition cannot
    hold and is SKIPPED — the author revises its tests and red-after is re-confirmed
    against the revised suite."""
    config = load_config()
    repo_root = str(find_project_root())
    reauthor = feedback is not None

    scripts = _script_gate_steps(config, gate_refs)
    if not scripts:
        logger.warning(
            "TDD: no script gate in the task-build gate list, so a green→red "
            "transition can't be confirmed; classic fallback for this task."
        )
        return TestAuthorResult(
            testable=False, degrade_kind=_DEGRADE_NO_GATE,
            summary="no deterministic test gate",
        )

    if not reauthor:
        green_before, _, _ = await _run_script_gates(scripts, repo_root)
        if not green_before:
            logger.warning(
                "TDD: the suite is not green before authoring, so a green→red "
                "transition can't be proven; classic fallback for this task."
            )
            return TestAuthorResult(
                testable=False, degrade_kind=_DEGRADE_NOT_GREEN,
                summary="suite not green before authoring",
            )

    allowed, disallowed = _test_author_tools(config)
    verdict = await author_tests(
        plan_text, model, _test_author_prompt(config), allowed, disallowed,
        feedback=feedback,
    )
    if not verdict.testable:
        logger.info("TDD: task judged untestable (%s); classic fallback.", verdict.summary)
        # The author's own UNTESTABLE judgement — a legitimate degrade, not a
        # red-confirm failure. Tag it so autonomous TDD degrades (not aborts).
        return verdict.model_copy(update={"degrade_kind": _DEGRADE_UNTESTABLE})

    green_after, red_output, full_run = await _run_script_gates(scripts, repo_root)
    if green_after:
        logger.warning(
            "TDD: the authored tests did not turn the suite red (born-green); the "
            "test proves nothing, so classic fallback for this task."
        )
        return TestAuthorResult(
            testable=False,
            degrade_kind=_DEGRADE_BORN_GREEN,
            summary="authored tests did not fail (born-green)",
            usage=verdict.usage,
        )

    snapshot = _hash_test_paths(test_paths, repo_root)
    return TestAuthorResult(
        testable=True,
        summary=verdict.summary,
        snapshot=snapshot,
        red_output=red_output,
        # The COMPLETE final RED run for the Phase 77b evidence layer; falls back to
        # the failures-only summary if the runner produced no captured full output.
        full_run=full_run or red_output,
        usage=verdict.usage,
    )


@task
@_audited_task("test_author")
async def test_author_task(
    plan_text: str, model: str, test_paths: list[str], gate_refs: list[str],
    feedback: str | None = None, attempt: int = 0,
) -> TestAuthorResult:
    # Runs ONCE per task (or once per re-author iteration), BEFORE the implement
    # loop (kept out of run_retry_block, which re-runs all producers each attempt).
    # The @task wrapper checkpoints the result so a mid-run resume REPLAYS the
    # authored verdict — the tests are not regenerated. The authored test files
    # persist in the working tree across resume, the same side-effect model as
    # implementation_task. config / repo_root are read inside _run_test_author (not
    # @task inputs), so the checkpoint cache key stays stable. `attempt` is a
    # cache-key discriminator only: each re-author (Phase 72b) is a distinct @task
    # invocation, so a resume replays the right iteration rather than collapsing them.
    return await _run_test_author(plan_text, model, test_paths, gate_refs, feedback)


@task
@_audited_task("coverage_critic")
async def critic_task(plan_text: str, model: str, attempt: int = 0) -> CoverageCriticResult:
    # The Phase 74 coverage critic, checkpointed so a resume replays the verdict.
    # `attempt` is a cache-key discriminator only: each critic round (one per
    # authored test version) is a distinct @task invocation, mirroring
    # test_author_task, so a resume replays the right round.
    return await _run_coverage_critic(plan_text, model)


# Default re-author guidance when the critic rejects but gives no specific note.
_CRITIC_DEFAULT_FEEDBACK = (
    "The tests do not meaningfully pin down this task's behaviour. Rewrite them to "
    "assert the observable behaviour through the public interface — they must fail "
    "if the behaviour is implemented wrongly; no vacuous, tautological, or "
    "shape-only checks."
)


# Resume words at the red-review pause that mean "implement against these tests".
# Anything that is neither an approve word nor an abort word is treated as
# re-author feedback.
_RED_REVIEW_APPROVE_WORDS = frozenset({"yes", "y", "approve", "ok", "lgtm", "go", "proceed"})


def _is_red_review_approve(decision) -> bool:
    return isinstance(decision, str) and decision.strip().lower() in _RED_REVIEW_APPROVE_WORDS


async def _author_with_review(
    task, task_plan: str, config, usage_by_task: dict, gate_refs: list[str], *,
    thread_id: str, audit, autonomous: bool, manual_checks: list | None = None,
    feedback: str | None = None, attempt: int = 0, rounds_info: dict | None = None,
) -> tuple[TestAuthorResult, int]:
    """Author tests for one task, with the Phase 74 coverage critic and the Phase
    72b supervised red-review pause + re-author escape.

    Per task: author → (testable?) → COVERAGE CRITIC judges the tests; a negative
    verdict re-authors with the critic's feedback, bounded by tdd_critic_max_attempts
    (still weak after the budget → proceed and record a manual check; never wedges).
    Then, if tdd_red_review is on and the run isn't autonomous, PAUSE for a human:
      - an approve word ('yes'/…)  → return the verdict; the implement loop runs;
      - 'abort'/'no'/'stop'        → raise BuildFailed → clean status="failed";
      - anything else              → treat as feedback and RE-AUTHOR (then re-critique).
    An untestable verdict returns immediately (classic fallback). Runs in the
    entrypoint frame so interrupt() is reachable; each re-author/critic round is a
    distinct test_author_task / critic_task @task (via `attempt`) so resume replays
    cleanly.

    Returns (verdict, attempt). `feedback`/`attempt` seed the loop so a caller can
    drive REPEATED rounds with a monotonic attempt counter (Phase 76's autonomous
    re-author cycle calls this once per round); the returned attempt is the value
    used for the LAST author call, so the next round passes attempt + 1 to keep
    every test_author_task / critic_task cache key unique even when the feedback
    string repeats. Supervised callers pass the defaults and ignore the count.

    `rounds_info` (Phase 77b): when a dict is passed, the converging process is
    recorded into it for the `test-author/summary.md` evidence — `critic_verdicts`
    (one {meaningful, feedback} per critic round), `critic_rounds`, and `red_review`
    (the supervised reviewer's outcome). Populated deterministically each call, so a
    resume that re-executes this frame fills it identically. None → no recording."""
    if rounds_info is not None:
        rounds_info.setdefault("critic_verdicts", [])
    while True:  # human-review loop (outer); each iteration re-resolves the tests
        # Coverage-critic loop (inner): author → critique → re-author until the
        # critic is satisfied, the critic is off, or its budget is exhausted.
        critic_rounds = 0
        while True:
            ta = await test_author_task(
                task_plan, _test_author_model(config),
                list(config.test_paths), gate_refs, feedback, attempt,
            )
            _record_usage(usage_by_task, "test_author", ta)
            if not ta.testable:
                return ta, attempt
            if not config.tdd_coverage_critic:
                break
            cr = await critic_task(task_plan, _coverage_critic_model(config), attempt)
            _record_usage(usage_by_task, "coverage_critic", cr)
            if rounds_info is not None:
                rounds_info["critic_verdicts"].append(
                    {"meaningful": cr.meaningful, "feedback": cr.feedback}
                )
            if cr.meaningful:
                break
            if critic_rounds >= config.tdd_critic_max_attempts:
                # Budget exhausted, tests still weak: proceed (never wedge) but flag
                # the criterion for manual verification (Phase 73 surfacing).
                logger.warning(
                    "TDD: coverage critic still flags weak tests for task %s after "
                    "%d re-author(s); proceeding, flagged as a manual check.",
                    task.id, critic_rounds,
                )
                if manual_checks is not None:
                    manual_checks.append({
                        "task_id": task.id,
                        "title": task.title,
                        "acceptance_criteria": task.acceptance_criteria,
                        "reason": f"coverage critic unresolved after "
                                  f"{critic_rounds} re-author(s): {cr.feedback}",
                    })
                break
            feedback = cr.feedback or _CRITIC_DEFAULT_FEEDBACK
            attempt += 1
            critic_rounds += 1

        if rounds_info is not None:
            rounds_info["critic_rounds"] = critic_rounds

        # Supervised red-review (Phase 72b). Suppressed when autonomous (Phase 76
        # replaces the human guard with the red-confirm hard gate + bounded
        # re-author in _run_autonomous_tdd_task).
        if autonomous or not config.tdd_red_review:
            return ta, attempt
        if audit is not None:
            emit_event(audit, thread_id, "interrupt",
                       payload={"kind": "red_review", "step_id": f"task:{task.id}"})
        decision = interrupt({
            "kind": "red_review",
            "task_id": task.id,
            "summary": ta.summary,
            "red_output": ta.red_output,
            "ask": (
                "Review the failing tests for this task. Reply 'yes' to implement "
                "against them, 'abort' to stop the run, or describe what to change "
                "to re-author the tests."
            ),
        })
        if _is_abort(decision):
            raise BuildFailed(
                f"task:{task.id}", attempt,
                "Aborted at red-review: the reviewer rejected the failing tests.",
            )
        if _is_red_review_approve(decision):
            if rounds_info is not None:
                rounds_info["red_review"] = "approved"
            return ta, attempt
        feedback = decision if isinstance(decision, str) and decision.strip() else None
        if rounds_info is not None:
            rounds_info["red_review"] = f"re-authored: {feedback or '(no note)'}"
        attempt += 1


async def _run_task_build_step(
    task, plan_result, impl_plan: str, qa_plan, gate_refs: list[str],
    diff_gate: dict, stage, hil, config, check_cancel, usage_by_task: dict,
    qa_holder: dict, *, thread_id: str, audit, autonomous: bool, retry=None,
    on_attempt=None,
) -> None:
    """Run ONE task's produce⇄gate build — the per-task tail shared by the
    supervised/non-TDD path and the Phase 76 autonomous-TDD path.

    `diff_gate` is the optional {"builtin:diff-gate": callable} freeze (empty for a
    classic / untestable task); when set, `gate_refs` already leads with
    "builtin:diff-gate". `retry` overrides stage.retry — autonomous TDD passes a
    copy with on_exhausted="abort" so the BOUNDED build raises BuildFailed on
    exhaustion (to be caught by the re-author cycle) instead of pausing for a
    human. A non-proceed build raises BuildFailed(step_id="task:<id>").

    `on_attempt` (Phase 77c) is the per-attempt evidence recorder for a TESTABLE
    TDD task, threaded to run_retry_block; a classic / untestable task passes None
    and writes no impl/ evidence."""
    producers, gates = _builtin_build_callables(
        config, plan_result, usage_by_task, qa_holder, thread_id,
        impl_plan=impl_plan, qa_plan=qa_plan,
    )
    gates = {**gates, **diff_gate}
    synthetic = BuildStep(
        id=f"task:{task.id}",
        produce=list(stage.produce),
        gate=gate_refs,
        ungated=stage.ungated or (not gate_refs),  # gate=[] → producer runs once
        retry=retry if retry is not None else stage.retry,
        human_in_loop=hil,
    )
    await _run_build_step(
        synthetic, config, impl_plan, check_cancel, usage_by_task,
        builtin_producers=producers,
        builtin_gates=gates,
        thread_id=thread_id,
        audit=audit,
        autonomous=autonomous,
        on_attempt=on_attempt,
    )


# Feedback the autonomous re-author cycle hands the test-author when the
# implementation cannot pass the frozen tests within its bounded budget — the tests
# themselves are the suspect (a human would re-author here; Phase 76 automates it).
_AUTONOMOUS_REAUTHOR_FEEDBACK = (
    "The implementation could not make these tests pass within its attempt budget. "
    "The tests may be wrong, over-constrained, or assert behaviour the task does "
    "not require. Revise the test file(s) to correctly and minimally pin the "
    "task's required behaviour through its public interface.\n\n"
    "Last failing gate output:\n{last}"
)


async def _run_autonomous_tdd_task(
    task, task_index: int, plan_result, task_plan: str, qa_plan,
    base_gate_refs: list[str], stage, hil, config, check_cancel,
    usage_by_task: dict, qa_holder: dict, *,
    thread_id: str, audit, manual_checks: list | None,
) -> None:
    """One task under autonomous TDD (Phase 76). No human is present, so the
    supervised guards (red-review / re-author) are replaced by machinery:

      - RED-CONFIRM IS A HARD GATE — a red-confirm failure (no script gate /
        non-green baseline / born-green; degrade_kind in _RED_CONFIRM_FAILURES)
        aborts the task (BuildFailed → status="failed") instead of silently
        degrading. A genuinely-untestable verdict (the author's own judgement)
        still degrades to the classic implement→qa path + a manual check
        (legitimate, and surfaced in the run result).
      - BOUNDED RE-AUTHOR — the implement build runs with a BOUNDED budget and
        on_exhausted="abort"; if it can't turn the frozen tests green within that
        budget the tests are re-authored (they may be wrong/over-constrained),
        capped at tdd_autonomous_reauthor_max rounds. Exhausting the cap aborts the
        task with a clear reason rather than looping a wrong frozen test to the
        autonomous safety ceiling.

    The diff-gate freeze (Phase 72) is autonomous-safe as-is and used unchanged.
    Runs in the entrypoint frame so its BuildFailed propagates to the body; the
    bounded build's @tasks replay positionally across a crash-resume, and each
    re-author round uses a strictly increasing `attempt` so every test_author_task
    / critic_task cache key stays unique even when the feedback string repeats."""
    repo_root = str(find_project_root())
    # on_exhausted="abort" so the bounded build raises BuildFailed (no human pause);
    # an empty hil so no producer/gate pauses fire (there is no human). The
    # run-level autonomous safety ceiling is still enforced via check_cancel, which
    # _run_build_step threads into every producer attempt.
    bounded_retry = stage.retry.model_copy(update={"on_exhausted": "abort"})
    bounded_hil = HumanInLoopConfig()

    attempt = 0
    reauthor_round = 0
    last_feedback: str | None = None
    while True:
        feedback = (
            None if reauthor_round == 0
            else _AUTONOMOUS_REAUTHOR_FEEDBACK.format(last=last_feedback or "(none)")
        )
        rounds_info: dict = {}
        ta, attempt = await _author_with_review(
            task, task_plan, config, usage_by_task, base_gate_refs,
            thread_id=thread_id, audit=audit, autonomous=True,
            manual_checks=manual_checks, feedback=feedback, attempt=attempt,
            rounds_info=rounds_info,
        )
        # Phase 77b evidence. Each autonomous re-author round OVERWRITES the folder;
        # the LAST write reflects the suite the implement build is about to run —
        # the final accepted tests on success, or the last authored set on abort.
        # Record the autonomous re-author count alongside the critic rounds.
        if reauthor_round:
            rounds_info["autonomous_reauthor_round"] = reauthor_round
        write_test_author_folder(
            thread_id, task_index, task, ta, list(config.test_paths), rounds_info,
        )

        if not ta.testable:
            if ta.degrade_kind in _RED_CONFIRM_FAILURES:
                # The green→red guarantee couldn't be established and there is no
                # human to eyeball it → hard abort rather than ship without a proof.
                raise BuildFailed(
                    f"task:{task.id}", attempt,
                    f"Autonomous TDD: red-confirm failed ({ta.degrade_kind}): "
                    f"{ta.summary}. With no human reviewer the green→red guarantee "
                    "can't be established, so the task is aborted rather than "
                    "shipped without a proven test.",
                )
            # Legitimately untestable (the author's judgement): degrade to the
            # classic implement→qa path (unbounded, like a non-TDD autonomous task)
            # and record a manual check (surfaced in the result).
            if manual_checks is not None:
                manual_checks.append({
                    "task_id": task.id,
                    "title": task.title,
                    "acceptance_criteria": task.acceptance_criteria,
                    "reason": ta.summary,
                })
            await _run_task_build_step(
                task, plan_result, task_plan, qa_plan, base_gate_refs, {},
                stage, hil, config, check_cancel, usage_by_task, qa_holder,
                thread_id=thread_id, audit=audit, autonomous=True,
            )
            return

        # Testable → freeze the authored tests + run a BOUNDED implement build.
        diff_gate = {
            "builtin:diff-gate": _make_diff_gate(
                list(config.test_paths), ta.snapshot, repo_root
            ),
        }
        gate_refs = ["builtin:diff-gate", *base_gate_refs]
        impl_plan = _compose_red_green(task_plan, ta.red_output)
        # Phase 77c per-attempt evidence. Each re-author round re-freezes a NEW
        # suite, so the recorder closes over THIS round's snapshot; the impl/ folder
        # is cleared on attempt 1 (the bounded build restarts the count each round),
        # leaving only attempts against the suite that ultimately ran.
        on_attempt = _impl_attempt_recorder(
            thread_id, task_index, task, list(config.test_paths), ta.snapshot, repo_root,
        )
        try:
            await _run_task_build_step(
                task, plan_result, impl_plan, qa_plan, gate_refs, diff_gate,
                stage, bounded_hil, config, check_cancel, usage_by_task, qa_holder,
                thread_id=thread_id, audit=audit, autonomous=False,
                retry=bounded_retry, on_attempt=on_attempt,
            )
            return  # the implementation turned the frozen tests green → task done
        except BuildFailed as exc:
            if reauthor_round >= config.tdd_autonomous_reauthor_max:
                raise BuildFailed(
                    f"task:{task.id}", exc.attempts,
                    f"Autonomous TDD: the implementation could not pass the "
                    f"authored tests after {reauthor_round} re-author(s) "
                    f"(cap {config.tdd_autonomous_reauthor_max}); the tests may be "
                    f"wrong or over-constrained. Last gate output: "
                    f"{exc.last_feedback or '(none)'}",
                ) from exc
            last_feedback = exc.last_feedback
            reauthor_round += 1
            attempt += 1  # keep every author / critic cache key strictly increasing


async def _run_task_loop(
    stage,
    decomposition,
    plan_result,
    config,
    check_cancel,
    usage_by_task: dict,
    qa_holder: dict,
    *,
    thread_id: str,
    audit,
    autonomous: bool = False,
    manual_checks: list | None = None,
) -> None:
    """The built-in `task-build` station: run the decomposed task list, one
    produce⇄gate build per task.

    Each task reuses _run_build_step with a synthetic BuildStep built from the
    `task-build` StageSpec (produce/gate/retry/human_in_loop), so per-task
    retry/feedback, human pauses, and the growable budget all come for free. The
    built-in implementation producer / qa gate are made task-aware by composing
    this task's context into the plan text. A task that exhausts its budget raises
    BuildFailed(step_id="task:<id>") → the entrypoint's clean status="failed".
    Runs in the entrypoint body so its interrupt()s are reachable.

    `manual_checks` (Phase 73): when tdd is on, each task DEGRADED to the classic
    path (test-author judged it untestable / born-green / no script gate) appends
    {task_id, title, acceptance_criteria, reason} here, so the entrypoint can record
    the criteria a human must verify by hand."""
    hil = _build_hil(stage)
    # 1-based flow order → the task-NN-<id> evidence folder index (Phase 77b).
    for task_index, task in enumerate(decomposition.tasks, 1):
        task_plan = _compose_task_plan(plan_result.plan_text, task)
        qa_plan = PlanResult(
            title=plan_result.title,
            type=plan_result.type,
            plan_text=_compose_task_qa(plan_result.plan_text, task),
        )
        base_gate_refs = list(stage.gate)

        # Phase 81: the decomposer marked this task non-testable (markup/CSS/docs/
        # config/rename). Skip the whole TDD station — test-author AND critic — and
        # run the classic implement→qa build, recording the same degraded evidence a
        # runtime testable=False produces (Phase 73 manual_checks + a degraded
        # test-author/ summary.md) WITHOUT paying the author leg. Only under
        # config.tdd; tdd-off already takes the classic path below. Handles both the
        # autonomous and supervised stations uniformly (passes `autonomous` through),
        # so the testable branches below are reached only for genuinely testable tasks.
        if config.tdd and not task.testable:
            ta = TestAuthorResult(
                testable=False, degrade_kind=_DEGRADE_UNTESTABLE,
                summary="non-testable per decompose",
            )
            write_test_author_folder(
                thread_id, task_index, task, ta, list(config.test_paths), {},
            )
            if manual_checks is not None:
                manual_checks.append({
                    "task_id": task.id,
                    "title": task.title,
                    "acceptance_criteria": task.acceptance_criteria,
                    "reason": "non-testable per decompose",
                })
            await _run_task_build_step(
                task, plan_result, task_plan, qa_plan, base_gate_refs, {}, stage, hil,
                config, check_cancel, usage_by_task, qa_holder,
                thread_id=thread_id, audit=audit, autonomous=autonomous,
                on_attempt=None,
            )
            continue

        # Phase 76: autonomous TDD has no human red-review / re-author guard, so it
        # runs a distinct station — a HARD red-confirm gate + a bounded re-author
        # cycle (see _run_autonomous_tdd_task). The supervised / non-TDD path below
        # is unchanged.
        if config.tdd and autonomous:
            await _run_autonomous_tdd_task(
                task, task_index, plan_result, task_plan, qa_plan, base_gate_refs,
                stage, hil, config, check_cancel, usage_by_task, qa_holder,
                thread_id=thread_id, audit=audit, manual_checks=manual_checks,
            )
            continue

        gate_refs = base_gate_refs
        diff_gate: dict = {}
        on_attempt = None  # Phase 77c: set only for a testable TDD task (below)
        impl_plan = task_plan  # the implementer's plan; gains the RED output below
        # TDD red-green (Phase 72): author tests ONCE before the implement loop,
        # confirm the green→red transition, then freeze them with the diff-gate
        # (prepended so a 'green' is only trusted against pristine tests). Gated on
        # config.tdd; a testable=False verdict (untestable / born-green / no script
        # gate) leaves gate_refs untouched → the classic implement→qa path.
        # The test-author always sees the clean task_plan, so its @task cache key is
        # stable across resumes; the RED output is injected only into impl_plan.
        if config.tdd:
            rounds_info: dict = {}
            ta, _ = await _author_with_review(
                task, task_plan, config, usage_by_task, gate_refs,
                thread_id=thread_id, audit=audit, autonomous=autonomous,
                manual_checks=manual_checks, rounds_info=rounds_info,
            )
            # Phase 73/77b: write the test-author/ evidence folder for EVERY TDD
            # task, so the run folder shows the decision for each (not only the ones
            # that got tests). For a testable task this captures the final accepted
            # suite + its RED run + freeze hash; an untestable task gets summary.md
            # only. Written ONCE here, after convergence — before the implement loop.
            write_test_author_folder(
                thread_id, task_index, task, ta, list(config.test_paths), rounds_info,
            )
            if ta.testable:
                repo_root = str(find_project_root())
                diff_gate = {
                    "builtin:diff-gate": _make_diff_gate(
                        list(config.test_paths), ta.snapshot, repo_root
                    ),
                }
                gate_refs = ["builtin:diff-gate", *gate_refs]
                # Phase 72b: give the implementer's first attempt the failing tests.
                impl_plan = _compose_red_green(task_plan, ta.red_output)
                # Phase 77c: record per-attempt impl evidence (full run + freeze
                # MATCH/MISMATCH) against this task's frozen baseline.
                on_attempt = _impl_attempt_recorder(
                    thread_id, task_index, task, list(config.test_paths),
                    ta.snapshot, repo_root,
                )
            elif manual_checks is not None:
                # Degraded to the classic path: its acceptance criterion is a manual
                # check (Phase 73). Surfaced in the result + a manual-checks.md artifact.
                manual_checks.append({
                    "task_id": task.id,
                    "title": task.title,
                    "acceptance_criteria": task.acceptance_criteria,
                    "reason": ta.summary,
                })
        await _run_task_build_step(
            task, plan_result, impl_plan, qa_plan, gate_refs, diff_gate, stage, hil,
            config, check_cancel, usage_by_task, qa_holder,
            thread_id=thread_id, audit=audit, autonomous=autonomous,
            on_attempt=on_attempt,
        )


async def _run_build_stage(
    stage, config, plan_result, check_cancel, usage_by_task: dict, qa_holder: dict,
    *, thread_id: str, audit, autonomous: bool = False,
) -> None:
    """A user-declared `build` stage: one produce⇄gate loop over the WHOLE tree.

    Same engine as the per-task station; produce/gate reference [defs.*] /
    [builtin.*] parts (builtin:implementation / builtin:qa exposed as the spine's
    own agents). A non-proceed outcome raises BuildFailed(step_id=stage.id)."""
    producers, gates = _builtin_build_callables(
        config, plan_result, usage_by_task, qa_holder, thread_id,
    )
    synthetic = BuildStep(
        id=stage.id,
        produce=list(stage.produce),
        gate=list(stage.gate),
        ungated=stage.ungated or (not stage.gate),
        retry=stage.retry,
        human_in_loop=_build_hil(stage),
    )
    await _run_build_step(
        synthetic, config, plan_result.plan_text, check_cancel, usage_by_task,
        builtin_producers=producers,
        builtin_gates=gates,
        thread_id=thread_id,
        audit=audit,
        autonomous=autonomous,
    )


async def _run_qa_stage(
    stage, config, plan_result, check_cancel, usage_by_task: dict, qa_holder: dict,
    *, thread_id: str,
) -> None:
    """A whole-diff QA stage: the built-in `qa` agent judges the full diff against
    the overall plan. A FAIL raises BuildFailed(step_id=stage.id) → the clean
    status="failed" return (no commit, no PR). Stores the verdict in qa_holder for
    the result dict. Used by the built-in `qa` stage and `uses = "builtin:qa"`."""
    check_cancel()
    qa_result = await qa_task(plan_result, config.resolved_model(stage.model))
    _record_usage(usage_by_task, "qa", qa_result)
    write_qa(thread_id, qa_result)
    qa_holder["qa"] = qa_result
    if qa_result.result != "PASS":
        raise BuildFailed(stage.id, 1, qa_result.failures or "")


async def _dispatch_stage(
    stage, *, config, plan_result, decomposition, check_cancel, usage_by_task: dict,
    qa_holder: dict, summary_holder: dict, thread_id: str, audit, autonomous: bool,
    manual_checks: list | None = None,
) -> None:
    """Run ONE post-branch stage, by id / effective type. Called in flow order
    from the entrypoint body (plan + decompose run earlier, so they are skipped by
    the caller). Stage @tasks keep stable names derived from stage id, so resume
    replays the same graph deterministically."""
    sid = stage.id
    uses = stage.uses

    # The per-task fan-out station: the built-in `task-build`, or a user stage
    # placing it via `uses`.
    if (stage.namespace == "builtin" and sid == "task-build") or uses == "builtin:task-build":
        await _run_task_loop(
            stage, decomposition, plan_result, config, check_cancel,
            usage_by_task, qa_holder, thread_id=thread_id, audit=audit,
            autonomous=autonomous, manual_checks=manual_checks,
        )
        return

    # Whole-diff QA: the built-in `qa` stage, or a user stage placing it.
    if (stage.namespace == "builtin" and sid == "qa") or uses == "builtin:qa":
        await _run_qa_stage(
            stage, config, plan_result, check_cancel, usage_by_task, qa_holder,
            thread_id=thread_id,
        )
        return

    # Built-in docs / summarize stages.
    if stage.namespace == "builtin" and sid == "docs":
        check_cancel()
        docs_result = await docs_task(
            plan_result.plan_text, config.resolved_model(stage.model)
        )
        _record_usage(usage_by_task, "docs", docs_result)
        return
    if stage.namespace == "builtin" and sid == "summarize":
        check_cancel()
        summary_result = await summarize_task(
            plan_result.plan_text, config.resolved_model(stage.model)
        )
        _record_usage(usage_by_task, "summarize", summary_result)
        # Phase 77d: under TDD the executed test-author/ + impl/ evidence supersedes
        # the implementer's manual test-plan, so test-plan.md is suppressed (strictly
        # gated on config.tdd; off → written unchanged).
        write_summary(thread_id, summary_result, tdd=config.tdd)
        summary_holder["summarize"] = summary_result
        return

    # A user-declared `build` stage (whole-tree produce/gate loop).
    if stage.effective_type == "build":
        await _run_build_stage(
            stage, config, plan_result, check_cancel, usage_by_task, qa_holder,
            thread_id=thread_id, audit=audit, autonomous=autonomous,
        )
        return

    if uses is not None:
        # builtin:qa / builtin:task-build are handled above; placing a [defs.*]
        # part as a standalone stage via `uses` is not yet wired at runtime.
        raise StepError(
            f"stage {sid!r}: `uses = {uses!r}` is only supported for "
            "builtin:qa / builtin:task-build in this version."
        )

    # A user script / ai_agent stage.
    et = stage.effective_type
    check_cancel()
    repo_root = str(find_project_root())
    if et == "script":
        # Non-zero exit raises StepError (abort) — same contract as a pre-hook.
        await _make_script_task(sid)(
            sid, stage.path, stage.timeout if stage.timeout is not None else 60, repo_root
        )
        return
    if et == "ai_agent":
        result = await _make_ai_agent_task(sid)(
            sid, stage.path, config.resolved_model(stage.model), repo_root,
            plan_result.plan_text, 0, None,
            stage.allowed_tools, stage.disallowed_tools, stage.timeout,
        )
        _record_usage(usage_by_task, sid, result)
        # A single-agent stage's human_in_loop is a bool: pause once after it runs
        # so a human can review. Same abort contract as an approval gate.
        if stage.human_in_loop is True and not autonomous:
            emit_event(audit, thread_id, "interrupt",
                       payload={"kind": "step_ai_agent_review", "step_id": sid})
            decision = interrupt({
                "kind": "step_ai_agent_review",
                "step_id": sid,
                "detail": result.detail,
                "attempt": 0,
            })
            if _is_abort(decision):
                raise StepGateAborted(sid)
        return

    raise StepError(f"unsupported stage {sid!r} (effective type {et!r}).")


@task
@_audited_task("planning")
async def planning_task(request: str, model: str) -> PlanResult:
    # `model` is required — every caller resolves it via config.resolved_model(...),
    # so a default here would be dead and a drift trap.
    return await plan(request, model=model)


# The decomposer. Runs after planning (and again after each plan-feedback
# regeneration), turning the approved plan into an ordered task list that
# _run_task_loop executes. Its DecompositionResult is checkpointed (on the serde
# allowlist), so a re-execution of the entrypoint body after the plan-approval
# interrupt replays it for free.
@task
@_audited_task("decompose")
async def decompose_task(
    plan_text: str, model: str, max_tasks: int = 0, tdd: bool = False
) -> DecompositionResult:
    return await decompose(plan_text, model, max_tasks, tdd=tdd)


# Pre-flight check. Runs FIRST in the workflow — before planning — so a
# dirty working tree fails fast with zero LLM cost and no wasted approval
# round. Defence in depth: create_branch_task also calls verify_clean_tree
# internally, since the tree could be dirtied between approval and branch
# creation (the user has time to make edits during plan review).
#
# After the tree check, run any user-defined pre-hook scripts from
# `.orchestrator/pre-hooks/` (configurable). A non-zero exit from any script
# raises PreHookError, which propagates out of the task and aborts the workflow —
# same pattern as DirtyTreeError from verify_clean_tree. The hook's stdout becomes
# the displayed abort reason.
@task
@_audited_task("preflight")
async def verify_clean_tree_task() -> None:
    await asyncio.to_thread(verify_clean_tree)
    _cfg = load_config()
    await asyncio.to_thread(ensure_on_main, _cfg.pr.base_branch)
    await asyncio.to_thread(run_pre_hooks, _cfg.pre_hooks.dir, _cfg.pre_hooks.timeout)


# Deterministic git task. Wraps the synchronous create_branch
# function with asyncio.to_thread so it doesn't block the event loop —
# subprocess.run is blocking, and even fast git commands shouldn't stall
# the loop. The @task wrapper means a successful branch creation is
# checkpointed: on resume, we don't re-run git checkout, we read the
# branch name back from the checkpoint and move on.
@task
@_audited_task("create_branch")
async def create_branch_task(
    plan_result: PlanResult, max_slug_length: int = 50, thread_id: str = ""
) -> str:
    return await asyncio.to_thread(create_branch, plan_result, max_slug_length, thread_id)


# Runs the implementation agent (Claude Agent SDK in a loop) to edit files per the
# plan. A generic retry-block producer: it emits a plain StepResult (its `detail`
# is ignored downstream), and the commit/PR summary + test_plan are produced
# separately by summarize_task. On a retry the failing gate's feedback arrives via
# `feedback`, appended to the user message under a standard heading
# (feedback_section). Implementation is the most expensive task by far (minutes of
# LLM time, real file edits), so the @task wrapper's resume-skip is the single
# biggest cost win the checkpointer gives us.
async def _run_implementation_producer(
    plan_text: str, feedback: str | None, model: str
) -> StepResult:
    """The implementation agent invocation, factored out of implementation_task.

    Keeping it separate lets the @task wrapper stay a pure checkpoint boundary:
    on resume the @task replays its cached StepResult and this expensive agent
    call is skipped. (It is also what the build tests fake when they resume
    mid-loop, so the real @task's replay semantics stay under test.)
    """
    _impl = load_config().part("builtin:implementation")
    # File-editing tools from [builtin.implementation] (the v2 part). `tools` is
    # the PartSpec alias for allowed_tools. Fall back to the producer role default
    # when the part is absent or leaves tools unset.
    _impl_allowed = ((_impl.allowed_tools or _impl.tools) if _impl else None) or [
        "Read", "Edit", "Write", "Bash",
    ]
    _impl_disallowed = (_impl.disallowed_tools if _impl else None) or []
    _impl_timeout = _impl.timeout if _impl else None
    parts = ["## Plan", "", plan_text]
    if feedback:
        # The producer formats the raw gate detail via the engine's standard helper.
        parts += ["", feedback_section(feedback)]
    return await run_structured_agent(
        system_prompt=load_prompt("implementation"),
        user_message="\n".join(parts),
        model=model,
        # No Git, no commit, no PR tools — the orchestrator owns those entirely.
        allowed_tools=_impl_allowed,
        disallowed_tools=_impl_disallowed,
        # cwd must be the target repo root — the agent edits files there.
        cwd=find_project_root(),
        timeout=_impl_timeout,
        emit_tool_name="emit_step_result",
        emit_tool_description=(
            "Emit the final result of this step. Call exactly once when the work "
            "is complete, with a one-line `summary` of what you changed. After "
            "calling, stop — the orchestrator takes over."
        ),
        emit_tool_fields={"summary": str},
        result_factory=lambda c, u: StepResult(
            step_id="implementation",
            kind="ai_agent",
            ok=True,
            detail=c.get("summary", "") or "",
            usage=u,
        ),
    )


@task
@_audited_task("implementation")
async def implementation_task(
    plan_text: str,
    feedback: str | None,
    model: str,
) -> StepResult:
    # `model` is required (resolved by the caller). `feedback` also has no default —
    # both call sites pass it positionally — so a required `model` can follow it
    # without a "non-default arg after default arg" error.
    return await _run_implementation_producer(plan_text, feedback, model)


# Read-only LLM task: the QA agent reviews the uncommitted diff against the
# approved plan and emits a PASS/FAIL verdict. No file edits, no git operations.
# On FAIL the build's retry loop re-runs the implementation producer with the
# failure text as feedback.
@task
@_audited_task("qa")
async def qa_task(
    plan_result: PlanResult, model: str
) -> QaResult:
    return await qa(plan_result, model=model)


# The summarizer. Runs ONCE after the impl→QA retry block passes, before commit.
# Reads the plan + `git diff HEAD` and emits the commit/PR summary + test_plan
# (the implementation producer is generic, so this read-only post-loop @task owns
# that structured output). Its SummaryResult is checkpointed (on the serde
# allowlist), so a crash before commit replays it.
@task
@_audited_task("summarize")
async def summarize_task(
    plan_text: str, model: str
) -> SummaryResult:
    return await summarize(plan_text, model)


# Documentation agent, a permanent spine task. Runs once after summarize, before
# commit — on the final, QA-passed code — so any doc edits land in the same commit.
# The prompt ships in the package (orchestrator/prompts/docs.md, tracked by git)
# and is loaded via load_prompt — the same loader as planning/implementation/qa,
# so it inherits the .orchestrator/prompts/ override path — rather than from
# .orchestrator/agents/ (gitignored), so a spine step never depends on a
# local-only file.
@task
@_audited_task("docs")
async def docs_task(
    plan_text: str, model: str
) -> StepResult:
    """Run the documentation agent against the QA-passed working tree.

    A @task like every other spine step: its StepResult is checkpointed, so a
    crash between docs and commit replays the docs result on resume (no LLM
    re-call). The package prompt is the system prompt; the agent reads
    `git diff HEAD` itself and edits ONLY documentation (.md) — it never edits
    source, including the workflow that orchestrates it. Returns a StepResult
    (already on the serde allowlist)."""
    return await run_structured_agent(
        system_prompt=load_prompt("docs"),
        user_message="\n".join(["## Plan", "", plan_text]),
        model=model,
        allowed_tools=["Read", "Edit", "Write", "Bash", "Grep"],
        disallowed_tools=[],
        cwd=find_project_root(),
        timeout=(load_config().stage("docs").timeout if load_config().stage("docs") else None),
        emit_tool_name="emit_step_result",
        emit_tool_description=(
            "Emit the final result of this step. Call exactly once when done, "
            "with a one-line `summary` of what you did. After calling, stop."
        ),
        emit_tool_fields={"summary": str},
        result_factory=lambda c, u: StepResult(
            step_id="docs",
            kind="ai_agent",
            ok=True,
            detail=c.get("summary", "") or "",
            usage=u,
        ),
    )


# Commit / push / PR are three idempotent tasks. Each step's success is
# checkpointed independently, so a failure at push or pr_create can be resumed via
# the resume_run MCP tool without re-committing or re-pushing work that already
# landed.
#
# push_task and pr_create_task take `sha` as an input even though they
# don't use it directly — including it in the inputs invalidates the
# @task cache key when the commit changes (e.g. if an earlier retry
# produced a different commit), forcing those downstream tasks to run
# fresh instead of returning stale cached results.
@task
@_audited_task("commit")
async def commit_task(
    branch: str, title: str, summary: str, base_branch: str | None = None
) -> str:
    """Stage + commit any uncommitted changes; return HEAD SHA.
    Idempotent: a clean tree with an existing ahead-of-base commit
    returns that commit's SHA without re-committing."""
    return await asyncio.to_thread(commit, branch, title, summary, base_branch)


@task
@_audited_task("push")
async def push_task(branch: str, sha: str, base_branch: str | None = None, auto_rebase: bool = True) -> None:
    """Push branch with upstream tracking. Idempotent (git push is a
    no-op when the remote is already up to date).

    Fetches origin first and rebases onto origin/<base_branch> if it
    moved since branch creation. Rebase conflicts surface as a UserActionError;
    set auto_rebase=False to skip and ask for manual rebase instead.
    """
    return await asyncio.to_thread(push, branch, base_branch, auto_rebase)


@task
@_audited_task("pr_create")
async def pr_create_task(
    branch: str,
    title: str,
    summary: str,
    test_plan: str,
    sha: str,
    base_branch: str | None = None,
    draft: bool = False,
    reviewers: list[str] | None = None,
    plan_type: str | None = None,
) -> str:
    """Open a PR and return its URL. Idempotent: if a PR already exists
    for this branch, returns its URL instead of opening another.

    `plan_type` (plan_result.type) is passed through to pr_create, which
    auto-derives the PR label from it."""
    return await asyncio.to_thread(
        pr_create, branch, title, summary, test_plan,
        base_branch, draft, reviewers or [], plan_type,
    )


# ---------------------------------------------------------------------------
# Entrypoint-body helpers carve the @entrypoint body into readable sections. They
# are plain `async def` (not @task), so the interrupt()s inside _plan_and_approve
# and _dispatch_stage run in the entrypoint frame — the same rule _run_build_step
# follows.
#
# Invariant: the @task names, their count, and their EXECUTION ORDER must stay
# fixed. LangGraph keys a task by name + call position, and calling @tasks from
# module-level helpers (here, and the flow-ordered _dispatch_stage loop) is the
# established pattern, so the task graph stays identical and resume/replay is
# unaffected.
# ---------------------------------------------------------------------------


async def _gate_checkpoint() -> None:
    """The version + pipeline-hash resume gates.

    record_version_task / record_pipeline_hash_task return the live values on a
    fresh run (and persist them) and the cached creation-time values on resume; a
    mismatch means the body or the v2 pipeline changed incompatibly since the run
    started. Both sides hash via load_config().pipeline (not a captured config) so
    a patched-load test or a default-config run computes the same hash. Raised from
    the entrypoint body (not a @task) so it propagates straight out of ainvoke
    without mutating the checkpoint — the run stays resumable once the code is
    reverted or the run abandoned.
    """
    stored_version = await record_version_task()
    if stored_version != WORKFLOW_VERSION:
        raise IncompatibleCheckpointError(stored_version, WORKFLOW_VERSION)
    current_hash = load_config().pipeline.manifest_hash()
    stored_hash = await record_pipeline_hash_task()
    if stored_hash != current_hash:
        raise IncompatiblePipelineError(stored_hash, current_hash)


async def _plan_and_approve(
    request: str,
    config: OrchestratorConfig,
    *,
    thread_id: str,
    audit,
    autonomous: bool,
    check_cancel,
    usage_by_task: dict,
) -> tuple[PlanResult, DecompositionResult]:
    """Plan → decompose → approval loop. Returns the approved (plan, decomposition).

    The loop runs until the user replies "yes"; any other reply is feedback that
    regenerates the plan (and re-decomposes, so the two never drift). The plan is
    decomposed BEFORE the approval interrupt so the task list is shown alongside
    the plan. interrupt() is reachable because this helper runs in the entrypoint
    frame. Planning is auto-approved under human_in_loop=false or autonomous mode.
    """
    plan_stage = config.stage("plan")
    decompose_stage = config.stage("decompose")
    plan_model = config.resolved_model(plan_stage.model if plan_stage else None)
    plan_hil = bool(plan_stage.human_in_loop) if plan_stage else True
    decompose_model = config.resolved_model(decompose_stage.model if decompose_stage else None)
    decompose_max_tasks = decompose_stage.max_tasks if decompose_stage else 0

    async def _run_planning(req: str) -> PlanResult:
        check_cancel()
        pr = await planning_task(req, plan_model)
        _record_usage(usage_by_task, "planning", pr)
        write_plan(thread_id, pr)
        return pr

    async def _run_decompose(pr: PlanResult) -> DecompositionResult:
        check_cancel()
        # Phase 78b: under TDD the test-author owns each task's tests, so tell the
        # decomposer not to emit a standalone test-writing task.
        d = await decompose_task(
            pr.plan_text, decompose_model, decompose_max_tasks, config.tdd
        )
        _record_usage(usage_by_task, "decompose", d)
        write_decomposition(thread_id, d)
        return d

    plan_result = await _run_planning(request)
    decomposition = await _run_decompose(plan_result)

    while True:
        if plan_hil and not autonomous:
            emit_event(audit, thread_id, "interrupt", payload={"kind": "plan_approval"})
            approval = interrupt({
                "kind": "plan_approval",
                "plan": plan_result.model_dump(),
                "tasks": [t.model_dump() for t in decomposition.tasks],
                "ask": "Approve this plan? Reply 'yes' or describe changes.",
            })
        else:
            approval = "yes"
        if approval == "yes":
            break
        plan_result = await _run_planning(f"{request}\n\nFeedback: {approval}")
        decomposition = await _run_decompose(plan_result)

    return plan_result, decomposition


async def _ship(
    plan_result: PlanResult,
    branch_name: str,
    summary_result: SummaryResult,
    config: OrchestratorConfig,
    *,
    thread_id: str,
    check_cancel,
) -> str:
    """commit → push → pr. Returns the PR url.

    summarize and docs already ran as flow stages (before the empty-diff guard),
    so this is purely the git ship rails. The three @tasks are idempotent and
    individually checkpointed, so a failure between commit/push/pr is resumable. No
    cancel checks once the commit has landed — aborting then would leave a
    half-shipped branch (use git, not the orchestrator).
    """
    # Each task's audit task_start/complete is emitted inside its @task (via
    # @_audited_task), so on resume a replayed task is not re-logged.
    check_cancel()
    sha = await commit_task(
        branch_name, plan_result.title, summary_result.summary, config.pr.base_branch
    )
    await push_task(branch_name, sha, config.pr.base_branch, config.git.auto_rebase)
    pr_url = await pr_create_task(
        branch_name,
        plan_result.title,
        summary_result.summary,
        summary_result.test_plan,
        sha,
        config.pr.base_branch,
        config.pr.draft,
        config.pr.reviewers,
        plan_result.type,
    )
    return pr_url


def _log_run_usage(usage_by_task: dict, thread: str, *, status: str | None) -> None:
    """Persist per-task + run-level usage at run end (Phase 80c).

    Two sinks, both best-effort: a per-task `usage` audit event (tokens by category
    + models + cost) so the compliance log carries spend, and a run-END rollup line
    in runs.jsonl so a run's total tokens/cost are durable and greppable next to its
    start. No-op when no usage was captured (e.g. fully-stubbed tests)."""
    rollup = run_usage_rollup(usage_by_task)
    if not rollup:
        return
    sink = _build_task_audit_sink()
    for task_name, totals in rollup["by_task"].items():
        emit_event(sink, thread, "usage", task_name=task_name, payload=totals)
    append_usage_rollup(thread, rollup, status=status)


def _finalize(usage_by_task: dict, thread: str, **fields) -> dict:
    """Assemble a workflow result dict: aggregate + persist usage, append it.

    Every workflow exit (succeeded / no_changes / failed / aborted / cancelled)
    ends by aggregating usage, writing it to the run folder, and returning a dict
    with a `usage` key. This collapses that shared tail; `fields` carries the
    per-status keys. `thread` is the run's thread_id used for write_usage (named
    distinctly so a result `thread_id` field can still be passed in `fields`).
    """
    usage = aggregate_usage(usage_by_task)
    write_usage(thread, usage)
    _log_run_usage(usage_by_task, thread, status=fields.get("status"))
    return {**fields, "usage": usage}


# build_workflow is a factory, not a module-level workflow definition.
# Why: AsyncSqliteSaver.from_conn_string returns an async context manager
# that opens the SQLite connection on entry and closes it on exit. The
# @entrypoint decorator captures the checkpointer at definition time, so
# the workflow MUST be defined inside the async-with block — there's no
# clean way to attach a still-opening connection to a module-level decorator.
# The asynccontextmanager wrapper lets callers do `async with build_workflow()`.
@asynccontextmanager
async def build_workflow(
    db_path: str | None = None,
    config: OrchestratorConfig | None = None,
) -> AsyncIterator:
    if config is None:
        config = load_config()
    raw_path = db_path if db_path is not None else config.db_path
    p = Path(raw_path)
    effective_db_path = str(p if p.is_absolute() else find_project_root() / p)

    async with AsyncSqliteSaver.from_conn_string(effective_db_path) as checkpointer:
        # AsyncSqliteSaver.from_conn_string doesn't accept a custom serde,
        # so we swap it in after construction. Both attributes need to
        # change: `serde` is the public one read by BaseCheckpointSaver,
        # `jsonplus_serde` is the internal one AsyncSqliteSaver uses
        # directly for some write paths.
        checkpointer.serde = _CUSTOM_SERDE
        checkpointer.jsonplus_serde = _CUSTOM_SERDE

        @entrypoint(checkpointer=checkpointer)
        async def workflow(request: str) -> dict:
            thread_id = get_config()["configurable"]["thread_id"]

            # One resolved flag for the whole run. Read once here so every
            # gate-suppression check below shares it.
            autonomous = config.fully_autonomous
            # Wall-clock budget is per-invocation (monotonic resets per process).
            # Fine because autonomous runs don't pause mid-flight — a run is one
            # continuous process; a resume after a crash starts a fresh budget.
            _run_started = time.monotonic()

            # Cancel-check helper closed over thread_id. Called before each task
            # (and threaded into builds as check_cancel, so it fires per producer
            # attempt). Raises WorkflowCancelled if the cancel_run MCP tool marked
            # this thread, OR — in autonomous mode — if the run crossed its
            # time/cost safety ceiling. The except clause at the bottom converts
            # either into a status="cancelled" dict.
            _warned_unpriced_models: set[str] = set()

            def _check_cancel() -> None:
                raise_if_cancelled(thread_id)
                if not autonomous:
                    return
                max_seconds = config.autonomous_max_seconds
                if max_seconds > 0 and (time.monotonic() - _run_started) > max_seconds:
                    raise AutonomousCeilingExceeded(thread_id, "autonomous_ceiling")
                max_cost = config.autonomous_max_cost_usd
                if max_cost > 0:
                    spent = 0.0
                    for entries in usage_by_task.values():
                        for u in entries:
                            c = u.cost_usd()
                            if c is not None:
                                spent += c
                            elif u.model not in _warned_unpriced_models:
                                logger.warning(
                                    "autonomous cost ceiling is set but model %r has "
                                    "no known price — the ceiling cannot account for "
                                    "its usage; relying on the time ceiling if set",
                                    u.model,
                                )
                                _warned_unpriced_models.add(u.model)
                    if spent > max_cost:
                        raise AutonomousCeilingExceeded(thread_id, "autonomous_ceiling")

            # Build the audit sink once per invocation. Each ainvoke() call (fresh
            # start or resume after interrupt) emits a "resume" event so the log
            # captures every interaction.
            _audit_log = str(find_project_root() / config.audit.log_path)
            _audit: AuditSink = (
                build_sink(_audit_log) if config.audit.enabled else NoopAuditSink()
            )
            emit_event(_audit, thread_id, "resume")

            # Accumulate token usage across all agent calls for the run.
            # Keys map to lists so retries (multiple impl/qa calls) are
            # all summed in the final aggregate. Defined OUTSIDE the
            # try block so the cancel-handler can still report whatever
            # tokens were spent before the cancel signal landed.
            usage_by_task: dict[str, list[TaskUsage]] = {
                "planning": [],
                "decompose": [],
                "test_author": [],
                "coverage_critic": [],
                "implementation": [],
                "qa": [],
                "summarize": [],
                "docs": [],
            }
            # Pre-declared so the BuildFailed handler can reference them in scope.
            # A build only runs after the branch, so by the time BuildFailed can be
            # raised both are set; these defaults just keep the names bound for the
            # except clause.
            plan_result: PlanResult | None = None
            branch_name: str | None = None

            try:
                # Resume gates (workflow-version + v2 pipeline hash). Raised from
                # the body, not a @task, so a mismatch leaves the checkpoint
                # untouched and resumable. See the helper.
                await _gate_checkpoint()

                _check_cancel()
                # Audit task events are emitted inside each @task (@_audited_task),
                # so they fire on real execution only — a resume that replays a
                # completed task no longer re-logs it.
                await verify_clean_tree_task()

                # Plan → decompose → approval loop (see _plan_and_approve). The
                # leading plan/decompose stages run HERE so the approval shows the
                # task list and a regeneration re-decomposes; they are SKIPPED in the
                # flow-dispatch loop below. Landmine #4: create_branch_task (the
                # first side effect) stays AFTER this — interrupt() re-executes the
                # body on resume and completed @tasks replay from cache.
                plan_result, decomposition = await _plan_and_approve(
                    request, config,
                    thread_id=thread_id, audit=_audit, autonomous=autonomous,
                    check_cancel=_check_cancel, usage_by_task=usage_by_task,
                )

                # Optional pre-branch approval gate ([branch].human_in_loop).
                if config.branch.human_in_loop and not autonomous:
                    emit_event(_audit, thread_id, "interrupt", payload={"kind": "branch_approval"})
                    interrupt({
                        "kind": "branch_approval",
                        "ask": "Proceed with branch creation?",
                    })

                # Fail loud on an empty decomposition: an approved plan that
                # produced zero tasks would otherwise run the task station as a
                # no-op and return status="no_changes", masking a decomposer/plan
                # failure. Raised before branch creation, so nothing is shipped.
                if not decomposition.tasks:
                    raise EmptyDecompositionError()

                _check_cancel()
                branch_name = await create_branch_task(
                    plan_result, config.branch.max_slug_length, thread_id
                )
                rename_with_branch(thread_id, branch_name)

                # Run the POST-BRANCH stages in flow order (plan/decompose already
                # ran above). _dispatch_stage routes each by id / effective type:
                # the task-build station, user builds, docs, summarize, a whole-diff
                # qa gate, and user script/ai_agent stages. A failing build or qa
                # gate raises BuildFailed → the clean status="failed" return below
                # (no commit, no PR). The holders stash the summary (for the ship
                # rails) and the latest QA verdict (for the result dict). Stage
                # @tasks keep stable, id-derived names + a deterministic flow order,
                # so resume replays the same graph.
                _qa_holder: dict[str, QaResult] = {}
                _summary_holder: dict[str, SummaryResult] = {}
                # Phase 73: TDD tasks that degraded to the classic path; their
                # acceptance criteria need manual verification.
                _manual_checks: list[dict] = []
                for stage in config.pipeline.stages:
                    if stage.id in ("plan", "decompose"):
                        continue
                    await _dispatch_stage(
                        stage,
                        config=config, plan_result=plan_result,
                        decomposition=decomposition, check_cancel=_check_cancel,
                        usage_by_task=usage_by_task, qa_holder=_qa_holder,
                        summary_holder=_summary_holder, thread_id=thread_id,
                        audit=_audit, autonomous=autonomous,
                        manual_checks=_manual_checks,
                    )

                # Phase 73: persist the manual-verification list (no-op if empty).
                write_manual_checks(thread_id, _manual_checks)

                # The latest QA verdict (last task's per-task QA, or a qa stage).
                # None only if the build was ungated AND no qa stage ran.
                qa_result = _qa_holder.get("qa")

                # Empty-diff resilience. If the build produced no diff (the producer
                # made no edits and nothing is ahead of base), there is nothing to
                # ship — committing would create an empty commit and a no-op PR.
                # Return a clean status="no_changes" instead, skipping the commit /
                # push / pr rails. Checked before the pr_approval gate so we never
                # ask "open a PR?" for an empty diff. Pre-commit, so return is safe.
                _check_cancel()
                if not await asyncio.to_thread(
                    working_tree_has_changes, config.pr.base_branch
                ):
                    return _finalize(
                        usage_by_task, thread_id,
                        status="no_changes",
                        plan=plan_result.model_dump(),
                        branch=branch_name,
                        qa=qa_result.model_dump() if qa_result else None,
                        # Phase 73: present only when some task degraded (else omitted,
                        # so non-TDD / all-testable results keep their exact shape).
                        **({"manual_checks": _manual_checks} if _manual_checks else {}),
                    )

                # Optional pre-PR gate ([pr].human_in_loop).
                if config.pr.human_in_loop and not autonomous:
                    emit_event(_audit, thread_id, "interrupt", payload={"kind": "pr_approval"})
                    interrupt({"kind": "pr_approval", "ask": "QA passed. Open a PR?"})

                # The summarize stage ran in the flow (require-summarize guarantees
                # it exists), populating the holder. Guard defensively in case a
                # custom pipeline ordered or stubbed it oddly.
                summary_result = _summary_holder.get("summarize") or SummaryResult(
                    summary=plan_result.title, test_plan=""
                )

                # commit → push → pr (see _ship).
                pr_url = await _ship(
                    plan_result, branch_name, summary_result, config,
                    thread_id=thread_id, check_cancel=_check_cancel,
                )
                return _finalize(
                    usage_by_task, thread_id,
                    status="succeeded",
                    plan=plan_result.model_dump(),
                    branch=branch_name,
                    # {summary, test_plan} shape unchanged for MCP/UI/tests.
                    implementation={
                        "summary": summary_result.summary,
                        "test_plan": summary_result.test_plan,
                    },
                    # None when the build was ungated or gated only on a non-qa gate.
                    qa=qa_result.model_dump() if qa_result else None,
                    pr_url=pr_url,
                    # Phase 73: criteria a human must verify (omitted when none).
                    **({"manual_checks": _manual_checks} if _manual_checks else {}),
                )

            except BuildFailed as exc:
                # A build step ran its full budget without a passing gate under
                # on_exhausted="abort" (or a human declined to keep retrying). Clean
                # status="failed" with the last gate feedback under `qa_failures`, no
                # commit, no PR — builds are pre-commit, so nothing is half-shipped.
                # plan_result/branch_name are guarded. failed_task_id is "task:<id>"
                # (the per-task station), a whole-diff qa stage id, or a user build
                # stage id.
                return _finalize(
                    usage_by_task, thread_id,
                    status="failed",
                    plan=plan_result.model_dump() if plan_result else None,
                    branch=branch_name,
                    failed_task_id=exc.step_id,
                    qa_failures=exc.last_feedback,
                )

            except StepGateAborted as exc:
                # An approval_gate step was resumed with an abort decision. Every gate
                # runs before the commit line, so nothing is half-shipped; branch_name
                # may not exist yet (gates can fire pre-branch).
                return _finalize(
                    usage_by_task, thread_id,
                    status="aborted",
                    thread_id=thread_id,
                    aborted_at=exc.step_id,
                )

            except WorkflowCancelled as exc:
                # A between-task check found the cancel flag set, or an autonomous run
                # tripped its safety ceiling — `reason` tells them apart
                # ("autonomous_ceiling" vs. a user cancel_run). Whatever was in
                # progress has completed (the SDK doesn't interrupt mid-task).
                reason = getattr(exc, "reason", "user_cancel")
                emit_event(_audit, thread_id, "cancel", payload={"reason": reason})
                return _finalize(
                    usage_by_task, thread_id,
                    status="cancelled",
                    thread_id=thread_id,
                    reason=reason,
                )

        yield workflow
