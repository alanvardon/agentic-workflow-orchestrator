"""Test-author agent — writes failing tests for ONE task, before implementation.

This is the role half of Phase 72's red-green separation: a DIFFERENT agent than
the implementer authors the tests, so the implementer can never write a test it
knows will pass (or weaken one on retry). It is an *agent loop* (like qa.py /
implementation), not a single LLM call: the author reads the task, designs a test
through the public interface, writes the test file(s), and runs the suite to
confirm the new tests fail.

What this module owns: the test-author prompt, its tools, the emit-tool schema,
and the `TestAuthorResult` factory. The surrounding red-green machinery — the
green→red transition check, the test-file snapshot the diff-gate freezes, and the
classic-fallback routing — lives in `workflow._run_test_author`, which calls this.

`TestAuthorResult` is on the workflow's serde allowlist so a mid-run resume
replays the authored result (the tests are NOT regenerated) — the whole point of
authoring once, before the retry loop.

Generic by design: nothing here is project-specific. The author judges
testability at runtime and emits `testable=False` with a reason when a behaviour
isn't unit-testable (or the criteria are too vague) — the workflow then falls back
to the classic implement→qa path for that task.
"""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import sys

from pydantic import BaseModel

from orchestrator.usage import TaskUsage
from orchestrator.prompt_loader import load_prompt

from orchestrator.agents.runner import run_structured_agent
from orchestrator.git_ops import REPO_ROOT


_TEST_AUTHOR_SYSTEM_PROMPT = load_prompt("test-author")

# The author's tools: it writes test files (Edit/Write) and runs the suite (Bash)
# to confirm red. Read for context. No git — the orchestrator owns that.
_DEFAULT_TOOLS = ["Read", "Edit", "Write", "Bash"]


class TestAuthorResult(BaseModel):
    """The outcome of authoring tests for one task.

    `testable`/`summary` come straight from the agent; `snapshot`/`red_output`
    are filled in by workflow._run_test_author after it confirms the green→red
    transition and hashes the frozen test files. The whole object is checkpointed
    (serde allowlist), so a resume replays it rather than re-authoring.
    """

    # Not a pytest test class despite the leading "Test" — tell the collector so.
    __test__ = False
    # Bump on incompatible shape changes; pure additions of optional fields don't.
    schema_version: int = 1
    # False → the task is not unit-testable (or the author couldn't write a
    # failing test); the workflow routes it to the classic implement→qa path.
    testable: bool
    # When testable is False, WHICH kind of degrade this is — set by
    # workflow._run_test_author, not the agent. Distinguishes a legitimate "not
    # unit-testable" judgement (a graceful classic fallback, even autonomous) from
    # a red-confirm FAILURE (no script gate / non-green baseline / born-green),
    # which autonomous TDD treats as a hard abort (Phase 76). None when testable.
    degrade_kind: str | None = None
    # The author's one-line note on success, or the UNTESTABLE reason.
    summary: str = ""
    # Hash of the test_paths globset right after authoring — the diff-gate's
    # frozen baseline. Empty when not testable.
    snapshot: str = ""
    # The failing-suite output captured at red-confirm time (for artifacts /
    # visibility). Empty when not testable. This is the failures-only summary that
    # feeds the implementer's first-attempt plan (_compose_red_green) and the
    # red-review display — short and focused on what to make pass.
    red_output: str = ""
    # The COMPLETE final RED run (every test, pass+fail; both streams) captured at
    # red-confirm time via the Phase 77a full-output capture — the evidence layer's
    # `results-test-run.md` (Phase 77b). Distinct from `red_output` (failures-only):
    # a full run proves what was EXECUTED, not just what failed. Empty when not
    # testable. Additive optional field → no schema_version bump.
    full_run: str = ""
    usage: TaskUsage | None = None


def _build_user_message(plan_text: str, feedback: str | None = None) -> str:
    """The per-task message for the author: the overall plan + this task's slice
    + acceptance criteria (composed upstream by _compose_task_plan).

    On a re-author (Phase 72b red-review escape) `feedback` carries the human's
    notes on why the previous tests were rejected; it is appended so the author
    revises its OWN tests accordingly. Revise the existing test files; do not add
    a parallel set."""
    parts = ["## Plan", "", plan_text]
    if feedback:
        parts += [
            "",
            "## Re-author feedback",
            "",
            "A human reviewed your previous tests for this task and asked for "
            "changes. Revise the test file(s) you wrote (do not leave the old "
            "versions alongside new ones) to address this:",
            "",
            feedback,
        ]
    return "\n".join(parts)


async def author_tests(
    plan_text: str,
    model: str,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
    feedback: str | None = None,
) -> TestAuthorResult:
    """Run the test-author agent and return its verdict.

    The agent writes test file(s) for the task and emits `testable` + a `reason`.
    It does NOT compute the snapshot or confirm red — that is the workflow's job
    (it owns the green→red transition and the freeze). The model and the prompt /
    tool overrides are resolved by the caller from the prompt frontmatter.

    `system_prompt`: the resolved prompt body+footer. None → the bundled/
    convention default (`_TEST_AUTHOR_SYSTEM_PROMPT`).
    `allowed_tools` / `disallowed_tools`: None → the author-role defaults
    (`_DEFAULT_TOOLS` / none); the workflow passes the prompt frontmatter's tools
    when it sets any.
    `feedback`: on a re-author (Phase 72b red-review escape), the human's notes on
    why the prior tests were rejected — appended to the user message.
    """
    return await run_structured_agent(
        system_prompt=system_prompt if system_prompt is not None else _TEST_AUTHOR_SYSTEM_PROMPT,
        user_message=_build_user_message(plan_text, feedback),
        model=model,
        allowed_tools=allowed_tools if allowed_tools is not None else _DEFAULT_TOOLS,
        disallowed_tools=disallowed_tools if disallowed_tools is not None else [],
        # Same repo root as implementation/QA — the author writes test files there.
        cwd=REPO_ROOT,
        timeout=None,
        emit_tool_name="emit_test_author_result",
        emit_tool_description=(
            "Emit the result of authoring tests for this task. Call exactly once "
            "when done. `testable` is true if you wrote one or more FAILING tests "
            "that exercise the task's behaviour through its public interface; "
            "false if the behaviour is not unit-testable or the criteria are too "
            "vague to test (set `reason` to why). `reason` is a one-line note: on "
            "success, what behaviour the tests cover; on testable=false, the "
            "UNTESTABLE reason. After calling, stop — the orchestrator takes over."
        ),
        emit_tool_fields={"testable": bool, "reason": str},
        result_factory=lambda captured, usage: TestAuthorResult(
            testable=_coerce_bool(captured.get("testable")),
            summary=(captured.get("reason") or ""),
            usage=usage,
        ),
    )


def _coerce_bool(raw: object) -> bool:
    """Interpret the agent's emitted `testable` as a bool, fail-closed.

    A non-canonical value is treated as NOT testable (→ classic fallback), never
    a ValidationError crash — mirrors qa._coerce_verdict / steps._coerce_passed."""
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("true", "1", "yes")


# Standalone test:
#   python -m orchestrator.agents.test_author "author tests for X"
if __name__ == "__main__":
    from orchestrator.config import OrchestratorConfig

    request = " ".join(sys.argv[1:]) or "author tests for the current task"

    async def _main() -> None:
        result = await author_tests(request, OrchestratorConfig().default_model)
        print(result.model_dump_json(indent=2))

    asyncio.run(_main())
