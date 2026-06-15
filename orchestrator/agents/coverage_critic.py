"""Coverage critic — judges whether a task's tests are MEANINGFUL (Phase 74).

The meaningfulness backstop of supervised TDD: after the test-author writes
failing tests for a task, a DIFFERENT, read-only agent reads them and judges
whether they actually pin down the task's behaviour — would they fail if the
behaviour were implemented wrongly? It catches tautological / vacuous / shape-only
tests that a deterministic test gate and the diff-gate can't (a vacuous test still
goes red→green). It NEVER writes or edits tests (that would reintroduce the
born-green circularity the test-author/diff-gate guard against); a `meaningful=False`
verdict routes BACK to the test-author to re-author with the critic's feedback.

Generic by design: nothing here is project-specific. The verdict flows through a
checkpointed @task (workflow.critic_task), so `CoverageCriticResult` is on the
workflow's serde allowlist.
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


_COVERAGE_CRITIC_SYSTEM_PROMPT = load_prompt("coverage-critic")

# Read-only: the critic reads tests + conventions and runs read commands; it never
# writes (no Edit/Write) — meaningfulness is judged from the test code itself.
_DEFAULT_TOOLS = ["Read", "Bash", "Grep"]


class CoverageCriticResult(BaseModel):
    """The critic's verdict on one task's tests.

    Checkpointed (serde allowlist), so a resume replays the verdict rather than
    re-running the critic. `meaningful=False` carries `feedback` the test-author
    uses to re-author."""

    # Bump on incompatible shape changes; pure additions of optional fields don't.
    schema_version: int = 1
    # True → the tests meaningfully pin down the task's behaviour; False → at least
    # one is vacuous/tautological/shape-only and the author should revise them.
    meaningful: bool
    # On False, a concrete note (which test is weak + what to assert); on True, a
    # one-line confirmation.
    feedback: str = ""
    usage: TaskUsage | None = None


def _build_user_message(plan_text: str) -> str:
    """The per-task message for the critic: the overall plan + this task's slice +
    acceptance criteria (composed upstream by _compose_task_plan)."""
    return "\n".join(["## Plan", "", plan_text])


async def critique_tests(
    plan_text: str,
    model: str,
    system_prompt: str | None = None,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
) -> CoverageCriticResult:
    """Run the coverage-critic agent on a task's just-authored tests.

    Read-only: it judges the test code, it does not run/await the implementation
    (the tests are red — the implementation doesn't exist yet). `system_prompt` /
    tools default to the bundled prompt + read-only role tools; the workflow passes
    the prompt frontmatter's overrides when set."""
    return await run_structured_agent(
        system_prompt=system_prompt if system_prompt is not None else _COVERAGE_CRITIC_SYSTEM_PROMPT,
        user_message=_build_user_message(plan_text),
        model=model,
        allowed_tools=allowed_tools if allowed_tools is not None else _DEFAULT_TOOLS,
        disallowed_tools=disallowed_tools if disallowed_tools is not None else [],
        cwd=REPO_ROOT,
        timeout=None,
        emit_tool_name="emit_coverage_critic_result",
        emit_tool_description=(
            "Emit your verdict on whether this task's tests are meaningful. Call "
            "exactly once when done. `meaningful` is true if the tests would fail "
            "when the behaviour is implemented wrongly (they pin down the task's "
            "behaviour through its public interface); false if one or more are "
            "vacuous, tautological, assert only the shape of data, or would pass "
            "against a stub. `feedback`: on false, name which test is weak and what "
            "behaviour it should assert instead; on true, a one-line confirmation."
        ),
        emit_tool_fields={"meaningful": bool, "feedback": str},
        result_factory=lambda captured, usage: CoverageCriticResult(
            meaningful=_coerce_bool(captured.get("meaningful")),
            feedback=(captured.get("feedback") or ""),
            usage=usage,
        ),
    )


def _coerce_bool(raw: object) -> bool:
    """Interpret the critic's `meaningful` verdict, FAIL-OPEN.

    Opposite stance to the test-author's fail-closed coercion: a non-canonical
    value is treated as `meaningful=True` (proceed), so critic confusion never
    blocks the run or churns the re-author loop — only an explicit negative routes
    back to re-authoring. The "never wedge" exhaustion policy (Phase 74) extends
    the same spirit."""
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("false", "0", "no")


# Standalone test:
#   python -m orchestrator.agents.coverage_critic "critique tests for X"
if __name__ == "__main__":
    from orchestrator.config import OrchestratorConfig

    request = " ".join(sys.argv[1:]) or "critique the tests for the current task"

    async def _main() -> None:
        result = await critique_tests(request, OrchestratorConfig().default_model)
        print(result.model_dump_json(indent=2))

    asyncio.run(_main())
