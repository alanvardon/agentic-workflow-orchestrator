"""QA agent — runs Claude Agent SDK to review uncommitted changes.

Like implementation.py this is an *agent loop*, not a single LLM call:
QA needs to read CLAUDE.md, read the plan, run `git diff HEAD`, inspect
specific files, and possibly run static checks before producing its
verdict. Multiple tool calls across multiple turns — that's the agent
loop shape.

What differs from implementation.py:
  - **Read-only tools.** No Edit, no Write. The QA agent must not be
    able to modify the working tree, even by accident. The allowlist is
    the hard gate; the system prompt's "do not fix anything" is the soft
    one.
  - **Structured output is a verdict, not a description.**
    `QaResult { result: Literal["PASS", "FAIL"], failures: str | None }`
    replaces the old `QA RESULT: PASS / QA RESULT: FAIL` sentinel and
    the separate `qa_failures.md` file. Both pieces of information now
    travel together as one typed object.

The agent loop itself (in-process emit tool, the `query()` loop, the
fail-closed guard, and usage extraction) lives in `run_structured_agent`.
This module supplies only the QA-specific prompt, tools, emit-tool schema,
and the QaResult factory — plus the scripted gate, which runs before any
LLM call.
"""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import sys

from pydantic import BaseModel
from typing import Literal

from orchestrator.usage import TaskUsage
from orchestrator.prompt_loader import load_prompt

from orchestrator.agents.planning import PlanResult
from orchestrator.agents.runner import run_structured_agent
from orchestrator.config import load_config
from orchestrator.git_ops import REPO_ROOT
from orchestrator.qa_scripts import run_qa_scripts


_QA_SYSTEM_PROMPT = load_prompt("qa")


class QaResult(BaseModel):
    # Bump on incompatible shape changes (renamed/removed fields); pure additions
    # of optional fields don't need a bump.
    schema_version: int = 1
    result: Literal["PASS", "FAIL"]
    failures: str | None = None
    # The QA agent's own account of what it reviewed and ran — the static checks
    # and checklist items it worked through, each with its ✓ PASS / ✗ FAIL (Phase
    # 77d). Drives the richer, QA-agent-only `qa.md`. Optional + defaulted, so it is
    # a pure-additive checkpoint change (no schema_version bump) and the scripted-
    # gate FAIL path, which returns before the agent runs, simply leaves it None.
    review: str | None = None
    usage: TaskUsage | None = None


def _coerce_verdict(raw: object) -> Literal["PASS", "FAIL"]:
    """Map the agent's emitted `result` onto the PASS/FAIL literal, fail-closed.

    The emit tool asks for the exact string 'PASS', but an LLM can drift
    ('pass', 'Passed', 'FAIL.'). Constructing QaResult straight from the raw
    string would raise a pydantic ValidationError inside qa_task and escape as
    a bare crash. Instead: only an exact (case-insensitive) 'PASS' passes;
    anything else is a FAIL whose feedback feeds the retry loop. Mirrors the
    gate path's `steps._coerce_passed` defensive posture."""
    return "PASS" if str(raw).strip().upper() == "PASS" else "FAIL"


def _build_user_message(plan: PlanResult) -> str:
    """Compose the per-run user message for the QA agent.

    QA only needs the plan — the diff comes from `git diff HEAD` which
    the agent runs itself via Bash. No mode switch (unlike
    implementation): QA always does the same thing.
    """
    return "\n".join(["## Plan", "", plan.plan_text])


async def qa(plan: PlanResult, model: str) -> QaResult:
    """Run the QA agent and return its structured verdict.

    Scripted gate runs first (before any LLM call). If any executable
    script in `qa_scripts_dir` exits non-zero, a FAIL is returned
    immediately with the script output embedded in `failures`.

    Read-only: the LLM agent has Read, Bash, Glob, Grep — explicitly no
    Edit or Write. The orchestrator (not the agent) decides what
    happens after a FAIL.
    """
    # --- Scripted QA gate -----------------------------------------------
    # Run before the LLM. Any non-zero exit from a script short-circuits
    # the whole QA phase: no prompt is built, no model is called.
    _config = load_config()
    _scripted_outcome = run_qa_scripts(
        repo_root=REPO_ROOT,
        qa_scripts_dir=_config.qa.scripts_dir,
        timeout=_config.qa.scripts_timeout,
    )
    if not _scripted_outcome.passed:
        return QaResult(
            result="FAIL",
            failures=_scripted_outcome.failure_report,
        )
    # --- End scripted gate ----------------------------------------------

    # Scripts passed → run the read-only QA agent. The agent loop, the
    # in-process emit tool, the fail-closed guard, and usage extraction all
    # live in run_structured_agent. The pinned MCP tool (emit_qa_result) is
    # appended to allowed_tools by the runner.
    #
    # The QA agent's tools/timeout come from the v2 [builtin.qa] part (the per-task
    # gate); a pipeline with no [builtin.qa] table falls back to the whole-diff
    # [stage.builtin.qa] stage, then to the read-only role default. (`tools` is the
    # PartSpec alias for allowed_tools.) The separate scripted gate above reads the
    # independent [qa] table (config.qa) — unchanged.
    _qa = _config.part("builtin:qa") or _config.stage("qa")
    _qa_allowed = (
        (getattr(_qa, "allowed_tools", None) or getattr(_qa, "tools", None))
        if _qa else None
    ) or ["Read", "Grep", "Bash"]
    _qa_disallowed = (getattr(_qa, "disallowed_tools", None) if _qa else None) or []
    _qa_timeout = getattr(_qa, "timeout", None) if _qa else None
    return await run_structured_agent(
        system_prompt=_QA_SYSTEM_PROMPT,
        user_message=_build_user_message(plan),
        model=model,
        # Read-only tools. The project's .claude/settings.json deny rules (via
        # setting_sources=["project"], set in the runner) still apply.
        allowed_tools=_qa_allowed,
        disallowed_tools=_qa_disallowed,
        # Same repo root as implementation — QA reviews changes in the
        # target repo's tree, not the orchestrator/ subdirectory.
        cwd=REPO_ROOT,
        timeout=_qa_timeout,
        emit_tool_name="emit_qa_result",
        emit_tool_description=(
            "Emit the final QA verdict. Call this exactly once when review is "
            "complete. `result` must be the exact string 'PASS' or 'FAIL'. "
            "`review` is your own account of what you reviewed and ran — each "
            "static check and checklist item with its ✓ PASS / ✗ FAIL; report only "
            "checks you ran yourself, never a re-summary of test suites you did not "
            "run. `failures` is an empty string on PASS, or a markdown failure "
            "report on FAIL. After calling, stop — the orchestrator takes over."
        ),
        # Schema uses plain `str` for `result` because the SDK's @tool
        # decorator takes simple Python types; the factory below maps that raw
        # string onto the Literal["PASS","FAIL"] fail-closed via _coerce_verdict
        # (a non-canonical string becomes FAIL, never a ValidationError crash).
        emit_tool_fields={"result": str, "review": str, "failures": str},
        result_factory=lambda captured, usage: QaResult(
            result=_coerce_verdict(captured.get("result")),
            review=(captured.get("review") or None),
            failures=(captured.get("failures") or None),
            usage=usage,
        ),
    )


# Standalone test:
#   python -m orchestrator.agents.qa "tiny test"
# Builds a fake plan, runs QA against whatever uncommitted changes are
# in the target repo right now, prints the verdict. Useful for
# iterating on the QA prompt without going through the whole workflow.
if __name__ == "__main__":
    from orchestrator.config import OrchestratorConfig

    request = " ".join(sys.argv[1:]) or "review whatever's currently uncommitted"

    async def _main() -> None:
        fake_plan = PlanResult(
            title="standalone qa test",
            type="feature",
            plan_text=request,
        )
        result = await qa(fake_plan, OrchestratorConfig().default_model)
        print(result.model_dump_json(indent=2))

    asyncio.run(_main())
