# load_dotenv reads the .env file in the current working directory and sets
# environment variables. AsyncAnthropic() (inside run_structured_completion) with
# no args picks up ANTHROPIC_API_KEY from os.environ, so this is how the key
# reaches the SDK. It must run before the first client is constructed.
from dotenv import load_dotenv
load_dotenv()

import asyncio
import sys

# Pydantic is a data-validation library. BaseModel gives us automatic
# type coercion and a clean __repr__ — no need to write __init__ or validate
# fields manually. The model acts as a typed contract between the planning
# agent and anything that consumes its output.
from pydantic import BaseModel, Field

from orchestrator.usage import TaskUsage
from orchestrator.agents.runner import run_structured_completion
from orchestrator.prompt_loader import load_prompt


_PLANNING_SYSTEM_PROMPT = load_prompt("planning")


# PlanResult is the structured output the planning agent returns.
# Wrapping the agent's response in a model means callers never have to
# inspect raw strings or dicts — they work with validated, typed attributes.
# Schema used as the emit_plan tool's input_schema. Excludes `usage` so
# the model is never asked to fill in token-count data.
class _PlanSchema(BaseModel):
    title: str
    type: str = Field(
        description="short kebab-case category like 'feature', 'fix', "
                    "'migration', 'config' — used as the branch prefix"
    )
    plan_text: str


class PlanResult(_PlanSchema):
    # Schema version for this result model. Bump on incompatible shape changes
    # (renamed/removed fields); pure additions of optional fields don't need a
    # bump. Defined here, NOT on _PlanSchema, so it never leaks into the emit_plan
    # tool's input_schema.
    schema_version: int = 1
    # Populated after the API call returns; not part of the LLM tool schema.
    usage: TaskUsage | None = None


async def plan(request: str, model: str) -> PlanResult:
    """Ask Claude to produce a plan, return it as a validated PlanResult.

    Uses Anthropic's tool-use-as-structured-output pattern:
      1. We declare a fake tool ("emit_plan") whose input schema matches
         PlanResult exactly.
      2. We force tool_choice to that tool, so the model MUST respond by
         "calling" it with arguments matching the schema.
      3. The tool's input is the validated structured output. No string
         parsing, no sentinel matching, no chance of malformed responses
         surviving past this function.

    This is the single biggest robustness win over the old coordinator,
    which relied on the model emitting `PLAN COMPLETE: title=X, type=Y` as
    free text and hoping the regex matched.

    The forced-tool-use plumbing + usage extraction lives in the shared
    run_structured_completion (sibling of run_structured_agent). _PlanSchema is the
    emit tool's input_schema (no usage); PlanResult is the validated result.
    """
    return await run_structured_completion(
        system_prompt=_PLANNING_SYSTEM_PROMPT,
        user_message=request,
        model=model,
        tool_name="emit_plan",
        tool_description="Emit the structured implementation plan for the requested change.",
        schema=_PlanSchema,
        result_model=PlanResult,
    )


# Allow `python -m orchestrator.agents.planning "add dark mode"` to run the
# function from the terminal. asyncio.run drives the async function from
# synchronous entry-point code.
if __name__ == "__main__":
    from orchestrator.config import OrchestratorConfig

    user_request = " ".join(sys.argv[1:]) or "add a dark mode toggle"
    result = asyncio.run(plan(user_request, OrchestratorConfig().default_model))
    print(result.model_dump_json(indent=2))
