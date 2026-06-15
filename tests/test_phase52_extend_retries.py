"""Phase 52 — grant more retries from the exhaustion approval gate.

When a build's retry budget runs out without the gate passing, a human at the
on_exhausted="approval_gate" prompt may now reply with a COUNT to grant N more
attempts and keep looping, instead of only choosing abort vs proceed-as-is.

Two slices:
1. Engine unit tests (fake producer/gate + a scripted interrupt_fn) — the
   dynamic-budget loop, the extend reply grammar, the max_total_attempts clamp,
   the ask text, and the suppressed double-prompt on the final budgeted attempt.
2. Integration — a declared build at a seam driven through real interrupts,
   including the headline resume-across-extend case (prior attempts replay from
   the checkpoint; only the newly-granted attempts run).
"""

import uuid

import pytest
from langgraph.types import Command

from orchestrator.manifest import RetryConfig, StepResult
from orchestrator.retry_block import (
    RetryBlock,
    _parse_extend,
    run_retry_block,
)

from tests.conftest import task_build_config


# --------------------------- engine-level fakes ---------------------------


def _producer(calls):
    async def run_producer(step_id, feedback):
        calls.append((step_id, feedback))
        return StepResult(step_id=step_id, kind="ai_agent", ok=True)
    return run_producer


def _gate(verdicts, calls):
    it = iter(verdicts)
    async def run_gate(step_id):
        calls.append(step_id)
        passed, detail = next(it)
        return StepResult(
            step_id=step_id, kind="script", ok=True, passed=passed, detail=detail
        )
    return run_gate


def _scripted_interrupt(replies):
    """An interrupt_fn that records each payload and returns the next scripted
    reply. Mimics LangGraph re-firing the same call site across extensions."""
    it = iter(replies)
    asks: list[dict] = []
    def interrupt_fn(payload):
        asks.append(payload)
        return next(it)
    interrupt_fn.asks = asks
    return interrupt_fn


def _fail(n):
    return [(False, f"f{i}") for i in range(1, n + 1)]


# --------------------------- _parse_extend grammar ---------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("2", 2),
        ("+2", 2),
        ("retry 2", 2),
        ("retry2", 2),
        ("more 2", 2),
        ("  3 ", 3),
        ("2 attempts", 2),
        ("1 attempt", 1),
        ("0", None),       # not positive
        ("-1", None),      # not a bare positive int
        ("yes", None),
        ("abort", None),
        ("proceed", None),
        ("", None),
        ("2.5", None),     # not an integer
        ("v2", None),
        (2, None),         # non-string
        (None, None),
    ],
)
def test_parse_extend_grammar(reply, expected):
    assert _parse_extend(reply) == expected


# --------------------------- dynamic budget ---------------------------


@pytest.mark.asyncio
async def test_extend_then_pass_counts_total_attempts():
    pcalls, gcalls = [], []
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    # attempt 1 fails → exhaust → grant 2 → attempts 2,3; gate passes on attempt 3.
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate([(False, "f1"), (False, "f2"), (True, "")], gcalls),
        interrupt_fn=_scripted_interrupt(["2"]),
    )
    assert result.ok and result.proceed
    assert result.attempts == 3                  # counts across the extension
    assert pcalls == [("impl", None), ("impl", "f1"), ("impl", "f2")]


@pytest.mark.asyncio
async def test_extend_exhaust_again_then_abort():
    pcalls, gcalls = [], []
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    intr = _scripted_interrupt(["1", "abort"])
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate(_fail(2), gcalls),
        interrupt_fn=intr,
    )
    assert not result.ok and not result.proceed  # aborted → failed run
    assert result.attempts == 2
    assert len(intr.asks) == 2                    # the gate re-fired after extend


@pytest.mark.asyncio
async def test_extend_then_proceed_ships_as_is():
    pcalls, gcalls = [], []
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate(_fail(2), gcalls),
        interrupt_fn=_scripted_interrupt(["1", "yes"]),
    )
    assert not result.ok and result.proceed       # proceeds despite never passing
    assert result.attempts == 2


@pytest.mark.parametrize("form", ["2", "+2", "retry 2", "more 2"])
@pytest.mark.asyncio
async def test_extend_reply_forms_all_grant(form):
    pcalls, gcalls = [], []
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate([(False, "f1"), (False, "f2"), (True, "")], gcalls),
        interrupt_fn=_scripted_interrupt([form]),
    )
    assert result.ok and result.attempts == 3


# --------------------------- regression: today's replies ---------------------------


@pytest.mark.asyncio
async def test_non_numeric_reply_proceeds_as_today():
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    result = await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "f1")], []),
        interrupt_fn=_scripted_interrupt(["looks fine, go"]),
    )
    assert not result.ok and result.proceed       # anything non-numeric → proceed
    assert result.attempts == 1


# --------------------------- max_total_attempts cap ---------------------------


@pytest.mark.asyncio
async def test_max_total_attempts_clamps_and_blocks_at_cap():
    pcalls, gcalls = [], []
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=2,
        on_exhausted="approval_gate", max_total_attempts=3,
    )
    # attempts 1,2 exhaust (remaining 1) → grant "5" clamps to 1 → attempt 3
    # exhausts at the cap (remaining 0) → "2" can't extend → proceed.
    intr = _scripted_interrupt(["5", "2"])
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate(_fail(3), gcalls),
        interrupt_fn=intr,
    )
    assert not result.ok and result.proceed
    assert result.attempts == 3                    # clamped to the cap, no further
    # The first ask advertises the headroom; the second (at the cap) does not.
    assert intr.asks[0]["remaining"] == 1
    assert "up to 1 more" in intr.asks[0]["ask"]
    assert intr.asks[1]["remaining"] == 0
    assert "number" not in intr.asks[1]["ask"]


@pytest.mark.asyncio
async def test_unbounded_ask_advertises_the_number_option():
    intr = _scripted_interrupt(["abort"])
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "f1")], []),
        interrupt_fn=intr,
    )
    assert intr.asks[0]["remaining"] is None
    assert "reply a number N" in intr.asks[0]["ask"]


# --------------------------- on_gate_failed double-prompt suppression ---------------------------


@pytest.mark.asyncio
async def test_gate_fail_pause_suppressed_on_final_budget_attempt_under_approval_gate():
    """Phase 51's per-attempt on_gate_failed pause must NOT fire on the final
    budgeted attempt when on_exhausted='approval_gate' — the richer exhaustion
    prompt owns that moment (no double prompt). It still fires on earlier
    attempts, and again after an extension grows the budget."""
    seen = []

    async def on_gate_failed(attempt, feedback):
        seen.append(attempt)
        return True

    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=2, on_exhausted="approval_gate"
    )
    # attempts 1,2 fail (budget 2). Pause fires on attempt 1 only (2 is the final
    # budgeted attempt → suppressed). Grant 1 → attempt 3 is the new final → also
    # suppressed. Then abort.
    await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate(_fail(3), []),
        on_gate_failed=on_gate_failed,
        interrupt_fn=_scripted_interrupt(["1", "abort"]),
    )
    assert seen == [1]


@pytest.mark.asyncio
async def test_gate_fail_pause_still_fires_on_final_attempt_under_abort():
    """Under on_exhausted='abort' there is no exhaustion prompt to own the
    moment, so the per-attempt pause keeps firing on every failing attempt
    (today's behaviour, unchanged)."""
    seen = []

    async def on_gate_failed(attempt, feedback):
        seen.append(attempt)
        return True

    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=2, on_exhausted="abort"
    )
    await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate(_fail(2), []),
        on_gate_failed=on_gate_failed,
    )
    assert seen == [1, 2]


# --------------------------- config plumbing ---------------------------


def test_retry_config_accepts_max_total_attempts():
    cfg = RetryConfig(max=3, on_exhausted="approval_gate", max_total_attempts=9)
    assert cfg.max_total_attempts == 9


def test_retry_config_max_total_attempts_defaults_unbounded():
    assert RetryConfig().max_total_attempts is None


# =========================== integration (real interrupts) ===========================
#
# Phase 68b: the extend-from-exhaustion behaviour is driven through the per-task
# `task-build` station (on_exhausted="approval_gate"), not a v1 [[steps.work]]
# build. The engine-level tests above cover the extend grammar / budget / clamp
# exhaustively; these two integration tests prove the WORKFLOW wires the engine's
# exhaustion interrupt through and resumes across an extension. (The proceed-as-is
# and max_total_attempts-clamp variants are covered at the engine level above:
# test_extend_then_proceed_ships_as_is / test_max_total_attempts_clamps_*.)


async def _drive_task_build(monkeypatch, tmp_path, *, qa_verdicts, max_retries, replies):
    """Drive the per-task station through plan approval + each exhaustion interrupt.

    Each ainvoke off the same checkpoint db is a resume — so prior attempts replay
    rather than re-run. `replies` is fed one per pause (first is plan-approval)."""
    from tests.test_phase56_per_task_loop import _Stubs, _patch, _cfg
    from orchestrator.workflow import build_workflow

    stubs = _Stubs(n_tasks=1, qa_verdicts=qa_verdicts)
    _patch(stubs, monkeypatch)
    oc = task_build_config(on_exhausted="approval_gate", max_retries=max_retries)

    db = str(tmp_path / "ckpt.db")
    config = _cfg()
    inputs = ["req"] + [Command(resume=r) for r in replies]

    result, interrupts = None, []
    async with build_workflow(db_path=db, config=oc) as wf:
        for inp in inputs:
            result = await wf.ainvoke(inp, config=config)
            if "__interrupt__" in result:
                interrupts.append(result["__interrupt__"][0].value)
    return result, interrupts, stubs


@pytest.mark.asyncio
async def test_integration_extend_then_pass_resumes_across_extend(monkeypatch, tmp_path):
    from orchestrator.agents.qa import QaResult

    # budget 2 exhausts (QA fails attempts 1,2) → grant 2 → attempts 3,4 → pass on 4.
    result, interrupts, stubs = await _drive_task_build(
        monkeypatch, tmp_path,
        qa_verdicts=[QaResult(result="FAIL", failures=f"f{i}") for i in (1, 2, 3)]
        + [QaResult(result="PASS")],
        max_retries=2, replies=["yes", "2"],
    )
    assert result["status"] == "succeeded"
    # Plan approval, then the exhaustion prompt advertising the number option.
    assert interrupts[-1]["kind"] == "retry_exhausted"
    assert "number" in interrupts[-1]["ask"]
    # Prior attempts replayed (no re-spend): 4 producer + 4 QA calls total.
    assert len(stubs.impl_plans) == 4
    assert stubs.qa_calls == 4


@pytest.mark.asyncio
async def test_integration_extend_exhaust_then_abort(monkeypatch, tmp_path):
    from orchestrator.agents.qa import QaResult

    # budget 1 exhausts → grant 1 → exhausts again → abort → failed run.
    result, interrupts, stubs = await _drive_task_build(
        monkeypatch, tmp_path,
        qa_verdicts=[QaResult(result="FAIL", failures="lint failed") for _ in range(2)],
        max_retries=1, replies=["yes", "1", "abort"],
    )
    assert result["status"] == "failed"
    assert result["failed_task_id"] == "task:t1"
    assert result["qa_failures"] == "lint failed"
    assert "pr_url" not in result
    # Two exhaustion prompts (initial + after the extension), then abort.
    assert [i["kind"] for i in interrupts] == [
        "plan_approval", "retry_exhausted", "retry_exhausted",
    ]
    assert len(stubs.impl_plans) == 2
