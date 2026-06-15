"""Phase 42 Part A — generic gate-retry engine tests.

Pure unit tests with fake producer/gate callables — no LLM, no langgraph
runtime, no workflow. Verifies the loop semantics, feedback passing, the
per-attempt human-gate hooks, the on_exhausted policies, and the fail-closed
guard on a missing gate verdict.
"""

import pydantic
import pytest

from orchestrator.manifest import StepResult
from orchestrator.retry_block import (
    RetryBlock,
    RetryConfigError,
    feedback_section,
    run_retry_block,
)


def _producer(calls):
    """A fake producer that records (step_id, feedback) and succeeds."""
    async def run_producer(step_id, feedback):
        calls.append((step_id, feedback))
        return StepResult(step_id=step_id, kind="ai_agent", ok=True)
    return run_producer


def _gate(verdicts, calls):
    """A fake gate that records the gate id and returns the next (passed, detail)
    verdict from `verdicts` on each call."""
    it = iter(verdicts)
    async def run_gate(step_id):
        calls.append(step_id)
        passed, detail = next(it)
        return StepResult(
            step_id=step_id, kind="script", ok=True, passed=passed, detail=detail
        )
    return run_gate


# --------------------------- happy path ---------------------------


@pytest.mark.asyncio
async def test_gate_passes_first_try():
    pcalls, gcalls = [], []
    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    result = await run_retry_block(
        block=block, run_producer=_producer(pcalls), run_gate=_gate([(True, "")], gcalls)
    )
    assert result.ok and result.proceed
    assert result.attempts == 1
    assert pcalls == [("impl", None)]  # ran once, no feedback on attempt 1
    assert gcalls == ["qa"]


@pytest.mark.asyncio
async def test_retry_then_pass_injects_raw_feedback():
    pcalls, gcalls = [], []
    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate([(False, "f1"), (False, "f2"), (True, "")], gcalls),
    )
    assert result.ok and result.proceed and result.attempts == 3
    # Attempt 1: no feedback. Retries carry the PRIOR failing gate's detail.
    assert pcalls == [("impl", None), ("impl", "f1"), ("impl", "f2")]


@pytest.mark.asyncio
async def test_ordered_gates_short_circuit():
    pcalls, gcalls = [], []
    block = RetryBlock(
        producers=["impl"], gates=["g1", "g2"], max_retries=1, on_exhausted="abort"
    )
    result = await run_retry_block(
        block=block, run_producer=_producer(pcalls), run_gate=_gate([(False, "bad")], gcalls)
    )
    assert gcalls == ["g1"]  # g2 is never invoked after g1 fails
    assert not result.ok and not result.proceed


# --------------------------- per-attempt approval gates ---------------------------


@pytest.mark.asyncio
async def test_on_producers_done_fires_each_attempt():
    pcalls, gcalls, pd = [], [], []

    async def on_producers_done(attempt):
        pd.append(attempt)

    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate([(False, "f1"), (True, "")], gcalls),
        on_producers_done=on_producers_done,
    )
    assert pd == [1, 2]  # fired each attempt, after producers, before gates


@pytest.mark.asyncio
async def test_on_gate_failed_can_abort():
    pcalls, gcalls = [], []

    async def on_gate_failed(attempt, feedback):
        return False  # stop instead of retrying

    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate([(False, "f1")], gcalls),
        on_gate_failed=on_gate_failed,
    )
    assert not result.ok and not result.proceed
    assert result.attempts == 1 and result.last_feedback == "f1"
    assert pcalls == [("impl", None)]  # did not spend a second attempt


@pytest.mark.asyncio
async def test_on_gate_failed_keep_going_retries():
    pcalls, gcalls = [], []

    async def on_gate_failed(attempt, feedback):
        return True  # keep retrying

    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    result = await run_retry_block(
        block=block,
        run_producer=_producer(pcalls),
        run_gate=_gate([(False, "f1"), (True, "")], gcalls),
        on_gate_failed=on_gate_failed,
    )
    assert result.ok and result.attempts == 2


# --------------------------- on_exhausted policies ---------------------------


@pytest.mark.asyncio
async def test_exhausted_abort():
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=2, on_exhausted="abort"
    )
    result = await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "f1"), (False, "f2")], []),
    )
    assert not result.ok and not result.proceed
    assert result.attempts == 2 and result.last_feedback == "f2"


@pytest.mark.asyncio
async def test_exhausted_proceed():
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=2, on_exhausted="proceed"
    )
    result = await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "f1"), (False, "f2")], []),
    )
    assert not result.ok and result.proceed  # proceed despite never passing


@pytest.mark.asyncio
async def test_exhausted_approval_gate_abort_and_proceed():
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    aborted = await run_retry_block(
        block=block, run_producer=_producer([]), run_gate=_gate([(False, "f1")], []),
        interrupt_fn=lambda payload: "abort",
    )
    assert not aborted.proceed

    proceeded = await run_retry_block(
        block=block, run_producer=_producer([]), run_gate=_gate([(False, "f1")], []),
        interrupt_fn=lambda payload: "yes",
    )
    assert proceeded.proceed


@pytest.mark.asyncio
async def test_approval_gate_requires_interrupt_fn():
    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="approval_gate"
    )
    with pytest.raises(RetryConfigError, match="interrupt"):
        await run_retry_block(
            block=block, run_producer=_producer([]), run_gate=_gate([(False, "f1")], [])
        )


# --------------------------- fail-closed + validation ---------------------------


@pytest.mark.asyncio
async def test_missing_gate_verdict_raises():
    async def bad_gate(step_id):
        return StepResult(step_id=step_id, kind="script", ok=True)  # passed=None

    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=2)
    with pytest.raises(RetryConfigError, match="no verdict"):
        await run_retry_block(
            block=block, run_producer=_producer([]), run_gate=bad_gate
        )


@pytest.mark.asyncio
async def test_check_cancel_invoked_between_steps():
    cancels = []
    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=1)
    await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(True, "")], []),
        check_cancel=lambda: cancels.append(1),
    )
    assert len(cancels) >= 2  # before the producer and before the gate


def test_max_retries_must_be_positive():
    with pytest.raises(pydantic.ValidationError):
        RetryBlock(producers=["impl"], gates=["qa"], max_retries=0)


def test_feedback_section_format():
    s = feedback_section("the diff omitted error handling")
    assert s.startswith("## Previous attempt feedback")
    assert "the diff omitted error handling" in s
