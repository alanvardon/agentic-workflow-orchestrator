"""Phase 77a — evidence foundations (generic plumbing).

Two additive, behaviour-preserving seams that the later TDD-evidence slices
(77b/77c) consume:

- `StepResult.full_output`: `_run_script_sync` keeps the COMPLETE runner log
  (every line, both streams), not just the `out[-500:]` tail it puts in
  `detail`. Without this the per-attempt / RED-run evidence is hollow.
- `run_retry_block(on_attempt=...)`: a per-attempt hook that fires once after
  the gates resolve EVERY attempt — including the GREEN (passing) one, which the
  existing hooks never see (`on_gate_failed` only fires on failure and the loop
  returns immediately on a pass).

Both are inert until a consumer is wired (77c): with the defaults, behaviour is
identical to before. LLM-free, like the phase 33 / 42 tests they extend.
"""

import pytest

from orchestrator.manifest import ScriptStep, StepResult
from orchestrator.retry_block import RetryBlock, run_retry_block
from orchestrator.steps import execute_script


# --------------------------- seam 1: full-output capture ---------------------------


@pytest.mark.asyncio
async def test_full_output_keeps_complete_log_not_just_tail(tmp_path):
    # A script whose stdout is comfortably larger than the 500-char `detail`
    # tail, so the head is provably truncated out of `detail` but survives in
    # `full_output`.
    script = tmp_path / "noisy.sh"
    script.write_text(
        '#!/bin/sh\nfor i in $(seq 1 200); do echo "line $i padding-padding-padding"; done\nexit 0\n'
    )
    script.chmod(0o755)
    result = await execute_script(ScriptStep(id="noisy", path="noisy.sh"), tmp_path)

    assert result.ok
    # detail is still the short tail: the last lines, not the first.
    assert len(result.detail) <= 500
    assert "line 1 " not in result.detail
    assert "line 200" in result.detail
    # full_output is the COMPLETE log: first line through last, untruncated.
    assert "line 1 " in result.full_output
    assert "line 200" in result.full_output
    assert len(result.full_output) > 500


@pytest.mark.asyncio
async def test_full_output_includes_both_streams_on_gate_fail(tmp_path):
    # A failing gate's `detail` is the failure report (stderr-first); `full_output`
    # must carry BOTH streams complete so the evidence shows the whole run.
    script = tmp_path / "fail.sh"
    script.write_text(
        '#!/bin/sh\necho "stdout-marker line"\necho "stderr-boom line" >&2\nexit 1\n'
    )
    script.chmod(0o755)
    result = await execute_script(
        ScriptStep(id="fail", path="fail.sh"), tmp_path, as_gate=True
    )

    assert result.ok and result.passed is False  # gate FAIL, not an abort
    assert "stdout-marker line" in result.full_output
    assert "stderr-boom line" in result.full_output


@pytest.mark.asyncio
async def test_full_output_present_on_success(tmp_path):
    script = tmp_path / "ok.sh"
    script.write_text('#!/bin/sh\necho hello-world\nexit 0\n')
    script.chmod(0o755)
    result = await execute_script(ScriptStep(id="ok", path="ok.sh"), tmp_path)
    assert "hello-world" in result.full_output


# --------------------------- seam 2: per-attempt on_attempt hook ---------------------------


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


@pytest.mark.asyncio
async def test_on_attempt_fires_on_green_first_try():
    seen = []

    async def on_attempt(attempt, passed, gate_results):
        seen.append((attempt, passed, [g.step_id for g in gate_results]))

    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    result = await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(True, "")], []),
        on_attempt=on_attempt,
    )
    assert result.ok and result.attempts == 1
    # The GREEN attempt is visible — the whole point of the hook.
    assert seen == [(1, True, ["qa"])]


@pytest.mark.asyncio
async def test_on_attempt_fires_every_attempt_including_the_green_one():
    seen = []

    async def on_attempt(attempt, passed, gate_results):
        seen.append((attempt, passed))

    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    result = await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "f1"), (False, "f2"), (True, "")], []),
        on_attempt=on_attempt,
    )
    assert result.ok and result.attempts == 3
    assert seen == [(1, False), (2, False), (3, True)]


@pytest.mark.asyncio
async def test_on_attempt_receives_failing_gate_result():
    seen = []

    async def on_attempt(attempt, passed, gate_results):
        seen.append((passed, gate_results[-1].detail))

    block = RetryBlock(
        producers=["impl"], gates=["qa"], max_retries=1, on_exhausted="abort"
    )
    await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "the failure log")], []),
        on_attempt=on_attempt,
    )
    # The hook gets the StepResult, so a consumer can persist its full_output/detail.
    assert seen == [(False, "the failure log")]


@pytest.mark.asyncio
async def test_on_attempt_fires_before_on_gate_failed():
    order = []

    async def on_attempt(attempt, passed, gate_results):
        order.append(("attempt", attempt))

    async def on_gate_failed(attempt, feedback):
        order.append(("gate_failed", attempt))
        return True

    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "f1"), (True, "")], []),
        on_attempt=on_attempt,
        on_gate_failed=on_gate_failed,
    )
    # Evidence is recorded for the attempt BEFORE the human gate-fail decision.
    assert order == [
        ("attempt", 1),
        ("gate_failed", 1),
        ("attempt", 2),
    ]


@pytest.mark.asyncio
async def test_on_attempt_absent_is_a_noop():
    # Default (no hook) preserves the pre-77a behaviour exactly.
    block = RetryBlock(producers=["impl"], gates=["qa"], max_retries=3)
    result = await run_retry_block(
        block=block,
        run_producer=_producer([]),
        run_gate=_gate([(False, "f1"), (True, "")], []),
    )
    assert result.ok and result.proceed and result.attempts == 2
