"""Generic gate-retry engine.

A *retry block* is a guarded loop: one or more PRODUCER steps that mutate the
working tree, plus one or more GATE steps that judge it and return pass/fail +
feedback. The engine re-runs the producers — with the failing gate's feedback
injected — until a gate passes or the retry budget is exhausted.

This module is the engine ONLY. It knows nothing about implementation, QA, or
the manifest: producers and gates are supplied as injected async callables, so
the same driver serves both the built-in spine and user-declared blocks in
orchestrator.toml.

`run_retry_block` is a plain async function (not a @task) and is meant to run in
the entrypoint body, because its optional approval gates call interrupt(), which
must run there — the same constraint as steps.run_seam.

Invariant: **gates fail closed.** A gate must report
`passed` True or False; a gate slot yielding `passed=None` is a configuration
error and raises (it never defaults to a pass).
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from orchestrator.errors import FatalError
from orchestrator.manifest import StepResult


# Resume/abort vocabulary, shared with run_seam's approval gates.
_ABORT_WORDS = frozenset({"abort", "no", "stop"})

# At the on_exhausted="approval_gate" prompt a human may grant MORE attempts by
# replying with a positive integer. Accepted forms: a bare int
# ("2"), "+N" ("+2"), "retry N", or "more N". An optional "attempt(s)" suffix is
# tolerated. Parsed strictly so a plan/feedback reply that isn't clearly a count
# falls through to "proceed".
_EXTEND_RE = re.compile(r"(?:retry|more|\+)?\s*(\d+)(?:\s*attempts?)?")

# Feedback injection (replaces the old mode="fix"). On attempt 1 a producer gets
# no feedback; on a retry the failing gate's detail is rendered under this
# heading for the producer to read and act on.
_FEEDBACK_HEADING = "## Previous attempt feedback"


class RetryConfigError(FatalError):
    """A retry block is mis-wired — e.g. a gate produced no verdict, or
    on_exhausted='approval_gate' was requested without an interrupt function."""


class RetryBlock(BaseModel):
    """Runtime spec for one retry block. Part B wires the built-in impl→QA loop
    onto this; Part C maps a declarative orchestrator.toml block onto it."""

    producers: list[str]
    gates: list[str]
    max_retries: int = Field(default=3, ge=1)
    on_exhausted: Literal["abort", "approval_gate", "proceed"] = "abort"
    # Optional hard ceiling on the TOTAL number of attempts a run may reach via
    # extensions. None = unbounded (a human must keep typing numbers).
    # A grant that would push past it is clamped; a grant at the cap can't extend.
    max_total_attempts: int | None = Field(default=None, ge=1)


class RetryBlockResult(BaseModel):
    ok: bool                          # a gate passed (clean success)
    proceed: bool                     # should the caller continue past the block?
    attempts: int                     # how many producer attempts ran
    last_feedback: str | None = None  # failing gate's detail on abort/exhaust


def feedback_section(detail: str) -> str:
    """The standard feedback block a producer receives on a retry. Consumers
    (Part B/C producers) append this to their user message; the engine passes
    the raw gate detail and lets the producer format it via this helper."""
    return f"{_FEEDBACK_HEADING}\n\n{detail}"


def _parse_extend(reply: object) -> int | None:
    """Parse an exhaustion reply as a request for N more attempts.

    Returns a positive int N if the reply is *clearly* a count ("2", "+2",
    "retry 2", "more 2"), else None (let the caller treat it as abort/proceed).
    Abort words are matched by the caller first, so they never reach here.
    """
    if not isinstance(reply, str):
        return None
    m = _EXTEND_RE.fullmatch(reply.strip().lower())
    if not m:
        return None
    n = int(m.group(1))
    return n if n > 0 else None


async def run_retry_block(
    *,
    block: RetryBlock,
    run_producer: Callable[[str, str | None], Awaitable[StepResult]],
    run_gate: Callable[[str], Awaitable[StepResult]],
    check_cancel: Callable[[], None] = lambda: None,
    on_producers_done: Callable[[int], Awaitable[None]] | None = None,
    on_attempt: Callable[[int, bool, list[StepResult]], Awaitable[None]] | None = None,
    on_gate_failed: Callable[[int, str], Awaitable[bool]] | None = None,
    interrupt_fn: Callable[[dict], Any] | None = None,
    autonomous: bool = False,
) -> RetryBlockResult:
    """Run a retry block to a verdict.

    `run_producer(step_id, feedback)` runs one producer; `feedback` is None on
    the first attempt and the failing gate's detail on retries. `run_gate(step_id)`
    runs one gate and returns a StepResult whose `passed` is True/False.

    Optional per-attempt approval gates (injected, so their interrupt() lives in the
    entrypoint):
      - on_producers_done(attempt): a pause after producers, before gates.
      - on_attempt(attempt, passed, gate_results): fires ONCE per attempt after the
        gates resolve, for EVERY attempt — including the passing (GREEN) one, which
        on_gate_failed never sees. `gate_results` are the gate StepResults evaluated
        this attempt (in order; ending at the first failing gate on a fail, or all
        gates on a pass), so a consumer can persist their full_output/verdict. Fires
        before on_gate_failed, so the attempt's evidence is recorded before any
        human decision. A read-only observer: its return value is ignored.
      - on_gate_failed(attempt, feedback) -> keep_going: a decision after a gate
        fails; return False to stop the block immediately.

    `interrupt_fn` is used only for on_exhausted="approval_gate".

    Under on_exhausted="approval_gate" a human may reply with a count (e.g. "2")
    to GRANT more attempts and keep looping, instead of only choosing abort vs
    proceed-as-is. The budget grows dynamically; `attempt` keeps counting across
    extensions. To avoid a double prompt on the last budgeted attempt (the
    on_gate_failed pause AND then this exhaustion gate), the per-attempt
    on_gate_failed pause is suppressed on the final budgeted attempt when
    on_exhausted="approval_gate" — the richer exhaustion prompt owns that moment.

    Returns a RetryBlockResult; `proceed` tells the caller whether to continue
    past the block (e.g. commit) or treat it as a failure.
    """
    feedback: str | None = None
    # In fully-autonomous mode the loop never exhausts — it re-produces with the
    # failing gate's feedback until a gate passes (or check_cancel stops
    # it, e.g. a safety ceiling). An unbounded budget makes the inner loop's
    # `attempt < budget` always true, so the exhaustion branch / on_exhausted are
    # never reached.
    budget = float("inf") if autonomous else block.max_retries
    attempt = 0

    while True:
        while attempt < budget:
            attempt += 1
            for pid in block.producers:
                check_cancel()
                await run_producer(pid, feedback)

            if on_producers_done is not None:
                await on_producers_done(attempt)

            gate_results: list[StepResult] = []
            attempt_passed = True
            for gid in block.gates:  # ordered; the first gate to fail wins
                check_cancel()
                gate_result = await run_gate(gid)
                gate_results.append(gate_result)
                if gate_result.passed is None:
                    raise RetryConfigError(
                        f"gate step {gid!r} returned no verdict (passed is None); "
                        "a gate must report pass/fail"
                    )
                if gate_result.passed is False:
                    feedback = gate_result.detail
                    attempt_passed = False
                    break

            # Per-attempt evidence hook, every attempt (pass included), before the
            # human gate-fail decision below.
            if on_attempt is not None:
                await on_attempt(attempt, attempt_passed, gate_results)

            if attempt_passed:
                # No gate failed → every gate passed.
                return RetryBlockResult(ok=True, proceed=True, attempts=attempt)

            # A gate failed. Optional human decision before spending another
            # attempt — but suppress it on the final budgeted attempt when the
            # exhaustion approval gate will own the decision (no double prompt).
            exhaustion_owns = (
                attempt >= budget and block.on_exhausted == "approval_gate"
            )
            if on_gate_failed is not None and not exhaustion_owns:
                keep_going = await on_gate_failed(attempt, feedback or "")
                if not keep_going:
                    return RetryBlockResult(
                        ok=False, proceed=False, attempts=attempt,
                        last_feedback=feedback,
                    )

        # Budget exhausted without a pass. on_exhausted decides; under
        # approval_gate a human may grant more attempts → grow the budget and
        # re-enter the loop from the current `attempt`.
        outcome = _handle_exhausted(block, feedback, interrupt_fn, attempt)
        if outcome[0] == "abort":
            return RetryBlockResult(
                ok=False, proceed=False, attempts=attempt, last_feedback=feedback
            )
        if outcome[0] == "proceed":
            return RetryBlockResult(
                ok=False, proceed=True, attempts=attempt, last_feedback=feedback
            )
        # ("extend", n): grant n more attempts and loop again.
        budget += outcome[1]


# Tagged outcomes from _handle_exhausted, consumed by run_retry_block's loop.
ExhaustedOutcome = tuple[str, int] | tuple[str]


def _handle_exhausted(
    block: RetryBlock,
    feedback: str | None,
    interrupt_fn: Callable[[dict], Any] | None,
    attempts: int,
) -> ExhaustedOutcome:
    """Decide what happens when the budget runs out. Returns a tagged outcome:
    ("abort",), ("proceed",), or ("extend", n) — the last only under
    on_exhausted="approval_gate" when a human grants n more attempts."""
    if block.on_exhausted == "proceed":
        return ("proceed",)

    if block.on_exhausted == "approval_gate":
        if interrupt_fn is None:
            raise RetryConfigError(
                "on_exhausted='approval_gate' requires an interrupt function"
            )
        # Headroom under the optional hard cap; None = unbounded.
        cap = block.max_total_attempts
        remaining = None if cap is None else max(0, cap - attempts)

        decision = interrupt_fn({
            "kind": "retry_exhausted",
            "attempts": attempts,
            "feedback": feedback,
            "remaining": remaining,
            "ask": _exhausted_ask(attempts, remaining),
        })

        if isinstance(decision, str) and decision.strip().lower() in _ABORT_WORDS:
            return ("abort",)

        grant = _parse_extend(decision)
        if grant is not None and remaining != 0:
            if remaining is not None:
                grant = min(grant, remaining)  # clamp to the hard cap
            return ("extend", grant)

        # Anything else (yes/proceed/empty, or a count with no headroom left).
        return ("proceed",)

    # default: "abort" — the caller treats proceed=False as a failed run.
    return ("abort",)


def _exhausted_ask(attempts: int, remaining: int | None) -> str:
    """The exhaustion prompt text — advertises the number option when there is
    headroom for more attempts."""
    if remaining is None:
        extend_clause = "reply a number N to grant N more attempts, "
    elif remaining > 0:
        extend_clause = (
            f"reply a number N (up to {remaining} more) to grant more attempts, "
        )
    else:
        extend_clause = ""  # at the max_total_attempts cap — no more extends
    return (
        f"Retry budget ({attempts}) exhausted. "
        "Reply 'abort' to stop, "
        f"{extend_clause}"
        "anything else to proceed as-is."
    )
