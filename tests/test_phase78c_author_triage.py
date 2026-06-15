"""Phase 78c — testability triage as the test-author's first, cheap move.

With TDD on by default, most of a DOM/UI-heavy project's tasks are not unit-
testable through the existing harness. Before 78c the test-author ran its full
read→write→run loop only to conclude "untestable" and degrade to classic — the
author spend (~$0.24) was wasted on the majority of tasks.

Option C folds the "is this testable?" check INTO the existing author (not a
separate pre-check agent / model call — that was option B): the author's FIRST
move is a cheap triage that can emit `testable=false` and bail BEFORE writing
tests or running the suite. The lever is the bundled prompt, so the whole
`testable=false` → classic-fallback plumbing (Phase 73) is reused unchanged; no
new agent, no schema/graph change.

These tests lock the prompt contract: triage is step 1, the early bail is
explicit, and — per the phase landmine — the author is biased toward PROCEEDING
when unsure (an over-eager bail would silently leave a real behaviour unTDD'd).
"""

from orchestrator.prompt_loader import load_prompt


def _body() -> str:
    return load_prompt("test-author").lower()


def test_triage_is_the_first_move():
    # Step 1 is the testability triage, explicitly before the expensive work.
    body = _body()
    assert "triage testability first" in body
    # Triage is positioned ahead of authoring/running in the "When invoked" flow.
    assert body.index("triage testability first") < body.index("author the tests")


def test_early_bail_skips_writing_and_running():
    # The clear-untestable branch bails WITHOUT writing a test or running the
    # suite — that early exit is the whole cost saving of option C.
    body = _body()
    assert "bail now" in body
    assert "do not explore the codebase further, write a test, or run the suite" in body
    # It still routes through the existing testable=false escape, not a new path.
    assert "testable=false" in body


def test_biased_toward_proceeding_when_unsure():
    # Phase 78c landmine: a triage is a heuristic, not a gate — bias toward
    # proceeding so an obvious-only bail can't strand a genuinely-testable task.
    body = _body()
    assert "unsure" in body and "proceed" in body
    assert "obvious" in body  # only bail when untestability is *obvious*


def test_structured_output_contract_survives():
    # The triage rewrite must not drop the emit-tool footer the orchestrator
    # relies on to capture the verdict.
    body = _body()
    assert "emit_test_author_result" in body
    assert "testable" in body and "reason" in body
