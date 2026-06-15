"""Per-run artifact folder.

Writes each run's plan, implementation summary, test-plan, QA verdict,
and token usage to disk under:

    .orchestrator/runs/{thread_id}/          ← before branch exists
    .orchestrator/runs/{thread_id}-{slug}/   ← after create_branch

All writes are best-effort: OSError is swallowed so a disk problem
never takes down the workflow. The checkpointer is the source of truth;
these files are for human debugging only.

Usage in workflow.py:
    from orchestrator.run_artifacts import (
        write_plan, write_summary, write_qa, write_usage,
        rename_with_branch,
    )
"""

import json
import logging
import shutil
import traceback
from pathlib import Path

from orchestrator.agents.decompose import DecompositionResult, Task
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.agents.summarize import SummaryResult
from orchestrator.agents.test_author import TestAuthorResult
from orchestrator.paths import find_project_root, iter_test_files

logger = logging.getLogger(__name__)


def _runs_dir() -> Path:
    """Resolve the runs directory lazily.

    Why: tests monkeypatch.chdir() after import, so a module-level
    constant would freeze to the real repo and leak test artifacts there.
    """
    return find_project_root() / ".orchestrator" / "runs"


def _run_dir(thread_id: str) -> Path:
    """Resolve the current artifact folder for a given thread_id.

    Checks for an existing {thread_id}-{slug} folder first (post-branch),
    then falls back to the bare {thread_id} folder (pre-branch or
    branch creation failed). Creates the folder if it doesn't exist yet.
    """
    runs = _runs_dir()
    runs.mkdir(parents=True, exist_ok=True)
    matches = sorted(runs.glob(f"{thread_id}-*"))
    if matches:
        return matches[0]
    return runs / thread_id


def write_plan(thread_id: str, plan: PlanResult) -> None:
    """Write plan.md — overwritten on each re-plan."""
    try:
        d = _run_dir(thread_id)
        d.mkdir(parents=True, exist_ok=True)
        content = f"# {plan.title}\n\n**Type:** {plan.type}\n\n{plan.plan_text}\n"
        (d / "plan.md").write_text(content, encoding="utf-8")
    except OSError:
        pass


def write_decomposition(thread_id: str, decomposition: DecompositionResult) -> None:
    """Write decomposition.md — the task list, for human review. Overwritten on
    re-plan. The checkpointed DecompositionResult is what the per-task loop
    actually executes; this artifact is the readable mirror of it."""
    try:
        d = _run_dir(thread_id)
        d.mkdir(parents=True, exist_ok=True)
        lines = ["# Task decomposition", ""]
        lines.append(f"**Complexity:** {getattr(decomposition, 'complexity', 'moderate')}")
        lines.append("")
        for i, t in enumerate(decomposition.tasks, 1):
            lines.append(f"## {i}. {t.title}  (`{t.id}`)")
            lines.append("")
            lines.append(t.description)
            if t.acceptance_criteria:
                lines.append("")
                lines.append(f"**Acceptance:** {t.acceptance_criteria}")
            lines.append("")
        (d / "decomposition.md").write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass


def write_summary(thread_id: str, summary: SummaryResult, *, tdd: bool = False) -> None:
    """Write summary.md, plus test-plan.md UNLESS tdd is on (Phase 77d).

    These come from the summarizer (post-retry-block), not the implementation
    producer — which is generic and reports no summary.

    Under TDD the implementer's manual test-plan is both redundant and was
    misleading (the Phase 75 run claimed "8 tests" for 9 and listed untested
    speculation): the executed `test-author/` + `impl/` evidence is the real,
    stronger verification record, so test-plan.md is suppressed. The gate is
    strictly `tdd` — with it off there is no automated suite, so the manual
    checklist still earns its place and is written exactly as before."""
    try:
        d = _run_dir(thread_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.md").write_text(summary.summary, encoding="utf-8")
        if not tdd:
            (d / "test-plan.md").write_text(summary.test_plan, encoding="utf-8")
    except OSError:
        pass


def write_qa(thread_id: str, qa: QaResult) -> None:
    """Write qa.md — the QA agent's own verdict and what it reviewed; latest
    attempt wins.

    QA-agent-only (Phase 77d): the verdict, the QA agent's account of the checks it
    ran (`review` → a `## Checks performed` section), and any failure detail — never
    the TDD red-green results, which live in the per-task `test-author/` + `impl/`
    evidence folders. A reader of qa.md sees what QA did and nothing it didn't run,
    so the two records never have to be disambiguated."""
    try:
        d = _run_dir(thread_id)
        d.mkdir(parents=True, exist_ok=True)
        review_section = (
            f"\n## Checks performed\n\n{qa.review}\n" if qa.review else ""
        )
        failures_section = (
            f"\n## Failures\n\n{qa.failures}\n" if qa.failures else ""
        )
        content = f"# QA Result: {qa.result}{review_section}{failures_section}\n"
        (d / "qa.md").write_text(content, encoding="utf-8")
    except OSError:
        pass


def _safe_id(task_id: str) -> str:
    """A filesystem-safe fragment from a task id (ids are slugs, but be defensive)."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in task_id) or "task"


def _task_folder_name(task_index: int, task_id: str) -> str:
    """`task-NN-<safe-id>` — the per-task evidence root, numbered in flow order
    (1-based, as the station iterates the decomposed tasks). The test-author/ folder
    (Phase 77b) and impl/attempt-N/ folders (Phase 77c) live under it."""
    return f"task-{task_index:02d}-{_safe_id(task_id)}"


def _copy_test_files(test_paths, dest_dir: Path) -> list[str]:
    """Copy every project test file matching `test_paths` into `dest_dir`, verbatim,
    preserving each file's path relative to the project root (so multiple files, or
    files in subdirs, never collide and a reviewer sees the real layout). Returns
    the copied relative paths. Generic — no assumption about filename or language;
    `iter_test_files` excludes the orchestrator's own workspace so prior tasks'
    evidence copies are never swept back in."""
    root = find_project_root()
    copied: list[str] = []
    for p in iter_test_files(test_paths, root):
        rel = p.relative_to(root)
        out = dest_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(p.read_bytes())
        copied.append(str(rel))
    return copied


def _test_author_summary_md(task: Task, ta: TestAuthorResult, info: dict, copied: list[str]) -> str:
    """Render test-author/summary.md: the testable verdict, the author's note, the
    coverage-critic verdict(s), # re-author rounds, and the red-review outcome — the
    converging process recorded as a log (Phase 77b). Defensive about missing
    `rounds_info` keys (an untestable task or a non-critic/non-review config)."""
    lines = [
        f"# Test author — {task.title} (`{task.id}`)",
        "",
        f"**Testable:** {ta.testable}",
    ]
    if ta.degrade_kind:
        lines.append(f"**Degrade kind:** {ta.degrade_kind}")
    lines += ["", ta.summary or "(no summary)", ""]

    critic_verdicts = info.get("critic_verdicts") or []
    if critic_verdicts:
        lines += ["## Coverage critic", ""]
        for i, cv in enumerate(critic_verdicts, 1):
            verdict = "meaningful" if cv.get("meaningful") else "weak"
            note = cv.get("feedback") or ""
            lines.append(f"{i}. **{verdict}**" + (f" — {note}" if note else ""))
        lines.append("")
    if "critic_rounds" in info:
        lines += [f"**Re-author rounds (critic):** {info['critic_rounds']}", ""]
    if "red_review" in info:
        lines += [f"**Red-review:** {info['red_review']}", ""]
    if "autonomous_reauthor_round" in info:
        lines += [
            f"**Autonomous re-author round:** {info['autonomous_reauthor_round']} "
            "(the implementation could not pass the prior suite within its budget)",
            "",
        ]
    if copied:
        lines += ["## Frozen test files", ""]
        lines += [f"- `{rel}`" for rel in copied]
        lines.append("")
    return "\n".join(lines)


def write_test_author_folder(
    thread_id: str, task_index: int, task: Task, ta: TestAuthorResult,
    test_paths, rounds_info: dict | None = None,
) -> None:
    """Write task-NN-<id>/test-author/ — the converged test-authoring evidence for
    one TDD task (Phase 77b; replaces the flat test-author-<id>.md).

    Written ONCE, after the authoring process converges (critic satisfied + any
    re-author done) — never per critic / re-author round; the rounds are recorded in
    summary.md as a log, but the copy + RED run + hash reflect the FINAL accepted
    suite only. For a TESTABLE task the folder is proof the discipline happened:

        task-NN-<id>/test-author/
        ├── <test_paths file(s)>   copied verbatim (any language / layout)
        ├── results-test-run.md    the COMPLETE RED run on those tests (Phase 77a)
        ├── test-snapshot-hash.md  the deterministic freeze baseline (_hash_test_paths)
        └── summary.md             verdict, critic verdict(s), # re-author, red-review

    EVERY TDD task gets the folder (Phase 73 surfacing): an untestable / degraded
    task gets summary.md only (verdict + reason), no tests / RED run / hash. The copy
    and the hash are taken from the same post-authoring tree state, so the freeze
    proof in 77c lines up. Best-effort; the checkpointed TestAuthorResult is the
    source of truth, so a disk error here never takes down the run."""
    try:
        d = _run_dir(thread_id) / _task_folder_name(task_index, task.id) / "test-author"
        d.mkdir(parents=True, exist_ok=True)

        copied: list[str] = []
        if ta.testable:
            copied = _copy_test_files(test_paths, d)
            (d / "results-test-run.md").write_text(
                f"# RED run — {task.id}\n\n"
                "The COMPLETE final test run on the frozen tests (every test, "
                "pass+fail; both streams). The suite is RED here — this is the spec "
                "the implementation must turn green.\n\n"
                f"```\n{ta.full_run or ta.red_output}\n```\n",
                encoding="utf-8",
            )
            (d / "test-snapshot-hash.md").write_text(
                f"# Frozen test snapshot — {task.id}\n\n"
                "The deterministic content+membership hash of the test_paths globset "
                "right after authoring — the baseline the diff-gate freezes and each "
                "impl attempt (Phase 77c) re-hashes to prove the tests weren't "
                "touched.\n\n"
                f"    {ta.snapshot}\n",
                encoding="utf-8",
            )

        (d / "summary.md").write_text(
            _test_author_summary_md(task, ta, rounds_info or {}, copied),
            encoding="utf-8",
        )
    except OSError:
        pass


def write_impl_attempt(
    thread_id: str,
    task_index: int,
    task: Task,
    attempt: int,
    passed: bool,
    gate_results: list,
    *,
    baseline: str,
    current_hash: str,
) -> None:
    """Write task-NN-<id>/impl/attempt-N/ — the GREEN-half evidence for ONE
    implement attempt of a TESTABLE TDD task (Phase 77c).

    Fired once per implement attempt via run_retry_block's on_attempt hook (Phase
    77a) — EVERY attempt, including the passing (GREEN) one, which the failure-only
    hooks never see. Two files:

        task-NN-<id>/impl/attempt-N/
        ├── test-results.md   the COMPLETE test run after this attempt (every gate's
        │                     captured full_output, Phase 77a) — proof the WHOLE
        │                     suite ran, not just the failing delta
        └── snapshot-hash.md  the test_paths re-hash this attempt vs the frozen 77b
                              baseline → MATCH ✓ / MISMATCH ✗ (the freeze proof; a
                              MISMATCH is the evidence of why the diff-gate failed)

    `attempt` is run_retry_block's own 1-based counter (continuous across a
    growable budget, so an extension keeps numbering rather than resetting). On
    `attempt <= 1` the impl/ folder is cleared first, so a fresh build leaves only
    the attempts against the suite that ultimately ran — notably each autonomous
    re-author round (Phase 76) re-freezes a NEW suite and restarts the attempt
    count, so without the clear a smaller final round would inherit stale,
    higher-numbered folders from a replaced suite.

    Best-effort; the checkpointed retry result is the source of truth, so a disk
    error here never takes down the run. Only the testable TDD path wires the
    on_attempt hook that calls this — the classic / untestable path writes
    nothing here."""
    try:
        impl_dir = (
            _run_dir(thread_id) / _task_folder_name(task_index, task.id) / "impl"
        )
        if attempt <= 1 and impl_dir.exists():
            shutil.rmtree(impl_dir, ignore_errors=True)
        d = impl_dir / f"attempt-{attempt}"
        d.mkdir(parents=True, exist_ok=True)

        # The COMPLETE test run: every gate that produced a captured runner log
        # (Phase 77a full_output). The synthetic diff-gate and LLM gates (e.g.
        # builtin:qa) carry no full_output, so they fall out naturally and what
        # remains is the actual test-suite run. On a freeze MISMATCH the diff-gate
        # (ordered first) fails before the suite runs → no log; record that.
        runs = [
            f"### {gr.step_id}\n\n```\n{gr.full_output}\n```"
            for gr in gate_results
            if getattr(gr, "full_output", "")
        ]
        body = "\n\n".join(runs) if runs else (
            "_No test run was captured this attempt — the freeze check (diff-gate) "
            "failed before the suite ran (the frozen tests were modified). See "
            "snapshot-hash.md._"
        )
        verdict = "GREEN ✓ (all gates passed)" if passed else "RED ✗ (a gate failed)"
        (d / "test-results.md").write_text(
            f"# Impl attempt {attempt} — {task.id}\n\n"
            f"**Gate verdict:** {verdict}\n\n"
            "The COMPLETE test run after this implementation attempt — every test, "
            "pass and fail, both streams — proof the whole suite ran against this "
            "attempt, not only the failing delta.\n\n"
            f"{body}\n",
            encoding="utf-8",
        )

        match = current_hash == baseline
        (d / "snapshot-hash.md").write_text(
            f"# Freeze check — {task.id} attempt {attempt}\n\n"
            f"**Result:** {'MATCH ✓' if match else 'MISMATCH ✗'}\n\n"
            "Re-hash of the test_paths globset this attempt vs the frozen baseline "
            "(../../test-author/test-snapshot-hash.md). A MATCH proves the "
            "implementer left the tests untouched and turned them green by changing "
            "the implementation only; a MISMATCH is the evidence of why the "
            "diff-gate failed this attempt.\n\n"
            f"- Baseline (frozen at authoring): `{baseline}`\n"
            f"- This attempt:                   `{current_hash}`\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def write_manual_checks(thread_id: str, items: list[dict]) -> None:
    """Write manual-checks.md — the acceptance criteria NOT proven by a test (P73).

    Each item is a degraded TDD task (the test-author judged it untestable / born-
    green / had no script gate, so it ran the classic implement→qa path): its
    title, id, the reason it degraded, and its acceptance_criteria — the spec a
    human must now verify by hand. No-op when there are none. Best-effort."""
    if not items:
        return
    try:
        d = _run_dir(thread_id)
        d.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Manual verification needed",
            "",
            "These tasks were not proven by an automated test (the test-author "
            "degraded them to the classic implement→qa path). Verify each "
            "acceptance criterion by hand:",
            "",
        ]
        for it in items:
            lines.append(f"## {it.get('title')}  (`{it.get('task_id')}`)")
            lines.append("")
            lines.append(f"**Why no test:** {it.get('reason') or '(unspecified)'}")
            lines.append("")
            lines.append(f"**Verify:** {it.get('acceptance_criteria') or '(none given)'}")
            lines.append("")
        (d / "manual-checks.md").write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass


def write_error(thread_id: str, exc: BaseException, *, failed_task: str | None = None) -> None:
    """Write error.md — a durable, greppable record of a run-terminating failure
    (Phase 80a, Sink B).

    Lands in the run's artifact folder beside plan.md / qa.md. Records the error
    type and message, the failed task (from the audit-log tail), any structured
    `cause` the runner's transcript feeder attached (e.g. the real billing_error
    the SDK collapsed to a useless subtype string), and the full traceback.

    Bulletproof on purpose: this runs on the failure path, so a problem WRITING it
    must never mask the original error. Any exception is logged at WARNING and
    swallowed (mirrors the audit sink's own guard)."""
    try:
        d = _run_dir(thread_id)
        d.mkdir(parents=True, exist_ok=True)
        cause = getattr(exc, "cause", None)
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        lines = [
            "# Run failed",
            "",
            f"**Error type:** {type(exc).__name__}",
        ]
        if failed_task:
            lines.append(f"**Failed task:** {failed_task}")
        lines += ["", "## Message", "", str(exc) or "(no message)", ""]
        if cause:
            lines += [
                "## Cause",
                "",
                "Recovered from the CLI transcript (the SDK discarded the real "
                "error). This is the actual reason the run failed:",
                "",
                "```json",
                json.dumps(cause, indent=2),
                "```",
                "",
            ]
        lines += ["## Traceback", "", "```", tb.rstrip(), "```", ""]
        (d / "error.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:  # never mask the original failure
        logger.warning("failed writing error.md for %s", thread_id, exc_info=True)


def write_usage(thread_id: str, usage: dict) -> None:
    """Write usage.json — final token/cost summary for the run."""
    try:
        d = _run_dir(thread_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "usage.json").write_text(
            json.dumps(usage, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def rename_with_branch(thread_id: str, branch: str) -> None:
    """Rename {thread_id}/ → {thread_id}-{slug}/ after branch creation.

    Slug is the part after the first '/' in the branch name, e.g.
    'feature/dark-mode-toggle' → 'dark-mode-toggle'.
    If the bare folder doesn't exist or rename fails, silently no-op.
    """
    try:
        slug = branch.split("/", 1)[-1] if "/" in branch else branch
        runs = _runs_dir()
        src = runs / thread_id
        dst = runs / f"{thread_id}-{slug}"
        if src.exists() and not dst.exists():
            src.rename(dst)
    except OSError:
        pass
