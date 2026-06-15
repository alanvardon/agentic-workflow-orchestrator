"""Deterministic git operations used by workflow tasks.

The orchestrator has a hard split:
  - cognition (planning, implementation, QA) → LLM-driven, probabilistic
  - control  (branch creation, commit, PR)   → subprocess, deterministic

This module owns the deterministic side. No prompts, no models, no
structured output — just shell commands wrapped in Python. The PR-creation
pipeline is three idempotent steps (commit, push, pr_create) so a failure at
any step is recoverable via @task caching.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from orchestrator.agents.planning import PlanResult
from orchestrator.config import load_config
from orchestrator.errors import UserActionError
from orchestrator.paths import find_project_root


REPO_ROOT = find_project_root()


class BranchCreationError(UserActionError):
    """Raised when create_branch can't safely create a new branch.

    Two cases collapse into this exception: cannot-reach-main and
    branch-already-exists. The message carries the detail. The orchestrator
    treats this as a terminal workflow failure — planning's checkpoint is
    preserved so you can fix the underlying issue and re-trigger without
    re-paying for the LLM call.

    (Dirty-tree failures raise the DirtyTreeError subclass below; see why
    there.)
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            action=(
                "Fix the underlying issue (branch already exists, network "
                "unreachable, etc.) then start a fresh implement_feature run."
            ),
        )


class DirtyTreeError(BranchCreationError):
    """Raised by verify_clean_tree when the working tree has uncommitted
    changes.

    A subclass of BranchCreationError so every existing `except
    BranchCreationError` still catches it — but its OWN type, because the
    pre-flight clean-tree check runs first in the workflow (before planning,
    long before any branch is created). Before this existed, a dirty tree
    surfaced in tracebacks/logs as a bare 'BranchCreationError', which read
    as a branch-creation failure when nothing of the sort had happened.
    """

    def __init__(self, message: str) -> None:
        UserActionError.__init__(
            self,
            message,
            action="Commit or stash your changes, then call resume_run.",
        )


def _sanitize_type(raw: str, max_len: int = 20) -> str:
    """Sanitize a free-form plan type into a safe git branch prefix.

    Strips to [a-z0-9-], truncates to max_len, rejects empty result.
    Needed because PlanResult.type is now a free-form str — a planner
    that emits type="add a tooltip" would otherwise produce an invalid
    git ref like "add a tooltip/the-slug".
    """
    sanitized = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    sanitized = sanitized[:max_len].rstrip("-")
    if not sanitized:
        raise BranchCreationError(
            f"plan type {raw!r} produced an empty branch prefix after sanitization"
        )
    return sanitized


def _slugify(title: str, max_len: int = 50) -> str:
    """Convert a plan title into a kebab-case branch slug.

    Rules (matching .claude/skills/create-feature-branch.md):
      - lowercase
      - non-alphanumeric runs → single hyphen
      - strip leading/trailing hyphens
      - truncate to max_len, and strip a trailing hyphen if the cut
        landed mid-word (otherwise you'd get `feature/some-title-`).

    Example: "LTV calculation rounding error" → "ltv-calculation-rounding-error"
    """
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:max_len].rstrip("-")


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run a git command in REPO_ROOT, raising on non-zero exit.

    capture_output keeps git's chatter off the orchestrator's stdout —
    workflow output shouldn't be mixed with raw shell noise. The caller
    reads .stdout / .stderr if it needs them.
    """
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def verify_clean_tree() -> None:
    """Raise DirtyTreeError if the working tree has uncommitted changes.

    Pre-flight check used at the very start of the workflow (so a dirty
    tree fails before paying for any LLM calls) and again inside
    create_branch (defence in depth — the tree might have become dirty
    between plan approval and branch creation).
    """
    status = _run(["git", "status", "--porcelain"])
    if status.stdout.strip():
        raise DirtyTreeError(
            f"working tree is dirty. Commit or stash your changes, then "
            f"retry.\n\n{status.stdout.strip()}"
        )


def working_tree_has_changes(base_branch: str | None = None) -> bool:
    """True if the run has anything to ship, False if the build produced no diff.

    Checked after the build (impl ⇄ qa) passes, before summarize / docs /
    commit / push / pr. "Anything to ship" means either an uncommitted
    change in the working tree (the fresh build's edits) OR the branch is already
    ahead of base (the resume-after-commit case, where a prior attempt committed
    but failed before push). A False return — a clean tree with no commits ahead
    of base — means the producer made no changes, so the workflow returns
    status="no_changes" instead of committing an empty branch and opening a PR.

    Mirrors commit()'s own dirty/ahead logic so the two never disagree about
    whether there is something to commit.
    """
    base_branch = _resolve_base(base_branch)
    dirty = bool(_run(["git", "status", "--porcelain"]).stdout.strip())
    ahead = int(
        _run(
            ["git", "rev-list", f"{base_branch}..HEAD", "--count"]
        ).stdout.strip()
        or "0"
    )
    return dirty or ahead > 0


def _strip_thread_prefix(thread_id: str) -> str:
    """Strip a leading word- prefix (e.g. 'run-', 'test-') from a thread id."""
    return re.sub(r"^[a-z]+-", "", thread_id)


def create_branch(plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
    """Create a feature branch from the base branch based on the plan's title and type.

    Returns the branch name on success. Raises:
      - DirtyTreeError if the working tree is dirty (uncommitted changes)
      - BranchCreationError if the base branch can't be reached (offline,
        fetch failure, ...) or the derived branch already exists

    Leaves HEAD on the new branch. Subsequent tasks (implementation,
    commit) assume that.

    Direct port of .claude/skills/create-feature-branch.md. Same five
    steps in the same order — review them side by side once.
    """
    base_branch = _resolve_base(None)

    # 1. Working tree must be clean. We can't safely switch branches
    # otherwise — uncommitted work would either be lost or carried into
    # the new branch (both bad).
    verify_clean_tree()

    # 2. Sync with origin base branch first. Branching from a stale local
    # base is how you end up with PRs that conflict from day one.
    try:
        _run(["git", "checkout", base_branch])
        _run(["git", "pull"])
    except subprocess.CalledProcessError as e:
        raise BranchCreationError(
            f"cannot reach {base_branch}: {(e.stderr or e.stdout).strip()}"
        ) from e

    # 3. Derive the branch name. `<type>/<kebab-slug>` — same scheme as
    # your current coordinator. Sanitize type since it's now free-form.
    type_prefix = _sanitize_type(plan.type)
    suffix = f"-{_strip_thread_prefix(thread_id)}" if thread_id else ""
    # Clamp to at least 8 so a small max_slug_length never produces a negative
    # budget (which would make s[:negative] silently slice from the END of the slug).
    slug_budget = max(8, max_slug_length - len(suffix))
    slug = _slugify(plan.title, slug_budget)
    branch_name = f"{type_prefix}/{slug}{suffix}"

    # 4. Refuse to clobber an existing branch. If the user retries a
    # request that already produced a branch, they need to either delete
    # it first or pick a different title.
    existing = _run(["git", "branch", "--list", branch_name])
    if existing.stdout.strip():
        raise BranchCreationError(f"branch already exists: {branch_name}")

    # 5. Create and switch.
    _run(["git", "checkout", "-b", branch_name])
    return branch_name


class CommitAndPrError(UserActionError):
    """Raised when any of commit/push/pr_create can't safely complete.

    commit/push/pr_create are three idempotent steps but share one error type
    because callers treat them uniformly: log, surface the thread_id, await a
    `resume_run` from the user once the underlying issue is fixed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            action=(
                "Fix the underlying issue (auth, network, pre-commit hook, "
                "etc.) then call resume_run(thread_id) — planning, "
                "implementation, and QA are cached and won't re-run."
            ),
        )


class PreHookError(UserActionError):
    """Raised when a pre-hook script exits with a non-zero status.

    The `output` attribute carries the script's stdout, which is surfaced
    as the abort reason so hook authors can write clear, actionable messages.
    The `returncode` is 124 when the script timed out (POSIX `timeout(1)`
    convention).
    """

    def __init__(self, script: str, output: str, returncode: int) -> None:
        super().__init__(
            f"pre-hook {script!r} failed (exit {returncode}): {output}",
            action=(
                f"Fix the failure in pre-hook {script!r} "
                f"(exit {returncode}), then call resume_run."
            ),
        )
        self.script = script
        self.output = output
        self.returncode = returncode


def _resolve_base(base_branch: str | None) -> str:
    """Return base_branch if given, else read config.pr.base_branch from orchestrator.toml."""
    return base_branch if base_branch is not None else load_config().pr.base_branch


def ensure_on_main(base_branch: str | None = None) -> None:
    """Switch to base_branch and pull if we're not already there.

    Called at the very start of the workflow (inside verify_clean_tree_task,
    after the clean-tree check) so the repo is in a known state before any
    LLM spend. If the user started the orchestrator from a feature branch
    with a clean tree, this moves them back to base_branch and updates it —
    the same checkout + pull that create_branch would do later, just earlier.

    No-op when already on base_branch (create_branch will still pull before
    branching). Raises BranchCreationError on checkout/pull failure.
    """
    base_branch = _resolve_base(base_branch)
    current = _current_branch()
    if current == base_branch:
        return
    try:
        _run(["git", "checkout", base_branch])
        _run(["git", "pull"])
    except subprocess.CalledProcessError as e:
        raise BranchCreationError(
            f"cannot switch to {base_branch}: {(e.stderr or e.stdout).strip()}"
        ) from e


def _current_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def _assert_on_branch(branch: str) -> None:
    """Refuse to touch the wrong branch.

    The single most important check we make at this layer — accidentally
    committing to main is the failure mode that takes the longest to
    recover from. Each of commit/push/pr_create calls this so the check is
    never skipped.
    """
    current = _current_branch()
    if current != branch:
        raise CommitAndPrError(
            f"wrong branch: expected {branch!r}, got {current!r}"
        )


def commit(branch: str, title: str, summary: str, base_branch: str | None = None) -> str:
    """Stage and commit any uncommitted changes; return the HEAD SHA.

    **Idempotent.** If the working tree is already clean AND the branch
    is ahead of base_branch (from config when not supplied), assume a
    previous attempt's commit is already in place and return HEAD's SHA
    without re-committing. This
    is the resume-after-failure path: if a previous workflow committed
    locally but failed before push/PR creation, the next attempt sees
    the commit already present and proceeds to push.

    If the working tree is clean AND the branch is NOT ahead of base,
    there's genuinely nothing to commit (the implementation phase made
    no edits, or the changes were already reverted) — raise.

    Returns:
        The commit SHA (full 40-char hex).
    """
    base_branch = _resolve_base(base_branch)
    _assert_on_branch(branch)

    dirty = bool(_run(["git", "status", "--porcelain"]).stdout.strip())
    ahead = int(
        _run(
            ["git", "rev-list", f"{base_branch}..HEAD", "--count"]
        ).stdout.strip()
        or "0"
    )

    if not dirty and ahead > 0:
        # Previous attempt's commit is still here. Return its SHA.
        return _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if not dirty and ahead == 0:
        raise CommitAndPrError("no changes to commit")

    # Stage everything. .gitignore excludes .env, .orchestrator/, caches;
    # project pre-commit hooks are the second line of defence. `git add
    # .` is the documented safe default for this project.
    _run(["git", "add", "."])

    # Compose the commit message. Type prefix from the branch name
    # (e.g. "feature/foo" → "feature"). Scope omitted — we can't infer
    # it deterministically.
    type_prefix = branch.split("/", 1)[0] if "/" in branch else "feature"
    subject = f"{type_prefix}: {title.lower()}"
    commit_msg = f"{subject}\n\n- {summary}"

    try:
        _run(
            [
                "git",
                "commit",
                "--author=Claude <claude@anthropic.com>",
                "-m",
                commit_msg,
            ]
        )
    except subprocess.CalledProcessError as e:
        raise CommitAndPrError(
            f"commit failed: {(e.stderr or e.stdout).strip()}"
        ) from e

    return _run(["git", "rev-parse", "HEAD"]).stdout.strip()


def push(branch: str, base_branch: str | None = None, auto_rebase: bool = True) -> None:
    """Push branch to origin with upstream tracking.

    **Idempotent.** `git push` is naturally a no-op when the remote is
    already up to date ("Everything up-to-date"), and `-u` re-asserting
    upstream tracking is a no-op if already set.

    **Conflict handling.** Fetches origin first, then checks
    whether `origin/<base_branch>` has advanced since branch creation.
    If so:
      - `auto_rebase=True` (default): runs `git rebase origin/<base_branch>`.
        If rebase conflicts, aborts and raises `UserActionError`.
      - `auto_rebase=False`: raises `UserActionError` immediately, asking the
        user to rebase manually then call `resume_run`.
    """
    base_branch = _resolve_base(base_branch)
    _assert_on_branch(branch)

    try:
        _run(["git", "fetch", "origin"])
    except subprocess.CalledProcessError as e:
        raise CommitAndPrError(
            f"fetch failed: {(e.stderr or e.stdout).strip()}"
        ) from e

    remote_base = f"origin/{base_branch}"
    behind = int(
        _run(["git", "rev-list", f"HEAD..{remote_base}", "--count"]).stdout.strip()
        or "0"
    )

    did_rebase = False
    if behind > 0:
        if not auto_rebase:
            raise UserActionError(
                f"{remote_base} has {behind} new commit(s); rebase manually then resume_run.",
                action=(
                    f"Run: git rebase {remote_base} on branch {branch!r}, "
                    "resolve any conflicts, then call resume_run."
                ),
            )
        try:
            _run(["git", "rebase", remote_base])
            did_rebase = True
        except subprocess.CalledProcessError:
            try:
                _run(["git", "rebase", "--abort"])
            except subprocess.CalledProcessError:
                pass
            raise UserActionError(
                f"{remote_base} moved and rebase conflicted; resolve manually then resume_run.",
                action=(
                    f"Run: git rebase {remote_base} on branch {branch!r}, "
                    "resolve conflicts, then call resume_run."
                ),
            )

    # A rebase rewrites local history. If this branch was pushed before (resume-
    # after-failure path), a plain push would be rejected as non-fast-forward.
    # --force-with-lease is safe: it refuses if the remote moved since our fetch,
    # so a genuinely concurrent push still fails loud rather than being overwritten.
    push_cmd = ["git", "push", "-u", "origin", branch]
    if did_rebase:
        push_cmd.insert(2, "--force-with-lease")
    try:
        _run(push_cmd)
    except subprocess.CalledProcessError as e:
        raise CommitAndPrError(
            f"push failed: {(e.stderr or e.stdout).strip()}"
        ) from e


# The PR label is auto-derived from the plan's type. Plan type "fix" maps to
# GitHub's conventional "bug" label; an unknown type yields no label.
_PLAN_TYPE_LABELS = {
    "feature": "feature",
    "fix": "bug",
    "refactor": "refactor",
    "chore": "chore",
}


def pr_create(
    branch: str,
    title: str,
    summary: str,
    test_plan: str,
    base_branch: str | None = None,
    draft: bool = False,
    reviewers: list[str] | None = None,
    plan_type: str | None = None,
) -> str:
    """Open a PR for the branch and return its URL.

    **Idempotent.** If a PR already exists for this branch (any state —
    open, closed, merged), return its URL instead of trying to open
    another. This handles the case where a previous attempt successfully
    opened the PR but the workflow failed afterwards (or where the user
    is `resume_run`-ing a thread whose pr_create_task already succeeded
    on a prior invocation — the LangGraph cache should cover that, but
    the in-function check is defence in depth).
    """
    base_branch = _resolve_base(base_branch)
    _assert_on_branch(branch)

    # Check for an existing open PR. `gh pr view <branch>` exits non-zero
    # if none exists — that's the signal to create one. Only reuse the URL
    # when state is OPEN; a closed PR means a previous attempt was rejected
    # and we should open a fresh one.
    try:
        existing = _run(["gh", "pr", "view", branch, "--json", "url,state"])
        data = json.loads(existing.stdout)
        if data.get("state") == "OPEN" and data.get("url"):
            return data["url"]
    except subprocess.CalledProcessError:
        # No PR exists; fall through to create one.
        pass

    body = (
        f"## Summary\n{summary}\n\n"
        f"## Test plan\n{test_plan}\n\n"
        "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )
    cmd = ["gh", "pr", "create", "--title", title, "--body", body, "--base", base_branch]
    if draft:
        cmd.append("--draft")
    for reviewer in (reviewers or []):
        cmd += ["--reviewer", reviewer]
    label = _PLAN_TYPE_LABELS.get(plan_type or "", "")
    if label:
        cmd += ["--label", label]
    try:
        result = _run(cmd)
    except subprocess.CalledProcessError as e:
        raise CommitAndPrError(
            f"gh pr create failed: {(e.stderr or e.stdout).strip()}"
        ) from e

    # `gh pr create` prints the URL on the last non-empty stdout line.
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise CommitAndPrError("gh pr create produced no output")
    return lines[-1].strip()


if __name__ == "__main__":
    # Standalone test for create_branch only:
    #   python -m orchestrator.git_ops "Stress test for variable rates" feature
    # No LLM call; doesn't touch the checkpointer. Just exercises the
    # subprocess plumbing against your real repo.
    #
    # commit/push/pr_create aren't exposed here because (a) their
    # inputs are awkward to pass as positional args (multi-line summary,
    # markdown test plan) and (b) their side effects — a real commit,
    # push, and PR opening on GitHub — make ad-hoc testing expensive.
    # Test them via the workflow end-to-end run instead.
    title = sys.argv[1] if len(sys.argv) > 1 else "test branch creation"
    branch_type = sys.argv[2] if len(sys.argv) > 2 else "feature"
    fake_plan = PlanResult(title=title, type=branch_type, plan_text="(test)")
    try:
        print(f"created: {create_branch(fake_plan)}")
    except BranchCreationError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
