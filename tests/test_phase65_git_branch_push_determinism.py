"""Phase 65 — git branch/push determinism edge cases.

Two findings:

#7  `_slugify` was called with a potentially-negative budget when the operator set a
    small `max_slug_length`. `s[:negative]` silently slices from the END of the slug,
    producing a garbled or empty branch name. Fixed by clamping the budget to
    max(8, max_slug_length - len(suffix)) before passing it to `_slugify`.

#8  After `git rebase` rewrites local history, a plain `git push` would be rejected
    as non-fast-forward if the branch was previously pushed (the resume-after-failure
    path). Fixed by using `--force-with-lease` after a rebase — safe because the
    lease refuses if the remote moved since our fetch.

See ../.misc_notes/remaining_phases/code_review_2026_06_04/phase_65_git_branch_push_determinism.md
"""

import subprocess

import pytest

from orchestrator.agents.planning import PlanResult
from orchestrator.errors import UserActionError
from orchestrator.git_ops import CommitAndPrError, _slugify, push


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


# ---------------------------------------------------------------------------
# #7 — _slugify budget clamp
# ---------------------------------------------------------------------------


def test_slugify_does_not_produce_end_slice_on_small_max_len():
    """The pre-65 bug: s[:negative] slices from the end. Ensure that never happens
    by testing _slugify directly with a small max_len."""
    # Any positive (or zero) max_len should not produce a trailing-hyphen slug
    result = _slugify("some-long-title-that-would-be-sliced", max_len=5)
    assert result  # non-empty
    assert not result.endswith("-")
    assert len(result) <= 5


def test_slugify_zero_max_len_gives_empty_string():
    """Zero budget → empty slug (caller is responsible for clamping before calling)."""
    # _slugify itself doesn't guard against 0/negative — the clamp lives in
    # create_branch. This test documents the _slugify contract.
    result = _slugify("any title", max_len=0)
    assert result == ""


def test_create_branch_small_max_slug_length_still_produces_valid_name(monkeypatch):
    """A small max_slug_length (e.g. 5) must yield a valid, non-empty branch name
    with no trailing hyphen — not a garbled end-slice artifact."""
    calls = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed("main\n")
        if "git checkout main" in joined:
            return _completed()
        if "git pull" in joined:
            return _completed()
        if "git branch --list" in joined:
            return _completed("")   # branch does not exist
        if "git checkout -b" in joined:
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    monkeypatch.setattr("orchestrator.git_ops.verify_clean_tree", lambda: None)
    monkeypatch.setattr(
        "orchestrator.git_ops._resolve_base", lambda b: "main"
    )

    from orchestrator.git_ops import create_branch

    plan = PlanResult(title="some long title here", type="feature", plan_text="")
    # max_slug_length=5 with a thread_id suffix of e.g. "-run123" would give a
    # negative budget pre-65. Now it is clamped to at least 8.
    branch = create_branch(plan, max_slug_length=5, thread_id="run-abc123")

    assert branch  # non-empty
    type_part, slug_part = branch.split("/", 1)
    assert type_part == "feature"
    assert slug_part  # non-empty slug
    assert not slug_part.startswith("-")
    assert not slug_part.endswith("-")


def test_create_branch_small_max_slug_length_no_thread_id(monkeypatch):
    """Without a thread_id suffix, a small max_slug_length still clamps correctly."""
    from orchestrator.git_ops import create_branch

    calls = []

    def fake_run(args):
        calls.append(args)
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed("main\n")
        if "git checkout main" in joined:
            return _completed()
        if "git pull" in joined:
            return _completed()
        if "git branch --list" in joined:
            return _completed("")
        if "git checkout -b" in joined:
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    monkeypatch.setattr("orchestrator.git_ops.verify_clean_tree", lambda: None)
    monkeypatch.setattr("orchestrator.git_ops._resolve_base", lambda b: "main")

    plan = PlanResult(title="a feature", type="feature", plan_text="")
    branch = create_branch(plan, max_slug_length=5)

    type_part, slug_part = branch.split("/", 1)
    assert slug_part  # non-empty
    assert not slug_part.endswith("-")


# ---------------------------------------------------------------------------
# #8 — --force-with-lease after rebase
# ---------------------------------------------------------------------------


def test_push_uses_force_with_lease_after_rebase(monkeypatch):
    """When origin/main moved and a rebase ran, the push must include --force-with-lease."""
    push_cmds = []

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed("feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            return _completed("2\n")  # 2 commits behind → rebase needed
        if "git rebase origin/main" in joined:
            return _completed()
        if "git push" in joined:
            push_cmds.append(args[:])
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    push("feature/test", auto_rebase=True)

    assert push_cmds, "push was never called"
    push_flags = push_cmds[0]
    assert "--force-with-lease" in push_flags, (
        f"expected --force-with-lease in push args after rebase, got: {push_flags}"
    )


def test_push_no_force_with_lease_without_rebase(monkeypatch):
    """When origin/main did NOT move (no rebase), push must NOT use --force-with-lease."""
    push_cmds = []

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed("feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            return _completed("0\n")  # up to date — no rebase
        if "git push" in joined:
            push_cmds.append(args[:])
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    push("feature/test")

    assert push_cmds, "push was never called"
    assert "--force-with-lease" not in push_cmds[0], (
        "should not add --force-with-lease when no rebase occurred"
    )


def test_push_lease_rejected_raises_commit_pr_error(monkeypatch):
    """If --force-with-lease is rejected (remote moved since our fetch), the error
    surfaces as CommitAndPrError, not an unhandled CalledProcessError."""

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed("feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            return _completed("1\n")   # behind → triggers rebase path
        if "git rebase origin/main" in joined:
            return _completed()
        if "git push" in joined:
            raise subprocess.CalledProcessError(
                1, args,
                stderr="! [rejected] feature/test -> feature/test (stale info)"
            )
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)

    with pytest.raises(CommitAndPrError, match="push failed"):
        push("feature/test", auto_rebase=True)


def test_push_force_with_lease_position_in_args(monkeypatch):
    """--force-with-lease must appear before -u in the push command so git accepts it."""
    push_cmds = []

    def fake_run(args):
        joined = " ".join(args)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _completed("feature/test\n")
        if "git fetch origin" in joined:
            return _completed()
        if "rev-list" in joined and "--count" in joined:
            return _completed("1\n")
        if "git rebase origin/main" in joined:
            return _completed()
        if "git push" in joined:
            push_cmds.append(args[:])
            return _completed()
        raise AssertionError(f"unexpected: {joined!r}")

    monkeypatch.setattr("orchestrator.git_ops._run", fake_run)
    push("feature/test", auto_rebase=True)

    assert push_cmds
    cmd = push_cmds[0]
    # git push --force-with-lease -u origin branch  (lease before -u)
    lease_idx = cmd.index("--force-with-lease")
    u_idx = cmd.index("-u")
    assert lease_idx < u_idx, f"--force-with-lease should precede -u; got {cmd}"
