"""Phase 18 idempotency tests.

Three slices:
1. Store helpers (reserve / lookup / purge) — direct unit tests against
   a tmp_path base_dir, no LangGraph involved.
2. Key validation — character class, length cap, dot-only refusal.
3. MCP-tool integration — `implement_feature` with a fresh key starts
   a workflow; the same key returns the existing thread's state
   (`replayed: True`) without spawning a new run.
"""

import asyncio
import os
import time
from pathlib import Path

import pytest

from orchestrator.manifest import StepResult
from orchestrator.agents.planning import PlanResult
from orchestrator.agents.qa import QaResult
from orchestrator.idempotency import (
    lookup,
    purge_older_than,
    reserve,
)


# ---------------------------------------------------------------------------
# Store helpers
# ---------------------------------------------------------------------------


def _dir(tmp_path: Path) -> Path:
    return tmp_path / "idempotency"


def test_reserve_fresh_key_returns_none(tmp_path):
    result = reserve("key-1", "run-abc", _dir(tmp_path))
    assert result is None


def test_reserve_duplicate_key_returns_existing_thread_id(tmp_path):
    d = _dir(tmp_path)
    reserve("key-1", "run-abc", d)
    result = reserve("key-1", "run-xyz", d)
    assert result == "run-abc"


def test_lookup_missing_key_returns_none(tmp_path):
    assert lookup("never-set", _dir(tmp_path)) is None


def test_lookup_after_reserve(tmp_path):
    d = _dir(tmp_path)
    reserve("key-1", "run-abc", d)
    assert lookup("key-1", d) == "run-abc"


def test_reserve_does_not_overwrite(tmp_path):
    """Even after several reserve() calls with different thread_ids,
    lookup must return the FIRST winner — the whole point of
    idempotency is that subsequent claims are no-ops."""
    d = _dir(tmp_path)
    reserve("key-1", "run-first", d)
    reserve("key-1", "run-second", d)
    reserve("key-1", "run-third", d)
    assert lookup("key-1", d) == "run-first"


def test_purge_older_than_removes_old_entries(tmp_path):
    d = _dir(tmp_path)
    reserve("old", "run-old", d)
    reserve("new", "run-new", d)

    # Backdate the "old" entry by 31 days.
    old_path = d / "old"
    old_ts = time.time() - 31 * 86400
    os.utime(old_path, (old_ts, old_ts))

    deleted = purge_older_than(30, d)
    assert deleted == 1
    assert lookup("old", d) is None
    assert lookup("new", d) == "run-new"


def test_purge_zero_days_removes_everything(tmp_path):
    d = _dir(tmp_path)
    reserve("a", "run-a", d)
    reserve("b", "run-b", d)
    # Push mtimes just slightly into the past so all entries are
    # "older than 0 days."
    for entry in d.iterdir():
        os.utime(entry, (time.time() - 1, time.time() - 1))
    assert purge_older_than(0, d) == 2


def test_purge_negative_days_raises(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        purge_older_than(-1, _dir(tmp_path))


def test_purge_missing_dir_returns_zero(tmp_path):
    # purge before any reserve — directory doesn't exist yet.
    assert purge_older_than(7, _dir(tmp_path)) == 0


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_key", [
    "with/slash",
    "with\\backslash",
    "../escape",
    "with space",
    "with:colon",
    "with;semi",
    "ünicode",
])
def test_invalid_keys_rejected(tmp_path, bad_key):
    with pytest.raises(ValueError, match="idempotency_key"):
        reserve(bad_key, "run-abc", _dir(tmp_path))


def test_empty_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        reserve("", "run-abc", _dir(tmp_path))


def test_overlong_key_rejected(tmp_path):
    overlong = "a" * 129
    with pytest.raises(ValueError, match="exceeds max"):
        reserve(overlong, "run-abc", _dir(tmp_path))


def test_dot_only_key_rejected(tmp_path):
    """A key consisting entirely of dots collides with `.` / `..` /
    `...` (current/parent directory specifiers). Refuse it explicitly
    even though `_KEY_RE` would otherwise let it through."""
    with pytest.raises(ValueError, match="directory specifier"):
        reserve(".", "run-abc", _dir(tmp_path))


@pytest.mark.parametrize("good_key", [
    "ci-job-789",
    "build.42",
    "webhook_2026-05-27",
    "abc",
    "a",
    "X" * 128,  # boundary: max length
])
def test_valid_keys_accepted(tmp_path, good_key):
    result = reserve(good_key, "run-abc", _dir(tmp_path))
    assert result is None
    assert lookup(good_key, _dir(tmp_path)) == "run-abc"


# ---------------------------------------------------------------------------
# MCP-tool integration
# ---------------------------------------------------------------------------


class _Stubs:
    def __init__(self) -> None:
        self.plan_calls: list[str] = []
        self.commit_called = False

    def verify_clean_tree(self) -> None:
        pass

    def ensure_on_main(self, base_branch: str = "main") -> None:
        pass

    async def plan(self, request: str, model: str = "claude-sonnet-4-6") -> PlanResult:
        self.plan_calls.append(request)
        n = len(self.plan_calls)
        return PlanResult(title=f"title-{n}", type="feature", plan_text=f"plan-{n}")

    def create_branch(self, plan: PlanResult, max_slug_length: int = 50, thread_id: str = "") -> str:
        return "feature/test"

    async def implementation_task(self, plan_text, feedback=None, model="claude-sonnet-4-6"):
        return StepResult(step_id="implementation", kind="ai_agent", ok=True)

    async def qa(self, plan, model="claude-sonnet-4-6") -> QaResult:
        return QaResult(result="PASS")

    def commit(self, branch, title, summary, base_branch="main") -> str:
        self.commit_called = True
        return "abc123def456"

    def push(self, branch, base_branch="main", auto_rebase=True) -> None:
        pass

    def pr_create(self, branch, title, summary, test_plan, base_branch="main", draft=False, reviewers=None, labels=None) -> str:
        return "https://github.com/test/pr/1"


def _patch(stubs: _Stubs, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("orchestrator.workflow.verify_clean_tree", stubs.verify_clean_tree)
    monkeypatch.setattr("orchestrator.workflow.ensure_on_main", stubs.ensure_on_main)
    monkeypatch.setattr("orchestrator.workflow.plan", stubs.plan)
    monkeypatch.setattr("orchestrator.workflow.create_branch", stubs.create_branch)
    monkeypatch.setattr("orchestrator.workflow.implementation_task", stubs.implementation_task)
    monkeypatch.setattr("orchestrator.workflow.qa", stubs.qa)
    monkeypatch.setattr("orchestrator.workflow.commit", stubs.commit)
    monkeypatch.setattr("orchestrator.workflow.push", stubs.push)
    monkeypatch.setattr("orchestrator.workflow.pr_create", stubs.pr_create)
    monkeypatch.chdir(tmp_path)
    Path(".orchestrator").mkdir(exist_ok=True)


@pytest.mark.asyncio
async def test_implement_feature_without_key_works_as_before(monkeypatch, tmp_path):
    """No idempotency_key → behave identically to pre-Phase-18."""
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("add a tooltip")
    assert result["status"] == "awaiting_approval"
    assert result["thread_id"].startswith("run-")
    assert "replayed" not in result


@pytest.mark.asyncio
async def test_implement_feature_fresh_key_runs_workflow(monkeypatch, tmp_path):
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    result = await implement_feature("add a tooltip", idempotency_key="ci-1")
    # Fresh key claims and runs the workflow as normal.
    assert result["status"] == "awaiting_approval"
    assert result.get("replayed") is not True  # first call is NOT a replay
    assert result["plan"]["plan_text"] == "plan-1"


@pytest.mark.asyncio
async def test_implement_feature_reused_key_replays(monkeypatch, tmp_path):
    """Two calls with the same key → same thread_id, replayed=True,
    and the planning stub is invoked only once."""
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    first = await implement_feature("add a tooltip", idempotency_key="ci-2")
    second = await implement_feature("add a tooltip", idempotency_key="ci-2")

    assert second["thread_id"] == first["thread_id"]
    assert second.get("replayed") is True
    assert second["status"] == "awaiting_approval"
    # Planning stub was invoked exactly once across the two calls.
    assert len(stubs.plan_calls) == 1


@pytest.mark.asyncio
async def test_implement_feature_reused_key_different_request_still_replays(
    monkeypatch, tmp_path
):
    """Idempotency is keyed solely on `idempotency_key` — the request
    text is ignored on the replay. A caller that wants a different
    workflow must supply a different key."""
    stubs = _Stubs()
    _patch(stubs, monkeypatch, tmp_path)

    from orchestrator.mcp_server import implement_feature

    first = await implement_feature("original request", idempotency_key="ci-3")
    second = await implement_feature("totally different request", idempotency_key="ci-3")

    assert second["thread_id"] == first["thread_id"]
    assert second.get("replayed") is True
    # Only the first request was sent to the planner.
    assert stubs.plan_calls == ["original request"]


# ---------------------------------------------------------------------------
# run_log integration
# ---------------------------------------------------------------------------


def test_append_run_omits_idempotency_key_when_none(tmp_path, monkeypatch):
    """Backwards compatibility: existing log shape stays compact when
    no key is supplied."""
    import json
    import orchestrator.run_log as run_log

    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_log, "_LOG_PATH", log_path)

    run_log.append_run("run-aaa", "request", "mcp")
    record = json.loads(log_path.read_text().strip())
    assert "idempotency_key" not in record


def test_append_run_includes_idempotency_key_when_given(tmp_path, monkeypatch):
    import json
    import orchestrator.run_log as run_log

    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_log, "_LOG_PATH", log_path)

    run_log.append_run("run-bbb", "request", "mcp", idempotency_key="ci-x")
    record = json.loads(log_path.read_text().strip())
    assert record["idempotency_key"] == "ci-x"
