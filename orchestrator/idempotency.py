"""Idempotency keys.

Caller-supplied `idempotency_key` on `implement_feature`: a second
call with the same key returns the existing thread's current state
instead of starting a fresh workflow. Useful when:
  - the MCP tool is invoked twice in a row (double-click, double-send)
  - a webhook retries on a flaky network connection
  - a CI job re-runs the same step

Store: a directory `.orchestrator/idempotency/<key>` where each file
contains the thread_id that claimed the key. The atomic primitive is
write-then-`os.link`: the thread_id is written to a private temp file,
then hard-linked onto the key path. `os.link` fails closed
(FileExistsError) if the key already exists — so exactly one of two
concurrent callers wins — and the published entry is content-complete
before it ever appears at the key path, so a loser always reads a full
thread_id (never an empty file, the flaw in the earlier create-empty-
then-write approach).

Why filesystem and not a SQLite table — same reason as the cancel
markers in `cancellation.py`. The LangGraph AsyncSqliteSaver holds a
write lock on the checkpoint db for the workflow's lifetime, and a
synchronous reader from a second module hits `database is locked`.
Filesystem entries have no shared lock.

When the orchestrator moves off a single dev machine onto infra, the
backend ports to a `idempotency_keys` table in the same Postgres
database that replaces `AsyncSqliteSaver`. The interface
(`reserve` / `lookup` / `purge_older_than`) stays; the bodies change.
The same production-port note that applies to the cancellation store
applies here.
"""

import os
import re
import time
from pathlib import Path
from uuid import uuid4

from orchestrator.config import load_config
from orchestrator.paths import find_project_root


# Idempotency keys are caller-supplied (CI job ids, webhook delivery
# ids, user-typed strings). We allow a slightly broader character set
# than thread_ids — periods are common in CI job ids (e.g. `build.42`)
# — but still reject slashes, dots-only sequences, and other
# path-traversal characters so a malicious or buggy caller can't
# write outside the idempotency directory.
_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_KEY_MAX_LEN = 128


def _validate_key(key: str) -> None:
    if not key:
        raise ValueError("idempotency_key must be a non-empty string")
    if len(key) > _KEY_MAX_LEN:
        raise ValueError(
            f"idempotency_key length {len(key)} exceeds max {_KEY_MAX_LEN}"
        )
    if not _KEY_RE.fullmatch(key):
        raise ValueError(
            f"idempotency_key {key!r} contains characters outside "
            "[A-Za-z0-9._-]; refusing to use as a filename."
        )
    # Defence in depth against `.` / `..` / `...` — _KEY_RE allows
    # dots, but a key consisting entirely of dots would collide with
    # the current-/parent-directory specials and is never a sensible
    # caller-supplied value.
    if set(key) == {"."}:
        raise ValueError(f"idempotency_key {key!r} is a directory specifier")


def _idempotency_dir(base_dir: Path | None = None) -> Path:
    """Resolve the directory holding idempotency entries.

    Defaults to `<config.db_path's parent>/idempotency` so the entries
    sit alongside the checkpoint db and the cancellation markers.
    Callers can pass an explicit base_dir for tests.
    """
    if base_dir is not None:
        return base_dir
    config = load_config()
    db_parent = Path(config.db_path).parent
    relative = db_parent / "idempotency"
    return relative if relative.is_absolute() else find_project_root() / relative


def _entry_path(key: str, base_dir: Path | None = None) -> Path:
    _validate_key(key)
    return _idempotency_dir(base_dir) / key


def reserve(
    key: str, thread_id: str, base_dir: Path | None = None
) -> str | None:
    """Atomically claim `key` for `thread_id`.

    Returns None on success (this caller is the first to use the key).
    Returns the existing thread_id (str) if the key was already claimed
    — the caller should use that thread_id instead of starting a new
    workflow.

    Race semantics: write-then-atomically-publish. The thread_id is written
    to a private temp file FIRST, then `os.link` hard-links it onto the key
    path. `os.link` fails closed (FileExistsError) if the key already exists,
    so exactly one of two concurrent callers wins (None) and the other reads
    the winner's id — and because the published entry is already
    content-complete, a losing caller can NEVER read a half-written (empty)
    file. (The old `O_CREAT|O_EXCL` created the key file empty and wrote the
    id in a second step, so a loser racing between the two could read "".)
    """
    path = _entry_path(key, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hidden + uniquely named (pid + uuid) so concurrent reservers never collide
    # on the temp, and `purge_older_than` / lookups never mistake it for an entry.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(thread_id)
        try:
            # Atomic publish: links the fully-written temp onto the key path, or
            # fails because someone already claimed it.
            os.link(str(tmp), str(path))
        except FileExistsError:
            # Race lost (or duplicate retry). The winner's entry is already
            # complete, so this read never sees an empty file.
            return path.read_text().strip()
        return None
    finally:
        # On the win path the content lives on at `path` (a second hard link);
        # on the lose path it was never published. Either way the temp goes.
        tmp.unlink(missing_ok=True)


def lookup(key: str, base_dir: Path | None = None) -> str | None:
    """Return the thread_id reserved for `key`, or None if not reserved."""
    path = _entry_path(key, base_dir)
    if not path.exists():
        return None
    return path.read_text().strip()


def purge_older_than(
    days: int, base_dir: Path | None = None
) -> int:
    """Delete idempotency entries older than `days` days (by mtime).

    Returns the count of entries deleted. Intended to be invoked from
    cron or an admin script — not on every reserve(), since the
    table-ish would grow forever otherwise and purging is cheap to run
    out-of-band.
    """
    if days < 0:
        raise ValueError(f"days must be non-negative, got {days}")
    cutoff = time.time() - days * 86400
    directory = _idempotency_dir(base_dir)
    if not directory.exists():
        return 0
    count = 0
    for entry in directory.iterdir():
        if entry.is_file() and entry.stat().st_mtime < cutoff:
            entry.unlink()
            count += 1
    return count
