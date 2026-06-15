"""Phase 64 — idempotency reserve() atomicity.

`reserve()` used `os.open(O_CREAT|O_EXCL)` to create the key file EMPTY, then
wrote the thread_id in a second step. A concurrent caller that lost the race
could hit FileExistsError and read the file in the window before the winner's
write landed — getting "". Phase 64 switches to write-then-hardlink (atomic
publish): the entry is fully written before it ever appears at the key path, so
a loser always reads a complete thread_id, never "".

See ../.misc_notes/remaining_phases/code_review_2026_06_04/phase_64_idempotency_reserve_atomicity.md
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from orchestrator.idempotency import lookup, purge_older_than, reserve


def _dir(tmp_path: Path) -> Path:
    return tmp_path / "idempotency"


def test_loser_reads_complete_thread_id_never_empty(tmp_path):
    """The classic lose path: key already claimed → the second reserve returns
    the winner's FULL id, never an empty string."""
    d = _dir(tmp_path)
    assert reserve("k", "run-winner", d) is None
    existing = reserve("k", "run-loser", d)
    assert existing == "run-winner"
    assert existing != ""


def test_concurrent_reservers_agree_on_one_winner(tmp_path):
    """Hammer the same key from many threads at once. Exactly one wins (None);
    every loser reads the SAME non-empty winner id; nobody reads ""."""
    d = _dir(tmp_path)
    n = 64
    barrier = Barrier(n)  # release all threads as simultaneously as possible

    def _claim(i: int):
        barrier.wait()
        return reserve("hot-key", f"run-{i}", d)

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(_claim, range(n)))

    winners = [r for r in results if r is None]
    losers = [r for r in results if r is not None]
    assert len(winners) == 1, f"expected exactly one winner, got {len(winners)}"
    assert len(losers) == n - 1
    # No loser ever saw a half-written (empty) entry...
    assert all(r for r in losers), "a loser read an empty thread_id"
    # ...and every loser agrees on the single persisted winner id.
    assert len(set(losers)) == 1
    assert losers[0] == lookup("hot-key", d)


def test_reserve_leaves_no_temp_files(tmp_path):
    """Both the win and lose paths clean up their temp file: the directory holds
    exactly one entry (the key), no leftover .tmp artifacts."""
    d = _dir(tmp_path)
    reserve("k", "run-a", d)   # win
    reserve("k", "run-b", d)   # lose
    entries = sorted(p.name for p in d.iterdir())
    assert entries == ["k"]


def test_temp_cleanup_does_not_disturb_purge(tmp_path):
    """purge_older_than counts only real entries — temp files (gone after each
    reserve) never inflate the count."""
    d = _dir(tmp_path)
    reserve("a", "run-a", d)
    reserve("b", "run-b", d)
    # Nothing old yet.
    assert purge_older_than(1, d) == 0
    # Everything purged with a zero cutoff — exactly the two real entries.
    assert purge_older_than(0, d) == 2
    assert sorted(p.name for p in d.iterdir()) == []
