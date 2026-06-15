"""Phase 68: the `flow` line parser.

Parses an Airflow-style pipeline-order string into an ordered sequence of stage
groups. `>>` is a sequential edge; `[a, b]` is a parallel group (fan-out, then
join). For example:

    "plan >> decompose >> [docs, gitleaks] >> summarize"

parses to groups:

    (("plan",), ("decompose",), ("docs", "gitleaks"), ("summarize",))

The Phase 68 executor runs the result SEQUENTIALLY — parallel groups run in
declared order — but the parsed shape already records the parallelism, so a
future concurrent runner needs no config-language change.

Pure and total: no I/O. Raises FlowSyntaxError on malformed input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from orchestrator.errors import FatalError


class FlowSyntaxError(FatalError):
    """The `flow` string is malformed (Phase 68). A config error: fix and re-run."""


# A stage id: starts alphanumeric, then alphanumerics / dot / underscore / hyphen.
# (Matches TOML bare-key-ish ids like `task-build`, `final-qa`, `plan`.)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class FlowGraph:
    """The parsed flow: an ordered tuple of parallel groups.

    Each inner tuple is a group whose members may (eventually) run in parallel;
    a singleton group is an ordinary sequential step.
    """

    groups: tuple[tuple[str, ...], ...]

    def ordered_ids(self) -> list[str]:
        """All stage ids in execution order (parallel groups flattened, declared
        order preserved). This is the sequence the Phase 68 executor runs."""
        return [sid for group in self.groups for sid in group]

    def ids(self) -> set[str]:
        return set(self.ordered_ids())

    def edges(self) -> list[tuple[str, str]]:
        """Sequential dependency edges: every id in group i precedes every id in
        group i+1. Useful for a future DAG executor / cycle reasoning. The flow
        form is inherently acyclic, so this never produces a cycle."""
        out: list[tuple[str, str]] = []
        for before, after in zip(self.groups, self.groups[1:]):
            for x in before:
                for y in after:
                    out.append((x, y))
        return out


def parse(flow: str) -> FlowGraph:
    """Parse a flow string into a FlowGraph, or raise FlowSyntaxError.

    Enforced: non-empty; no empty segments around `>>`; well-formed `[...]`
    groups; valid ids; and no id appearing more than once across the whole flow.
    """
    if not isinstance(flow, str) or not flow.strip():
        raise FlowSyntaxError("`flow` must be a non-empty string.")

    raw_tokens = [t.strip() for t in flow.split(">>")]
    if any(t == "" for t in raw_tokens):
        raise FlowSyntaxError(
            f"malformed flow {flow!r}: empty segment around '>>' "
            "(leading, trailing, or doubled '>>')."
        )

    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        ids = _parse_token(tok, flow)
        for sid in ids:
            if sid in seen:
                raise FlowSyntaxError(
                    f"stage {sid!r} appears more than once in flow {flow!r}."
                )
            seen.add(sid)
        groups.append(tuple(ids))

    return FlowGraph(groups=tuple(groups))


def _parse_token(tok: str, flow: str) -> list[str]:
    """A flow token is either a single id or a bracketed parallel group."""
    if tok.startswith("[") or tok.endswith("]"):
        if not (tok.startswith("[") and tok.endswith("]")):
            raise FlowSyntaxError(
                f"malformed parallel group {tok!r} in flow {flow!r} "
                "(unbalanced brackets)."
            )
        inner = tok[1:-1].strip()
        if not inner:
            raise FlowSyntaxError(f"empty parallel group '[]' in flow {flow!r}.")
        ids = [p.strip() for p in inner.split(",")]
        if any(p == "" for p in ids):
            raise FlowSyntaxError(
                f"empty id in parallel group {tok!r} in flow {flow!r} "
                "(trailing or doubled comma)."
            )
        for sid in ids:
            _validate_id(sid, flow)
        return ids

    _validate_id(tok, flow)
    return [tok]


def _validate_id(sid: str, flow: str) -> None:
    if not _ID_RE.match(sid):
        raise FlowSyntaxError(f"invalid stage id {sid!r} in flow {flow!r}.")
