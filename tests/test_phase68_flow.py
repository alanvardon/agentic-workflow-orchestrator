"""Phase 68 — flow-line parser tests.

Pure unit tests for orchestrator.flow.parse: ordering, parallel groups, and the
fail-loud syntax errors. No config, no workflow, no LLM.
"""

import pytest

from orchestrator.errors import FatalError
from orchestrator.flow import FlowGraph, FlowSyntaxError, parse


def test_simple_chain_orders_stages():
    g = parse("plan >> decompose >> task-build >> docs")
    assert g.ordered_ids() == ["plan", "decompose", "task-build", "docs"]
    assert g.ids() == {"plan", "decompose", "task-build", "docs"}


def test_singletons_are_one_id_groups():
    g = parse("a >> b")
    assert g.groups == (("a",), ("b",))


def test_parallel_group_kept_as_one_group():
    g = parse("plan >> [docs, gitleaks] >> summarize")
    assert g.groups == (("plan",), ("docs", "gitleaks"), ("summarize",))
    # ordered_ids flattens declared order
    assert g.ordered_ids() == ["plan", "docs", "gitleaks", "summarize"]


def test_edges_fan_out_and_join():
    g = parse("a >> [b, c] >> d")
    edges = set(g.edges())
    assert edges == {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}


def test_whitespace_is_tolerated():
    g = parse("  plan   >>   [ docs ,  gitleaks ]>>summarize ")
    assert g.ordered_ids() == ["plan", "docs", "gitleaks", "summarize"]


def test_hyphen_dot_underscore_ids_allowed():
    g = parse("task-build >> final.qa >> a_b")
    assert g.ordered_ids() == ["task-build", "final.qa", "a_b"]


def test_flowgraph_is_frozen():
    g = parse("a >> b")
    assert isinstance(g, FlowGraph)
    with pytest.raises(Exception):
        g.groups = ()  # type: ignore[misc]


# ── fail-loud syntax errors ──────────────────────────────────────────────────

def test_flow_syntax_error_is_a_fatal_error():
    # so the loader/MCP server classifies it as non-retriable config breakage
    assert issubclass(FlowSyntaxError, FatalError)


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_empty_flow_rejected(bad):
    with pytest.raises(FlowSyntaxError):
        parse(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "a >> >> b",   # doubled
        ">> a",        # leading
        "a >>",        # trailing
    ],
)
def test_empty_segments_rejected(bad):
    with pytest.raises(FlowSyntaxError):
        parse(bad)


def test_duplicate_stage_in_flow_rejected():
    with pytest.raises(FlowSyntaxError):
        parse("a >> b >> a")


def test_duplicate_across_parallel_group_rejected():
    with pytest.raises(FlowSyntaxError):
        parse("a >> [b, a]")


@pytest.mark.parametrize(
    "bad",
    [
        "a >> [b, c",     # unbalanced open
        "a >> b, c]",     # unbalanced close
        "a >> []",        # empty group
        "a >> [b,]",      # trailing comma
        "a >> [,b]",      # leading comma
        "a >> [b,,c]",    # doubled comma
    ],
)
def test_malformed_groups_rejected(bad):
    with pytest.raises(FlowSyntaxError):
        parse(bad)


@pytest.mark.parametrize("bad", ["a >> -b", "a >> .b", "a >> b!c", "a >> b c"])
def test_invalid_ids_rejected(bad):
    with pytest.raises(FlowSyntaxError):
        parse(bad)
