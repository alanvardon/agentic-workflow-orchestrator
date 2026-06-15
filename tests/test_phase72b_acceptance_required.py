"""Phase 72b — Task.acceptance_criteria is required.

Every vertical slice (Phase 70) must carry the observable-behaviour spec the
test-author writes its tests against, so the field is no longer optional. The
decomposer prompt already demanded it; this pins the model to match.
"""

import pytest
from pydantic import ValidationError

from orchestrator.agents.decompose import DecompositionResult, Task


def test_task_requires_acceptance_criteria():
    with pytest.raises(ValidationError):
        Task(id="t1", title="T", description="do the thing")


def test_task_with_acceptance_criteria_is_valid():
    t = Task(id="t1", title="T", description="do the thing", acceptance_criteria="it works")
    assert t.acceptance_criteria == "it works"


def test_decomposition_schema_version_bumped():
    r = DecompositionResult(
        tasks=[Task(id="t1", title="T", description="d", acceptance_criteria="w")]
    )
    assert r.schema_version == 2
