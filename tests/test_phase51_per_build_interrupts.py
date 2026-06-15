"""Phase 51 — the build's human pauses move onto the build step's own config.

The build now carries `human_in_loop = { after_producer, on_gate_fail }`
(replacing the global [workflow.implementation]/[workflow.qa] flags). This covers
the model + load-time guard; the interrupt/resume flow is in
test_phase42_spine_gates.
"""

import pytest

from orchestrator.config import load_config
from orchestrator.manifest import BuildStep, HumanInLoopConfig, ManifestError, load_manifest


def _write(p, body):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_human_in_loop_defaults_off():
    b = BuildStep(produce=["implementation"], gate=["qa"], id="build")
    assert b.human_in_loop == HumanInLoopConfig()
    assert b.human_in_loop.after_producer is False
    assert b.human_in_loop.on_gate_fail is False


def test_build_human_in_loop_roundtrips(tmp_path):
    _write(
        tmp_path / "orchestrator.toml",
        """
[[steps.work]]
id      = "build"
type    = "build"
produce = ["implementation"]
gate    = ["qa"]
human_in_loop = { after_producer = true, on_gate_fail = true }
""",
    )
    build = load_manifest(project_root=tmp_path).for_seam("work")[0]
    assert isinstance(build, BuildStep)
    assert build.human_in_loop.after_producer is True
    assert build.human_in_loop.on_gate_fail is True


def test_build_human_in_loop_typo_rejected(tmp_path):
    # extra="forbid" on HumanInLoopConfig catches a typo'd key at load.
    _write(
        tmp_path / "orchestrator.toml",
        """
[[steps.work]]
id      = "build"
type    = "build"
produce = ["implementation"]
gate    = ["qa"]
human_in_loop = { after_producer = true, typo = true }
""",
    )
    with pytest.raises(ManifestError, match="(?i)extra|invalid step"):
        load_manifest(project_root=tmp_path)


@pytest.mark.parametrize("table", ["implementation", "qa"])
def test_v1_workflow_table_rejected(tmp_path, table):
    # The whole v1 [workflow.*] dialect (incl. its human_in_loop flags) is gone —
    # load_config fails loud with a migration message. The build's pauses now live
    # on the task-build stage's human_in_loop = { after_producer, on_gate_fail }.
    _write(
        tmp_path / "orchestrator.toml",
        f"[workflow.{table}]\nhuman_in_loop = true\n",
    )
    with pytest.raises(ValueError, match="v1 orchestrator.toml"):
        load_config(tmp_path / "orchestrator.toml")
