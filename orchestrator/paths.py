"""Project root discovery.

Walk up from CWD to find the nearest .git directory. This works for any
git repo regardless of where the orchestrator package is installed,
making it safe for both the drop-in case (orchestrator/ folder lives
inside the target repo) and the extracted case (orchestrator installed
globally, target repo lives elsewhere). __file__-based resolution would
break the extracted case once the package lives in site-packages.
"""

from pathlib import Path


# The orchestrator's own per-run workspace, directly under the project root. Its
# contents are artifacts (NOT project source), so anything matching a project glob
# inside it must be ignored — notably the verbatim test copies the Phase 77b
# evidence layer writes here, which would otherwise pollute the test_paths freeze
# hash and break the diff-gate.
ORCHESTRATOR_DIRNAME = ".orchestrator"


def find_project_root() -> Path:
    """Return the nearest ancestor directory containing a .git folder.

    Falls back to CWD if no .git is found (e.g. running outside a git repo
    in tests). Never raises.
    """
    current = Path.cwd().resolve()
    for path in [current, *current.parents]:
        if (path / ".git").exists():
            return path
    return current


def iter_test_files(globset, root: Path) -> list[Path]:
    """Every PROJECT file matching any project-root-relative glob in `globset`,
    sorted and de-duplicated, EXCLUDING the orchestrator's own workspace.

    The single source of truth for resolving a `test_paths` globset to real files:
    used by the diff-gate freeze hash and by the Phase 77b evidence copier, so both
    see exactly the same set. Files under `.orchestrator/` (run artifacts, incl.
    77b's verbatim test copies) are skipped — they are evidence, not project tests,
    and `**` would otherwise sweep them back in.
    """
    matched = set()
    for pattern in globset:
        for p in root.glob(pattern):
            if p.is_file() and ORCHESTRATOR_DIRNAME not in p.relative_to(root).parts:
                matched.add(p)
    return sorted(matched)
