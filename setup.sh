#!/usr/bin/env bash
# One-shot installer for the drop-in orchestrator.
#
# Run it from anywhere — it locates itself, sets up a Python environment,
# installs the package, and wires up the target project's .mcp.json.
#
# Two environment styles:
#
#   # 1. Just run it — if pyenv is installed it ASKS for a virtualenv name
#   #    (press Enter to fall back to an in-folder orchestrator/.venv):
#   ./orchestrator/setup.sh
#
#   # 2. Skip the prompt by naming the env up front (scriptable / CI). The env
#   #    is created on PY_VERSION (default 3.12) if it doesn't exist yet:
#   PYENV_ENV=agentic-workflow-orchestrator ./orchestrator/setup.sh
#   PYENV_ENV=myenv PY_VERSION=3.12.8 ./orchestrator/setup.sh
#
# Then add ANTHROPIC_API_KEY to <project-root>/.env and restart Claude Code.
set -euo pipefail

ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$ORCH_DIR")"

# If PYENV_ENV wasn't passed, ask for it — but only when pyenv is installed AND
# we have a terminal to read from. Type a virtualenv name to install into it, or
# press Enter to fall back to the in-folder .venv. Non-interactive runs (no TTY,
# e.g. CI or a piped install) skip the prompt and use the default, so they keep
# working unattended.
if [ -z "${PYENV_ENV:-}" ] && [ -t 0 ] && command -v pyenv >/dev/null 2>&1; then
  existing="$(pyenv virtualenvs --bare 2>/dev/null | grep -v '/envs/' | tr '\n' ' ')"
  [ -n "$existing" ] && echo "Existing pyenv virtualenvs: $existing"
  printf 'Enter a pyenv virtualenv name to install into (blank = local .venv): '
  read -r PYENV_ENV
fi

if [ -n "${PYENV_ENV:-}" ]; then
  # ── pyenv virtualenv route ─────────────────────────────────────────────────
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "PYENV_ENV is set but 'pyenv' is not on PATH." >&2
    exit 1
  fi
  PY_VERSION="${PY_VERSION:-3.12}"
  if ! pyenv virtualenvs --bare 2>/dev/null | grep -qx "$PYENV_ENV"; then
    echo "Creating pyenv virtualenv '$PYENV_ENV' on Python $PY_VERSION …"
    pyenv virtualenv "$PY_VERSION" "$PYENV_ENV"
  else
    echo "Using existing pyenv virtualenv '$PYENV_ENV'."
  fi
  PYTHON="$(pyenv prefix "$PYENV_ENV")/bin/python"
else
  # ── in-folder venv route (default) ─────────────────────────────────────────
  VENV="$ORCH_DIR/.venv"
  echo "Creating venv at $VENV …"
  python3 -m venv "$VENV"
  PYTHON="$VENV/bin/python"
fi

echo "Installing orchestrator into $PYTHON …"
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install -e "$ORCH_DIR"

# Absolute interpreter path — pyenv/uv auto-activation doesn't apply to the MCP
# subprocess spawn, so the bare "python" shim would fail to import the package.
# cwd pins the run to PROJECT_ROOT, which is what find_project_root() walks up
# from — so the orchestrator branches/commits/PRs into THIS repo.
ENTRY=$(cat <<EOF
"orchestrator": {
      "command": "$PYTHON",
      "args": ["-m", "orchestrator.mcp_server"],
      "cwd": "$PROJECT_ROOT"
    }
EOF
)

MCP="$PROJECT_ROOT/.mcp.json"
if [ -f "$MCP" ]; then
  echo
  echo "An $MCP already exists — add this entry under \"mcpServers\" yourself:"
  echo
  echo "    $ENTRY"
else
  printf '{\n  "mcpServers": {\n    %s\n  }\n}\n' "$ENTRY" > "$MCP"
  echo "Wrote $MCP"
fi

echo
echo "Done. Next:"
echo "  1. Put ANTHROPIC_API_KEY in $PROJECT_ROOT/.env"
echo "  2. Restart Claude Code so it picks up the MCP server."
