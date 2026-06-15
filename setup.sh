#!/usr/bin/env bash
# One-shot installer for the drop-in orchestrator.
#
# Run it from anywhere — it locates itself, builds a venv inside this folder,
# installs the package, and wires up the target project's .mcp.json:
#
#   ./orchestrator/setup.sh
#
# Then add ANTHROPIC_API_KEY to <project-root>/.env and restart Claude Code.
set -euo pipefail

ORCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$ORCH_DIR")"
VENV="$ORCH_DIR/.venv"

echo "Installing orchestrator into $VENV …"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install -e "$ORCH_DIR"

# Absolute venv python path — pyenv/uv auto-activation doesn't apply to the MCP
# subprocess spawn, so the bare "python" shim would fail to import the package.
ENTRY=$(cat <<EOF
"orchestrator": {
      "command": "$VENV/bin/python",
      "args": ["-m", "orchestrator.mcp_server"]
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
