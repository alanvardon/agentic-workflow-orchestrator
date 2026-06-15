# Agentic Workflow Orchestrator

Automates code changes end-to-end — from a plain-English request to an open PR.
You describe what you want, review the plan, and the orchestrator writes the code,
checks it, commits, and opens the PR.

The core idea: everything **deterministic** (git operations, branching, commits,
PRs, state persistence) is handled by Python. Everything that requires **judgment**
(writing the plan, editing files, reviewing the diff) is handled by Claude. The two
never mix.

Works on any git repo via per-project prompts and config.

---

## Why split it this way

AI is non-deterministic — the same prompt can produce different output on different
runs. That's fine for judgment calls, but a problem for operations that have one
right answer. An LLM handling git introduces real risk: wrong base branch,
accidental force-pushes, malformed commits, credentials in diffs. Keeping those
steps in plain Python means the failure surface is small and auditable.

- **Deterministic Python owns:** branch creation, commit, push, PR open, state
  persistence, scripted checks ([git_ops.py](orchestrator/git_ops.py),
  [workflow.py](orchestrator/workflow.py))
- **Claude owns:** planning, writing code, reviewing whether the diff matches the
  plan ([agents/](orchestrator/agents/))

The full spine is:
`verify clean tree → plan → decompose → per-task build (implement ⇄ QA) → docs → summarize → branch → commit → push → open PR`

Each step is checkpointed. If the process crashes mid-run it resumes from the last
completed step — no re-running expensive LLM calls from scratch.

---

## Dependencies

### System tools (not pip-installable — must already exist)

| Dependency | Why | Check |
|---|---|---|
| **Python ≥ 3.12** | uses `tomllib` and modern typing | `python3 --version` |
| **git** | the deterministic spine shells out to `git` | `git --version` |
| **GitHub CLI (`gh`)**, authenticated | the PR step runs `gh pr create` | `gh auth status` |
| **Node.js + the `claude` CLI** | the implementation & QA agents run inside a Claude Code subprocess via `claude_agent_sdk.query()`, which spawns the `claude` binary | `claude --version` |

> The `claude` CLI is the easy-to-miss one. `pip install` pulls the *Python*
> `claude-agent-sdk`, but that SDK drives the **Claude Code CLI** as a subprocess —
> Node + `claude` must be installed and on PATH separately.

### Python packages (installed by `pip install -e .`)

Runtime: `anthropic`, `langgraph`, `langgraph-checkpoint-sqlite`, `langsmith`,
`pydantic>=2`, `python-dotenv`, `aiosqlite`, `mcp`, `claude-agent-sdk`, `pyyaml`.
Dev (`pip install -e ".[dev]"`): `pytest`, `pytest-asyncio`, `ipython`.

### Credentials

| Var | Required? | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | **yes** | every model call |
| `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` | optional | tracing (set `LANGSMITH_TRACING=true` to enable) |

Provided via a `.env` at the project root (read from CWD by `load_dotenv()`) or
exported in your shell.

---

## Install (drop-in)

The orchestrator runs *against* a target git repo and lives as a folder inside it.
It is not published to PyPI — you clone it in.

```bash
cd ~/your-project                       # the repo you want to automate

git clone <this-repo-url> orchestrator  # drop it in
# optional: rm -rf orchestrator/.git    # freeze as a snapshot

python3 -m venv orchestrator/.venv      # isolated venv
orchestrator/.venv/bin/pip install -e orchestrator

printf 'ANTHROPIC_API_KEY=%s\n' "$YOUR_KEY" > .env   # credentials at the project root
```

Or run `./orchestrator/setup.sh` from the project root, which does the venv +
install and writes `.mcp.json` (or prints the snippet to merge).

---

## Use it in Claude Code (preferred)

Add an MCP server entry pointing at the **venv's** python by absolute path — pyenv/uv
auto-activation does NOT apply to MCP subprocess spawns, so the bare `python` shim
fails to find the package.

`.mcp.json` at the project root:
```json
{
  "mcpServers": {
    "orchestrator": {
      "command": "/abs/path/to/your-project/orchestrator/.venv/bin/python",
      "args": ["-m", "orchestrator.mcp_server"]
    }
  }
}
```

Restart Claude Code. Five tools become available:

| Tool | What it does |
|---|---|
| `implement_feature(request, approve_plan?, base_branch?)` | start a run; pauses at plan approval |
| `approve_plan(thread_id, response)` | `"yes"` to proceed, or feedback to revise the plan |
| `resume_run(thread_id, force?)` | continue a failed run after fixing the cause |
| `cancel_run(thread_id)` | graceful stop at the next step boundary |
| `run_status(thread_id)` | progress of a backgrounded run |

Sanity-check without Claude Code:
```bash
npx @modelcontextprotocol/inspector \
  /abs/path/to/orchestrator/.venv/bin/python -m orchestrator.mcp_server
```

### Or the CLI (good for a first smoke test)

```bash
cd ~/your-project
orchestrator/.venv/bin/implement-feature "add a tooltip explaining LTV"
```

---

## Customize per project

The orchestrator finds the project root by walking up to the nearest `.git`, then
reads project-specific overrides if present — none are required:

```
your-project/
├── .git/
├── .env                       # credentials (gitignore)
├── .mcp.json                  # MCP server config (commit)
├── orchestrator.toml          # pipeline / model / tool config (commit; optional)
├── CLAUDE.md                  # project rules; auto-picked-up by the impl agent
├── .orchestrator/             # runtime state, auto-created
│   ├── checkpoints.db             (gitignore)
│   ├── runs/ , runs.jsonl         (gitignore — audit artifacts)
│   ├── prompts/<step>.md          (commit — override built-in prompts)
│   ├── qa/NN-*.sh                 (commit — scripted QA gates, run before the LLM judge)
│   └── pre-hooks/NN-*.sh          (commit — pre-flight checks)
└── ...your code...
```

- **Prompts:** drop `planning.md` / `implementation.md` / `qa.md` / `docs.md` /
  `summarize.md` into `.orchestrator/prompts/` to replace the bundled defaults. A
  prompt's YAML frontmatter (`model`, `tools`) drives that step; extra keys
  (`name`, `description`, …) are ignored, so a downloaded Claude Code subagent drops
  in as-is.
- **Config:** `orchestrator.toml` tunes the pipeline, per-step models, per-agent
  tool profiles (`allowed_tools`), retry budgets, and approval gates. Omit it
  entirely for the default spine. See [orchestrator.example.toml](orchestrator.example.toml).
- **Tool profiles** are config-driven: implementation gets read/write, QA is
  read-only, and no agent ever touches git — the orchestrator owns that entirely.

---

## How it works

- **Approval gates** — plan approval, branch creation, post-implementation, on QA
  failure, and pre-PR are independently toggleable. Run fully supervised, fully
  autonomous, or anything in between.
- **Two-layer QA** — scripts in `.orchestrator/qa/` run first (deterministic; a
  non-zero exit aborts), then a read-only Claude agent reviews the diff against the
  plan and returns PASS/FAIL. On FAIL the build retries with the failure notes.
- **Audit trails** — each run gets a folder under `.orchestrator/runs/` (plan, QA
  verdict, token usage); every step's I/O is checkpointed to
  `.orchestrator/checkpoints.db`; optional LangSmith traces carry per-call cost
  ([tracing.py](orchestrator/tracing.py), [usage.py](orchestrator/usage.py)).
- **Recovery** — commit, push, and PR are separate checkpointed steps. If push
  fails (e.g. auth expired), the commit is preserved; fix the issue and
  `resume_run` continues from push.

---

## Verify a fresh install

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # the suite runs offline (no live model calls)
```

---

## Scope

**This is:** a working example of an agent implementation layer for one workflow (PR
creation), runnable on your own repos. **This isn't:** a general-purpose framework
(the spine is fixed; what changes per project is prompts, scripts, config, and the
`flow`), enterprise-ready, or a product.

**Known limits:** the QA judge shares a model family with the implementation agent,
so the scripted gates are the stronger check; one plan per request (no parallel
fan-out); the spine (clean-tree → … → PR) is hard-coded in Python while the
region between branch and summarize is config-driven (`flow` + `[stage.*]`).

---

## Troubleshooting

- **`spawn claude ENOENT` / agent never produces output** → the `claude` CLI isn't on
  the MCP subprocess's PATH. Install Claude Code; verify `claude --version`.
- **`ModuleNotFoundError: orchestrator`** → `.mcp.json` points at the system `python`,
  not the venv's. Use the absolute `.venv/bin/python` path.
- **`gh pr create failed: ... no auth`** → run `gh auth login`. The commit is
  preserved; `resume_run(thread_id)` continues from push.
- **Missing `ANTHROPIC_API_KEY`** → `.env` isn't where you run from. It's read from
  CWD; put it at the project root or export the var.
