# issuetopr

An LLM agent that turns a labeled GitHub issue into a pull request. Label an issue with `agent-fix` and a GitHub Actions workflow clones the repo, hands the issue to an LLM with a small set of tools (read, write, grep, list, run tests), and opens a PR with the proposed fix.

## How it works

1. A GitHub Action fires when an issue is labeled `agent-fix` (`.github/workflows/agent.yml`).
2. `agent.main` reads the issue, shallow-clones the repo, and ranks candidate files via keyword grep (`agent/search.py`).
3. `agent.loop` drives a tool-use loop against any litellm-compatible model. The model emits one JSON tool call per turn:
   - `read_file`, `write_file`, `grep_code`, `list_directory`
   - `run_shell` — restricted to test runners (`pytest`, `npm test`, `go test`, `cargo test`, `make test`)
   - `finish` — returns a `done` or `uncertain` status with a summary
4. Based on the result, the agent either pushes the branch and opens a PR, opens a draft PR for partial fixes, or comments on the issue when no changes were made.

## Setup

### 1. Install the workflow in a target repo

Copy `.github/workflows/agent.yml` into the repo you want the agent to act on.

### 2. Configure repository secrets and variables

In the target repo's **Settings → Secrets and variables → Actions**:

**Secrets** (set whichever providers you use):
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`

`GITHUB_TOKEN` is supplied automatically by Actions.

**Variables**:
- `LLM_PROVIDER` — litellm-format model id (e.g. `anthropic/claude-sonnet-4-6`, `openai/gpt-4o`, `groq/llama-3.3-70b-versatile`)
- `MAX_ITERATIONS` — optional, defaults to `15`

### 3. Trigger the agent

Create or open an issue and add the `agent-fix` label. The workflow runs, and the agent posts a PR or a comment back on the issue.

## Local run

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in values
export $(grep -v '^#' .env | xargs)
python -m agent.main
```

Required environment: `GITHUB_TOKEN`, `REPO_NAME`, `ISSUE_NUMBER`, `LLM_PROVIDER`, and the API key for the chosen provider.

## Tests

```bash
pytest
```

## Project layout

```
agent/
  main.py          # entry point: orchestrates clone → loop → PR
  loop.py          # LLM tool-use loop and JSON parsing
  tools.py         # sandboxed file + shell tools exposed to the LLM
  search.py        # keyword-based candidate file ranking
  repo.py          # clone, branch, commit, push helpers
  github_client.py # PyGithub wrapper for issues, PRs, labels
.github/workflows/
  agent.yml        # GitHub Actions trigger
tests/             # pytest suite
```

## Safety notes

- `run_shell` is allowlisted to test-runner prefixes only.
- File tool paths are constrained to the cloned repo via path resolution checks.
- The agent works in a shallow clone in a temp directory; the original checkout is untouched.
- If the LLM claims `done` but produced no file changes, the status is downgraded to `uncertain` and a comment is posted instead of a PR.
