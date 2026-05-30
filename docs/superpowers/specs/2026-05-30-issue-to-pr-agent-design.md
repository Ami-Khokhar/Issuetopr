# GitHub Issue → PR Agent — Design Spec

**Date:** 2026-05-30  
**Status:** Approved

---

## Overview

An autonomous bug-fix agent that monitors a GitHub repository for issues labeled `agent-fix`, reads the issue, clones the repo, iteratively explores and edits files using a ReAct-style tool-calling loop, runs the test suite, and opens a pull request with the result. If tests fail or the agent is uncertain, a draft PR is opened with an explanation.

---

## Scope

- **In scope:** Bug fix issues only. The agent makes targeted code changes to fix a described bug.
- **Out of scope:** New features, dependency upgrades, refactors, multi-repo issues.

---

## Architecture

```
GitHub Issue labeled "agent-fix"
        │
        ▼
[GitHub Actions Workflow — .github/workflows/agent.yml]
        │  env: GITHUB_TOKEN, LLM_PROVIDER, <provider_api_key>
        ▼
[agent/main.py]  — orchestrates the full pipeline
    ├── 1. Fetch issue (title, body, repo) via GitHub API
    ├── 2. Clone repo to temp directory
    ├── 3. Extract keywords from issue → grep repo → build file candidate list
    ├── 4. ReAct agent loop (loop.py)
    │       Tools: read_file, write_file, grep_code, list_directory, run_shell, finish
    │       LLM: configured via LLM_PROVIDER (litellm)
    ├── 5. finish(status="done")   → commit + open ready PR
    └── 6. finish(status="uncertain") or timeout → commit + open draft PR + comment on issue
```

---

## File Structure

```
issuetopr/
├── agent/
│   ├── main.py           # Entry point: reads env vars, runs pipeline
│   ├── github_client.py  # GitHub API: fetch issue, create branch, open PR, post comment
│   ├── repo.py           # Clone repo, manage temp dir, git commit/push
│   ├── search.py         # Keyword extraction from issue + grep to produce file candidates
│   ├── tools.py          # Tool implementations: read_file, write_file, grep_code, list_directory, run_shell
│   └── loop.py           # ReAct agent loop: LLM call → tool dispatch → observe → repeat
├── .github/
│   └── workflows/
│       └── agent.yml     # Triggered on issue label "agent-fix"
├── pyproject.toml        # Dependencies: litellm, PyGithub, gitpython
└── .env.example          # Required env var documentation
```

### Component boundaries

- `loop.py` is pure: it accepts a tool registry, LLM config, and initial context, and returns a structured result. It has no knowledge of GitHub or git.
- `github_client.py` owns all GitHub API calls. No other module touches the GitHub API.
- `tools.py` tools operate only within the cloned temp directory. No network access from within the loop.
- `repo.py` owns all git operations (clone, branch, commit, push).

---

## LLM Provider Abstraction

Uses [`litellm`](https://github.com/BerriAI/litellm) for provider-agnostic LLM calls.

**Configuration via environment variables:**

| Variable | Example value | Required |
|---|---|---|
| `LLM_PROVIDER` | `anthropic/claude-sonnet-4-6` | Yes |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | If using Anthropic |
| `OPENAI_API_KEY` | `sk-...` | If using OpenAI |

Any provider supported by litellm works without code changes.

---

## ReAct Agent Loop

### System prompt

```
You are a bug-fix agent. You have been given a GitHub issue describing a bug.
Your job is to:
1. Understand the bug from the issue.
2. Locate the relevant source files.
3. Make the minimal code change to fix the bug.
4. Verify the fix by running the test suite.
5. Call finish() when done.

Work methodically. Prefer small, targeted changes. Do not refactor unrelated code.
```

### Initial context (passed at loop start)

1. Issue title and body
2. File candidate list from keyword/grep search
3. Root directory listing (from `list_directory("")`)

### Tools

| Tool | Parameters | Description |
|---|---|---|
| `read_file` | `path: str` | Read a file in the cloned repo |
| `write_file` | `path: str, content: str` | Overwrite a file with new content |
| `grep_code` | `pattern: str, path: str = ""` | Search for a regex pattern across files |
| `list_directory` | `path: str = ""` | List files and directories at a path |
| `run_shell` | `command: str` | Run a shell command (tests/linting only — see safety rules) |
| `finish` | `status: "done" \| "uncertain", summary: str` | Signal completion or inability to fix |

### Loop termination

- `finish(status="done")` called → success path
- `finish(status="uncertain")` called → draft PR path
- `MAX_ITERATIONS` (default: 15) reached without `finish` → treated as `uncertain`

### `run_shell` safety

Allowlisted command prefixes (case-insensitive):
- `pytest`, `python -m pytest`
- `npm test`, `npm run test`
- `go test`
- `cargo test`
- `make test`

Any other command returns an error string to the LLM. Timeout: 5 minutes per call.

---

## Keyword/Grep Search (Step 3)

1. Extract significant words from the issue title and body (strip stop words, deduplicate).
2. Run `grep -r` across the repo for each keyword, collect matching file paths.
3. Deduplicate and rank by hit count.
4. Pass the top N (default: 20) as the "file candidate list" to the LLM at loop start.

This gives the LLM a warm start without requiring embeddings infrastructure.

---

## PR Creation

### Success path (`finish(status="done")`)

- Branch name: `agent/fix-issue-{issue_number}`
- PR title: `fix: {issue_title} (#{issue_number})`
- PR body: agent summary + changed files list + "Closes #{issue_number}"
- PR labels: `agent-generated`
- PR state: ready for review

### Uncertain path

- Same branch and title, but PR opened as **draft**
- PR body: agent summary explaining what was tried and where it got stuck
- Comment posted on original issue: `"Agent opened draft PR #{pr_number} — needs human review."`

---

## GitHub Actions Workflow

```yaml
name: Issue to PR Agent
on:
  issues:
    types: [labeled]

jobs:
  run-agent:
    if: github.event.label.name == 'agent-fix'
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e .
      - run: python -m agent.main
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          REPO_NAME: ${{ github.repository }}
          LLM_PROVIDER: ${{ vars.LLM_PROVIDER }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Error Handling Summary

| Situation | Behavior |
|---|---|
| Issue has no reproducible code path | Agent calls `finish(uncertain)` → draft PR |
| Test command not found | `run_shell` returns error; LLM tries alternatives |
| Disallowed shell command | `run_shell` returns error; LLM tries allowed variant |
| Tests fail after fix | Agent can retry; if still failing at `finish`, opens draft PR |
| Max iterations hit | Treated as `uncertain`; draft PR opened |
| GitHub API failure | Exception propagates; Actions job fails with log |
| LLM API failure | Exception propagates; Actions job fails with log |

---

## Dependencies

```toml
[project]
dependencies = [
    "litellm>=1.40",
    "PyGithub>=2.3",
    "gitpython>=3.1",
]
```

---

## Success Criteria

1. Labeling an issue `agent-fix` triggers the workflow within 30 seconds.
2. The agent opens a PR (ready or draft) for any issue it processes.
3. On a simple, well-described bug, the agent produces a working fix that passes tests.
4. On an ambiguous issue, the agent opens a draft PR with a useful explanation rather than crashing.
5. Disallowed shell commands are rejected gracefully without aborting the loop.
