# GitHub Issue → PR Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python agent that triggers from a GitHub Actions label event, reads a GitHub issue, uses a configurable LLM in a ReAct tool-calling loop to locate and fix a bug, runs tests, then opens a ready or draft PR.

**Architecture:** A linear pipeline in `agent/main.py` orchestrates six focused modules: issue fetching (`github_client.py`), repo cloning (`repo.py`), keyword search (`search.py`), tool implementations (`tools.py`), and the ReAct loop (`loop.py`). The loop is pure — it has no GitHub or git knowledge — making it independently testable. `litellm` routes LLM calls to any configured provider.

**Tech Stack:** Python 3.11+, litellm, PyGithub, GitPython, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies and build config |
| `.env.example` | Environment variable documentation |
| `agent/__init__.py` | Package marker |
| `agent/search.py` | Keyword extraction + grep → ranked file candidates |
| `agent/tools.py` | Tool implementations (read/write/grep/ls/shell) sandboxed to repo dir |
| `agent/loop.py` | ReAct agent loop: send messages → dispatch tools → repeat until `finish` |
| `agent/repo.py` | Clone repo to temp dir, git operations |
| `agent/github_client.py` | GitHub API: fetch issue, create PR, post comment |
| `agent/main.py` | Entry point: reads env vars, wires all components together |
| `.github/workflows/agent.yml` | Actions workflow triggered on `agent-fix` label |
| `tests/conftest.py` | Shared fixtures (tmp git repo) |
| `tests/test_search.py` | Unit tests for `search.py` |
| `tests/test_tools.py` | Unit tests for `tools.py` |
| `tests/test_loop.py` | Unit tests for `loop.py` (LLM mocked) |
| `tests/test_repo.py` | Integration tests for `repo.py` (uses local git) |
| `tests/test_github_client.py` | Unit tests for `github_client.py` (PyGithub mocked) |
| `tests/test_main.py` | Integration tests for `main.py` (all I/O mocked) |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `agent/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `.env.example`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "issuetopr"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "litellm>=1.40",
    "PyGithub>=2.3",
    "gitpython>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[tool.hatch.build.targets.wheel]
packages = ["agent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `agent/__init__.py` and `tests/__init__.py`**

Both files are empty. Run:

```bash
mkdir -p agent tests
touch agent/__init__.py tests/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
import subprocess
import pytest


@pytest.fixture
def tmp_repo(tmp_path):
    """Temporary directory initialized as a git repo with one commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text("test repo")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path
```

- [ ] **Step 4: Create `.env.example`**

```bash
# Required
GITHUB_TOKEN=your_github_personal_access_token
REPO_NAME=owner/repo-name
ISSUE_NUMBER=42

# LLM Configuration (litellm format: provider/model)
LLM_PROVIDER=anthropic/claude-sonnet-4-6

# Provider API Keys — set whichever you use
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional
MAX_ITERATIONS=15
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: installs litellm, PyGithub, gitpython, pytest, pytest-mock without errors.

- [ ] **Step 6: Verify pytest discovers no tests yet**

```bash
pytest --collect-only
```

Expected: `no tests ran` (or empty collection — no errors).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml agent/__init__.py tests/__init__.py tests/conftest.py .env.example
git commit -m "chore: project scaffolding"
```

---

## Task 2: `search.py` — Keyword Extraction and Grep

**Files:**
- Create: `tests/test_search.py`
- Create: `agent/search.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_search.py`:

```python
from pathlib import Path
from agent.search import extract_keywords, grep_files, find_candidate_files


def test_extract_keywords_filters_stop_words():
    text = "the function is broken and it returns None"
    result = extract_keywords(text)
    assert "the" not in result
    assert "and" not in result
    assert "function" in result
    assert "returns" in result
    assert "None" in result


def test_extract_keywords_deduplicates():
    text = "function function function"
    result = extract_keywords(text)
    assert result.count("function") == 1


def test_extract_keywords_minimum_length():
    # Words shorter than 3 chars (after regex) should be excluded by the pattern
    text = "if it is ok do it"
    result = extract_keywords(text)
    # "if", "it", "is", "ok" are 2 chars or stop words — none should appear
    assert not any(len(w) < 3 for w in result)


def test_grep_files_finds_matching_files(tmp_path):
    (tmp_path / "foo.py").write_text("def calculate_tax(amount):\n    return amount * 0.1\n")
    (tmp_path / "bar.py").write_text("def greet(name):\n    return f'hello {name}'\n")
    result = grep_files(tmp_path, ["calculate_tax"])
    assert any("foo.py" in f for f in result)
    assert not any("bar.py" in f for f in result)


def test_grep_files_ranks_by_hit_count(tmp_path):
    # foo.py has 3 keyword hits, bar.py has 1
    (tmp_path / "foo.py").write_text("error error error\n")
    (tmp_path / "bar.py").write_text("error\n")
    result = grep_files(tmp_path, ["error"])
    assert result[0].endswith("foo.py") or result[0].endswith("bar.py")  # both found
    assert len(result) == 2


def test_grep_files_respects_top_n(tmp_path):
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text("keyword\n")
    result = grep_files(tmp_path, ["keyword"], top_n=3)
    assert len(result) == 3


def test_find_candidate_files_returns_relative_paths(tmp_path):
    (tmp_path / "module.py").write_text("def login_user(username):\n    pass\n")
    result = find_candidate_files("login_user function is broken", tmp_path)
    assert any("module.py" in f for f in result)
    # Paths must be relative (no tmp_path prefix)
    assert not any(str(tmp_path) in f for f in result)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.search'`

- [ ] **Step 3: Implement `agent/search.py`**

```python
import re
import subprocess
from pathlib import Path

STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "with", "this", "that", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "could", "may", "might", "can",
    "from", "by", "as", "we", "you", "he", "she", "they", "my", "our",
    "your", "his", "her", "their", "its", "also", "when", "where",
    "which", "who", "how", "if", "then", "else", "return", "true", "false",
}


def extract_keywords(text: str) -> list[str]:
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{2,}', text)
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        lower = word.lower()
        if lower not in STOP_WORDS and lower not in seen:
            seen.add(lower)
            result.append(word)
    return result


def grep_files(repo_path: Path, keywords: list[str], top_n: int = 20) -> list[str]:
    extensions = ["*.py", "*.js", "*.ts", "*.go", "*.java", "*.rb", "*.rs", "*.cpp", "*.c", "*.h"]
    include_flags = [flag for ext in extensions for flag in ("--include", ext)]
    hits: dict[str, int] = {}
    for keyword in keywords:
        try:
            proc = subprocess.run(
                ["grep", "-rl", *include_flags, keyword, str(repo_path)],
                capture_output=True, text=True, timeout=30,
            )
            for path in proc.stdout.strip().splitlines():
                path = path.strip()
                if path:
                    try:
                        rel = str(Path(path).relative_to(repo_path))
                        hits[rel] = hits.get(rel, 0) + 1
                    except ValueError:
                        continue
        except subprocess.TimeoutExpired:
            continue
    sorted_hits = sorted(hits.items(), key=lambda x: x[1], reverse=True)
    return [path for path, _ in sorted_hits[:top_n]]


def find_candidate_files(issue_text: str, repo_path: Path, top_n: int = 20) -> list[str]:
    keywords = extract_keywords(issue_text)
    return grep_files(repo_path, keywords, top_n)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_search.py -v
```

Expected: all 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agent/search.py tests/test_search.py
git commit -m "feat: add keyword extraction and grep file search"
```

---

## Task 3: `tools.py` — Tool Implementations

**Files:**
- Create: `tests/test_tools.py`
- Create: `agent/tools.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tools.py`:

```python
import pytest
from pathlib import Path
from agent.tools import Tools, ToolError


@pytest.fixture
def tools(tmp_path):
    return Tools(tmp_path)


def test_read_file_returns_content(tools, tmp_path):
    (tmp_path / "hello.py").write_text("print('hello')")
    assert tools.read_file("hello.py") == "print('hello')"


def test_read_file_raises_on_missing(tools):
    with pytest.raises(ToolError, match="File not found"):
        tools.read_file("nonexistent.py")


def test_write_file_creates_file(tools, tmp_path):
    result = tools.write_file("new_file.py", "x = 1\n")
    assert "new_file.py" in result
    assert (tmp_path / "new_file.py").read_text() == "x = 1\n"


def test_write_file_creates_nested_directory(tools, tmp_path):
    tools.write_file("subdir/deep/module.py", "pass\n")
    assert (tmp_path / "subdir" / "deep" / "module.py").exists()


def test_write_file_blocked_outside_repo(tools):
    with pytest.raises(ToolError, match="outside the repository"):
        tools.write_file("../escape.py", "malicious")


def test_grep_code_finds_matches(tools, tmp_path):
    (tmp_path / "source.py").write_text("def calculate(x):\n    return x * 2\n")
    result = tools.grep_code("calculate")
    assert "source.py" in result
    assert "calculate" in result


def test_grep_code_no_matches(tools, tmp_path):
    (tmp_path / "source.py").write_text("def add(x, y):\n    return x + y\n")
    result = tools.grep_code("zzznotfound")
    assert "No matches found" in result


def test_list_directory_lists_files(tools, tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "subdir").mkdir()
    result = tools.list_directory("")
    assert "a.py" in result
    assert "b.py" in result
    assert "subdir" in result


def test_list_directory_subdirectory(tools, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("")
    result = tools.list_directory("pkg")
    assert "mod.py" in result


def test_run_shell_allowed_command(tools, tmp_path):
    (tmp_path / "test_dummy.py").write_text("def test_pass():\n    assert True\n")
    result = tools.run_shell("pytest test_dummy.py -v")
    assert "test_pass" in result or "passed" in result


def test_run_shell_disallowed_command_returns_error(tools):
    result = tools.run_shell("rm -rf /")
    assert "Error" in result
    assert "not allowed" in result


def test_run_shell_disallowed_command_does_not_execute(tools, tmp_path):
    sentinel = tmp_path / "deleted.txt"
    sentinel.write_text("still here")
    tools.run_shell(f"rm {sentinel}")
    assert sentinel.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tools'`

- [ ] **Step 3: Implement `agent/tools.py`**

```python
import subprocess
from pathlib import Path


ALLOWED_COMMAND_PREFIXES = [
    "pytest",
    "python -m pytest",
    "npm test",
    "npm run test",
    "go test",
    "cargo test",
    "make test",
]


class ToolError(Exception):
    pass


class Tools:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()

    def _safe_path(self, path: str) -> Path:
        resolved = (self.repo_path / path).resolve()
        if not str(resolved).startswith(str(self.repo_path)):
            raise ToolError(f"Path {path!r} is outside the repository")
        return resolved

    def read_file(self, path: str) -> str:
        p = self._safe_path(path)
        if not p.exists():
            raise ToolError(f"File not found: {path}")
        return p.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    def grep_code(self, pattern: str, path: str = "") -> str:
        search_path = self._safe_path(path) if path else self.repo_path
        proc = subprocess.run(
            ["grep", "-rn", pattern, str(search_path)],
            capture_output=True, text=True, timeout=30,
        )
        lines = []
        for line in proc.stdout.strip().splitlines():
            if line.startswith(str(self.repo_path)):
                line = line[len(str(self.repo_path)) + 1:]
            lines.append(line)
        return "\n".join(lines) if lines else "No matches found"

    def list_directory(self, path: str = "") -> str:
        p = self._safe_path(path) if path else self.repo_path
        if not p.exists():
            raise ToolError(f"Path not found: {path}")
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = [
            f"[dir]  {e.name}" if e.is_dir() else f"[file] {e.name}"
            for e in entries
        ]
        return "\n".join(lines) if lines else "(empty directory)"

    def run_shell(self, command: str) -> str:
        cmd_stripped = command.strip()
        allowed = any(
            cmd_stripped.lower().startswith(prefix.lower())
            for prefix in ALLOWED_COMMAND_PREFIXES
        )
        if not allowed:
            return (
                f"Error: Command not allowed. "
                f"Allowed prefixes: {', '.join(ALLOWED_COMMAND_PREFIXES)}"
            )
        try:
            proc = subprocess.run(
                cmd_stripped,
                shell=True,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (proc.stdout + proc.stderr).strip()
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 5 minutes"
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: all 12 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add sandboxed tool implementations"
```

---

## Task 4: `loop.py` — ReAct Agent Loop

**Files:**
- Create: `tests/test_loop.py`
- Create: `agent/loop.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_loop.py`:

```python
import json
from unittest.mock import MagicMock, patch, call
from agent.loop import run_loop, LoopResult


def _make_tool_response(name: str, args: dict, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    resp = MagicMock()
    resp.choices[0].message = msg
    return resp


def _make_finish_response(status: str, summary: str, call_id: str = "call_fin"):
    return _make_tool_response("finish", {"status": status, "summary": summary}, call_id)


def _make_text_response(content: str):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    resp = MagicMock()
    resp.choices[0].message = msg
    return resp


def test_loop_returns_done_when_finish_called():
    dispatched = []

    def dispatch(name, args):
        dispatched.append(name)
        return "result"

    with patch("agent.loop.litellm.completion", return_value=_make_finish_response("done", "Fixed the bug")):
        result = run_loop("anthropic/claude-sonnet-4-6", "context", dispatch, lambda p: None)

    assert result.status == "done"
    assert result.summary == "Fixed the bug"
    assert result.iterations == 1


def test_loop_returns_uncertain_when_finish_uncertain():
    with patch("agent.loop.litellm.completion", return_value=_make_finish_response("uncertain", "Could not find the bug")):
        result = run_loop("anthropic/claude-sonnet-4-6", "context", lambda n, a: "", lambda p: None)

    assert result.status == "uncertain"
    assert "Could not find" in result.summary


def test_loop_dispatches_tool_then_finishes():
    responses = [
        _make_tool_response("read_file", {"path": "foo.py"}, "call_1"),
        _make_finish_response("done", "Fixed it", "call_2"),
    ]

    dispatched = []
    def dispatch(name, args):
        dispatched.append((name, args))
        return "file contents"

    with patch("agent.loop.litellm.completion", side_effect=responses):
        result = run_loop("openai/gpt-4o", "context", dispatch, lambda p: None)

    assert result.status == "done"
    assert result.iterations == 2
    assert dispatched == [("read_file", {"path": "foo.py"})]


def test_loop_tracks_written_files():
    responses = [
        _make_tool_response("write_file", {"path": "main.py", "content": "x=1"}, "call_w"),
        _make_finish_response("done", "Done", "call_f"),
    ]

    written = []
    def dispatch(name, args):
        return "ok"

    with patch("agent.loop.litellm.completion", side_effect=responses):
        result = run_loop("openai/gpt-4o", "context", dispatch, written.append)

    assert "main.py" in written


def test_loop_returns_uncertain_at_max_iterations():
    finish_never = _make_tool_response("read_file", {"path": "x.py"}, "call_x")

    with patch("agent.loop.litellm.completion", return_value=finish_never):
        result = run_loop("openai/gpt-4o", "context", lambda n, a: "data", lambda p: None, max_iterations=3)

    assert result.status == "uncertain"
    assert result.iterations == 3
    assert "maximum iterations" in result.summary


def test_loop_returns_uncertain_on_text_response_no_tool_calls():
    with patch("agent.loop.litellm.completion", return_value=_make_text_response("I give up")):
        result = run_loop("openai/gpt-4o", "context", lambda n, a: "", lambda p: None)

    assert result.status == "uncertain"
    assert "I give up" in result.summary


def test_loop_handles_tool_dispatch_error_gracefully():
    responses = [
        _make_tool_response("read_file", {"path": "bad.py"}, "call_e"),
        _make_finish_response("done", "Recovered", "call_f"),
    ]

    def dispatch(name, args):
        if name == "read_file":
            raise ValueError("File too large")
        return "ok"

    with patch("agent.loop.litellm.completion", side_effect=responses):
        result = run_loop("openai/gpt-4o", "context", dispatch, lambda p: None)

    assert result.status == "done"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_loop.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.loop'`

- [ ] **Step 3: Implement `agent/loop.py`**

```python
import json
from dataclasses import dataclass, field
from typing import Callable
import litellm


@dataclass
class LoopResult:
    status: str
    summary: str
    iterations: int
    changed_files: list[str] = field(default_factory=list)


SYSTEM_PROMPT = (
    "You are a bug-fix agent. You have been given a GitHub issue describing a bug.\n"
    "Your job is to:\n"
    "1. Understand the bug from the issue.\n"
    "2. Locate the relevant source files using your tools.\n"
    "3. Make the minimal code change to fix the bug.\n"
    "4. Verify the fix by running the test suite.\n"
    "5. Call finish() when done.\n\n"
    "Work methodically. Prefer small, targeted changes. Do not refactor unrelated code.\n"
    "If you cannot confidently fix the bug, call finish with status='uncertain' and explain why."
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the cloned repository",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative file path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": "Search for a pattern in the repository files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Optional subdirectory (default: repo root)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional subdirectory (default: repo root)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command (tests/linting only). "
                "Allowed prefixes: pytest, python -m pytest, npm test, npm run test, "
                "go test, cargo test, make test"
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that you are done. "
                "status='done' if fix is complete and tests pass, "
                "'uncertain' if you could not confidently fix the bug."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["done", "uncertain"]},
                    "summary": {"type": "string", "description": "What you did or why you're uncertain"},
                },
                "required": ["status", "summary"],
            },
        },
    },
]


def run_loop(
    model: str,
    initial_context: str,
    tool_dispatch: Callable[[str, dict], str],
    track_write: Callable[[str], None],
    max_iterations: int = 15,
) -> LoopResult:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_context},
    ]

    for iteration in range(1, max_iterations + 1):
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return LoopResult(
                status="uncertain",
                summary=message.content or "Agent stopped without calling finish()",
                iterations=iteration,
            )

        assistant_msg: dict = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ],
        }
        messages.append(assistant_msg)

        for tc in message.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if name == "finish":
                return LoopResult(
                    status=args["status"],
                    summary=args["summary"],
                    iterations=iteration,
                )

            if name == "write_file":
                track_write(args.get("path", ""))

            try:
                result = tool_dispatch(name, args)
            except Exception as exc:
                result = f"Error: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return LoopResult(
        status="uncertain",
        summary=f"Reached maximum iterations ({max_iterations}) without finishing.",
        iterations=max_iterations,
    )
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_loop.py -v
```

Expected: all 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agent/loop.py tests/test_loop.py
git commit -m "feat: add ReAct agent loop with tool dispatch"
```

---

## Task 5: `repo.py` — Git Operations

**Files:**
- Create: `tests/test_repo.py`
- Create: `agent/repo.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_repo.py`:

```python
import subprocess
from pathlib import Path
import pytest
from agent.repo import Repo


@pytest.fixture
def bare_repo(tmp_path):
    """A bare git repo that can be cloned from."""
    bare = tmp_path / "origin.git"
    bare.mkdir()
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    # Create a working clone to add an initial commit
    work = tmp_path / "work"
    subprocess.run(["git", "clone", str(bare), str(work)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=work, check=True, capture_output=True)
    (work / "README.md").write_text("init")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=work, check=True, capture_output=True)
    return bare


def test_repo_clones_and_creates_branch(bare_repo):
    repo = Repo(str(bare_repo), "agent/fix-issue-1")
    try:
        assert repo.path.exists()
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo.path, capture_output=True, text=True,
        )
        assert result.stdout.strip() == "agent/fix-issue-1"
    finally:
        repo.cleanup()


def test_changed_files_after_write(bare_repo):
    repo = Repo(str(bare_repo), "agent/fix-issue-2")
    try:
        (repo.path / "new_file.py").write_text("x = 1\n")
        changed = repo.changed_files()
        assert "new_file.py" in changed
    finally:
        repo.cleanup()


def test_changed_files_empty_when_nothing_changed(bare_repo):
    repo = Repo(str(bare_repo), "agent/fix-issue-3")
    try:
        assert repo.changed_files() == []
    finally:
        repo.cleanup()


def test_cleanup_removes_temp_directory(bare_repo):
    repo = Repo(str(bare_repo), "agent/fix-issue-4")
    path = repo.path
    repo.cleanup()
    assert not path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_repo.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.repo'`

- [ ] **Step 3: Implement `agent/repo.py`**

```python
import shutil
import subprocess
import tempfile
from pathlib import Path


class Repo:
    def __init__(self, clone_url: str, branch: str):
        self._tmpdir = tempfile.mkdtemp(prefix="issuetopr-")
        self.path = Path(self._tmpdir) / "repo"
        self.branch = branch
        subprocess.run(
            ["git", "clone", "--depth=1", clone_url, str(self.path)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=str(self.path), check=True, capture_output=True,
        )

    def changed_files(self) -> list[str]:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(self.path), capture_output=True, text=True,
        )
        files = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line:
                files.append(line[2:].strip())
        return sorted(files)

    def commit_and_push(self, message: str, github_token: str, repo_name: str) -> None:
        subprocess.run(
            ["git", "config", "user.email", "agent@issuetopr.bot"],
            cwd=str(self.path), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Issue to PR Agent"],
            cwd=str(self.path), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(self.path), check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(self.path), check=True, capture_output=True,
        )
        remote = f"https://x-access-token:{github_token}@github.com/{repo_name}.git"
        subprocess.run(
            ["git", "push", remote, self.branch],
            cwd=str(self.path), check=True, capture_output=True,
        )

    def cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_repo.py -v
```

Expected: all 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agent/repo.py tests/test_repo.py
git commit -m "feat: add git repo clone and management"
```

---

## Task 6: `github_client.py` — GitHub API

**Files:**
- Create: `tests/test_github_client.py`
- Create: `agent/github_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_github_client.py`:

```python
from unittest.mock import MagicMock, patch, call
import pytest
from agent.github_client import GitHubClient


@pytest.fixture
def mock_gh():
    with patch("agent.github_client.Github") as MockGithub:
        mock_instance = MagicMock()
        MockGithub.return_value = mock_instance
        yield mock_instance


def test_get_issue_returns_issue(mock_gh):
    mock_issue = MagicMock()
    mock_issue.title = "Bug: login fails"
    mock_gh.get_repo.return_value.get_issue.return_value = mock_issue

    client = GitHubClient("token123")
    issue = client.get_issue("owner/repo", 42)

    mock_gh.get_repo.assert_called_with("owner/repo")
    mock_gh.get_repo.return_value.get_issue.assert_called_with(42)
    assert issue.title == "Bug: login fails"


def test_get_default_branch(mock_gh):
    mock_gh.get_repo.return_value.default_branch = "main"
    client = GitHubClient("token123")
    assert client.get_default_branch("owner/repo") == "main"


def test_create_pr_ready(mock_gh):
    mock_repo = mock_gh.get_repo.return_value
    mock_repo.get_labels.return_value = []
    mock_pr = MagicMock()
    mock_pr.html_url = "https://github.com/owner/repo/pull/1"
    mock_repo.create_pull.return_value = mock_pr

    client = GitHubClient("token123")
    pr = client.create_pr(
        "owner/repo",
        title="fix: login bug (#42)",
        body="Summary\n\nCloses #42",
        head_branch="agent/fix-issue-42",
        base_branch="main",
        draft=False,
        labels=["agent-generated"],
    )

    mock_repo.create_pull.assert_called_once_with(
        title="fix: login bug (#42)",
        body="Summary\n\nCloses #42",
        head="agent/fix-issue-42",
        base="main",
        draft=False,
    )
    assert pr.html_url == "https://github.com/owner/repo/pull/1"


def test_create_pr_draft(mock_gh):
    mock_repo = mock_gh.get_repo.return_value
    mock_repo.get_labels.return_value = []
    mock_repo.create_pull.return_value = MagicMock()

    client = GitHubClient("token123")
    client.create_pr(
        "owner/repo",
        title="fix: login bug (#42)",
        body="body",
        head_branch="agent/fix-issue-42",
        base_branch="main",
        draft=True,
    )

    mock_repo.create_pull.assert_called_once_with(
        title="fix: login bug (#42)",
        body="body",
        head="agent/fix-issue-42",
        base="main",
        draft=True,
    )


def test_post_issue_comment(mock_gh):
    mock_issue = MagicMock()
    mock_gh.get_repo.return_value.get_issue.return_value = mock_issue

    client = GitHubClient("token123")
    client.post_issue_comment("owner/repo", 42, "Agent opened draft PR #5.")

    mock_issue.create_comment.assert_called_once_with("Agent opened draft PR #5.")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_github_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.github_client'`

- [ ] **Step 3: Implement `agent/github_client.py`**

```python
from github import Github, GithubException


class GitHubClient:
    def __init__(self, token: str):
        self._gh = Github(token)

    def get_issue(self, repo_name: str, issue_number: int):
        return self._gh.get_repo(repo_name).get_issue(issue_number)

    def get_default_branch(self, repo_name: str) -> str:
        return self._gh.get_repo(repo_name).default_branch

    def create_pr(
        self,
        repo_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
        draft: bool = False,
        labels: list[str] | None = None,
    ):
        repo = self._gh.get_repo(repo_name)
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=draft,
        )
        if labels:
            try:
                existing = {label.name for label in repo.get_labels()}
                for label in labels:
                    if label not in existing:
                        repo.create_label(label, "0075ca")
                pr.add_to_labels(*labels)
            except GithubException:
                pass
        return pr

    def post_issue_comment(self, repo_name: str, issue_number: int, body: str) -> None:
        self._gh.get_repo(repo_name).get_issue(issue_number).create_comment(body)
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_github_client.py -v
```

Expected: all 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agent/github_client.py tests/test_github_client.py
git commit -m "feat: add GitHub API client"
```

---

## Task 7: `main.py` — Orchestration

**Files:**
- Create: `tests/test_main.py`
- Create: `agent/main.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_main.py`:

```python
import os
from unittest.mock import MagicMock, patch, call
import pytest
from agent.loop import LoopResult


def _env(overrides=None):
    base = {
        "GITHUB_TOKEN": "tok",
        "REPO_NAME": "owner/repo",
        "ISSUE_NUMBER": "42",
        "LLM_PROVIDER": "anthropic/claude-sonnet-4-6",
        "MAX_ITERATIONS": "5",
    }
    if overrides:
        base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def patch_all():
    mock_issue = MagicMock()
    mock_issue.title = "Bug: divide by zero"
    mock_issue.body = "calling divide(0) crashes the app"
    mock_issue.number = 42

    mock_gh = MagicMock()
    mock_gh.get_issue.return_value = mock_issue
    mock_gh.get_default_branch.return_value = "main"
    mock_pr = MagicMock()
    mock_pr.html_url = "https://github.com/owner/repo/pull/99"
    mock_pr.number = 99
    mock_gh.create_pr.return_value = mock_pr

    mock_repo = MagicMock()
    mock_repo.path = MagicMock()

    with (
        patch("agent.main.GitHubClient", return_value=mock_gh) as pgh,
        patch("agent.main.Repo", return_value=mock_repo) as prepo,
        patch("agent.main.Tools") as ptools,
        patch("agent.main.find_candidate_files", return_value=["divide.py"]) as psearch,
        patch("agent.main.run_loop") as ploop,
    ):
        ptools.return_value.list_directory.return_value = "[file] divide.py"
        yield {
            "gh": mock_gh,
            "repo": mock_repo,
            "tools": ptools.return_value,
            "loop": ploop,
            "pr": mock_pr,
        }


def test_success_opens_ready_pr(patch_all, monkeypatch):
    for k, v in _env().items():
        monkeypatch.setenv(k, v)

    patch_all["repo"].changed_files.return_value = ["divide.py"]
    patch_all["loop"].return_value = LoopResult(
        status="done", summary="Fixed divide by zero", iterations=3
    )

    from agent.main import main
    main()

    patch_all["gh"].create_pr.assert_called_once()
    call_kwargs = patch_all["gh"].create_pr.call_args.kwargs
    assert call_kwargs["draft"] is False
    assert "agent-generated" in call_kwargs["labels"]
    assert "Closes #42" in call_kwargs["body"]


def test_uncertain_opens_draft_pr(patch_all, monkeypatch):
    for k, v in _env().items():
        monkeypatch.setenv(k, v)

    patch_all["repo"].changed_files.return_value = ["divide.py"]
    patch_all["loop"].return_value = LoopResult(
        status="uncertain", summary="Could not reproduce", iterations=2
    )

    from agent.main import main
    main()

    call_kwargs = patch_all["gh"].create_pr.call_args.kwargs
    assert call_kwargs["draft"] is True
    patch_all["gh"].post_issue_comment.assert_called_once()
    comment = patch_all["gh"].post_issue_comment.call_args.args[2]
    assert "draft PR" in comment


def test_done_with_no_changes_opens_draft_pr(patch_all, monkeypatch):
    for k, v in _env().items():
        monkeypatch.setenv(k, v)

    patch_all["repo"].changed_files.return_value = []
    patch_all["loop"].return_value = LoopResult(
        status="done", summary="Looks fixed already", iterations=1
    )

    from agent.main import main
    main()

    # No changes → uncertain → no PR, just a comment
    patch_all["gh"].create_pr.assert_not_called()
    patch_all["gh"].post_issue_comment.assert_called_once()


def test_uncertain_with_no_changes_posts_comment_only(patch_all, monkeypatch):
    for k, v in _env().items():
        monkeypatch.setenv(k, v)

    patch_all["repo"].changed_files.return_value = []
    patch_all["loop"].return_value = LoopResult(
        status="uncertain", summary="Gave up early", iterations=1
    )

    from agent.main import main
    main()

    patch_all["gh"].create_pr.assert_not_called()
    patch_all["gh"].post_issue_comment.assert_called_once()


def test_repo_cleanup_called_on_success(patch_all, monkeypatch):
    for k, v in _env().items():
        monkeypatch.setenv(k, v)

    patch_all["repo"].changed_files.return_value = ["file.py"]
    patch_all["loop"].return_value = LoopResult(
        status="done", summary="Fixed", iterations=1
    )

    from agent.main import main
    main()

    patch_all["repo"].cleanup.assert_called_once()


def test_repo_cleanup_called_on_error(patch_all, monkeypatch):
    for k, v in _env().items():
        monkeypatch.setenv(k, v)

    patch_all["loop"].side_effect = RuntimeError("LLM error")

    from agent.main import main
    with pytest.raises(RuntimeError, match="LLM error"):
        main()

    patch_all["repo"].cleanup.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.main'`

- [ ] **Step 3: Implement `agent/main.py`**

```python
import os
from agent.github_client import GitHubClient
from agent.repo import Repo
from agent.search import find_candidate_files
from agent.tools import Tools
from agent.loop import run_loop


def _build_context(issue_title: str, issue_body: str, candidates: list[str], root_listing: str) -> str:
    cand_text = "\n".join(candidates) if candidates else "(none found)"
    return (
        f"## GitHub Issue\n\n"
        f"**Title:** {issue_title}\n\n"
        f"**Body:**\n{issue_body}\n\n"
        f"---\n\n"
        f"## File Candidates (from keyword search)\n\n{cand_text}\n\n"
        f"---\n\n"
        f"## Repository Root\n\n{root_listing}\n\n"
        f"---\n\n"
        f"Start by reading the most relevant file(s). Then find and fix the bug."
    )


def main() -> None:
    github_token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["REPO_NAME"]
    issue_number = int(os.environ["ISSUE_NUMBER"])
    model = os.environ["LLM_PROVIDER"]
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "15"))

    gh = GitHubClient(github_token)
    issue = gh.get_issue(repo_name, issue_number)
    default_branch = gh.get_default_branch(repo_name)
    branch_name = f"agent/fix-issue-{issue_number}"
    clone_url = f"https://x-access-token:{github_token}@github.com/{repo_name}.git"

    repo = Repo(clone_url, branch_name)
    try:
        tools = Tools(repo.path)
        candidates = find_candidate_files(f"{issue.title}\n{issue.body or ''}", repo.path)
        root_listing = tools.list_directory("")
        context = _build_context(issue.title, issue.body or "", candidates, root_listing)

        written_files: list[str] = []

        def dispatch(name: str, args: dict) -> str:
            if name == "read_file":
                return tools.read_file(args["path"])
            if name == "write_file":
                return tools.write_file(args["path"], args["content"])
            if name == "grep_code":
                return tools.grep_code(args["pattern"], args.get("path", ""))
            if name == "list_directory":
                return tools.list_directory(args.get("path", ""))
            if name == "run_shell":
                return tools.run_shell(args["command"])
            return f"Unknown tool: {name}"

        result = run_loop(model, context, dispatch, written_files.append, max_iterations)

        changed = repo.changed_files()

        if result.status == "done" and not changed:
            result.status = "uncertain"
            result.summary = "Agent reported done but made no file changes. " + result.summary

        changed_md = "\n".join(f"- `{f}`" for f in changed) if changed else "_(no files changed)_"

        if result.status == "done":
            commit_msg = f"fix: {issue.title[:60]} (#{issue_number})"
            repo.commit_and_push(commit_msg, github_token, repo_name)
            body = (
                f"## Summary\n\n{result.summary}\n\n"
                f"## Changed Files\n\n{changed_md}\n\n"
                f"Closes #{issue_number}"
            )
            pr = gh.create_pr(
                repo_name,
                title=f"fix: {issue.title} (#{issue_number})",
                body=body,
                head_branch=branch_name,
                base_branch=default_branch,
                draft=False,
                labels=["agent-generated"],
            )
            print(f"Opened PR: {pr.html_url}")
        elif changed:
            commit_msg = f"wip: partial fix for #{issue_number}"
            repo.commit_and_push(commit_msg, github_token, repo_name)
            body = (
                f"## Agent Status: Needs Review\n\n{result.summary}\n\n"
                f"## Changed Files\n\n{changed_md}\n\n"
                f"> Draft PR opened by agent. Please review and complete the fix."
            )
            pr = gh.create_pr(
                repo_name,
                title=f"fix: {issue.title} (#{issue_number})",
                body=body,
                head_branch=branch_name,
                base_branch=default_branch,
                draft=True,
                labels=["agent-generated"],
            )
            gh.post_issue_comment(
                repo_name, issue_number,
                f"Agent opened draft PR #{pr.number} — needs human review. {pr.html_url}",
            )
            print(f"Opened draft PR: {pr.html_url}")
        else:
            gh.post_issue_comment(
                repo_name, issue_number,
                f"Agent could not fix this issue: {result.summary}",
            )
            print("No changes made; posted comment on issue.")
    finally:
        repo.cleanup()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest tests/test_main.py -v
```

Expected: all 6 tests PASSED.

- [ ] **Step 5: Run the full test suite**

```bash
pytest -v
```

Expected: all tests PASSED across all modules.

- [ ] **Step 6: Commit**

```bash
git add agent/main.py tests/test_main.py
git commit -m "feat: add orchestration entry point"
```

---

## Task 8: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/agent.yml`

- [ ] **Step 1: Create the workflow directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Create `.github/workflows/agent.yml`**

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
      - name: Checkout agent code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -e .

      - name: Run agent
        run: python -m agent.main
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          REPO_NAME: ${{ github.repository }}
          LLM_PROVIDER: ${{ vars.LLM_PROVIDER }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          MAX_ITERATIONS: ${{ vars.MAX_ITERATIONS || '15' }}
```

- [ ] **Step 3: Verify the workflow YAML is valid**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/agent.yml'))" && echo "YAML valid"
```

Expected: `YAML valid`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/agent.yml
git commit -m "feat: add GitHub Actions workflow for agent-fix label"
```

---

## Task 9: Final Verification

- [ ] **Step 1: Run all tests**

```bash
pytest -v
```

Expected: all tests PASSED (no failures, no errors).

- [ ] **Step 2: Verify the module runs without import errors**

```bash
python -c "from agent.main import main; print('import OK')"
```

Expected: `import OK`

- [ ] **Step 3: Verify the entry point is invocable (env vars missing = KeyError, not ImportError)**

```bash
python -m agent.main 2>&1 | head -5 || true
```

Expected: output contains `KeyError: 'GITHUB_TOKEN'` — confirming the module loads and reaches env var reading before failing.

- [ ] **Step 4: Review the project structure**

```bash
find . -not -path './.git/*' -not -path './__pycache__/*' | sort
```

Expected output includes:
```
./.env.example
./.github/workflows/agent.yml
./agent/__init__.py
./agent/github_client.py
./agent/loop.py
./agent/main.py
./agent/repo.py
./agent/search.py
./agent/tools.py
./docs/superpowers/plans/2026-05-30-issue-to-pr-agent.md
./docs/superpowers/specs/2026-05-30-issue-to-pr-agent-design.md
./pyproject.toml
./tests/__init__.py
./tests/conftest.py
./tests/test_github_client.py
./tests/test_loop.py
./tests/test_main.py
./tests/test_repo.py
./tests/test_search.py
./tests/test_tools.py
```

---

## Setup Checklist (for running in a real repo)

Before the agent can work end-to-end on GitHub:

1. Add `agent-fix` label to the target repo (Issues → Labels → New label)
2. Set `LLM_PROVIDER` as a repo variable (Settings → Variables → Actions)
3. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` as a repo secret (Settings → Secrets → Actions)
4. Push this repo to GitHub — the workflow activates automatically on the next labeled issue
