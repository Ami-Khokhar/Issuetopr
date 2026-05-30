import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional
import litellm


@dataclass
class LoopResult:
    status: str
    summary: str
    iterations: int
    changed_files: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are a bug-fix agent. You have been given a GitHub issue describing a bug.

Your job is to:
1. Understand the bug from the issue.
2. Locate the relevant source files using your tools.
3. Make the minimal code change to fix the bug.
4. Verify the fix by running the test suite.
5. Call finish when done.

Work methodically. Prefer small, targeted changes. Do not refactor unrelated code.
If you cannot confidently fix the bug, call finish with status='uncertain' and explain why.

## Available Tools

You interact with the codebase by emitting a tool call. To call a tool, respond with ONLY a
single JSON object on its own line, with no surrounding prose, no explanation, no markdown.

Format: {"tool": "TOOL_NAME", "args": {...}}

Available tools:
- read_file       args: {"path": "relative/path.py"}
- write_file      args: {"path": "relative/path.py", "content": "full new file contents"}
- grep_code       args: {"pattern": "regex", "path": "optional/subdir"}
- list_directory  args: {"path": "optional/subdir"}
- run_shell       args: {"command": "pytest"} - allowed prefixes only: pytest, python -m pytest, npm test, npm run test, go test, cargo test, make test
- finish          args: {"status": "done" | "uncertain", "summary": "what you did or why uncertain"}

## Response Rules

- Your entire response MUST be a single JSON object: {"tool": "...", "args": {...}}
- No prose. No explanation. No markdown code fences. No multiple tool calls.
- After each call you will see "OBSERVATION: <result>" - then emit the next tool call.
- When the fix is complete and tests pass, call finish with status="done".
"""


_FN_TAG_INLINE_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*?\})\s*>", re.DOTALL)
_FN_TAG_BODY_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)\s*>(\{.*?\})\s*</function>", re.DOTALL)
_FENCED_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _escape_string_newlines(text: str) -> str:
    """Escape raw newlines, tabs, and CRs that appear inside JSON string literals.

    LLMs often emit JSON where multi-line strings (e.g. file content for write_file)
    contain literal newlines instead of \\n. That violates JSON spec. This walker
    re-escapes them so json.loads succeeds.
    """
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            out.append(c)
            out.append(text[i + 1])
            i += 2
            continue
        if c == '"':
            in_string = not in_string
            out.append(c)
            i += 1
            continue
        if in_string and c == "\n":
            out.append("\\n")
        elif in_string and c == "\r":
            out.append("\\r")
        elif in_string and c == "\t":
            out.append("\\t")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _try_load(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        try:
            obj = json.loads(_escape_string_newlines(s))
        except (json.JSONDecodeError, ValueError):
            return None
    return obj if isinstance(obj, dict) else None


def _normalize(name: str, args) -> Optional[dict]:
    if not isinstance(args, dict):
        return None
    return {"tool": name, "args": args}


def _parse_tool_call(content: str) -> Optional[dict]:
    """Extract a {tool, args} call from a free-form LLM response.

    Handles direct JSON, fenced JSON, Llama-style <function=NAME{...}> tags,
    and bare balanced-brace substrings.
    """
    if not content:
        return None
    text = content.strip()

    # 1. Direct JSON object: {"tool": ..., "args": ...}
    obj = _try_load(text)
    if obj and "tool" in obj:
        return _normalize(obj["tool"], obj.get("args", {}))

    # 2. Llama native: <function=NAME{...args}>  (args inside open tag)
    m = _FN_TAG_INLINE_RE.search(text)
    if m:
        args = _try_load(m.group(2)) or {}
        return _normalize(m.group(1), args)

    # 3. Llama native: <function=NAME>{...args}</function>  (args between tags)
    m = _FN_TAG_BODY_RE.search(text)
    if m:
        args = _try_load(m.group(2)) or {}
        return _normalize(m.group(1), args)

    # 4. Fenced JSON
    for fm in _FENCED_RE.finditer(text):
        obj = _try_load(fm.group(1))
        if obj and "tool" in obj:
            return _normalize(obj["tool"], obj.get("args", {}))

    # 5. First balanced {...} containing "tool"
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    obj = _try_load(text[start:i + 1])
                    if obj and "tool" in obj:
                        return _normalize(obj["tool"], obj.get("args", {}))
                    break
        start = text.find("{", start + 1)

    return None


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
        response = litellm.completion(model=model, messages=messages)
        content = response.choices[0].message.content or ""

        call = _parse_tool_call(content)

        if call is None:
            return LoopResult(
                status="uncertain",
                summary=content or "Agent stopped without producing a valid tool call",
                iterations=iteration,
            )

        name = call["tool"]
        args = call["args"]

        messages.append({"role": "assistant", "content": json.dumps(call)})

        if name == "finish":
            return LoopResult(
                status=args.get("status", "uncertain"),
                summary=args.get("summary", ""),
                iterations=iteration,
            )

        if name == "write_file":
            track_write(args.get("path", ""))

        try:
            result = tool_dispatch(name, args)
        except Exception as exc:
            result = f"Error: {exc}"

        messages.append({"role": "user", "content": f"OBSERVATION: {result}"})

    return LoopResult(
        status="uncertain",
        summary=f"Reached maximum iterations ({max_iterations}) without finishing.",
        iterations=max_iterations,
    )
