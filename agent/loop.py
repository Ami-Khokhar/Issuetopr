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
