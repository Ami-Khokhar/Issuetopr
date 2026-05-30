import subprocess
from pathlib import Path


ALLOWED_COMMAND_PREFIXES = [
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
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
