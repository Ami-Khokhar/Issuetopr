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
            cwd=str(self.path), check=True, capture_output=True, text=True,
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
