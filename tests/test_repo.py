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
