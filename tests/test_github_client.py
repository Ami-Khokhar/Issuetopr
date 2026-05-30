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
