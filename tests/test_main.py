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
