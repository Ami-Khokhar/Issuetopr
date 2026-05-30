from typing import Optional
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
        labels: Optional[list] = None,
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
