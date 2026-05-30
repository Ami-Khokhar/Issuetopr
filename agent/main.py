import os
from agent.github_client import GitHubClient
from agent.repo import Repo
from agent.search import find_candidate_files
from agent.tools import Tools
from agent.loop import run_loop


def _build_context(issue_title: str, issue_body: str, candidates: list[str], root_listing: str) -> str:
    cand_text = "\n".join(candidates) if candidates else "(none found)"
    return (
        f"## Issue\n**Title:** {issue_title}\n\n**Body:**\n{issue_body}\n\n"
        f"## Candidate files (keyword search)\n{cand_text}\n\n"
        f"## Repo root\n{root_listing}\n\n"
        f"Read the most relevant file, then fix the bug."
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
        candidates = find_candidate_files(f"{issue.title}\n{issue.body or ''}", repo.path, top_n=10)
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
