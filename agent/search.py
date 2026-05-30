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
                ["grep", "-rF", *include_flags, keyword, str(repo_path)],
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
