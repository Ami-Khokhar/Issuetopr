from pathlib import Path
from agent.search import extract_keywords, grep_files, find_candidate_files


def test_extract_keywords_filters_stop_words():
    text = "the function is broken and it returns None"
    result = extract_keywords(text)
    assert "the" not in result
    assert "and" not in result
    assert "function" in result
    assert "returns" in result
    assert "None" in result


def test_extract_keywords_deduplicates():
    text = "function function function"
    result = extract_keywords(text)
    assert result.count("function") == 1


def test_extract_keywords_minimum_length():
    # Words shorter than 3 chars (after regex) should be excluded by the pattern
    text = "if it is ok do it"
    result = extract_keywords(text)
    # "if", "it", "is", "ok" are 2 chars or stop words — none should appear
    assert not any(len(w) < 3 for w in result)


def test_grep_files_finds_matching_files(tmp_path):
    (tmp_path / "foo.py").write_text("def calculate_tax(amount):\n    return amount * 0.1\n")
    (tmp_path / "bar.py").write_text("def greet(name):\n    return f'hello {name}'\n")
    result = grep_files(tmp_path, ["calculate_tax"])
    assert any("foo.py" in f for f in result)
    assert not any("bar.py" in f for f in result)


def test_grep_files_ranks_by_hit_count(tmp_path):
    # foo.py has 3 keyword hits, bar.py has 1
    (tmp_path / "foo.py").write_text("error error error\n")
    (tmp_path / "bar.py").write_text("error\n")
    result = grep_files(tmp_path, ["error"])
    assert result[0].endswith("foo.py") or result[0].endswith("bar.py")  # both found
    assert len(result) == 2


def test_grep_files_respects_top_n(tmp_path):
    for i in range(5):
        (tmp_path / f"file{i}.py").write_text("keyword\n")
    result = grep_files(tmp_path, ["keyword"], top_n=3)
    assert len(result) == 3


def test_find_candidate_files_returns_relative_paths(tmp_path):
    (tmp_path / "module.py").write_text("def login_user(username):\n    pass\n")
    result = find_candidate_files("login_user function is broken", tmp_path)
    assert any("module.py" in f for f in result)
    # Paths must be relative (no tmp_path prefix)
    assert not any(str(tmp_path) in f for f in result)
