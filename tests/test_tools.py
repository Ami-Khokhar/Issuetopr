import pytest
from pathlib import Path
from agent.tools import Tools, ToolError


@pytest.fixture
def tools(tmp_path):
    return Tools(tmp_path)


def test_read_file_returns_content(tools, tmp_path):
    (tmp_path / "hello.py").write_text("print('hello')")
    assert tools.read_file("hello.py") == "print('hello')"


def test_read_file_raises_on_missing(tools):
    with pytest.raises(ToolError, match="File not found"):
        tools.read_file("nonexistent.py")


def test_write_file_creates_file(tools, tmp_path):
    result = tools.write_file("new_file.py", "x = 1\n")
    assert "new_file.py" in result
    assert (tmp_path / "new_file.py").read_text() == "x = 1\n"


def test_write_file_creates_nested_directory(tools, tmp_path):
    tools.write_file("subdir/deep/module.py", "pass\n")
    assert (tmp_path / "subdir" / "deep" / "module.py").exists()


def test_write_file_blocked_outside_repo(tools):
    with pytest.raises(ToolError, match="outside the repository"):
        tools.write_file("../escape.py", "malicious")


def test_grep_code_finds_matches(tools, tmp_path):
    (tmp_path / "source.py").write_text("def calculate(x):\n    return x * 2\n")
    result = tools.grep_code("calculate")
    assert "source.py" in result
    assert "calculate" in result


def test_grep_code_no_matches(tools, tmp_path):
    (tmp_path / "source.py").write_text("def add(x, y):\n    return x + y\n")
    result = tools.grep_code("zzznotfound")
    assert "No matches found" in result


def test_list_directory_lists_files(tools, tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "subdir").mkdir()
    result = tools.list_directory("")
    assert "a.py" in result
    assert "b.py" in result
    assert "subdir" in result


def test_list_directory_subdirectory(tools, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("")
    result = tools.list_directory("pkg")
    assert "mod.py" in result


def test_run_shell_allowed_command(tools, tmp_path):
    (tmp_path / "test_dummy.py").write_text("def test_pass():\n    assert True\n")
    result = tools.run_shell("python3 -m pytest test_dummy.py -v")
    assert "test_pass" in result or "passed" in result


def test_run_shell_disallowed_command_returns_error(tools):
    result = tools.run_shell("rm -rf /")
    assert "Error" in result
    assert "not allowed" in result


def test_run_shell_disallowed_command_does_not_execute(tools, tmp_path):
    sentinel = tmp_path / "deleted.txt"
    sentinel.write_text("still here")
    tools.run_shell(f"rm {sentinel}")
    assert sentinel.exists()
