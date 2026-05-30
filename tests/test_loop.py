import json
from unittest.mock import MagicMock, patch
from agent.loop import run_loop, _parse_tool_call


def _make_response(content: str):
    msg = MagicMock()
    msg.content = content
    resp = MagicMock()
    resp.choices[0].message = msg
    return resp


def _call(tool: str, args: dict) -> str:
    return json.dumps({"tool": tool, "args": args})


def test_loop_returns_done_when_finish_called():
    with patch(
        "agent.loop.litellm.completion",
        return_value=_make_response(_call("finish", {"status": "done", "summary": "Fixed the bug"})),
    ):
        result = run_loop("groq/llama-3.3-70b-versatile", "context", lambda n, a: "", lambda p: None)

    assert result.status == "done"
    assert result.summary == "Fixed the bug"
    assert result.iterations == 1


def test_loop_returns_uncertain_when_finish_uncertain():
    with patch(
        "agent.loop.litellm.completion",
        return_value=_make_response(_call("finish", {"status": "uncertain", "summary": "Could not find the bug"})),
    ):
        result = run_loop("groq/llama-3.3-70b-versatile", "context", lambda n, a: "", lambda p: None)

    assert result.status == "uncertain"
    assert "Could not find" in result.summary


def test_loop_dispatches_tool_then_finishes():
    responses = [
        _make_response(_call("read_file", {"path": "foo.py"})),
        _make_response(_call("finish", {"status": "done", "summary": "Fixed it"})),
    ]

    dispatched = []
    def dispatch(name, args):
        dispatched.append((name, args))
        return "file contents"

    with patch("agent.loop.litellm.completion", side_effect=responses):
        result = run_loop("openai/gpt-4o", "context", dispatch, lambda p: None)

    assert result.status == "done"
    assert result.iterations == 2
    assert dispatched == [("read_file", {"path": "foo.py"})]


def test_loop_tracks_written_files():
    responses = [
        _make_response(_call("write_file", {"path": "main.py", "content": "x=1"})),
        _make_response(_call("finish", {"status": "done", "summary": "Done"})),
    ]

    written = []
    with patch("agent.loop.litellm.completion", side_effect=responses):
        run_loop("openai/gpt-4o", "context", lambda n, a: "ok", written.append)

    assert "main.py" in written


def test_loop_returns_uncertain_at_max_iterations():
    never_finish = _make_response(_call("read_file", {"path": "x.py"}))

    with patch("agent.loop.litellm.completion", return_value=never_finish):
        result = run_loop("openai/gpt-4o", "context", lambda n, a: "data", lambda p: None, max_iterations=3)

    assert result.status == "uncertain"
    assert result.iterations == 3
    assert "maximum iterations" in result.summary


def test_loop_returns_uncertain_on_unparseable_response():
    with patch("agent.loop.litellm.completion", return_value=_make_response("I give up, nothing to do here")):
        result = run_loop("openai/gpt-4o", "context", lambda n, a: "", lambda p: None)

    assert result.status == "uncertain"
    assert "I give up" in result.summary


def test_loop_handles_tool_dispatch_error_gracefully():
    responses = [
        _make_response(_call("read_file", {"path": "bad.py"})),
        _make_response(_call("finish", {"status": "done", "summary": "Recovered"})),
    ]

    def dispatch(name, args):
        if name == "read_file":
            raise ValueError("File too large")
        return "ok"

    with patch("agent.loop.litellm.completion", side_effect=responses):
        result = run_loop("openai/gpt-4o", "context", dispatch, lambda p: None)

    assert result.status == "done"


def test_parse_direct_json():
    out = _parse_tool_call('{"tool": "read_file", "args": {"path": "x.py"}}')
    assert out == {"tool": "read_file", "args": {"path": "x.py"}}


def test_parse_llama_native_function_tag():
    raw = '<function=grep_code{"pattern": "grep_files"}></function>'
    out = _parse_tool_call(raw)
    assert out == {"tool": "grep_code", "args": {"pattern": "grep_files"}}


def test_parse_llama_function_tag_with_inner_args():
    raw = '<function=read_file>{"path": "agent/search.py"}</function>'
    out = _parse_tool_call(raw)
    assert out == {"tool": "read_file", "args": {"path": "agent/search.py"}}


def test_parse_fenced_json():
    raw = 'Here is my call:\n```json\n{"tool": "list_directory", "args": {"path": ""}}\n```'
    out = _parse_tool_call(raw)
    assert out == {"tool": "list_directory", "args": {"path": ""}}


def test_parse_returns_none_for_garbage():
    assert _parse_tool_call("Just some prose with no tool call.") is None
    assert _parse_tool_call("") is None
