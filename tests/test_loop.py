import json
from unittest.mock import MagicMock, patch, call
from agent.loop import run_loop, LoopResult


def _make_tool_response(name: str, args: dict, call_id: str = "call_1"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    resp = MagicMock()
    resp.choices[0].message = msg
    return resp


def _make_finish_response(status: str, summary: str, call_id: str = "call_fin"):
    return _make_tool_response("finish", {"status": status, "summary": summary}, call_id)


def _make_text_response(content: str):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    resp = MagicMock()
    resp.choices[0].message = msg
    return resp


def test_loop_returns_done_when_finish_called():
    dispatched = []

    def dispatch(name, args):
        dispatched.append(name)
        return "result"

    with patch("agent.loop.litellm.completion", return_value=_make_finish_response("done", "Fixed the bug")):
        result = run_loop("anthropic/claude-sonnet-4-6", "context", dispatch, lambda p: None)

    assert result.status == "done"
    assert result.summary == "Fixed the bug"
    assert result.iterations == 1


def test_loop_returns_uncertain_when_finish_uncertain():
    with patch("agent.loop.litellm.completion", return_value=_make_finish_response("uncertain", "Could not find the bug")):
        result = run_loop("anthropic/claude-sonnet-4-6", "context", lambda n, a: "", lambda p: None)

    assert result.status == "uncertain"
    assert "Could not find" in result.summary


def test_loop_dispatches_tool_then_finishes():
    responses = [
        _make_tool_response("read_file", {"path": "foo.py"}, "call_1"),
        _make_finish_response("done", "Fixed it", "call_2"),
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
        _make_tool_response("write_file", {"path": "main.py", "content": "x=1"}, "call_w"),
        _make_finish_response("done", "Done", "call_f"),
    ]

    written = []
    def dispatch(name, args):
        return "ok"

    with patch("agent.loop.litellm.completion", side_effect=responses):
        result = run_loop("openai/gpt-4o", "context", dispatch, written.append)

    assert "main.py" in written


def test_loop_returns_uncertain_at_max_iterations():
    finish_never = _make_tool_response("read_file", {"path": "x.py"}, "call_x")

    with patch("agent.loop.litellm.completion", return_value=finish_never):
        result = run_loop("openai/gpt-4o", "context", lambda n, a: "data", lambda p: None, max_iterations=3)

    assert result.status == "uncertain"
    assert result.iterations == 3
    assert "maximum iterations" in result.summary


def test_loop_returns_uncertain_on_text_response_no_tool_calls():
    with patch("agent.loop.litellm.completion", return_value=_make_text_response("I give up")):
        result = run_loop("openai/gpt-4o", "context", lambda n, a: "", lambda p: None)

    assert result.status == "uncertain"
    assert "I give up" in result.summary


def test_loop_handles_tool_dispatch_error_gracefully():
    responses = [
        _make_tool_response("read_file", {"path": "bad.py"}, "call_e"),
        _make_finish_response("done", "Recovered", "call_f"),
    ]

    def dispatch(name, args):
        if name == "read_file":
            raise ValueError("File too large")
        return "ok"

    with patch("agent.loop.litellm.completion", side_effect=responses):
        result = run_loop("openai/gpt-4o", "context", dispatch, lambda p: None)

    assert result.status == "done"
