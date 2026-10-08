"""Agent pieces that need no database."""

import pytest

from mpc.agents.personal import _paths_in
from mpc.agents.runtime import ToolError, normalize, validate
from mpc.agents.templates import TEMPLATES, TOOL_SPECS, template_for
from mpc.llm.base import Completion, ToolSpec, message_text
from mpc.llm.fake import ScriptedAdapter, ScriptExhausted

READ = ToolSpec("read_file", "", {"type": "object"})
FINISH = ToolSpec("finish", "", {"type": "object"})


def test_scripted_adapter_replays_steps_then_stops():
    a = ScriptedAdapter([{"tool": "read_file", "input": {"path": "x.py"}}, {"text": "done"}], "worker")
    c1 = a.tool_call([{"role": "user", "content": "go"}], [READ, FINISH])
    assert [t.name for t in c1.tool_calls] == ["read_file"] and c1.tool_calls[0].input == {"path": "x.py"}
    assert c1.raw_content[-1]["type"] == "tool_use"
    c2 = a.tool_call([], [READ, FINISH])
    assert c2.tool_calls == [] and c2.text == "done"
    c3 = a.tool_call([], [READ, FINISH])
    assert "exhausted" in c3.text and c3.tool_calls == []


def test_scripted_adapter_refuses_tools_the_template_lacks():
    a = ScriptedAdapter([{"tool": "write_file", "input": {}}], "reviewer")
    with pytest.raises(ScriptExhausted):
        a.tool_call([], [READ, FINISH])


def test_assistant_message_is_raw_content_unchanged():
    raw = [{"type": "thinking", "thinking": "", "signature": "abc"}, {"type": "text", "text": "hi"}]
    c = Completion(text="hi", model="m", provider="p", raw_content=raw)
    assert c.assistant_message() == {"role": "assistant", "content": raw}
    assert message_text({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]}) == "ok"


@pytest.mark.parametrize("bad", ["", "/etc/passwd", "C:/Windows/x", "../outside.py", "a/../../b", "."])
def test_normalize_rejects_escapes(bad):
    with pytest.raises(ToolError):
        normalize(bad)


def test_normalize_keeps_dotted_dirs():
    assert normalize("./backend/app/main.py") == "backend/app/main.py"
    assert normalize(".github/workflows/ci.yml") == ".github/workflows/ci.yml"
    assert normalize("backend\\app\\x.py") == "backend/app/x.py"


def test_validate_catches_syntax_errors():
    assert validate("backend/a.py", "def f(:\n") and not validate("backend/a.py", "def f():\n    return 1\n")
    assert validate("frontend/a.tsx", "export default function A( { return <div>; }")
    assert not validate("frontend/a.ts", "export const x: number = 1;\n")
    assert validate("openapi/x.yaml", "a: [1, 2") and not validate("openapi/x.yaml", "a: 1\n")


def test_templates_per_team_and_modes():
    assert len(TEMPLATES) == 6
    assert template_for("worker", "backend").write_paths == ("backend/**",)
    assert template_for("worker", "frontend").mode == "curated_fork"
    assert template_for("worker", "testing").mode == "isolated"
    assert "run_tests" in template_for("worker", "testing").tools
    assert "run_tests" not in template_for("worker", "backend").tools
    for team in ("frontend", "backend", "testing"):
        rv = template_for("reviewer", team)
        assert rv.mode == "isolated" and rv.write_paths == () and "write_file" not in rv.tools
    assert all(name in TOOL_SPECS for t in TEMPLATES.values() for name in t.tools)


def test_review_paths_parsed_from_command():
    assert _paths_in("please look at frontend/src/api/auth.ts, and backend/app/main.py.") == [
        "frontend/src/api/auth.ts",
        "backend/app/main.py",
    ]
