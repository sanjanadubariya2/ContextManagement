"""The Anthropic adapter against a stubbed SDK client (no network, no key)."""

from anthropic.types.beta import BetaMessage

from mpc.llm.anthropic_provider import FALLBACK_BETA, AnthropicAdapter
from mpc.llm.base import ToolSpec


class _Stub:
    def __init__(self, message):
        self.message, self.calls = message, []
        self.beta = self
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return self.message


def _msg(content, stop="tool_use"):
    return BetaMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    })


def test_tool_call_request_shape_and_raw_content():
    content = [
        {"type": "text", "text": "Reading the routes first."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "backend/app/main.py"}},
    ]
    stub = _Stub(_msg(content))
    a = AnthropicAdapter(model="claude-opus-5-5", client=stub)
    tool = ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}})
    c = a.tool_call([{"role": "user", "content": "go"}], [tool], system="sys")

    sent = stub.calls[0]
    assert sent["model"] == "claude-opus-5-5" and sent["system"] == "sys"
    assert sent["betas"] == [FALLBACK_BETA] and sent["fallbacks"] == "default"
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["tool_choice"] == {"type": "auto"}  # forced tool choice is rejected on this model
    assert sent["tools"][0]["name"] == "read_file" and "input_schema" in sent["tools"][0]

    assert [t.name for t in c.tool_calls] == ["read_file"]
    assert c.tool_calls[0].input == {"path": "backend/app/main.py"}
    assert c.input_tokens == 120 and c.output_tokens == 30
    # The assistant turn goes back verbatim on the next request.
    assert c.assistant_message()["content"] == [
        {"type": "text", "text": "Reading the routes first."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "backend/app/main.py"}},
    ]


def test_small_model_skips_effort_and_fallbacks():
    stub = _Stub(_msg([{"type": "text", "text": "ok"}], stop="end_turn"))
    a = AnthropicAdapter(model="claude-haiku-4-5", client=stub)
    c = a.complete([{"role": "user", "content": "hi"}])
    sent = stub.calls[0]
    assert "output_config" not in sent and "fallbacks" not in sent
    assert c.text == "ok" and c.tool_calls == []
