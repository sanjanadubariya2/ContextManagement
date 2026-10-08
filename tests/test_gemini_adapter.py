"""The Gemini adapter against a stubbed google-genai client (no network, no key)."""

from google.genai import types

from mpc.llm.base import ToolSpec, tool_result
from mpc.llm.gemini_provider import GEMINI_BLOCK, GeminiAdapter, GeminiEmbedder


class _Models:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.responses.pop(0)

    def embed_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[3.0, 4.0]) for _ in contents])


class _Client:
    def __init__(self, responses=()):
        self.models = _Models(responses)


def _resp(parts, finish="STOP", prompt=100, out=20, thoughts=5):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts),
                                    finish_reason=types.FinishReason[finish])],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, candidates_token_count=out, thoughts_token_count=thoughts),
    )


READ = ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})


def test_tool_loop_round_trip_keeps_thought_signature():
    signed = types.Part(function_call=types.FunctionCall(name="read_file", args={"path": "backend/app/main.py"}),
                        thought_signature=b"sig-bytes")
    client = _Client([
        _resp([types.Part(text="Reading first."), signed]),
        _resp([types.Part(text="Done.")]),
    ])
    a = GeminiAdapter(model="gemini-3.8-flash", client=client)

    c1 = a.tool_call([{"role": "user", "content": "go"}], [READ], system="sys")
    sent = client.models.calls[0]
    cfg = sent["config"]
    assert sent["model"] == "gemini-3.8-flash" and cfg.system_instruction == "sys"
    assert cfg.automatic_function_calling.disable is True  # our loop runs the tools
    decl = cfg.tools[0].function_declarations[0]
    assert decl.name == "read_file" and decl.parameters_json_schema == READ.input_schema
    assert [t.name for t in c1.tool_calls] == ["read_file"]
    assert c1.tool_calls[0].input == {"path": "backend/app/main.py"}
    assert c1.stop_reason == "tool_use" and c1.text == "Reading first."
    assert c1.input_tokens == 100 and c1.output_tokens == 25  # thinking tokens are billed output

    messages = [
        {"role": "user", "content": "go"},
        c1.assistant_message(),
        {"role": "user", "content": [tool_result(c1.tool_calls[0], "app = FastAPI()")]},
    ]
    c2 = a.tool_call(messages, [READ])
    contents = client.models.calls[1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[1].thought_signature == b"sig-bytes"  # model turn sent back unchanged
    fr = contents[2].parts[0].function_response
    assert fr.name == "read_file" and fr.response == {"output": "app = FastAPI()"}
    assert c2.stop_reason == "end_turn" and c2.text == "Done." and c2.tool_calls == []
    assert c2.raw_content[0]["type"] == GEMINI_BLOCK


def test_tool_errors_and_foreign_history_translate():
    client = _Client([_resp([types.Part(text="ok")])])
    a = GeminiAdapter(model="m", client=client)
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},  # plain text from Postgres
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "no file x", "is_error": True}]},
    ]
    a.complete(history)
    contents = client.models.calls[0]["contents"]
    assert contents[1].role == "model" and contents[1].parts[0].text == "earlier answer"
    assert contents[2].parts[0].function_call.name == "read_file"
    assert contents[3].parts[0].function_response.response == {"error": "no file x"}


def test_safety_stop_is_a_refusal_and_thoughts_are_not_text():
    client = _Client([_resp([types.Part(text="hidden reasoning", thought=True)], finish="SAFETY")])
    c = GeminiAdapter(model="m", client=client).complete([{"role": "user", "content": "x"}])
    assert c.stop_reason == "refusal" and c.refusal_category == "safety" and c.text == ""


def test_blocked_prompt_without_candidates():
    resp = types.GenerateContentResponse(
        candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
    c = GeminiAdapter(model="m", client=_Client([resp])).complete([{"role": "user", "content": "x"}])
    assert c.stop_reason == "refusal" and c.tool_calls == []


def test_embedder_requests_1024_dims_and_normalises():
    client = _Client()
    vecs = GeminiEmbedder(model="gemini-embedding-001", client=client).embed(["a", "b"])
    cfg = client.models.calls[0]["config"]
    assert cfg.output_dimensionality == 1024
    assert vecs == [[0.6, 0.8], [0.6, 0.8]]
