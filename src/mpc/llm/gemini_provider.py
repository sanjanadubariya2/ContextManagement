"""Gemini via Google's official google-genai SDK.

The agents speak the adapter's neutral message shape (Messages-API style
blocks: text, tool_use, tool_result). This module translates it to Gemini
contents and back. Gemini's own model turns are carried through unchanged:
thinking models attach thought signatures to their function calls and
require them on the next request, so a model turn is never rebuilt from text.

Credentials: GEMINI_API_KEY (or GOOGLE_API_KEY), read by the SDK.
"""

import math
from typing import Any

from google import genai
from google.genai import types

from mpc.config import EMBED_DIM, get_settings
from mpc.llm.base import ChatMessage, Completion, ToolCall, ToolSpec
from mpc.llm.embeddings import get_embedder

GEMINI_BLOCK = "gemini_content"  # raw_content block type carrying a Gemini Content
_REFUSALS = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}


def _client() -> genai.Client:
    return genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY


def _to_contents(messages: list[ChatMessage]) -> list[types.Content]:
    contents: list[types.Content] = []
    calls: dict[str, tuple[str, str | None]] = {}  # our call id -> (name, Gemini's id)
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            native = next((b for b in blocks if b.get("type") == GEMINI_BLOCK), None)
            if native is not None:
                contents.append(native["content"])  # verbatim, thought signatures included
                calls.update(native["calls"])
                continue
            parts = []
            for b in blocks:  # history written by another provider, or plain text turns
                if b.get("type") == "text" and b.get("text"):
                    parts.append(types.Part(text=b["text"]))
                elif b.get("type") == "tool_use":
                    calls[b["id"]] = (b["name"], None)
                    parts.append(types.Part(function_call=types.FunctionCall(name=b["name"], args=b.get("input") or {})))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue

        if isinstance(content, str):
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            continue
        parts = []
        for b in content:
            if b.get("type") == "text":
                parts.append(types.Part(text=b.get("text", "")))
            elif b.get("type") == "tool_result":
                name, gid = calls.get(b["tool_use_id"], ("tool", None))
                key = "error" if b.get("is_error") else "output"
                out = b.get("content")
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=gid, name=name, response={key: out if isinstance(out, str) else str(out)},
                )))
        if parts:
            contents.append(types.Content(role="user", parts=parts))
    return contents


class GeminiAdapter:
    provider = "gemini"

    def __init__(self, model: str | None = None, client: genai.Client | None = None):
        self.model = model or get_settings().gemini_model
        self.client = client or _client()
        self._embedder = get_embedder()

    def _request(self, messages, *, system, model, max_tokens, tools=None) -> Completion:
        model = model or self.model
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens or get_settings().llm_max_tokens,
            # The agents run their own tool loops; the SDK must never call functions itself.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description,
                                          parameters_json_schema=t.input_schema)
                for t in tools
            ])] if tools else None,
        )
        resp = self.client.models.generate_content(model=model, contents=_to_contents(messages), config=config)

        usage = resp.usage_metadata
        in_tok = (usage.prompt_token_count or 0) if usage else 0
        out_tok = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None:
            reason = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
            return Completion(text="", model=model, provider=self.provider, stop_reason="refusal",
                              input_tokens=in_tok, output_tokens=out_tok,
                              refusal_category=str(reason) if reason else "no candidate")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        call_map: dict[str, tuple[str, str | None]] = {}
        for i, part in enumerate(cand.content.parts or []):
            if part.function_call is not None:
                fc = part.function_call
                cid = fc.id or f"gemini_{i}_{fc.name}"
                tool_calls.append(ToolCall(id=cid, name=fc.name, input=dict(fc.args or {})))
                call_map[cid] = (fc.name, fc.id)
            elif part.text and not part.thought:
                text_parts.append(part.text)

        finish = cand.finish_reason.name if cand.finish_reason is not None else "STOP"
        if finish in _REFUSALS:
            stop, refusal = "refusal", finish.lower()
        elif finish == "MAX_TOKENS":
            stop, refusal = "max_tokens", None
        else:
            stop, refusal = ("tool_use" if tool_calls else "end_turn"), None

        text = "".join(text_parts)
        raw: list[dict[str, Any]] = [{"type": GEMINI_BLOCK, "content": cand.content, "calls": call_map}]
        return Completion(text=text, model=model, provider=self.provider, stop_reason=stop,
                          tool_calls=tool_calls, input_tokens=in_tok, output_tokens=out_tok,
                          refusal_category=refusal, raw_content=raw)

    def complete(self, messages, *, system=None, model=None, max_tokens=None) -> Completion:
        return self._request(messages, system=system, model=model, max_tokens=max_tokens)

    def tool_call(self, messages, tools: list[ToolSpec], *, system=None, model=None, max_tokens=None) -> Completion:
        return self._request(messages, system=system, model=model, max_tokens=max_tokens, tools=tools)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)


class GeminiEmbedder:
    """Semantic embeddings from gemini-embedding-001, truncated to EMBED_DIM and
    re-normalised (the model's vectors are only unit length at full size)."""

    name = "gemini"

    def __init__(self, model: str | None = None, dim: int = EMBED_DIM, client: genai.Client | None = None):
        self.model = model or get_settings().gemini_embed_model
        self.dim = dim
        self.client = client or _client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            res = self.client.models.embed_content(
                model=self.model, contents=texts[i:i + 100],
                config=types.EmbedContentConfig(output_dimensionality=self.dim, task_type="SEMANTIC_SIMILARITY"),
            )
            for e in res.embeddings:
                norm = math.sqrt(sum(v * v for v in e.values)) or 1.0
                out.append([v / norm for v in e.values])
        return out
