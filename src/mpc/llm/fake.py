"""Deterministic offline providers for tests and runs without credentials.

FakeAdapter echoes; ScriptedAdapter replays a scenario's planned tool calls,
one per call, so an agent run is repeatable end to end without an LLM.
"""

import hashlib
from typing import Any

from mpc.llm.base import ChatMessage, Completion, ToolCall, ToolSpec, message_text
from mpc.llm.embeddings import get_embedder
from mpc.tokens import estimate_tokens


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return message_text(m)
    return ""


def _input_tokens(messages: list[ChatMessage], system: str | None) -> int:
    return sum(estimate_tokens(message_text(m)) for m in messages) + estimate_tokens(system or "")


class FakeAdapter:
    provider = "fake"

    def __init__(self, model: str = "fake-1"):
        self.model = model
        self._embedder = get_embedder()

    def complete(self, messages, *, system=None, model=None, max_tokens=None) -> Completion:
        prompt = _last_user_text(messages)
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        ctx = f", system context {estimate_tokens(system or '')} tokens" if system else ""
        text = f"[fake reply {digest}] Received {estimate_tokens(prompt)} tokens{ctx}: {prompt[:120]}"
        return Completion(
            text=text,
            model=model or self.model,
            provider=self.provider,
            stop_reason="end_turn",
            input_tokens=_input_tokens(messages, system),
            output_tokens=estimate_tokens(text),
            raw_content=[{"type": "text", "text": text}],
        )

    def tool_call(
        self, messages, tools: list[ToolSpec], *, system=None, model=None, max_tokens=None
    ) -> Completion:
        """Calls the first tool whose name appears in the last user message."""
        prompt = _last_user_text(messages)
        base = self.complete(messages, system=system, model=model)
        for tool in tools:
            if tool.name in prompt:
                call = ToolCall(id=f"fake_{tool.name}", name=tool.name, input={})
                base.tool_calls = [call]
                base.stop_reason = "tool_use"
                base.raw_content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": {}}
                )
                break
        return base

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)


class ScriptExhausted(Exception):
    pass


class ScriptedAdapter:
    """Replays planned steps: each step is {tool: name, input: {...}} or {text: "..."}.

    One step per call. A step naming a tool the caller did not offer is an
    error, so a script cannot make an agent do something its template forbids.
    """

    provider = "scripted"

    def __init__(self, steps: list[dict[str, Any]], role: str = "agent"):
        self.model = f"script:{role}"
        self.steps = list(steps)
        self.role = role
        self.calls = 0
        self._embedder = get_embedder()

    def _next(self) -> dict[str, Any] | None:
        if not self.steps:
            return None
        self.calls += 1
        return self.steps.pop(0)

    def complete(self, messages, *, system=None, model=None, max_tokens=None) -> Completion:
        return self.tool_call(messages, [], system=system, model=model)

    def tool_call(self, messages, tools, *, system=None, model=None, max_tokens=None) -> Completion:
        step = self._next()
        if step is None:
            text = f"[{self.role} script exhausted]"
            return Completion(text=text, model=self.model, provider=self.provider,
                              stop_reason="end_turn", raw_content=[{"type": "text", "text": text}],
                              input_tokens=_input_tokens(messages, system))
        text = step.get("text", "")
        raw: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        calls: list[ToolCall] = []
        if "tool" in step:
            offered = {t.name for t in tools}
            if step["tool"] not in offered:
                raise ScriptExhausted(
                    f"{self.role} script step {self.calls} calls {step['tool']!r}, "
                    f"which this agent does not offer ({', '.join(sorted(offered))})"
                )
            call = ToolCall(id=f"script_{self.role}_{self.calls}", name=step["tool"],
                            input=dict(step.get("input") or {}))
            calls.append(call)
            raw.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.input})
        return Completion(
            text=text,
            model=self.model,
            provider=self.provider,
            stop_reason="tool_use" if calls else "end_turn",
            tool_calls=calls,
            input_tokens=_input_tokens(messages, system),
            output_tokens=estimate_tokens(str(step)),
            raw_content=raw,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)
