"""Deterministic offline provider for tests and runs without credentials."""

import hashlib

from mpc.llm.base import ChatMessage, Completion, ToolCall, ToolSpec
from mpc.llm.embeddings import get_embedder
from mpc.tokens import estimate_tokens


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], str):
            return m["content"]
    return ""


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
        in_tokens = sum(estimate_tokens(str(m["content"])) for m in messages)
        return Completion(
            text=text,
            model=model or self.model,
            provider=self.provider,
            stop_reason="end_turn",
            input_tokens=in_tokens + estimate_tokens(system or ""),
            output_tokens=estimate_tokens(text),
        )

    def tool_call(
        self, messages, tools: list[ToolSpec], *, system=None, model=None, max_tokens=None
    ) -> Completion:
        """Calls the first tool whose name appears in the last user message."""
        prompt = _last_user_text(messages)
        base = self.complete(messages, system=system, model=model)
        for tool in tools:
            if tool.name in prompt:
                base.tool_calls = [ToolCall(id=f"fake_{tool.name}", name=tool.name, input={})]
                base.stop_reason = "tool_use"
                break
        return base

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)
