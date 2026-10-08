"""Claude via the official Anthropic SDK.

Defaults: claude-opus-5-5, thinking left at its adaptive default, effort set
explicitly (Opus 5.5 defaults to medium), and server-side refusal fallbacks
enabled (`fallbacks: "default"`), which can be turned off with
MPC_LLM_FALLBACKS=false. Credentials resolve the SDK's usual way
(ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile).
"""

from typing import Any

import anthropic

from mpc.config import get_settings
from mpc.llm.base import ChatMessage, Completion, ToolCall, ToolSpec
from mpc.llm.embeddings import get_embedder

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept output_config.effort and the "default" fallback mode.
_EFFORT_MODELS = ("claude-opus-5", "claude-fable-5", "claude-sonnet-5", "claude-opus-4-")
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, model: str | None = None, client: anthropic.Anthropic | None = None):
        s = get_settings()
        self.model = model or s.llm_model
        self.client = client or anthropic.Anthropic()
        self._embedder = get_embedder()

    def _request(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None,
        model: str | None,
        max_tokens: int | None,
        tools: list[ToolSpec] | None = None,
    ) -> Completion:
        s = get_settings()
        model = model or self.model
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or s.llm_max_tokens,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if model.startswith(_EFFORT_MODELS):
            params["output_config"] = {"effort": s.llm_effort}
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]
            params["tool_choice"] = {"type": "auto"}

        if s.llm_fallbacks and model in _FALLBACK_MODELS:
            response = self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **params
            )
        else:
            response = self.client.messages.create(**params)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        refusal = None
        if response.stop_reason == "refusal" and getattr(response, "stop_details", None):
            refusal = response.stop_details.category

        return Completion(
            text="".join(text_parts),
            model=response.model,
            provider=self.provider,
            stop_reason=response.stop_reason,
            tool_calls=calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            refusal_category=refusal,
            raw_content=[b.model_dump(mode="json", exclude_none=True) for b in response.content],
        )

    def complete(self, messages, *, system=None, model=None, max_tokens=None) -> Completion:
        return self._request(messages, system=system, model=model, max_tokens=max_tokens)

    def tool_call(
        self, messages, tools, *, system=None, model=None, max_tokens=None
    ) -> Completion:
        return self._request(
            messages, system=system, model=model, max_tokens=max_tokens, tools=tools
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Anthropic has no embeddings endpoint; delegate to the configured embedder.
        return self._embedder.embed(texts)
