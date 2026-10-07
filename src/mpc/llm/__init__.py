"""LLM adapter selection."""

import os

from mpc.config import get_settings
from mpc.llm.base import Completion, LLMAdapter, ToolCall, ToolSpec

__all__ = ["Completion", "LLMAdapter", "ToolCall", "ToolSpec", "get_adapter"]


def _has_anthropic_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    home = os.path.expanduser("~")
    return os.path.isdir(os.path.join(home, ".config", "anthropic"))


def get_adapter(provider: str | None = None, model: str | None = None) -> LLMAdapter:
    provider = provider or get_settings().llm_provider
    if provider == "auto":
        provider = "anthropic" if _has_anthropic_credentials() else "fake"
    if provider == "anthropic":
        from mpc.llm.anthropic_provider import AnthropicAdapter

        return AnthropicAdapter(model=model)
    if provider == "fake":
        from mpc.llm.fake import FakeAdapter

        return FakeAdapter()
    raise ValueError(f"unknown LLM provider: {provider}")
