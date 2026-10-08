"""LLM adapter selection.

MPC_LLM_PROVIDER=auto (default) uses Gemini when GEMINI_API_KEY or
GOOGLE_API_KEY is set, else Anthropic when its credentials exist, else stops
with an error. There is no silent fallback to the fake provider: it runs only
when MPC_LLM_PROVIDER=fake is set explicitly (the test suite does this).
"""

import os

from mpc.config import get_settings
from mpc.llm.base import Completion, LLMAdapter, ToolCall, ToolSpec

__all__ = ["Completion", "LLMAdapter", "LLMConfigError", "ToolCall", "ToolSpec", "get_adapter", "resolve_provider"]

PROVIDERS = ("auto", "gemini", "anthropic", "fake")


class LLMConfigError(RuntimeError):
    pass


def gemini_credentials() -> str | None:
    """Which variable holds a Gemini key, or None. Never returns the secret."""
    get_settings()  # loads .env into the environment first
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return var
    return None


def anthropic_credentials() -> str | None:
    get_settings()
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        if os.environ.get(var):
            return var
    if os.path.isdir(os.path.join(os.path.expanduser("~"), ".config", "anthropic")):
        return "ant auth login profile"
    return None


def _missing_key_hint(var: str) -> str:
    from mpc.config import REPO_ROOT

    env = REPO_ROOT / ".env"
    if env.is_file() and env.stat().st_size == 0:
        return (f"{env} exists but is empty on disk; if your editor shows the key, save the "
                "file (Ctrl+S)")
    if not env.is_file():
        return f"create {env} with a line {var}=..."
    return f"{env} has no {var} line; add {var}=..."


def resolve_provider(provider: str | None = None) -> tuple[str, str]:
    """(provider, reason) for the configured MPC_LLM_PROVIDER."""
    configured = provider or get_settings().llm_provider
    if configured not in PROVIDERS:
        raise LLMConfigError(f"MPC_LLM_PROVIDER={configured!r}; use one of {', '.join(PROVIDERS)}")
    if configured == "fake":
        return "fake", "MPC_LLM_PROVIDER=fake (offline stand-in, for tests)"
    gem, ant = gemini_credentials(), anthropic_credentials()
    if configured == "auto":
        if gem:
            return "gemini", f"auto: key from {gem}"
        if ant:
            return "anthropic", f"auto: credentials from {ant}"
        raise LLMConfigError(f"No LLM API key found: {_missing_key_hint('GEMINI_API_KEY')}.")
    if configured == "gemini":
        if not gem:
            raise LLMConfigError(f"MPC_LLM_PROVIDER=gemini but no Gemini key: {_missing_key_hint('GEMINI_API_KEY')}.")
        return "gemini", f"key from {gem}"
    if not ant:
        raise LLMConfigError(f"MPC_LLM_PROVIDER=anthropic but no key: {_missing_key_hint('ANTHROPIC_API_KEY')}.")
    return "anthropic", f"credentials from {ant}"


def model_for(provider: str) -> str:
    s = get_settings()
    return {"gemini": s.gemini_model, "anthropic": s.llm_model}.get(provider, "fake-1")


def get_adapter(provider: str | None = None, model: str | None = None) -> LLMAdapter:
    provider, _ = resolve_provider(provider)
    if provider == "gemini":
        from mpc.llm.gemini_provider import GeminiAdapter

        return GeminiAdapter(model=model)
    if provider == "anthropic":
        from mpc.llm.anthropic_provider import AnthropicAdapter

        return AnthropicAdapter(model=model)
    from mpc.llm.fake import FakeAdapter

    return FakeAdapter()
