"""Provider-neutral LLM adapter interface: complete, tool_call, embed.

Everything above this layer (agents, CLI, Memorizer, ...) talks to
`LLMAdapter` only, so the provider can be swapped without touching callers.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    refusal_category: str | None = None


# Messages use the common {"role": "user"|"assistant", "content": str} shape.
ChatMessage = dict[str, Any]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMAdapter(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...

    def tool_call(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
