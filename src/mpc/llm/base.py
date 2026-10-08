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
    # The assistant turn exactly as returned (text, tool_use, thinking blocks...).
    # Tool loops append it unchanged: current Claude models reject a history whose
    # earlier assistant turns (including their thinking blocks) were edited.
    raw_content: list[dict[str, Any]] = field(default_factory=list)

    def assistant_message(self) -> "ChatMessage":
        return {"role": "assistant", "content": self.raw_content or [{"type": "text", "text": self.text}]}


# Messages use {"role": "user"|"assistant", "content": str | list[block]} where
# blocks follow the Messages API shapes: text, tool_use, tool_result.
ChatMessage = dict[str, Any]


def tool_result(call: ToolCall, content: str, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": call.id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def message_text(message: ChatMessage) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            c = b.get("content")
            parts.append(c if isinstance(c, str) else str(c))
    return "\n".join(parts)


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
