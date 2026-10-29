"""Torch-free LLM interface shared by future agent code and local backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

StopReason = Literal["eos", "max_tokens"]


class LLMError(ValueError):
    """A generation request is invalid or the backend cannot safely generate."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    stop_reason: StopReason

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "stop_reason": self.stop_reason,
        }


class LLM(Protocol):
    """Minimal replaceable text-generation contract.

    Implementations may be local or remote. Agent code should depend on this
    protocol rather than on Transformer internals or a vendor SDK.
    """

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult: ...
