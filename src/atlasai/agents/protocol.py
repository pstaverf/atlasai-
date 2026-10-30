"""Strict JSON action protocol for the basic agent.

The model may propose only two actions: ``think`` records a short note for the
next loop, ``respond`` returns the final user-facing answer. There is no shell,
file, network, or arbitrary tool action at this stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from atlasai.agents.errors import ProtocolError

ActionType = Literal["think", "respond"]
_ALLOWED_ACTIONS: set[str] = {"think", "respond"}
_MAX_RAW_CHARS = 65_536


@dataclass(frozen=True, slots=True)
class AgentAction:
    action: ActionType
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action, "text": self.text}


def parse_action(raw: str, *, max_text_chars: int = 4096) -> AgentAction:
    """Parse exactly one JSON object with ``action`` and ``text``.

    Markdown fences, extra prose, multiple objects, unknown keys, and unknown
    actions are rejected instead of being guessed or executed.
    """
    if not isinstance(raw, str):
        raise ProtocolError("Model output must be a string")
    if len(raw) > _MAX_RAW_CHARS:
        raise ProtocolError(f"Model output exceeds {_MAX_RAW_CHARS} characters")
    text = raw.strip()
    if not text:
        raise ProtocolError("Model output is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Model output must be one JSON object")
    if set(payload) != {"action", "text"}:
        raise ProtocolError("Model action must contain exactly the keys action and text")
    action = payload["action"]
    answer = payload["text"]
    if not isinstance(action, str) or action not in _ALLOWED_ACTIONS:
        raise ProtocolError("action must be 'think' or 'respond'")
    if not isinstance(answer, str) or not answer.strip():
        raise ProtocolError("text must be a non-empty string")
    if len(answer) > max_text_chars:
        raise ProtocolError(f"text exceeds {max_text_chars} characters")
    if action == "think":
        return AgentAction(action="think", text=answer)
    return AgentAction(action="respond", text=answer)
