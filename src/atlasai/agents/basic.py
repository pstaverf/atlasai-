"""A small, bounded think → act → observe agent.

This stage intentionally has no external tools. The LLM proposes a JSON action;
the agent validates it, records an observation, and either loops or returns the
final answer. Invalid output is reported and retried within the step limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from atlasai.agents.errors import AgentError, ProtocolError
from atlasai.agents.protocol import AgentAction, parse_action
from atlasai.core.model.base import LLM, GenerationResult, LLMError

AgentStatus = Literal["completed", "max_steps", "model_error"]

_SYSTEM_PROMPT = (
    "Агент Atlas. Верни один JSON: "
    '{"action":"think","text":"заметка"} или {"action":"respond","text":"итог"}. '
    "Команды не выполняй."
)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_steps: int = 4
    max_task_chars: int = 2000
    max_action_chars: int = 4096
    max_observation_chars: int = 512

    def validate(self) -> None:
        bounds = (
            ("max_steps", self.max_steps, 1, 32),
            ("max_task_chars", self.max_task_chars, 1, 8192),
            ("max_action_chars", self.max_action_chars, 1, 8192),
            ("max_observation_chars", self.max_observation_chars, 1, 2048),
        )
        for name, value, minimum, maximum in bounds:
            if isinstance(value, bool) or not isinstance(value, int):
                raise AgentError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise AgentError(f"{name} must be between {minimum} and {maximum}")


@dataclass(frozen=True, slots=True)
class AgentStep:
    index: int
    raw_output: str
    action: AgentAction | None
    observation: str
    prompt_tokens: int
    completion_tokens: int
    stop_reason: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "raw_output": self.raw_output,
            "action": self.action.to_dict() if self.action is not None else None,
            "observation": self.observation,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class AgentResult:
    task: str
    status: AgentStatus
    final_answer: str | None
    steps: tuple[AgentStep, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "status": self.status,
            "final_answer": self.final_answer,
            "steps": [step.to_dict() for step in self.steps],
            "error": self.error,
        }


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_prompt(task: str, steps: tuple[AgentStep, ...], step_index: int) -> str:
    """Keep the prompt compact and matching the foundation corpus format."""
    lines = [_SYSTEM_PROMPT, f"Задача: {task}", f"Шаг: {step_index}"]
    if steps:
        lines.append(f"Наблюдение: {steps[-1].observation}")
    lines.append("Ответ:")
    return "\n".join(lines)


class BasicAgent:
    """Run a bounded agent loop over any :class:`LLM` implementation."""

    def __init__(self, llm: LLM, config: AgentConfig | None = None) -> None:
        self.llm = llm
        self.config = config if config is not None else AgentConfig()
        self.config.validate()

    def run(self, task: str) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            raise AgentError("task must be a non-empty string")
        if len(task) > self.config.max_task_chars:
            raise AgentError(f"task exceeds {self.config.max_task_chars} characters")
        cleaned_task = task.strip()
        steps: list[AgentStep] = []
        for index in range(1, self.config.max_steps + 1):
            prompt = build_prompt(cleaned_task, tuple(steps), index)
            try:
                generation: GenerationResult = self.llm.generate(prompt)
            except LLMError as exc:
                return AgentResult(
                    task=cleaned_task,
                    status="model_error",
                    final_answer=None,
                    steps=tuple(steps),
                    error=str(exc),
                )
            try:
                action = parse_action(generation.text, max_text_chars=self.config.max_action_chars)
            except ProtocolError as exc:
                observation = _clip(
                    f"Ошибка протокола: {exc}. Верни один JSON объект: action и text.",
                    self.config.max_observation_chars,
                )
                steps.append(
                    AgentStep(
                        index=index,
                        raw_output=generation.text,
                        action=None,
                        observation=observation,
                        prompt_tokens=generation.prompt_tokens,
                        completion_tokens=generation.completion_tokens,
                        stop_reason=generation.stop_reason,
                        error=str(exc),
                    )
                )
                continue
            if action.action == "respond":
                steps.append(
                    AgentStep(
                        index=index,
                        raw_output=generation.text,
                        action=action,
                        observation="Ответ принят.",
                        prompt_tokens=generation.prompt_tokens,
                        completion_tokens=generation.completion_tokens,
                        stop_reason=generation.stop_reason,
                    )
                )
                return AgentResult(
                    task=cleaned_task,
                    status="completed",
                    final_answer=action.text,
                    steps=tuple(steps),
                )
            observation = _clip(
                f"Заметка записана. Инструменты не вызывались. {action.text}",
                self.config.max_observation_chars,
            )
            steps.append(
                AgentStep(
                    index=index,
                    raw_output=generation.text,
                    action=action,
                    observation=observation,
                    prompt_tokens=generation.prompt_tokens,
                    completion_tokens=generation.completion_tokens,
                    stop_reason=generation.stop_reason,
                )
            )
        return AgentResult(
            task=cleaned_task,
            status="max_steps",
            final_answer=None,
            steps=tuple(steps),
            error=f"Agent reached max_steps={self.config.max_steps} without a respond action",
        )
