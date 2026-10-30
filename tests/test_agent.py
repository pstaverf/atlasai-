from __future__ import annotations

import json

import pytest

from atlasai.agents.basic import AgentConfig, BasicAgent, build_prompt
from atlasai.agents.errors import AgentError
from atlasai.core.model.base import GenerationResult, LLMError


class FakeLLM:
    def __init__(self, outputs: list[str] | Exception) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        if isinstance(self.outputs, Exception):
            raise self.outputs
        text = self.outputs.pop(0)
        return GenerationResult(
            text=text, prompt_tokens=10, completion_tokens=5, stop_reason="max_tokens"
        )


def test_respond_completes_in_one_step() -> None:
    llm = FakeLLM(['{"action":"respond","text":"Готово"}'])
    result = BasicAgent(llm).run("  Что делает pwd?  ")
    assert result.status == "completed"
    assert result.final_answer == "Готово"
    assert len(result.steps) == 1
    assert result.steps[0].action is not None
    assert "Что делает pwd?" in llm.prompts[0]
    assert json.loads(json.dumps(result.to_dict()))["status"] == "completed"


def test_think_then_respond_uses_observation() -> None:
    llm = FakeLLM(
        [
            '{"action":"think","text":"Нужно проверить URL"}',
            '{"action":"respond","text":"Проверь URL и статус"}',
        ]
    )
    result = BasicAgent(llm).run("План запроса")
    assert result.status == "completed"
    assert len(result.steps) == 2
    assert "Заметка записана" in result.steps[0].observation
    assert "Наблюдение:" in llm.prompts[1]
    assert "Нужно проверить URL" in llm.prompts[1]


def test_protocol_error_is_observed_and_retried() -> None:
    llm = FakeLLM(["not json", '{"action":"respond","text":"Исправлено"}'])
    result = BasicAgent(llm).run("Задача")
    assert result.status == "completed"
    assert result.steps[0].error is not None
    assert "Ошибка протокола" in llm.prompts[1]


def test_max_steps_is_explicit_failure() -> None:
    llm = FakeLLM(['{"action":"think","text":"думаю"}'] * 2)
    result = BasicAgent(llm, AgentConfig(max_steps=2)).run("Задача")
    assert result.status == "max_steps"
    assert result.final_answer is None
    assert len(result.steps) == 2
    assert "max_steps=2" in (result.error or "")


def test_model_error_is_reported() -> None:
    llm = FakeLLM(LLMError("backend unavailable"))
    result = BasicAgent(llm).run("Задача")
    assert result.status == "model_error"
    assert result.error == "backend unavailable"
    assert result.steps == ()


@pytest.mark.parametrize("task", ["", "   ", "x" * 11])
def test_invalid_task_is_rejected(task: str) -> None:
    with pytest.raises(AgentError):
        BasicAgent(FakeLLM([]), AgentConfig(max_task_chars=10)).run(task)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(AgentError):
        BasicAgent(FakeLLM([]), AgentConfig(max_steps=0))


def test_build_prompt_is_compact_and_ordered() -> None:
    prompt = build_prompt("Задача", (), 1)
    assert prompt.index("Задача:") < prompt.index("Шаг: 1") < prompt.index("Ответ:")
    assert "один JSON" in prompt
