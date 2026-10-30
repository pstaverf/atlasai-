from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from atlasai.cli import main
from atlasai.core.model.base import GenerationResult


class FakeBackend:
    parameter_count = 123

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        return GenerationResult(
            text=self.outputs.pop(0),
            prompt_tokens=1,
            completion_tokens=1,
            stop_reason="max_tokens",
        )


def test_agent_run_completed(
    tiny_config: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(['{"action":"respond","text":"Ответ"}'])
    monkeypatch.setattr("atlasai.cli._load_backend", lambda args, config: backend)
    code = main(["agent", "run", "--config", str(tiny_config), "--task", "Вопрос"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["command"] == "agent.run"
    assert payload["status"] == "completed"
    assert payload["final_answer"] == "Ответ"
    assert payload["parameter_count"] == 123


def test_agent_run_max_steps_returns_one(
    tiny_config: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(['{"action":"think","text":"заметка"}'])
    monkeypatch.setattr("atlasai.cli._load_backend", lambda args, config: backend)
    code = main(
        ["agent", "run", "--config", str(tiny_config), "--task", "Вопрос", "--max-steps", "1"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["status"] == "max_steps"


def test_agent_cli_explains_missing_torch(tiny_config: Path) -> None:
    code = (
        "import sys; sys.modules['torch']=None; "
        "from atlasai.cli import main; raise SystemExit(main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "agent", "run", "--config", str(tiny_config), "--task", "x"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "PyTorch is required" in json.loads(result.stderr)["error"]


def test_agent_core_imports_without_torch() -> None:
    code = (
        "import sys; sys.modules['torch']=None; "
        "from atlasai.agents import BasicAgent, AgentConfig; "
        "print(BasicAgent is not None, AgentConfig().max_steps)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=15
    )
    assert result.stdout.strip() == "True 4"
