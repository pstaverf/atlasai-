from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from atlasai import __version__
from atlasai.cli import main

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "atlas.toml"


def test_config_command_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["config", "--config", str(REPO_CONFIG)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["runtime"]["device"] == "cpu"
    assert payload["model"]["backend"] == "atlas-local"


def test_invalid_config_returns_one_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('[model]\nbackend = "ollama"\n', encoding="utf-8")
    exit_code = main(["config", "--config", str(path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "atlas-local" in json.loads(captured.err)["error"]
    assert "Traceback" not in captured.err


def test_missing_config_returns_one(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    exit_code = main(["doctor", "--config", str(tmp_path / "missing.toml")])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not found" in json.loads(captured.err)["error"]


def test_doctor_does_not_claim_model_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "atlas.toml"
    path.write_text(
        "[model]\n"
        'tokenizer_path = "missing-tokenizer.json"\n'
        'checkpoint_path = "missing-model.pt"\n',
        encoding="utf-8",
    )
    exit_code = main(["doctor", "--config", str(path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["stage"] == 1
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False
    assert payload["model_status"] == "not_implemented"
    assert payload["artifacts"]["checkpoint"]["exists"] is False


def test_existing_artifact_is_not_reported_as_loaded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    path = tmp_path / "atlas.toml"
    path.write_text(f'[model]\ncheckpoint_path = "{checkpoint.name}"\n', encoding="utf-8")
    exit_code = main(["doctor", "--config", str(path)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["artifacts"]["checkpoint"]["exists"] is True
    assert payload["model_loaded"] is False


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_subcommand_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2


def test_python_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "atlasai", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == f"atlasai {__version__}"
