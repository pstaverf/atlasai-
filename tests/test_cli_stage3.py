from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from atlasai.cli import main
from atlasai.core.tokenizer.bpe import BPETokenizer

REPO = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO / "configs" / "smoke.toml"
CORPUS = REPO / "examples" / "corpus.txt"


def test_model_train_and_evaluate_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    tokenizer = tmp_path / "tokenizer.json"
    checkpoint = tmp_path / "model.pt"
    BPETokenizer().save(tokenizer)
    common = [
        "--config",
        str(SMOKE_CONFIG),
        "--input",
        str(CORPUS),
        "--tokenizer",
        str(tokenizer),
        "--checkpoint",
        str(checkpoint),
    ]
    assert (
        main(["model", "train", *common, "--max-steps", "2", "--batch-size", "2", "--quiet"]) == 0
    )
    trained = json.loads(capsys.readouterr().out)
    assert trained["steps_completed"] == 2
    assert trained["parameter_count"] > 0
    assert checkpoint.is_file()
    assert Path(trained["metrics_path"]).is_file()
    assert main(["model", "evaluate", *common, "--batch-size", "2"]) == 0
    evaluated = json.loads(capsys.readouterr().out)
    assert evaluated["validation"]["loss"] == pytest.approx(trained["best_validation"]["loss"])
    assert evaluated["checkpoint"]["tokenizer_sha256"]


def test_model_output_cannot_overwrite_tokenizer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    tokenizer = tmp_path / "tokenizer.json"
    BPETokenizer().save(tokenizer)
    code = main(
        [
            "model",
            "train",
            "--config",
            str(SMOKE_CONFIG),
            "--input",
            str(CORPUS),
            "--tokenizer",
            str(tokenizer),
            "--checkpoint",
            str(tokenizer),
            "--max-steps",
            "1",
        ]
    )
    assert code == 1
    assert "overwrite" in json.loads(capsys.readouterr().err)["error"]


def test_base_cli_does_not_import_torch() -> None:
    code = (
        "import sys; sys.modules['torch']=None; "
        "from atlasai.cli import main; raise SystemExit(main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "config", "--config", str(SMOKE_CONFIG)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert json.loads(result.stdout)["model"]["backend"] == "atlas-local"


def test_model_cli_explains_missing_torch(capsys: pytest.CaptureFixture[str]) -> None:
    # Run in a child so blocking torch does not poison the rest of the test process.
    code = (
        "import sys; sys.modules['torch']=None; "
        "from atlasai.cli import main; raise SystemExit(main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "model", "train", "--config", str(SMOKE_CONFIG)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "PyTorch is required" in json.loads(result.stderr)["error"]
