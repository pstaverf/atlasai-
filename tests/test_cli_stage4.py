from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from atlasai.cli import main
from atlasai.core.tokenizer.bpe import BPETokenizer


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    pytest.importorskip("torch")
    from atlasai.config import TransformerConfig
    from atlasai.core.model.transformer import AtlasTransformer
    from atlasai.training.checkpoint import save_checkpoint

    tokenizer = tmp_path / "tokenizer.json"
    checkpoint = tmp_path / "model.pt"
    BPETokenizer().save(tokenizer)
    model = AtlasTransformer(
        vocab_size=259,
        context_length=16,
        config=TransformerConfig(d_model=8, num_layers=1, num_heads=2, dropout=0.0),
    )
    save_checkpoint(checkpoint, model, tokenizer_path=tokenizer, step=0, validation_loss=1.0)
    return tokenizer, checkpoint


def test_llm_info_and_generate_cli(
    tmp_path: Path, tiny_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tokenizer, checkpoint = _artifacts(tmp_path)
    common = [
        "--config",
        str(tiny_config),
        "--tokenizer",
        str(tokenizer),
        "--checkpoint",
        str(checkpoint),
    ]
    assert main(["llm", "info", *common]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["backend"] == "atlas-local"
    assert info["parameter_count"] > 0
    assert main(["llm", "generate", *common, "--prompt", "Atlas", "--max-new-tokens", "3"]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["completion_tokens"] <= 3
    assert generated["parameter_count"] == info["parameter_count"]


def test_llm_missing_checkpoint_returns_one(
    tmp_path: Path, tiny_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    tokenizer = tmp_path / "tokenizer.json"
    BPETokenizer().save(tokenizer)
    code = main(
        [
            "llm",
            "generate",
            "--config",
            str(tiny_config),
            "--tokenizer",
            str(tokenizer),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--prompt",
            "hello",
        ]
    )
    assert code == 1
    assert "Checkpoint not found" in json.loads(capsys.readouterr().err)["error"]


def test_llm_cli_explains_missing_torch(tiny_config: Path) -> None:
    code = (
        "import sys; sys.modules['torch']=None; "
        "from atlasai.cli import main; raise SystemExit(main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "llm", "info", "--config", str(tiny_config)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "PyTorch is required" in json.loads(result.stderr)["error"]
