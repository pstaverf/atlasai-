from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not installed or cannot load", allow_module_level=True)

from atlasai.config import TransformerConfig
from atlasai.core.model.transformer import AtlasTransformer
from atlasai.core.tokenizer.bpe import BPETokenizer
from atlasai.training.checkpoint import load_checkpoint, load_model_checkpoint, save_checkpoint
from atlasai.training.errors import CheckpointError


def _artifacts(tmp_path: Path) -> tuple[AtlasTransformer, Path, Path]:
    torch.manual_seed(3)
    model = AtlasTransformer(
        vocab_size=259,
        context_length=8,
        config=TransformerConfig(d_model=8, num_layers=1, num_heads=2, dropout=0.0),
    )
    tokenizer = tmp_path / "tokenizer.json"
    BPETokenizer().save(tokenizer)
    return model, tokenizer, tmp_path / "model.pt"


def test_save_load_and_reconstruct(tmp_path: Path) -> None:
    model, tokenizer, path = _artifacts(tmp_path)
    info = save_checkpoint(path, model, tokenizer_path=tokenizer, step=4, validation_loss=1.5)
    restored, restored_info = load_model_checkpoint(path, tokenizer_path=tokenizer)
    assert info == restored_info
    assert restored.config == model.config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    assert not list(tmp_path.glob("*.tmp"))


def test_load_into_matching_model(tmp_path: Path) -> None:
    model, tokenizer, path = _artifacts(tmp_path)
    save_checkpoint(path, model, tokenizer_path=tokenizer, step=1, validation_loss=2.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    info = load_checkpoint(path, model, tokenizer_path=tokenizer)
    assert info.step == 1
    assert float(model.lm_head.weight.abs().sum().item()) > 0


def test_changed_tokenizer_is_rejected(tmp_path: Path) -> None:
    model, tokenizer, path = _artifacts(tmp_path)
    save_checkpoint(path, model, tokenizer_path=tokenizer, step=1, validation_loss=2.0)
    tokenizer.write_text(tokenizer.read_text() + "\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="fingerprint"):
        load_model_checkpoint(path, tokenizer_path=tokenizer)


def test_architecture_mismatch_is_rejected(tmp_path: Path) -> None:
    model, tokenizer, path = _artifacts(tmp_path)
    save_checkpoint(path, model, tokenizer_path=tokenizer, step=1, validation_loss=2.0)
    other = AtlasTransformer(
        vocab_size=259,
        context_length=8,
        config=TransformerConfig(d_model=16, num_layers=1, num_heads=2, dropout=0.0),
    )
    with pytest.raises(CheckpointError, match="architecture"):
        load_checkpoint(path, other, tokenizer_path=tokenizer)


def test_corrupt_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="safely load"):
        load_model_checkpoint(path, tokenizer_path=tmp_path / "tokenizer.json")


def test_nonfinite_weights_are_rejected(tmp_path: Path) -> None:
    model, tokenizer, path = _artifacts(tmp_path)
    save_checkpoint(path, model, tokenizer_path=tokenizer, step=1, validation_loss=2.0)
    payload = torch.load(path, weights_only=True)
    payload["model_state"]["lm_head.weight"][0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="non-finite"):
        load_model_checkpoint(path, tokenizer_path=tokenizer)


def test_incomplete_architecture_is_rejected(tmp_path: Path) -> None:
    model, tokenizer, path = _artifacts(tmp_path)
    save_checkpoint(path, model, tokenizer_path=tokenizer, step=1, validation_loss=2.0)
    payload = torch.load(path, weights_only=True)
    del payload["architecture"]["dropout"]
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="missing or unknown"):
        load_model_checkpoint(path, tokenizer_path=tokenizer)


def test_unsafe_pickle_is_rejected(tmp_path: Path) -> None:
    class Unsafe:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return eval, ("1 + 1",)

    path = tmp_path / "unsafe.pt"
    torch.save({"payload": Unsafe()}, path)
    with pytest.raises(CheckpointError, match="safely load"):
        load_model_checkpoint(path, tokenizer_path=tmp_path / "tokenizer.json")


def test_checkpoint_cannot_overwrite_tokenizer(tmp_path: Path) -> None:
    model, tokenizer, _ = _artifacts(tmp_path)
    with pytest.raises(CheckpointError, match="overwrite"):
        save_checkpoint(tokenizer, model, tokenizer_path=tokenizer, step=0, validation_loss=1.0)
