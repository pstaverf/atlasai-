from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not installed or cannot load", allow_module_level=True)

from atlasai.config import TrainingConfig, TransformerConfig
from atlasai.core.model.transformer import AtlasTransformer
from atlasai.core.tokenizer.bpe import PAD_ID, BPETokenizer
from atlasai.training.checkpoint import load_model_checkpoint
from atlasai.training.dataset import DatasetStats, TokenBlock, TokenDataset
from atlasai.training.errors import TrainingError
from atlasai.training.evaluation import EvaluationResult, evaluate, token_loss
from atlasai.training.trainer import configure_torch, train_model


def _model() -> AtlasTransformer:
    configure_torch(seed=11, num_threads=1)
    return AtlasTransformer(
        vocab_size=259,
        context_length=8,
        config=TransformerConfig(d_model=16, num_layers=1, num_heads=4, dropout=0.0),
    )


def _dataset() -> TokenDataset:
    block = TokenBlock(input_ids=tuple(range(3, 11)), target_ids=tuple(range(4, 12)))
    train = (block,) * 4
    val = (block,)
    return TokenDataset(
        train=train,
        val=val,
        stats=DatasetStats(
            documents=1,
            total_tokens=45,
            train_blocks=len(train),
            val_blocks=len(val),
            context_length=8,
            stride=8,
            dropped_tokens=0,
        ),
    )


def _tokenizer(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.json"
    BPETokenizer().save(path)
    return path


def test_training_reduces_loss_and_reloads_best(tmp_path: Path) -> None:
    model = _model()
    tokenizer = _tokenizer(tmp_path)
    output = tmp_path / "model.pt"
    config = TrainingConfig(
        batch_size=4,
        max_steps=40,
        learning_rate=0.01,
        weight_decay=0.0,
        eval_every=10,
        log_every=10,
    )
    result = train_model(
        model,
        _dataset(),
        tokenizer_path=tokenizer,
        checkpoint_path=output,
        config=config,
        seed=11,
        num_threads=1,
    )
    initial = result.initial_validation.loss
    best = result.best_validation.loss
    assert best < initial * 0.8
    assert result.final_train_loss < initial
    assert output.is_file()
    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert len(metrics["history"]) == 5
    restored, info = load_model_checkpoint(output, tokenizer_path=tokenizer)
    repeated = evaluate(restored, _dataset().val, batch_size=4, device=torch.device("cpu"))
    assert info.step == result.best_step
    assert repeated.loss == pytest.approx(best, abs=1e-6)


def test_training_is_reproducible(tmp_path: Path) -> None:
    config = TrainingConfig(batch_size=4, max_steps=5, eval_every=5, learning_rate=0.001)
    tokenizer = _tokenizer(tmp_path)
    first = _model()
    second = _model()
    first_result = train_model(
        first,
        _dataset(),
        tokenizer_path=tokenizer,
        checkpoint_path=tmp_path / "first.pt",
        config=config,
        seed=11,
        num_threads=1,
    )
    second_result = train_model(
        second,
        _dataset(),
        tokenizer_path=tokenizer,
        checkpoint_path=tmp_path / "second.pt",
        config=config,
        seed=11,
        num_threads=1,
    )
    assert first_result.best_validation == second_result.best_validation
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=1e-6, atol=1e-7)


def test_overlapping_windows_are_rejected(tmp_path: Path) -> None:
    dataset = _dataset()
    overlapping = replace(dataset, stats=replace(dataset.stats, stride=4))
    with pytest.raises(TrainingError, match="token overlap"):
        train_model(
            _model(),
            overlapping,
            tokenizer_path=_tokenizer(tmp_path),
            checkpoint_path=tmp_path / "model.pt",
            config=TrainingConfig(max_steps=1),
            seed=1,
            num_threads=1,
        )


def test_empty_validation_is_rejected(tmp_path: Path) -> None:
    dataset = _dataset()
    with pytest.raises(TrainingError, match="non-empty"):
        train_model(
            _model(),
            replace(dataset, val=()),
            tokenizer_path=_tokenizer(tmp_path),
            checkpoint_path=tmp_path / "model.pt",
            config=TrainingConfig(max_steps=1),
            seed=1,
            num_threads=1,
        )


def test_token_loss_ignores_padding() -> None:
    logits = torch.zeros(1, 3, 259)
    targets = torch.tensor([[PAD_ID, 3, 4]], dtype=torch.long)
    loss, count = token_loss(logits, targets)
    assert count == 2
    assert float(loss.item()) == pytest.approx(2 * math.log(259))


def test_perplexity_overflow_is_json_null() -> None:
    payload = EvaluationResult(loss=1000.0, tokens=1, batches=1).to_dict()
    assert payload["perplexity"] is None
    json.dumps(payload, allow_nan=False)


def test_empty_evaluation_is_rejected() -> None:
    with pytest.raises(TrainingError, match="empty split"):
        evaluate(_model(), [], batch_size=1, device=torch.device("cpu"))
