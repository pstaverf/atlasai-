from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not installed or cannot load", allow_module_level=True)

from atlasai.config import TransformerConfig
from atlasai.core.model.base import LLM, LLMError
from atlasai.core.model.local import AtlasLocalModel
from atlasai.core.model.transformer import AtlasTransformer
from atlasai.core.tokenizer.bpe import BOS_ID, EOS_ID, PAD_ID, BPETokenizer
from atlasai.training.checkpoint import save_checkpoint


def _backend(
    tmp_path: Path, *, max_new_tokens: int = 8, temperature: float = 0.0, seed: int = 42
) -> AtlasLocalModel:
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "model.pt"
    BPETokenizer().save(tokenizer_path)
    torch.manual_seed(5)
    model = AtlasTransformer(
        vocab_size=259,
        context_length=16,
        config=TransformerConfig(d_model=8, num_layers=1, num_heads=2, dropout=0.0),
    )
    save_checkpoint(
        checkpoint_path, model, tokenizer_path=tokenizer_path, step=0, validation_loss=1.0
    )
    return AtlasLocalModel(
        checkpoint_path,
        tokenizer_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )


def test_local_model_satisfies_llm_protocol(tmp_path: Path) -> None:
    llm: LLM = _backend(tmp_path)
    result = llm.generate("Atlas", max_new_tokens=3)
    assert result.prompt_tokens > 0
    assert result.completion_tokens <= 3
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    assert result.stop_reason in {"eos", "max_tokens"}
    assert isinstance(result.text, str)


def test_greedy_generation_is_deterministic(tmp_path: Path) -> None:
    first = _backend(tmp_path / "first")
    second = _backend(tmp_path / "second")
    assert first.generate("hello", max_new_tokens=4) == second.generate("hello", max_new_tokens=4)


def test_sampling_is_reproducible_with_same_seed(tmp_path: Path) -> None:
    first = _backend(tmp_path / "first", temperature=0.8, seed=123)
    second = _backend(tmp_path / "second", temperature=0.8, seed=123)
    assert first.generate("hello", max_new_tokens=6) == second.generate("hello", max_new_tokens=6)


def test_eos_stops_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(tmp_path)

    def force_eos(logits: torch.Tensor, temperature: float) -> int:
        return EOS_ID

    monkeypatch.setattr(backend, "_next_token", force_eos)
    result = backend.generate("hello", max_new_tokens=5)
    assert result.stop_reason == "eos"
    assert result.completion_tokens == 0
    assert result.text == ""


def test_long_prompt_is_truncated_to_context(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    result = backend.generate("a" * 1000, max_new_tokens=2)
    assert result.prompt_tokens > backend.context_length
    assert result.completion_tokens <= 2


def test_request_cannot_exceed_backend_limit(tmp_path: Path) -> None:
    backend = _backend(tmp_path, max_new_tokens=4)
    with pytest.raises(LLMError, match="max_new_tokens"):
        backend.generate("hello", max_new_tokens=5)


@pytest.mark.parametrize("temperature", [-0.1, 2.1, float("nan"), True])
def test_invalid_temperature(tmp_path: Path, temperature: object) -> None:
    backend = _backend(tmp_path)
    with pytest.raises(LLMError):
        backend.generate("hello", temperature=temperature)  # type: ignore[arg-type]


def test_invalid_prompt(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    with pytest.raises(LLMError, match="prompt"):
        backend.generate(123)  # type: ignore[arg-type]


def test_control_tokens_are_masked(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    logits = torch.zeros(259)
    logits[PAD_ID] = 100.0
    logits[BOS_ID] = 90.0
    assert backend._next_token(logits, 0.0) not in {PAD_ID, BOS_ID}
    assert backend._next_token(logits, 1.0) not in {PAD_ID, BOS_ID}


def test_info_reports_parameters(tmp_path: Path) -> None:
    info = _backend(tmp_path).info()
    assert info["backend"] == "atlas-local"
    assert isinstance(info["parameter_count"], int)
    assert info["parameter_count"] > 0
