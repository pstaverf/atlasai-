from __future__ import annotations

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not installed or cannot load", allow_module_level=True)

from atlasai.config import ConfigError, TransformerConfig
from atlasai.core.model.transformer import AtlasTransformer


def _model(*, dropout: float = 0.1, tie: bool = True) -> AtlasTransformer:
    torch.manual_seed(7)
    return AtlasTransformer(
        vocab_size=17,
        context_length=8,
        config=TransformerConfig(
            d_model=8, num_layers=1, num_heads=2, dropout=dropout, tie_embeddings=tie
        ),
    )


def test_logits_shape_and_finiteness() -> None:
    model = _model(dropout=0.0).eval()
    ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 4, 17)
    assert bool(torch.isfinite(logits).all().item())
    assert model.parameter_count() > 0


def test_future_tokens_cannot_change_past_logits() -> None:
    model = _model().eval()
    ids = torch.arange(3, 11, dtype=torch.long).unsqueeze(0)
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([[16, 15, 14, 13]], dtype=torch.long)
    with torch.no_grad():
        original = model(ids)
        altered = model(changed)
    torch.testing.assert_close(original[:, :4], altered[:, :4], rtol=0, atol=0)
    assert not torch.allclose(original[:, 4:], altered[:, 4:])


def test_gradients_reach_every_parameter() -> None:
    model = _model(dropout=0.0)
    ids = torch.arange(3, 11, dtype=torch.long).unsqueeze(0)
    loss = model(ids).square().mean()
    torch.autograd.backward(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all().item()), name
        assert float(parameter.grad.abs().sum().item()) > 0, name


def test_tied_embeddings_reduce_parameters() -> None:
    tied = _model()
    untied = _model(tie=False)
    assert tied.parameter_count() < untied.parameter_count()
    assert tied.lm_head.weight is tied.token_embedding.weight


def test_eval_is_deterministic_with_dropout() -> None:
    model = _model().eval()
    ids = torch.tensor([[3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(model(ids), model(ids), rtol=0, atol=0)


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([3, 4], dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
        torch.empty((0, 2), dtype=torch.long),
        torch.arange(3, 12, dtype=torch.long).unsqueeze(0),
        torch.tensor([[17]], dtype=torch.long),
        torch.tensor([[-1]], dtype=torch.long),
        torch.tensor([[3.0]], dtype=torch.float32),
    ],
)
def test_invalid_inputs(ids: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        _model()(ids)


@pytest.mark.parametrize(
    "config",
    [
        TransformerConfig(d_model=9, num_heads=2),
        TransformerConfig(num_layers=0),
        TransformerConfig(dropout=1.0),
        TransformerConfig(dropout=float("nan")),
        TransformerConfig(tie_embeddings=1),  # type: ignore[arg-type]
    ],
)
def test_invalid_architecture(config: TransformerConfig) -> None:
    with pytest.raises(ConfigError):
        AtlasTransformer(vocab_size=17, context_length=8, config=config)
