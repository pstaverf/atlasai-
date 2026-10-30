from __future__ import annotations

from pathlib import Path

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch is not installed or cannot load", allow_module_level=True)

from atlasai.config import load_config
from atlasai.core.model.transformer import AtlasTransformer

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config_name", ["atlas.toml", "smoke.toml"])
def test_default_model_has_at_least_ten_million_parameters(config_name: str) -> None:
    config = load_config(REPO / "configs" / config_name)
    torch.manual_seed(0)
    model = AtlasTransformer(
        vocab_size=config.tokenizer.vocab_size,
        context_length=config.model.context_length,
        config=config.transformer,
    )
    assert model.parameter_count() >= 10_000_000
