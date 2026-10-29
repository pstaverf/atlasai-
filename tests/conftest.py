from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "examples" / "corpus.txt"


@pytest.fixture
def tiny_config(tmp_path: Path) -> Path:
    """A fast CPU config for CLI tests; production defaults are much larger."""
    path = tmp_path / "tiny.toml"
    path.write_text(
        f"""
[runtime]
device = "cpu"
seed = 42
num_threads = 1

[model]
backend = "atlas-local"
tokenizer_path = "tokenizer.json"
checkpoint_path = "model.pt"
context_length = 64
max_new_tokens = 8
temperature = 0.0

[data]
input_paths = ["{CORPUS.as_posix()}"]
stride = 64
val_fraction = 0.2

[transformer]
d_model = 8
num_layers = 1
num_heads = 2
dropout = 0.0
tie_embeddings = true

[training]
batch_size = 2
max_steps = 2
learning_rate = 0.001
eval_every = 1
log_every = 1
""",
        encoding="utf-8",
    )
    return path
