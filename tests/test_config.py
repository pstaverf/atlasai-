from __future__ import annotations

from pathlib import Path

import pytest

from atlasai.config import MAX_CONFIG_BYTES, ConfigError, load_config, parse_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "atlas.toml"


def test_default_config_loads_without_artifacts() -> None:
    config = load_config(REPO_CONFIG)
    assert config.runtime.device == "cpu"
    assert config.runtime.num_threads == 1
    assert config.model.backend == "atlas-local"
    assert config.model.context_length == 128
    assert config.tokenizer.vocab_size == 512
    assert config.data.input_paths == (REPO_CONFIG.parents[1] / "examples" / "corpus.txt",)
    assert not config.model.checkpoint_path.exists()


def test_relative_paths_are_resolved_from_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "atlas.toml"
    path.write_text(
        '[runtime]\ndata_dir = "../data"\n[model]\ncheckpoint_path = "../weights/model.pt"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.runtime.data_dir == tmp_path / "data"
    assert config.model.checkpoint_path == tmp_path / "weights" / "model.pt"


def test_empty_config_uses_defaults() -> None:
    config = parse_config({}, base_dir=Path("/tmp/atlas-config"))
    assert config.runtime.device == "cpu"
    assert config.model.backend == "atlas-local"
    assert config.runtime.data_dir == Path("/tmp/atlas-config/data")
    assert config.tokenizer.min_frequency == 2
    assert config.data.input_paths == ()


def test_missing_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "missing.toml")


def test_invalid_toml_is_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[runtime\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path)


def test_invalid_utf8_is_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(path)


def test_oversized_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "large.toml"
    path.write_bytes(b" " * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigError, match="too large"):
        load_config(path)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"unknown": {}}, "Unknown keys in root"),
        ({"runtime": {"unknown": 1}}, "Unknown keys in runtime"),
        ({"model": {"unknown": 1}}, "Unknown keys in model"),
        ({"runtime": []}, "runtime must be a TOML table"),
        ({"runtime": {"device": "cuda"}}, "runtime.device"),
        ({"model": {"backend": "ollama"}}, "model.backend"),
        ({"model": {"backend": "openai"}}, "model.backend"),
        ({"runtime": {"num_threads": 0}}, "num_threads"),
        ({"runtime": {"num_threads": True}}, "num_threads must be an integer"),
        ({"runtime": {"seed": -1}}, "seed"),
        ({"model": {"context_length": 4}}, "context_length"),
        ({"model": {"max_new_tokens": 0}}, "max_new_tokens"),
        ({"model": {"temperature": -0.1}}, "temperature"),
        ({"model": {"temperature": 2.1}}, "temperature"),
        ({"model": {"temperature": True}}, "temperature must be a number"),
        ({"runtime": {"data_dir": ""}}, "data_dir"),
        ({"runtime": {"data_dir": "bad\x00path"}}, "NUL"),
        ({"tokenizer": {"vocab_size": 258}}, "vocab_size"),
        ({"tokenizer": {"min_frequency": 0}}, "min_frequency"),
        ({"tokenizer": {"unknown": 1}}, "Unknown keys in tokenizer"),
        ({"data": {"input_paths": "not-a-list"}}, "input_paths"),
        ({"data": {"input_paths": [""]}}, "input_paths"),
        ({"data": {"stride": 0}}, "stride"),
        ({"data": {"val_fraction": 1.0}}, "val_fraction"),
        ({"data": {"unknown": 1}}, "Unknown keys in data"),
    ],
)
def test_invalid_values_are_rejected(data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(data, base_dir=Path("/tmp"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_temperature_is_rejected(value: str) -> None:
    with pytest.raises(ConfigError, match="finite"):
        parse_config({"model": {"temperature": float(value)}}, base_dir=Path("/tmp"))


def test_integer_temperature_is_accepted() -> None:
    config = parse_config({"model": {"temperature": 1}}, base_dir=Path("/tmp"))
    assert config.model.temperature == 1.0


def test_config_view_is_json_serializable() -> None:
    import json

    config = load_config(REPO_CONFIG)
    decoded = json.loads(json.dumps(config.to_dict()))
    assert decoded["model"]["backend"] == "atlas-local"
    assert Path(decoded["runtime"]["data_dir"]).is_absolute()
