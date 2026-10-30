"""Strict, dependency-free TOML configuration for Atlas AI.

Loading a configuration never creates directories, imports torch, loads a model,
or contacts the network. Missing model artifacts are therefore not config errors;
they are expected before the training stage.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_CONFIG_PATH = Path("configs/atlas.toml")
MAX_CONFIG_BYTES = 1024 * 1024


class ConfigError(ValueError):
    """A configuration file is missing, malformed, or contains invalid values."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Resource settings. Paths are absolute when produced by ``load_config``."""

    device: Literal["cpu"] = "cpu"
    seed: int = 42
    num_threads: int = 1
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Local artifact paths and decoding limits; the LLM adapter arrives at stage 4."""

    backend: Literal["atlas-local"] = "atlas-local"
    tokenizer_path: Path = Path("checkpoints/tokenizer.json")
    checkpoint_path: Path = Path("checkpoints/atlas.pt")
    context_length: int = 64
    max_new_tokens: int = 128
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Byte-level BPE training defaults."""

    vocab_size: int = 512
    min_frequency: int = 2


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Corpus and next-token block preparation defaults."""

    input_paths: tuple[Path, ...] = ()
    validation_input_paths: tuple[Path, ...] = ()
    stride: int = 64
    val_fraction: float = 0.1


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Small pre-norm decoder; vocabulary and context come from the artifacts."""

    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    tie_embeddings: bool = True

    def validate(self) -> None:
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int):
            raise ConfigError("transformer.d_model must be an integer")
        if not 8 <= self.d_model <= 512:
            raise ConfigError("transformer.d_model must be between 8 and 512")
        if isinstance(self.num_heads, bool) or not isinstance(self.num_heads, int):
            raise ConfigError("transformer.num_heads must be an integer")
        if not 1 <= self.num_heads <= 16 or self.d_model % self.num_heads:
            raise ConfigError("transformer.num_heads must be 1..16 and divide d_model")
        if isinstance(self.num_layers, bool) or not isinstance(self.num_layers, int):
            raise ConfigError("transformer.num_layers must be an integer")
        if not 1 <= self.num_layers <= 12:
            raise ConfigError("transformer.num_layers must be between 1 and 12")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ConfigError("transformer.dropout must be finite and in [0, 1)")
        if not isinstance(self.tie_embeddings, bool):
            raise ConfigError("transformer.tie_embeddings must be a boolean")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TransformerConfig:
        _check_keys(
            values,
            {"d_model", "num_layers", "num_heads", "dropout", "tie_embeddings"},
            "transformer",
        )
        defaults = cls()
        result = cls(
            d_model=_integer(values, "d_model", defaults.d_model, minimum=8, maximum=512),
            num_layers=_integer(values, "num_layers", defaults.num_layers, minimum=1, maximum=12),
            num_heads=_integer(values, "num_heads", defaults.num_heads, minimum=1, maximum=16),
            dropout=_number(values, "dropout", defaults.dropout, minimum=0.0, maximum=0.999),
            tie_embeddings=_boolean(values, "tie_embeddings", defaults.tie_embeddings),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Bounded CPU training defaults; no distributed or mixed-precision machinery."""

    batch_size: int = 1
    max_steps: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 20
    log_every: int = 10

    def validate(self) -> None:
        integer_bounds = (
            ("batch_size", self.batch_size, 1, 128),
            ("max_steps", self.max_steps, 1, 100_000),
            ("eval_every", self.eval_every, 1, 100_000),
            ("log_every", self.log_every, 1, 100_000),
        )
        for name, value, minimum, maximum in integer_bounds:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"training.{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ConfigError(f"training.{name} must be between {minimum} and {maximum}")
        number_bounds = (
            ("learning_rate", self.learning_rate, 1e-8, 1.0),
            ("weight_decay", self.weight_decay, 0.0, 1.0),
            ("grad_clip", self.grad_clip, 0.0, 1000.0),
        )
        for name, number, lower, upper in number_bounds:
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or not lower <= number <= upper
            ):
                raise ConfigError(f"training.{name} must be finite and between {lower} and {upper}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TrainingConfig:
        _check_keys(
            values,
            {
                "batch_size",
                "max_steps",
                "learning_rate",
                "weight_decay",
                "grad_clip",
                "eval_every",
                "log_every",
            },
            "training",
        )
        defaults = cls()
        result = cls(
            batch_size=_integer(values, "batch_size", defaults.batch_size, minimum=1, maximum=128),
            max_steps=_integer(values, "max_steps", defaults.max_steps, minimum=1, maximum=100_000),
            learning_rate=_number(
                values, "learning_rate", defaults.learning_rate, minimum=1e-8, maximum=1.0
            ),
            weight_decay=_number(
                values, "weight_decay", defaults.weight_decay, minimum=0.0, maximum=1.0
            ),
            grad_clip=_number(values, "grad_clip", defaults.grad_clip, minimum=0.0, maximum=1000.0),
            eval_every=_integer(
                values, "eval_every", defaults.eval_every, minimum=1, maximum=100_000
            ),
            log_every=_integer(values, "log_every", defaults.log_every, minimum=1, maximum=100_000),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class AtlasConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view, including normalized absolute paths."""
        return {
            "runtime": {
                "device": self.runtime.device,
                "seed": self.runtime.seed,
                "num_threads": self.runtime.num_threads,
                "data_dir": str(self.runtime.data_dir),
                "checkpoint_dir": str(self.runtime.checkpoint_dir),
            },
            "model": {
                "backend": self.model.backend,
                "tokenizer_path": str(self.model.tokenizer_path),
                "checkpoint_path": str(self.model.checkpoint_path),
                "context_length": self.model.context_length,
                "max_new_tokens": self.model.max_new_tokens,
                "temperature": self.model.temperature,
            },
            "tokenizer": {
                "vocab_size": self.tokenizer.vocab_size,
                "min_frequency": self.tokenizer.min_frequency,
            },
            "data": {
                "input_paths": [str(path) for path in self.data.input_paths],
                "validation_input_paths": [str(path) for path in self.data.validation_input_paths],
                "stride": self.data.stride,
                "val_fraction": self.data.val_fraction,
            },
            "transformer": {
                "d_model": self.transformer.d_model,
                "num_layers": self.transformer.num_layers,
                "num_heads": self.transformer.num_heads,
                "dropout": self.transformer.dropout,
                "tie_embeddings": self.transformer.tie_embeddings,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "max_steps": self.training.max_steps,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "grad_clip": self.training.grad_clip,
                "eval_every": self.training.eval_every,
                "log_every": self.training.log_every,
            },
        }


def _table(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a TOML table")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{name} contains a non-string key")
        result[key] = item
    return result


def _check_keys(table: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in {name}: {', '.join(unknown)}")


def _integer(
    table: Mapping[str, object], key: str, default: int, *, minimum: int, maximum: int
) -> int:
    value = table.get(key, default)
    # bool subclasses int in Python, but `num_threads = true` is a config mistake.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigError(f"{key} must be between {minimum} and {maximum}")
    return value


def _number(
    table: Mapping[str, object], key: str, default: float, *, minimum: float, maximum: float
) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ConfigError(f"{key} must be finite and between {minimum} and {maximum}")
    return result


def _boolean(table: Mapping[str, object], key: str, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _path(table: Mapping[str, object], key: str, default: Path, base_dir: Path) -> Path:
    value = table.get(key, str(default))
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{key} must be a non-empty path without NUL bytes")
    try:
        candidate = Path(value).expanduser()
        return (base_dir / candidate).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"Cannot resolve {key}: {exc}") from exc


def _path_list(
    table: Mapping[str, object], key: str, default: tuple[Path, ...], base_dir: Path
) -> tuple[Path, ...]:
    value = table.get(key, [str(path) for path in default])
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a list of paths")
    result: list[Path] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or "\x00" in item:
            raise ConfigError(f"{key} must contain non-empty paths without NUL bytes")
        try:
            result.append((base_dir / Path(item).expanduser()).resolve(strict=False))
        except (OSError, RuntimeError) as exc:
            raise ConfigError(f"Cannot resolve {key}: {exc}") from exc
    return tuple(result)


def parse_config(data: Mapping[str, object], *, base_dir: Path) -> AtlasConfig:
    """Validate a TOML mapping and resolve relative paths against ``base_dir``."""
    _check_keys(data, {"runtime", "model", "tokenizer", "data", "transformer", "training"}, "root")
    runtime_defaults = RuntimeConfig()
    model_defaults = ModelConfig()
    tokenizer_defaults = TokenizerConfig()
    data_defaults = DataConfig()
    runtime_table = _table(data.get("runtime", {}), "runtime")
    model_table = _table(data.get("model", {}), "model")
    tokenizer_table = _table(data.get("tokenizer", {}), "tokenizer")
    data_table = _table(data.get("data", {}), "data")
    transformer_table = _table(data.get("transformer", {}), "transformer")
    training_table = _table(data.get("training", {}), "training")
    _check_keys(
        runtime_table,
        {"device", "seed", "num_threads", "data_dir", "checkpoint_dir"},
        "runtime",
    )
    _check_keys(
        model_table,
        {
            "backend",
            "tokenizer_path",
            "checkpoint_path",
            "context_length",
            "max_new_tokens",
            "temperature",
        },
        "model",
    )
    _check_keys(tokenizer_table, {"vocab_size", "min_frequency"}, "tokenizer")
    _check_keys(
        data_table,
        {"input_paths", "validation_input_paths", "stride", "val_fraction"},
        "data",
    )

    device = runtime_table.get("device", runtime_defaults.device)
    if not isinstance(device, str) or device != "cpu":
        raise ConfigError("runtime.device must be 'cpu'; GPU support is not implemented yet")
    backend = model_table.get("backend", model_defaults.backend)
    if not isinstance(backend, str) or backend != "atlas-local":
        raise ConfigError("model.backend must be 'atlas-local'; external backends are not used")

    runtime = RuntimeConfig(
        device="cpu",
        seed=_integer(runtime_table, "seed", runtime_defaults.seed, minimum=0, maximum=2**32 - 1),
        num_threads=_integer(
            runtime_table, "num_threads", runtime_defaults.num_threads, minimum=1, maximum=64
        ),
        data_dir=_path(runtime_table, "data_dir", runtime_defaults.data_dir, base_dir),
        checkpoint_dir=_path(
            runtime_table, "checkpoint_dir", runtime_defaults.checkpoint_dir, base_dir
        ),
    )
    model = ModelConfig(
        backend="atlas-local",
        tokenizer_path=_path(
            model_table, "tokenizer_path", model_defaults.tokenizer_path, base_dir
        ),
        checkpoint_path=_path(
            model_table, "checkpoint_path", model_defaults.checkpoint_path, base_dir
        ),
        context_length=_integer(
            model_table, "context_length", model_defaults.context_length, minimum=8, maximum=4096
        ),
        max_new_tokens=_integer(
            model_table, "max_new_tokens", model_defaults.max_new_tokens, minimum=1, maximum=1024
        ),
        temperature=_number(
            model_table, "temperature", model_defaults.temperature, minimum=0.0, maximum=2.0
        ),
    )
    tokenizer = TokenizerConfig(
        vocab_size=_integer(
            tokenizer_table,
            "vocab_size",
            tokenizer_defaults.vocab_size,
            minimum=259,
            maximum=65_536,
        ),
        min_frequency=_integer(
            tokenizer_table,
            "min_frequency",
            tokenizer_defaults.min_frequency,
            minimum=1,
            maximum=1_000_000,
        ),
    )
    data_config = DataConfig(
        input_paths=_path_list(data_table, "input_paths", data_defaults.input_paths, base_dir),
        validation_input_paths=_path_list(
            data_table,
            "validation_input_paths",
            data_defaults.validation_input_paths,
            base_dir,
        ),
        stride=_integer(data_table, "stride", data_defaults.stride, minimum=1, maximum=4096),
        val_fraction=_number(
            data_table, "val_fraction", data_defaults.val_fraction, minimum=0.0, maximum=0.999
        ),
    )
    return AtlasConfig(
        runtime=runtime,
        model=model,
        tokenizer=tokenizer,
        data=data_config,
        transformer=TransformerConfig.from_mapping(transformer_table),
        training=TrainingConfig.from_mapping(training_table),
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AtlasConfig:
    """Load TOML from disk. Expected failures are wrapped in ``ConfigError``."""
    config_path = Path(path).expanduser()
    try:
        config_path = config_path.resolve(strict=False)
        size = config_path.stat().st_size
        if size > MAX_CONFIG_BYTES:
            raise ConfigError(f"Config is too large: {size} bytes (limit {MAX_CONFIG_BYTES})")
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Config must be UTF-8: {config_path}: {exc}") from exc
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"Cannot read config {config_path}: {exc}") from exc
    return parse_config(data, base_dir=config_path.parent)
