"""Strict, dependency-free TOML configuration for the stage-1 scaffold.

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
    """Settings reserved for our own local model; no inference happens at stage 1."""

    backend: Literal["atlas-local"] = "atlas-local"
    tokenizer_path: Path = Path("checkpoints/tokenizer.json")
    checkpoint_path: Path = Path("checkpoints/atlas.pt")
    context_length: int = 128
    max_new_tokens: int = 128
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class AtlasConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

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


def _path(table: Mapping[str, object], key: str, default: Path, base_dir: Path) -> Path:
    value = table.get(key, str(default))
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{key} must be a non-empty path without NUL bytes")
    try:
        candidate = Path(value).expanduser()
        return (base_dir / candidate).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"Cannot resolve {key}: {exc}") from exc


def parse_config(data: Mapping[str, object], *, base_dir: Path) -> AtlasConfig:
    """Validate a TOML mapping and resolve relative paths against ``base_dir``."""
    _check_keys(data, {"runtime", "model"}, "root")
    runtime_defaults = RuntimeConfig()
    model_defaults = ModelConfig()
    runtime_table = _table(data.get("runtime", {}), "runtime")
    model_table = _table(data.get("model", {}), "model")
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
    return AtlasConfig(runtime=runtime, model=model)


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
