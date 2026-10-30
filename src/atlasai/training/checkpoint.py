"""Versioned model checkpoints with restricted, weights-only deserialization.

Checkpoints contain model weights and provenance, not optimizer state. They can
be evaluated or used for inference; resuming an interrupted optimizer is not
implemented. Untrusted arbitrary Python pickle loading is never enabled.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from atlasai import __version__
from atlasai.config import ConfigError, TransformerConfig
from atlasai.core.model.transformer import AtlasTransformer
from atlasai.training.errors import CheckpointError

CHECKPOINT_FORMAT = "atlas-transformer-checkpoint"
CHECKPOINT_VERSION = 1
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
_PAYLOAD_KEYS = {
    "format",
    "version",
    "atlas_version",
    "torch_version",
    "architecture",
    "context_length",
    "vocab_size",
    "tokenizer_sha256",
    "step",
    "validation_loss",
    "model_state",
}


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    step: int
    validation_loss: float
    tokenizer_sha256: str
    atlas_version: str
    torch_version: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointError(f"Cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _integer(payload: dict[str, object], key: str, *, minimum: int, maximum: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CheckpointError(
            f"Checkpoint {key} must be an integer between {minimum} and {maximum}"
        )
    return value


def _read_payload(path: Path) -> dict[str, object]:
    try:
        size = path.stat().st_size
        if size > MAX_CHECKPOINT_BYTES:
            raise CheckpointError(
                f"Checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte safety limit"
            )
        payload: object = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError as exc:
        raise CheckpointError(f"Checkpoint not found: {path}") from exc
    except (OSError, RuntimeError, ValueError, pickle.PickleError, EOFError) as exc:
        raise CheckpointError(f"Cannot safely load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise CheckpointError("Checkpoint payload must be a string-keyed object")
    result: dict[str, object] = dict(payload)
    if set(result) != _PAYLOAD_KEYS:
        raise CheckpointError("Checkpoint payload has missing or unknown fields")
    version = result["version"]
    if (
        result["format"] != CHECKPOINT_FORMAT
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version != CHECKPOINT_VERSION
    ):
        raise CheckpointError("Unsupported checkpoint format or version")
    for key in ("atlas_version", "torch_version", "tokenizer_sha256"):
        if not isinstance(result[key], str):
            raise CheckpointError(f"Checkpoint {key} must be a string")
    digest = result["tokenizer_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise CheckpointError("Checkpoint tokenizer_sha256 is invalid")
    _integer(result, "step", minimum=0, maximum=100_000)
    _integer(result, "context_length", minimum=1, maximum=4096)
    _integer(result, "vocab_size", minimum=3, maximum=65_536)
    loss = result["validation_loss"]
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(loss)
        or loss < 0
    ):
        raise CheckpointError("Checkpoint validation_loss must be finite and non-negative")
    return result


def _model_from_payload(payload: dict[str, object]) -> AtlasTransformer:
    architecture = payload["architecture"]
    if not isinstance(architecture, dict) or not all(isinstance(key, str) for key in architecture):
        raise CheckpointError("Checkpoint architecture must be a string-keyed object")
    if set(architecture) != {"d_model", "num_layers", "num_heads", "dropout", "tie_embeddings"}:
        raise CheckpointError("Checkpoint architecture has missing or unknown fields")
    try:
        config = TransformerConfig.from_mapping(dict(architecture))
        return AtlasTransformer(
            vocab_size=_integer(payload, "vocab_size", minimum=3, maximum=65_536),
            context_length=_integer(payload, "context_length", minimum=1, maximum=4096),
            config=config,
        )
    except (ConfigError, ValueError) as exc:
        raise CheckpointError(f"Invalid checkpoint architecture: {exc}") from exc


def _state(payload: dict[str, object]) -> dict[str, torch.Tensor]:
    raw = payload["model_state"]
    if not isinstance(raw, dict) or not raw:
        raise CheckpointError("Checkpoint model_state must be a non-empty tensor mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise CheckpointError("Checkpoint model_state must map names to tensors")
        if not bool(torch.isfinite(value).all().item()):
            raise CheckpointError(f"Checkpoint contains non-finite weights: {key}")
        state[key] = value
    return state


def _info(payload: dict[str, object]) -> CheckpointInfo:
    # _read_payload has validated these fields.
    step = payload["step"]
    loss = payload["validation_loss"]
    digest = payload["tokenizer_sha256"]
    atlas_version = payload["atlas_version"]
    torch_version = payload["torch_version"]
    if (
        not isinstance(step, int)
        or not isinstance(loss, (int, float))
        or not isinstance(digest, str)
        or not isinstance(atlas_version, str)
        or not isinstance(torch_version, str)
    ):
        raise CheckpointError("Invalid checkpoint metadata types")
    return CheckpointInfo(step, float(loss), digest, atlas_version, torch_version)


def _check_tokenizer(payload: dict[str, object], tokenizer_path: Path) -> None:
    if payload["tokenizer_sha256"] != file_sha256(tokenizer_path):
        raise CheckpointError(
            "Tokenizer fingerprint differs from the checkpoint's training tokenizer"
        )


def _apply_state(model: AtlasTransformer, payload: dict[str, object]) -> CheckpointInfo:
    try:
        model.load_state_dict(_state(payload), strict=True)
    except (RuntimeError, CheckpointError) as exc:
        raise CheckpointError(f"Checkpoint weights are incompatible: {exc}") from exc
    model.eval()
    return _info(payload)


def save_checkpoint(
    path: str | Path,
    model: AtlasTransformer,
    *,
    tokenizer_path: str | Path,
    step: int,
    validation_loss: float,
) -> CheckpointInfo:
    """Atomically save CPU weights, architecture, and the exact tokenizer fingerprint."""
    target = Path(path).expanduser().resolve()
    tokenizer = Path(tokenizer_path).expanduser().resolve()
    if target == tokenizer:
        raise CheckpointError("Checkpoint output must not overwrite the tokenizer")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 100_000:
        raise CheckpointError("step must be an integer between 0 and 100000")
    if (
        isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(validation_loss)
        or validation_loss < 0
    ):
        raise CheckpointError("validation_loss must be finite and non-negative")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "atlas_version": __version__,
        "torch_version": str(torch.__version__),
        "architecture": asdict(model.config),
        "context_length": model.context_length,
        "vocab_size": model.vocab_size,
        "tokenizer_sha256": file_sha256(tokenizer),
        "step": step,
        "validation_loss": validation_loss,
        "model_state": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
    }
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except (OSError, RuntimeError, ValueError, pickle.PickleError) as exc:
        raise CheckpointError(f"Cannot save checkpoint {target}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise CheckpointError(
                    f"Cannot clean temporary checkpoint {temporary}: {exc}"
                ) from exc
    return _info(payload)


def load_checkpoint(
    path: str | Path, model: AtlasTransformer, *, tokenizer_path: str | Path
) -> CheckpointInfo:
    """Load weights into a matching model; reject a changed tokenizer or architecture."""
    payload = _read_payload(Path(path).expanduser().resolve())
    _check_tokenizer(payload, Path(tokenizer_path).expanduser().resolve())
    expected = _model_from_payload(payload)
    if (
        expected.config != model.config
        or expected.context_length != model.context_length
        or expected.vocab_size != model.vocab_size
    ):
        raise CheckpointError("Checkpoint architecture does not match the supplied model")
    return _apply_state(model, payload)


def load_model_checkpoint(
    path: str | Path, *, tokenizer_path: str | Path
) -> tuple[AtlasTransformer, CheckpointInfo]:
    """Reconstruct the architecture from safe metadata and load its weights on CPU."""
    payload = _read_payload(Path(path).expanduser().resolve())
    _check_tokenizer(payload, Path(tokenizer_path).expanduser().resolve())
    model = _model_from_payload(payload)
    return model, _apply_state(model, payload)
