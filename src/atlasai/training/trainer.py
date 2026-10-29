"""Bounded, reproducible CPU training for the small Atlas decoder."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from atlasai.config import ConfigError, TrainingConfig
from atlasai.core.model.transformer import AtlasTransformer
from atlasai.training.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from atlasai.training.dataset import TokenDataset
from atlasai.training.errors import TrainingError
from atlasai.training.evaluation import EvaluationResult, blocks_to_tensors, evaluate, token_loss


@dataclass(frozen=True, slots=True)
class TrainingResult:
    steps_completed: int
    parameter_count: int
    duration_seconds: float
    initial_validation: EvaluationResult
    best_validation: EvaluationResult
    best_step: int
    final_train_loss: float
    checkpoint_path: str
    metrics_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "steps_completed": self.steps_completed,
            "parameter_count": self.parameter_count,
            "duration_seconds": self.duration_seconds,
            "initial_validation": self.initial_validation.to_dict(),
            "best_validation": self.best_validation.to_dict(),
            "best_step": self.best_step,
            "final_train_loss": self.final_train_loss,
            "checkpoint_path": self.checkpoint_path,
            "metrics_path": self.metrics_path,
        }


def configure_torch(*, seed: int, num_threads: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise TrainingError("seed must be an unsigned 32-bit integer")
    if (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, int)
        or not 1 <= num_threads <= 64
    ):
        raise TrainingError("num_threads must be between 1 and 64")
    torch.set_num_threads(num_threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_model(
    model: AtlasTransformer,
    dataset: TokenDataset,
    *,
    tokenizer_path: str | Path,
    checkpoint_path: str | Path,
    config: TrainingConfig,
    seed: int,
    num_threads: int,
    verbose: bool = False,
) -> TrainingResult:
    """Train with AdamW, keep the best validation checkpoint, and verify reload.

    The stage-2 corpus is held in memory. Non-overlapping windows are required so
    train and validation do not share target tokens. This is a pipeline smoke test,
    not a document-level generalization benchmark.
    """
    try:
        config.validate()
    except ConfigError as exc:
        raise TrainingError(str(exc)) from exc
    configure_torch(seed=seed, num_threads=num_threads)
    if not dataset.train or not dataset.val:
        raise TrainingError("Training requires non-empty train and validation splits")
    context = dataset.stats.context_length
    if context != model.context_length:
        raise TrainingError("Dataset context_length does not match the model")
    if dataset.stats.stride != context:
        raise TrainingError(
            "Training requires stride == context_length to prevent train/validation token overlap"
        )
    tokenizer = Path(tokenizer_path).expanduser().resolve()
    output = Path(checkpoint_path).expanduser().resolve()
    file_sha256(tokenizer)  # Fail before spending compute if the tokenizer is missing.
    if output == tokenizer:
        raise TrainingError("Checkpoint output must not overwrite the tokenizer")

    device = torch.device("cpu")
    model.to(device)
    train_inputs, train_targets = blocks_to_tensors(dataset.train, device=device)
    if int(train_targets.min().item()) < 0 or int(train_targets.max().item()) >= model.vocab_size:
        raise TrainingError("Dataset targets contain a token outside the model vocabulary")
    initial = evaluate(model, dataset.val, batch_size=config.batch_size, device=device)
    best = initial
    best_step = 0
    save_checkpoint(output, model, tokenizer_path=tokenizer, step=0, validation_loss=initial.loss)
    history: list[dict[str, object]] = [
        {"step": 0, "train_loss": None, "validation": initial.to_dict()}
    ]
    if verbose:
        print(f"step=0 validation_loss={initial.loss:.6f}", file=sys.stderr)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameters = list(model.parameters())
    started = time.perf_counter()
    step = 0
    window_loss = 0.0
    window_batches = 0
    final_train_loss = 0.0
    model.train()
    while step < config.max_steps:
        order = torch.randperm(len(dataset.train), generator=generator).tolist()
        for offset in range(0, len(order), config.batch_size):
            if step >= config.max_steps:
                break
            indices = torch.tensor(order[offset : offset + config.batch_size], dtype=torch.long)
            inputs = train_inputs[indices]
            targets = train_targets[indices]
            optimizer.zero_grad(set_to_none=True)
            try:
                summed_loss, count = token_loss(model(inputs), targets)
                loss = summed_loss / count
                if not bool(torch.isfinite(loss).item()):
                    raise TrainingError("Training produced non-finite loss")
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=config.grad_clip if config.grad_clip > 0 else float("inf"),
                    error_if_nonfinite=True,
                )
                optimizer.step()
            except RuntimeError as exc:
                raise TrainingError(f"Optimizer step {step + 1} failed: {exc}") from exc
            step += 1
            batch_loss = float(loss.item())
            window_loss += batch_loss
            window_batches += 1
            if verbose and step % config.log_every == 0:
                print(f"step={step} train_loss={batch_loss:.6f}", file=sys.stderr)
            if step % config.eval_every == 0 or step == config.max_steps:
                validation = evaluate(
                    model, dataset.val, batch_size=config.batch_size, device=device
                )
                mean_train_loss = window_loss / window_batches
                final_train_loss = mean_train_loss
                history.append(
                    {
                        "step": step,
                        "train_loss": mean_train_loss,
                        "validation": validation.to_dict(),
                    }
                )
                if verbose:
                    print(
                        f"step={step} validation_loss={validation.loss:.6f}",
                        file=sys.stderr,
                    )
                if validation.loss < best.loss:
                    best = validation
                    best_step = step
                    save_checkpoint(
                        output,
                        model,
                        tokenizer_path=tokenizer,
                        step=step,
                        validation_loss=best.loss,
                    )
                window_loss = 0.0
                window_batches = 0
                model.train()

    info = load_checkpoint(output, model, tokenizer_path=tokenizer)
    reloaded = evaluate(model, dataset.val, batch_size=config.batch_size, device=device)
    if abs(reloaded.loss - best.loss) > 1e-6:
        raise TrainingError("Reloaded best checkpoint did not reproduce its validation loss")
    result = TrainingResult(
        steps_completed=step,
        parameter_count=model.parameter_count(),
        duration_seconds=time.perf_counter() - started,
        initial_validation=initial,
        best_validation=best,
        best_step=best_step,
        final_train_loss=final_train_loss,
        checkpoint_path=str(output),
        metrics_path=str(output.with_suffix(".metrics.json")),
    )
    report = {
        "result": result.to_dict(),
        "checkpoint": info.to_dict(),
        "training": asdict(config),
        "dataset": dataset.stats.to_dict(),
        "seed": seed,
        "num_threads": num_threads,
        "history": history,
    }
    try:
        Path(result.metrics_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        raise TrainingError(f"Cannot write training metrics: {exc}") from exc
    return result
