"""Token-weighted next-token evaluation on held-out, non-overlapping windows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from atlasai.core.model.transformer import AtlasTransformer
from atlasai.core.tokenizer.bpe import PAD_ID
from atlasai.training.dataset import TokenBlock
from atlasai.training.errors import TrainingError


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    tokens: int
    batches: int

    def to_dict(self) -> dict[str, float | int | None]:
        # A finite but enormous loss can overflow exp. Report null, not invalid JSON infinity.
        perplexity: float | None
        try:
            perplexity = math.exp(self.loss)
        except OverflowError:
            perplexity = None
        return {
            "loss": self.loss,
            "perplexity": perplexity,
            "tokens": self.tokens,
            "batches": self.batches,
        }


def blocks_to_tensors(
    blocks: Sequence[TokenBlock], *, device: torch.device
) -> tuple[Tensor, Tensor]:
    if not blocks:
        raise TrainingError("Cannot evaluate or train on an empty split")
    return (
        torch.tensor([block.input_ids for block in blocks], dtype=torch.long, device=device),
        torch.tensor([block.target_ids for block in blocks], dtype=torch.long, device=device),
    )


def token_loss(logits: Tensor, targets: Tensor) -> tuple[Tensor, int]:
    """Return summed cross-entropy and number of non-padding target tokens."""
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise TrainingError("Expected logits [B,T,V] and targets [B,T]")
    if targets.dtype != torch.long:
        raise TrainingError("targets must have dtype torch.long")
    count = int((targets != PAD_ID).sum().item())
    if count == 0:
        raise TrainingError("A batch must contain at least one non-padding target")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="sum",
    )
    return loss, count


def evaluate(
    model: AtlasTransformer,
    blocks: Sequence[TokenBlock],
    *,
    batch_size: int,
    device: torch.device,
) -> EvaluationResult:
    """Put model in eval mode and compute loss without building a gradient graph."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise TrainingError("batch_size must be a positive integer")
    inputs, targets = blocks_to_tensors(blocks, device=device)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(blocks), batch_size):
            stop = start + batch_size
            logits = model(inputs[start:stop])
            loss, count = token_loss(logits, targets[start:stop])
            if not bool(torch.isfinite(loss).item()):
                raise TrainingError("Evaluation produced non-finite loss")
            total_loss += float(loss.item())
            total_tokens += count
            batches += 1
    return EvaluationResult(loss=total_loss / total_tokens, tokens=total_tokens, batches=batches)
