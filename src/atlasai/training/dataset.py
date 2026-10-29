"""Prepare next-token-prediction blocks from UTF-8 text documents.

This module is deliberately PyTorch-free. The trainer converts these blocks to
tensors without changing tokenizer or corpus preparation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from atlasai.core.tokenizer.bpe import EOS_ID, BPETokenizer


class DatasetError(ValueError):
    """Corpus loading or block preparation failed."""


@dataclass(frozen=True, slots=True)
class TokenBlock:
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DatasetStats:
    documents: int
    total_tokens: int
    train_blocks: int
    val_blocks: int
    context_length: int
    stride: int
    dropped_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "documents": self.documents,
            "total_tokens": self.total_tokens,
            "train_blocks": self.train_blocks,
            "val_blocks": self.val_blocks,
            "context_length": self.context_length,
            "stride": self.stride,
            "dropped_tokens": self.dropped_tokens,
        }


@dataclass(frozen=True, slots=True)
class TokenDataset:
    train: tuple[TokenBlock, ...]
    val: tuple[TokenBlock, ...]
    stats: DatasetStats

    def __len__(self) -> int:
        return len(self.train) + len(self.val)


def read_documents(paths: Sequence[Path]) -> Iterator[str]:
    """Yield each UTF-8 text file as one document."""
    if not paths:
        raise DatasetError("At least one input document is required")
    for path in paths:
        if not path.is_file():
            raise DatasetError(f"Input document not found: {path}")
        try:
            yield path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DatasetError(f"Cannot read document {path}: {exc}") from exc


def build_blocks(
    token_ids: Sequence[int], *, context_length: int, stride: int
) -> tuple[list[TokenBlock], int]:
    """Create shifted input/target windows and return the unused tail length."""
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        raise DatasetError("context_length must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise DatasetError("stride must be an integer")
    if not 1 <= stride <= context_length:
        raise DatasetError("stride must be between 1 and context_length")

    block_size = context_length + 1
    if len(token_ids) < block_size:
        return [], len(token_ids)
    blocks: list[TokenBlock] = []
    last_start = 0
    for start in range(0, len(token_ids) - block_size + 1, stride):
        window = token_ids[start : start + block_size]
        blocks.append(TokenBlock(input_ids=tuple(window[:-1]), target_ids=tuple(window[1:])))
        last_start = start
    dropped = len(token_ids) - (last_start + block_size)
    return blocks, dropped


def split_blocks(
    blocks: Sequence[TokenBlock], *, val_fraction: float, seed: int
) -> tuple[tuple[TokenBlock, ...], tuple[TokenBlock, ...]]:
    """Deterministically shuffle blocks into train and validation splits."""
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)):
        raise DatasetError("val_fraction must be a number")
    fraction = float(val_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise DatasetError("val_fraction must be finite and in [0, 1)")
    if fraction == 0.0:
        return tuple(blocks), ()
    if len(blocks) < 2:
        raise DatasetError("At least two blocks are required for a validation split")

    val_count = int(len(blocks) * fraction)
    val_count = max(1, min(val_count, len(blocks) - 1))
    indices = list(range(len(blocks)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:val_count])
    train = tuple(block for index, block in enumerate(blocks) if index not in val_indices)
    val = tuple(block for index, block in enumerate(blocks) if index in val_indices)
    return train, val


def load_dataset(
    paths: Sequence[Path],
    tokenizer: BPETokenizer,
    *,
    context_length: int,
    stride: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> TokenDataset:
    """Read documents, insert EOS boundaries, and build train/validation blocks."""
    documents = list(read_documents(paths))
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document))
        token_ids.append(EOS_ID)

    blocks, dropped = build_blocks(token_ids, context_length=context_length, stride=stride)
    if not blocks:
        raise DatasetError(
            f"Corpus produced {len(token_ids)} tokens; need at least {context_length + 1} "
            "for one block"
        )
    train, val = split_blocks(blocks, val_fraction=val_fraction, seed=seed)
    stats = DatasetStats(
        documents=len(documents),
        total_tokens=len(token_ids),
        train_blocks=len(train),
        val_blocks=len(val),
        context_length=context_length,
        stride=stride,
        dropped_tokens=dropped,
    )
    return TokenDataset(train=train, val=val, stats=stats)
