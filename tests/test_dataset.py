from __future__ import annotations

from pathlib import Path

import pytest

from atlasai.core.tokenizer.bpe import EOS_ID, BPETokenizer
from atlasai.training.dataset import (
    DatasetError,
    TokenBlock,
    build_blocks,
    load_dataset,
    read_documents,
    split_blocks,
)


def test_build_blocks_shifts_targets() -> None:
    blocks, dropped = build_blocks(list(range(10)), context_length=3, stride=3)
    assert dropped == 0
    assert blocks[0] == TokenBlock(input_ids=(0, 1, 2), target_ids=(1, 2, 3))
    assert len(blocks) == 3


def test_build_blocks_overlap_and_tail() -> None:
    blocks, dropped = build_blocks(list(range(5)), context_length=3, stride=1)
    assert len(blocks) == 2
    assert dropped == 0
    blocks, dropped = build_blocks(list(range(5)), context_length=3, stride=3)
    assert len(blocks) == 1
    assert dropped == 1


def test_build_blocks_insufficient_tokens() -> None:
    blocks, dropped = build_blocks([1, 2, 3], context_length=3, stride=1)
    assert blocks == []
    assert dropped == 3


@pytest.mark.parametrize("stride", [0, 4])
def test_invalid_stride(stride: int) -> None:
    with pytest.raises(DatasetError):
        build_blocks(list(range(10)), context_length=3, stride=stride)


def _blocks(count: int) -> list[TokenBlock]:
    return [TokenBlock(input_ids=(index,), target_ids=(index + 1,)) for index in range(count)]


def test_split_is_deterministic() -> None:
    blocks = _blocks(10)
    first = split_blocks(blocks, val_fraction=0.2, seed=42)
    second = split_blocks(blocks, val_fraction=0.2, seed=42)
    assert first == second
    assert len(first[0]) == 8
    assert len(first[1]) == 2


def test_split_without_validation() -> None:
    train, val = split_blocks(_blocks(3), val_fraction=0.0, seed=1)
    assert len(train) == 3
    assert val == ()


def test_split_requires_two_blocks() -> None:
    with pytest.raises(DatasetError, match="At least two blocks"):
        split_blocks(_blocks(1), val_fraction=0.5, seed=1)


def test_load_dataset_inserts_eos_between_documents(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("aa", encoding="utf-8")
    second.write_text("bb", encoding="utf-8")
    dataset = load_dataset(
        [first, second],
        BPETokenizer(),
        context_length=2,
        stride=2,
        val_fraction=0.0,
    )
    assert dataset.stats.documents == 2
    assert dataset.stats.total_tokens == 6
    all_tokens = [token for block in dataset.train for token in block.input_ids]
    assert EOS_ID in all_tokens


def test_read_missing_document(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        list(read_documents([tmp_path / "missing.txt"]))


def test_read_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(DatasetError, match="Cannot read document"):
        list(read_documents([path]))


def test_load_dataset_requires_one_full_block(tmp_path: Path) -> None:
    path = tmp_path / "tiny.txt"
    path.write_text("a", encoding="utf-8")
    with pytest.raises(DatasetError, match="need at least"):
        load_dataset([path], BPETokenizer(), context_length=8, stride=8)
