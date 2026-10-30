from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from atlasai.cli import main
from atlasai.config import load_config
from atlasai.core.tokenizer.bpe import EOS_ID, BPETokenizer
from atlasai.training.dataset import DatasetError, load_dataset

REPO = Path(__file__).resolve().parents[1]
FOUNDATION_CONFIG = REPO / "configs" / "foundation.toml"
CORPUS = REPO / "examples" / "foundation"
ANSWER_RE = re.compile(r"^Ответ: (\{.*\})$", re.MULTILINE)


def test_foundation_config_uses_separate_validation_documents() -> None:
    config = load_config(FOUNDATION_CONFIG)
    assert len(config.data.input_paths) == 10
    assert len(config.data.validation_input_paths) == 10
    assert not set(config.data.input_paths) & set(config.data.validation_input_paths)
    assert all(path.is_file() for path in config.data.input_paths)
    assert all(path.is_file() for path in config.data.validation_input_paths)
    assert config.data.stride == config.model.context_length == 256
    assert config.model.checkpoint_path.name == "foundation.pt"


def test_foundation_examples_are_valid_agent_actions() -> None:
    for split in ("train", "val"):
        for path in sorted((CORPUS / split).glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            answers = ANSWER_RE.findall(text)
            assert answers, path
            for raw in answers:
                action = json.loads(raw)
                assert set(action) == {"action", "text"}
                assert action["action"] in {"think", "respond"}
                assert isinstance(action["text"], str) and action["text"].strip()


def test_validation_examples_are_not_copied_from_training() -> None:
    train_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((CORPUS / "train").glob("*.txt"))
    )
    for path in sorted((CORPUS / "val").glob("*.txt")):
        for example in path.read_text(encoding="utf-8").split("\n\n"):
            if example.strip():
                assert example not in train_text


def test_explicit_validation_split_does_not_mix_documents(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("a" * 12, encoding="utf-8")
    val.write_text("b" * 12, encoding="utf-8")
    dataset = load_dataset(
        [train],
        BPETokenizer(),
        context_length=4,
        stride=4,
        validation_paths=[val],
    )
    assert dataset.stats.documents == 2
    assert dataset.train and dataset.val
    train_tokens = {token for block in dataset.train for token in block.input_ids}
    val_tokens = {token for block in dataset.val for token in block.input_ids}
    assert not (train_tokens & val_tokens) - {EOS_ID}


def test_explicit_validation_split_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "same.txt"
    path.write_text("abcdef", encoding="utf-8")
    with pytest.raises(DatasetError, match="must not also be training"):
        load_dataset([path], BPETokenizer(), context_length=2, stride=2, validation_paths=[path])


def test_explicit_validation_requires_full_block(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("abcdef", encoding="utf-8")
    val.write_text("x", encoding="utf-8")
    with pytest.raises(DatasetError, match="Validation corpus produced no full blocks"):
        load_dataset([train], BPETokenizer(), context_length=4, stride=4, validation_paths=[val])


def test_foundation_dataset_inspect_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    assert (
        main(
            [
                "tokenizer",
                "train",
                "--config",
                str(FOUNDATION_CONFIG),
                "--output",
                str(tokenizer),
                "--vocab-size",
                "259",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "dataset",
                "inspect",
                "--config",
                str(FOUNDATION_CONFIG),
                "--tokenizer",
                str(tokenizer),
            ]
        )
        == 0
    )
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 20
    assert stats["train_blocks"] > 0
    assert stats["val_blocks"] > 0


def test_empty_validation_override_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "dataset",
            "inspect",
            "--config",
            str(FOUNDATION_CONFIG),
            "--validation-input",
        ]
    )
    assert code == 1
    assert "at least one document" in json.loads(capsys.readouterr().err)["error"]
