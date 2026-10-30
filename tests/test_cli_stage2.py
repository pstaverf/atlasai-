from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlasai.cli import main

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "atlas.toml"
EXAMPLE_CORPUS = Path(__file__).resolve().parents[1] / "examples" / "corpus.txt"


def test_tokenizer_train_encode_decode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    exit_code = main(
        [
            "tokenizer",
            "train",
            "--config",
            str(REPO_CONFIG),
            "--input",
            str(EXAMPLE_CORPUS),
            "--output",
            str(tokenizer_path),
            "--vocab-size",
            "320",
        ]
    )
    assert exit_code == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained["vocab_size"] <= 320
    assert tokenizer_path.is_file()

    text = "Привет Atlas 🙂"
    assert (
        main(
            [
                "tokenizer",
                "encode",
                "--config",
                str(REPO_CONFIG),
                "--tokenizer",
                str(tokenizer_path),
                "--text",
                text,
            ]
        )
        == 0
    )
    encoded = json.loads(capsys.readouterr().out)
    ids = ",".join(str(token_id) for token_id in encoded["ids"])
    assert (
        main(
            [
                "tokenizer",
                "decode",
                "--config",
                str(REPO_CONFIG),
                "--tokenizer",
                str(tokenizer_path),
                "--ids",
                ids,
            ]
        )
        == 0
    )
    decoded = json.loads(capsys.readouterr().out)
    assert decoded["text"] == text


def test_dataset_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    assert (
        main(
            [
                "tokenizer",
                "train",
                "--config",
                str(REPO_CONFIG),
                "--input",
                str(EXAMPLE_CORPUS),
                "--output",
                str(tokenizer_path),
                "--vocab-size",
                "320",
            ]
        )
        == 0
    )
    capsys.readouterr()
    exit_code = main(
        [
            "dataset",
            "inspect",
            "--config",
            str(REPO_CONFIG),
            "--input",
            str(EXAMPLE_CORPUS),
            "--tokenizer",
            str(tokenizer_path),
            "--context-length",
            "32",
            "--stride",
            "32",
            "--val-fraction",
            "0.2",
        ]
    )
    assert exit_code == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 1
    assert stats["train_blocks"] > 0
    assert stats["val_blocks"] > 0
    assert stats["first_train_block"] is not None


def test_decode_invalid_ids_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    assert (
        main(
            [
                "tokenizer",
                "train",
                "--config",
                str(REPO_CONFIG),
                "--input",
                str(EXAMPLE_CORPUS),
                "--output",
                str(tokenizer_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    exit_code = main(
        [
            "tokenizer",
            "decode",
            "--config",
            str(REPO_CONFIG),
            "--tokenizer",
            str(tokenizer_path),
            "--ids",
            "1,not-an-int",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Invalid token id list" in json.loads(captured.err)["error"]


def test_missing_tokenizer_returns_one(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    exit_code = main(
        [
            "tokenizer",
            "encode",
            "--config",
            str(REPO_CONFIG),
            "--tokenizer",
            str(tmp_path / "missing.json"),
            "--text",
            "hello",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not found" in json.loads(captured.err)["error"]
