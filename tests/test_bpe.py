from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlasai.core.tokenizer.bpe import (
    BASE_VOCAB_SIZE,
    BOS_ID,
    BYTE_OFFSET,
    EOS_ID,
    PAD_ID,
    BPETokenizer,
    TokenizerError,
    pretokenize,
    train_bpe,
)


@pytest.mark.parametrize(
    "text",
    ["", " ", "hello world", "Привет мир", "emoji 🙂 and 中文", "a\tb\nc  d", "  leading"],
)
def test_pretokenize_is_lossless(text: str) -> None:
    assert "".join(pretokenize(text)) == text


@pytest.mark.parametrize(
    "text",
    ["hello world", "Привет, Atlas! 🙂", "中文测试", "line\nline\tend  ", ""],
)
def test_base_tokenizer_roundtrip(text: str) -> None:
    tokenizer = BPETokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_is_deterministic_and_roundtrips() -> None:
    corpus = ["hello world hello world\n", "Привет мир Привет мир\n"] * 8
    first = train_bpe(corpus, vocab_size=320, min_frequency=2)
    second = train_bpe(corpus, vocab_size=320, min_frequency=2)
    assert first.merges == second.merges
    assert BASE_VOCAB_SIZE < first.vocab_size <= 320
    text = "hello world Привет мир 🙂"
    assert first.decode(first.encode(text)) == text


def test_training_can_reduce_token_count() -> None:
    corpus = ["hello hello hello hello world world world world\n"] * 16
    base = BPETokenizer()
    trained = train_bpe(corpus, vocab_size=320, min_frequency=2)
    text = "hello hello world world"
    assert len(trained.encode(text)) < len(base.encode(text))


def test_min_frequency_stops_early() -> None:
    tokenizer = train_bpe(["abcdef"], vocab_size=300, min_frequency=100)
    assert tokenizer.vocab_size == BASE_VOCAB_SIZE


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(TokenizerError, match="empty corpus"):
        train_bpe([], vocab_size=300)


@pytest.mark.parametrize("vocab_size", [258, 65_537, True])
def test_invalid_vocab_size(vocab_size: object) -> None:
    with pytest.raises(TokenizerError):
        train_bpe(["hello"], vocab_size=vocab_size)  # type: ignore[arg-type]


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    tokenizer = train_bpe(["hello world hello world\n"] * 8, vocab_size=300)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.encode("hello world") == tokenizer.encode("hello world")


def test_load_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "other", "version": 1}), encoding="utf-8")
    with pytest.raises(TokenizerError, match="format"):
        BPETokenizer.load(path)


def test_load_rejects_forward_merge_reference(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "format": "atlas-bpe",
                "version": 1,
                "special_tokens": ["<pad>", "<bos>", "<eos>"],
                "vocab_size": BASE_VOCAB_SIZE + 1,
                "merges": [[BASE_VOCAB_SIZE, BYTE_OFFSET]],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TokenizerError, match="existing non-special token"):
        BPETokenizer.load(path)


def test_duplicate_merge_is_rejected() -> None:
    with pytest.raises(TokenizerError, match="Duplicate merge"):
        BPETokenizer([(BYTE_OFFSET, BYTE_OFFSET + 1), (BYTE_OFFSET, BYTE_OFFSET + 1)])


def test_encode_special_tokens() -> None:
    tokenizer = BPETokenizer()
    assert tokenizer.encode("a", add_bos=True, add_eos=True) == [
        BOS_ID,
        BYTE_OFFSET + ord("a"),
        EOS_ID,
    ]


def test_decode_special_tokens() -> None:
    tokenizer = BPETokenizer()
    ids = [BOS_ID, EOS_ID, PAD_ID]
    assert tokenizer.decode(ids) == ""
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<bos><eos><pad>"


def test_decode_unknown_id() -> None:
    with pytest.raises(TokenizerError, match="Unknown token id"):
        BPETokenizer().decode([BASE_VOCAB_SIZE])


def test_decode_invalid_utf8() -> None:
    with pytest.raises(TokenizerError, match="valid UTF-8"):
        BPETokenizer().decode([BYTE_OFFSET + 255])


def test_decode_invalid_utf8_can_be_replaced_for_generation() -> None:
    assert BPETokenizer().decode([BYTE_OFFSET + 255], errors="replace") == "\ufffd"
