"""A small byte-level BPE tokenizer implemented without third-party libraries.

Design choices for the experimental Atlas model:

* UTF-8 bytes are the base alphabet, so any valid Unicode text is representable.
* Merges are learned inside whitespace-aware pre-token chunks. This keeps the
  training state proportional to unique chunks rather than the whole corpus.
* Merge selection is deterministic: highest weighted frequency, then the
  lexicographically smallest token-id pair.
* The serialized format is JSON so it can be inspected and versioned easily.

This is intentionally simpler than GPT-2's regex pre-tokenizer. It is suitable
for small experiments, not a claim of production tokenizer quality.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import suppress
from itertools import pairwise
from pathlib import Path
from typing import Literal

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>")
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = len(SPECIAL_TOKENS)
BASE_VOCAB_SIZE = BYTE_OFFSET + 256
MAX_VOCAB_SIZE = 65_536
MAX_TOKENIZER_BYTES = 32 * 1024 * 1024
FORMAT_NAME = "atlas-bpe"
FORMAT_VERSION = 1

_PRETOKENIZE_RE = re.compile(r"\S+\s*|\s+")


class TokenizerError(ValueError):
    """Tokenizer training, serialization, encoding, or decoding failed."""


def pretokenize(text: str) -> list[str]:
    """Split text into lossless chunks: non-space runs plus following spaces."""
    return _PRETOKENIZE_RE.findall(text)


def _byte_ids(chunk: str) -> list[int]:
    try:
        return [BYTE_OFFSET + value for value in chunk.encode("utf-8")]
    except UnicodeEncodeError as exc:
        raise TokenizerError(f"Cannot encode chunk as UTF-8: {exc}") from exc


def _merge_sequence(sequence: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Merge non-overlapping occurrences of ``pair`` from left to right."""
    result: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            result.append(new_id)
            index += 2
        else:
            result.append(sequence[index])
            index += 1
    return result


def _validate_merge_pair(value: object, next_id: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TokenizerError("Each merge must be a pair of token ids")
    left, right = value
    for token_id in (left, right):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TokenizerError("Merge token ids must be integers")
        if not BYTE_OFFSET <= token_id < next_id:
            raise TokenizerError(
                f"Merge id {token_id} must refer to an existing non-special token below {next_id}"
            )
    return int(left), int(right)


class BPETokenizer:
    """Immutable vocabulary and merge rules learned by :func:`train_bpe`."""

    def __init__(self, merges: Iterable[Sequence[int]] = ()) -> None:
        normalized: list[tuple[int, int]] = []
        self._token_bytes: list[bytes] = [bytes([value]) for value in range(256)]
        self._rank: dict[tuple[int, int], int] = {}
        for index, raw_pair in enumerate(merges):
            new_id = BASE_VOCAB_SIZE + index
            pair = _validate_merge_pair(raw_pair, new_id)
            if pair in self._rank:
                raise TokenizerError(f"Duplicate merge rule: {pair}")
            self._token_bytes.append(
                self._token_bytes[pair[0] - BYTE_OFFSET] + self._token_bytes[pair[1] - BYTE_OFFSET]
            )
            self._rank[pair] = new_id
            normalized.append(pair)
        self.merges: tuple[tuple[int, int], ...] = tuple(normalized)

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    def _apply_merges(self, token_ids: list[int]) -> list[int]:
        if not self._rank:
            return token_ids
        while len(token_ids) > 1:
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None
            for pair in pairwise(token_ids):
                rank = self._rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            token_ids = _merge_sequence(token_ids, best_pair, self._rank[best_pair])
        return token_ids

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TokenizerError("text must be a string")
        result: list[int] = [BOS_ID] if add_bos else []
        for chunk in pretokenize(text):
            result.extend(self._apply_merges(_byte_ids(chunk)))
        if add_eos:
            result.append(EOS_ID)
        return result

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: Literal["strict", "replace"] = "strict",
    ) -> str:
        if errors not in {"strict", "replace"}:
            raise TokenizerError("errors must be 'strict' or 'replace'")
        buffer = bytearray()
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TokenizerError("Token ids must be integers")
            if PAD_ID <= token_id < BYTE_OFFSET:
                if not skip_special_tokens:
                    buffer.extend(SPECIAL_TOKENS[token_id].encode("utf-8"))
            elif BYTE_OFFSET <= token_id < self.vocab_size:
                buffer.extend(self._token_bytes[token_id - BYTE_OFFSET])
            else:
                raise TokenizerError(f"Unknown token id: {token_id}")
        try:
            return buffer.decode("utf-8", errors=errors)
        except UnicodeDecodeError as exc:
            raise TokenizerError(f"Token ids do not form valid UTF-8 text: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "special_tokens": list(SPECIAL_TOKENS),
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temporary_name: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        except OSError as exc:
            raise TokenizerError(f"Cannot save tokenizer to {target}: {exc}") from exc
        finally:
            if temporary_name is not None:
                with suppress(OSError):
                    Path(temporary_name).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        source = Path(path).expanduser()
        try:
            size = source.stat().st_size
            if size > MAX_TOKENIZER_BYTES:
                raise TokenizerError(
                    f"Tokenizer file is too large: {size} bytes (limit {MAX_TOKENIZER_BYTES})"
                )
            raw = json.loads(source.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise TokenizerError(f"Tokenizer file not found: {source}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"Cannot read tokenizer {source}: {exc}") from exc

        if not isinstance(raw, dict):
            raise TokenizerError("Tokenizer JSON must be an object")
        if raw.get("format") != FORMAT_NAME:
            raise TokenizerError(f"Unsupported tokenizer format: {raw.get('format')!r}")
        version = raw.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version != FORMAT_VERSION:
            raise TokenizerError(f"Unsupported tokenizer version: {version!r}")
        if raw.get("special_tokens") != list(SPECIAL_TOKENS):
            raise TokenizerError("Tokenizer special tokens do not match this implementation")
        raw_merges = raw.get("merges")
        if not isinstance(raw_merges, list):
            raise TokenizerError("Tokenizer merges must be a list")
        merges = [
            _validate_merge_pair(pair, BASE_VOCAB_SIZE + index)
            for index, pair in enumerate(raw_merges)
        ]
        vocab_size = raw.get("vocab_size")
        if vocab_size is not None and (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or vocab_size != BASE_VOCAB_SIZE + len(merges)
        ):
            raise TokenizerError("Tokenizer vocab_size does not match its merge rules")
        return cls(merges)


def train_bpe(
    documents: Iterable[str], *, vocab_size: int = 512, min_frequency: int = 2
) -> BPETokenizer:
    """Learn byte-level BPE merges from text documents.

    Training stops early if no remaining pair meets ``min_frequency``; the
    resulting vocabulary can therefore be smaller than ``vocab_size``.
    """
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TokenizerError("vocab_size must be an integer")
    if not BASE_VOCAB_SIZE <= vocab_size <= MAX_VOCAB_SIZE:
        raise TokenizerError(f"vocab_size must be between {BASE_VOCAB_SIZE} and {MAX_VOCAB_SIZE}")
    if isinstance(min_frequency, bool) or not isinstance(min_frequency, int) or min_frequency < 1:
        raise TokenizerError("min_frequency must be a positive integer")

    frequencies: Counter[str] = Counter()
    for document in documents:
        if not isinstance(document, str):
            raise TokenizerError("Training documents must be strings")
        frequencies.update(pretokenize(document))
    if not frequencies:
        raise TokenizerError("Cannot train a tokenizer on an empty corpus")

    sequences = {chunk: _byte_ids(chunk) for chunk in frequencies}
    target_merges = vocab_size - BASE_VOCAB_SIZE
    merges: list[tuple[int, int]] = []
    while len(merges) < target_merges:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for chunk, frequency in frequencies.items():
            sequence = sequences[chunk]
            for pair in pairwise(sequence):
                pair_counts[pair] += frequency
        if not pair_counts:
            break
        best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
        if pair_counts[best_pair] < min_frequency:
            break
        new_id = BASE_VOCAB_SIZE + len(merges)
        merges.append(best_pair)
        for chunk, sequence in sequences.items():
            sequences[chunk] = _merge_sequence(sequence, best_pair, new_id)
    return BPETokenizer(merges)
