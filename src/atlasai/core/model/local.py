"""Own local LLM backend: safe checkpoint loading plus autoregressive decoding.

This is intentionally a simple full-context decoder without a KV cache. It is
correct and easy to inspect, but long generations on CPU will be slow. Optimize
only after profiling, per the project rules.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

from atlasai.core.model.base import GenerationResult, LLMError, StopReason
from atlasai.core.tokenizer.bpe import BOS_ID, EOS_ID, PAD_ID, BPETokenizer
from atlasai.training.checkpoint import CheckpointInfo, load_model_checkpoint


class AtlasLocalModel:
    """LLM implementation backed only by our own tokenizer and Transformer."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_threads: int = 1,
        seed: int = 42,
    ) -> None:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= 1024
        ):
            raise LLMError("max_new_tokens must be between 1 and 1024")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0.0 <= temperature <= 2.0
        ):
            raise LLMError("temperature must be finite and between 0 and 2")
        if (
            isinstance(num_threads, bool)
            or not isinstance(num_threads, int)
            or not 1 <= num_threads <= 64
        ):
            raise LLMError("num_threads must be between 1 and 64")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise LLMError("seed must be an unsigned 32-bit integer")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.model, self.checkpoint_info = load_model_checkpoint(
            checkpoint_path, tokenizer_path=tokenizer_path
        )
        self.model.eval()
        torch.set_num_threads(num_threads)
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count()

    @property
    def context_length(self) -> int:
        return self.model.context_length

    def _next_token(self, logits: Tensor, temperature: float) -> int:
        scores = logits.detach().clone()
        if not bool(torch.isfinite(scores).all().item()):
            raise LLMError("Model produced non-finite generation logits")
        # PAD and BOS are control tokens, not text. EOS is allowed and stops decoding.
        scores[PAD_ID] = float("-inf")
        scores[BOS_ID] = float("-inf")
        if temperature == 0.0:
            return int(torch.argmax(scores).item())
        probabilities = torch.softmax(scores / temperature, dim=-1)
        return int(torch.multinomial(probabilities, 1, generator=self._generator).item())

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        if not isinstance(prompt, str):
            raise LLMError("prompt must be a string")
        limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= self.max_new_tokens
        ):
            raise LLMError(f"max_new_tokens must be between 1 and {self.max_new_tokens}")
        decoding_temperature = self.temperature if temperature is None else temperature
        if (
            isinstance(decoding_temperature, bool)
            or not isinstance(decoding_temperature, (int, float))
            or not math.isfinite(decoding_temperature)
            or not 0.0 <= decoding_temperature <= 2.0
        ):
            raise LLMError("temperature must be finite and between 0 and 2")

        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        prompt_tokens = len(prompt_ids)
        # Keep the tail when the prompt is longer than the trained context window.
        context_ids = prompt_ids[-self.context_length :]
        completion: list[int] = []
        stop_reason: StopReason = "max_tokens"
        with torch.no_grad():
            for _ in range(limit):
                inputs = torch.tensor([context_ids], dtype=torch.long)
                logits = self.model(inputs)
                next_token = self._next_token(logits[0, -1], float(decoding_temperature))
                if next_token == EOS_ID:
                    stop_reason = "eos"
                    break
                completion.append(next_token)
                context_ids.append(next_token)
                context_ids = context_ids[-self.context_length :]
        return GenerationResult(
            # An undertrained byte-level model can emit incomplete UTF-8. Show U+FFFD
            # explicitly instead of crashing or pretending the bytes were valid text.
            text=self.tokenizer.decode(completion, errors="replace"),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(completion),
            stop_reason=stop_reason,
        )

    def info(self) -> dict[str, object]:
        info: CheckpointInfo = self.checkpoint_info
        return {
            "backend": "atlas-local",
            "parameter_count": self.parameter_count,
            "vocab_size": self.model.vocab_size,
            "context_length": self.context_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "checkpoint": info.to_dict(),
        }
