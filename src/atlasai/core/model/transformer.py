"""A small causal Transformer decoder implemented directly with PyTorch ops.

This module implements the architecture used by the local LLM adapter. Attention,
residual blocks, normalization, and the language-model head are explicit so the
training path can be inspected and tested without ``nn.Transformer``.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atlasai.config import TransformerConfig


class CausalSelfAttention(nn.Module):
    """Multi-head attention whose position t can only read positions <= t."""

    def __init__(self, config: TransformerConfig, context_length: int) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)
        self.causal_mask: Tensor
        self.register_buffer(
            "causal_mask",
            torch.ones(context_length, context_length, dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = cast(Tensor, self.qkv(x)).chunk(3, dim=-1)
        shape = (batch, length, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Bool buffer is moved by model.to(device), and sliced for shorter prompts.
        allowed = self.causal_mask[:length, :length]
        scores = scores.masked_fill(~allowed, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        attended = (probabilities @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return cast(Tensor, self.residual_dropout(self.projection(attended)))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.layers(x))


class DecoderBlock(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, config: TransformerConfig, context_length: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config, context_length)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return cast(Tensor, x + self.feed_forward(self.feed_forward_norm(x)))


class AtlasTransformer(nn.Module):
    """Token/position embeddings -> causal decoder blocks -> next-token logits."""

    def __init__(self, *, vocab_size: int, context_length: int, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        if (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or not 3 <= vocab_size <= 65_536
        ):
            raise ValueError("vocab_size must be an integer between 3 and 65536")
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or not 1 <= context_length <= 4096
        ):
            raise ValueError("context_length must be an integer between 1 and 4096")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, context_length) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [non-empty batch, non-empty sequence]")
        if input_ids.shape[1] > self.context_length:
            raise ValueError("input sequence exceeds model context_length")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("input_ids and model must be on the same device")
        # Explicit bounds checks produce useful errors instead of a CUDA assert later.
        if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= self.vocab_size:
            raise ValueError("input_ids contain a token outside the vocabulary")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        return cast(Tensor, self.lm_head(self.final_norm(x)))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
