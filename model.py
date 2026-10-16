"""Transformer Encoder for news text classification."""

from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first tensors.

    Input/output shape: [batch_size, sequence_length, embedding_dim]
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        self.dropout = nn.Dropout(p=dropout)

        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(max_len, embedding_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(positions * div_term)
        # Works for odd embedding dimensions as well.
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, embedding_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds positional encoding max_len "
                f"({self.pe.size(1)}). Increase max_len or lower max_length."
            )
        x = x + self.pe[:, :seq_len, :].to(dtype=x.dtype)
        return self.dropout(x)


class TransformerNewsClassifier(nn.Module):
    """A small Transformer Encoder classifier for AG_NEWS.

    Input token IDs: [batch_size, sequence_length]
    Output logits:   [batch_size, num_classes]
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        embedding_dim: int = 100,
        nhead: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        if embedding_dim % nhead != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by nhead ({nhead})."
            )
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2 (PAD and UNK tokens).")

        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            # Avoid an irrelevant nested-tensor optimization warning with odd head counts (e.g. 5).
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                "Expected token_ids with shape [batch_size, sequence_length], "
                f"received shape {tuple(token_ids.shape)}."
            )

        # [batch, sequence] -> [batch, sequence, embedding_dim]
        padding_mask = token_ids.eq(self.pad_idx)
        x = self.embedding(token_ids) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)

        # Prevent padding positions from being used as keys/values by attention.
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Mean-pool only real tokens; ignore padding tokens.
        valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        summed = (x * valid_mask).sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths

        return self.classifier(self.dropout(pooled))
