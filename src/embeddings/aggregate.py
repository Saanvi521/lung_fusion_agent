"""Tile -> slide aggregation: mean_pool (no training, used by 02b) and ABMILPooling/ABMILClassifier (trained, used by 02c)."""
from __future__ import annotations

import numpy as np


def mean_pool(tile_embeddings: np.ndarray) -> np.ndarray:
    """(N_tiles, D) -> (D,). Parameter-free baseline aggregation."""
    if tile_embeddings.shape[0] == 0:
        raise ValueError("no tiles to aggregate — check tissue-mask threshold for this slide")
    return tile_embeddings.mean(axis=0)


# torch is optional here so mean_pool still works without it installed.
try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


class ABMILPooling(nn.Module if torch is not None else object):
    """Gated attention-based MIL pooling (Ilse et al. 2018): learns a per-tile weight instead of averaging."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_v = nn.Linear(embed_dim, hidden_dim)
        self.attn_u = nn.Linear(embed_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, tile_embeddings: "torch.Tensor") -> tuple:
        # tile_embeddings: (N_tiles, D)
        a = torch.tanh(self.attn_v(tile_embeddings)) * torch.sigmoid(self.attn_u(tile_embeddings))
        scores = self.attn_w(a).squeeze(-1)  # (N_tiles,)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(-1) * tile_embeddings).sum(dim=0)  # (D,)
        return pooled, weights


class ABMILClassifier(nn.Module if torch is not None else object):
    """ABMILPooling + a linear head, so the attention weights can be fit with real 7-class supervision in 02c_train_abmil.py."""

    def __init__(self, embed_dim: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.pool = ABMILPooling(embed_dim, hidden_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, tile_embeddings: "torch.Tensor") -> tuple:
        pooled, weights = self.pool(tile_embeddings)
        logits = self.classifier(pooled)
        return logits, pooled, weights
