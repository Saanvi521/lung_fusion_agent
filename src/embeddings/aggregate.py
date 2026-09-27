"""Tile -> slide aggregation.

Default is mean pooling across a slide's tile embeddings: simple, no extra
trainable parameters, and appropriate given how few patients we have (146
train patients is not much to also fit an attention-pooling model on top
of). ABMIL is included as an optional ablation, not the default -- see
report.md for the justification.
"""
from __future__ import annotations

import numpy as np


def mean_pool(tile_embeddings: np.ndarray) -> np.ndarray:
    """(N_tiles, D) -> (D,). The default, primary aggregation."""
    if tile_embeddings.shape[0] == 0:
        raise ValueError("no tiles to aggregate — check tissue-mask threshold for this slide")
    return tile_embeddings.mean(axis=0)


# --- Optional ablation --------------------------------------------------
# Imported lazily so mean_pool (the default, always-used path) works even
# in environments without torch installed, e.g. quick unit tests.
try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object


class ABMILPooling(nn.Module if torch is not None else object):
    """Attention-based MIL pooling (Ilse et al. 2018), gated variant.

    Learns a per-tile attention weight, then returns the attention-weighted
    sum of tile embeddings. Only worth trying once mean pooling is working
    end-to-end and if time allows — flag this as an ablation in the report,
    not the headline aggregation.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_v = nn.Linear(embed_dim, hidden_dim)
        self.attn_u = nn.Linear(embed_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        # tile_embeddings: (N_tiles, D)
        a = torch.tanh(self.attn_v(tile_embeddings)) * torch.sigmoid(self.attn_u(tile_embeddings))
        scores = self.attn_w(a).squeeze(-1)  # (N_tiles,)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * tile_embeddings).sum(dim=0)  # (D,)
