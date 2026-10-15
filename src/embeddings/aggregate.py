"""Tile -> slide aggregation.

- ``mean_pool``: parameter-free baseline (used by 02b).
- ``MultiStreamABMIL``: gated attention MIL (Ilse et al., 2018) in the configuration
  used by current pathology-FM papers (UNI / MIL-Lab recipe): per-feature
  standardization -> input dropout -> FC projection -> dropout -> gated attention ->
  linear head. It accepts one *or several* foundation-model streams for the same
  tiles, so the same class is the single-FM ABMIL baseline and the multi-FM
  fusion model. This keeps the classifier family identical across rows of the
  comparison table.

Why this differs from the first version: attention and the classifier used to act
directly on the raw, unnormalised 1536/2560/4608-dim features with no dropout, so the
network memorised the ~110 training slides within ~10 epochs and the validation
cross-entropy then rose as its predictions became over-confident. Note the parameter
count is NOT much smaller here at proj_dim=256 (~0.46M for UNI2-h vs ~0.40M before);
what changes is the regularisation: per-FM feature standardisation, input dropout,
dropout after the projection and inside attention, plus (in the trainer) AdamW weight
decay and random tile sub-sampling. proj_dim=128 (searched by 02g) also halves the size.
"""
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
    nn = None

FUSION_MODES = ("tile_concat", "gated", "late")

if torch is not None:

    class FeatureStandardizer(nn.Module):
        """Per-dimension (x - mean) / std with statistics fitted on TRAINING tiles only.

        Stored as buffers so they travel with ``state_dict`` (and with the early-stopping
        checkpoint). Different foundation models have very different feature scales;
        without this the same learning rate is a much bigger effective step for one FM
        than another.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.register_buffer("mean", torch.zeros(dim))
            self.register_buffer("std", torch.ones(dim))

        def set_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
            self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
            self.std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return (x - self.mean) / self.std

    class GatedAttention(nn.Module):
        """a_i = w^T (tanh(V h_i) * sigmoid(U h_i)); softmax over the tiles of one bag."""

        def __init__(self, in_dim: int, attn_dim: int, dropout: float):
            super().__init__()
            self.v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
            self.u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
            self.w = nn.Linear(attn_dim, 1)

        def forward(self, h: "torch.Tensor") -> "torch.Tensor":  # h: (N, in_dim) -> (N,)
            return torch.softmax(self.w(self.v(h) * self.u(h)).squeeze(-1), dim=0)

    class _Stream(nn.Module):
        """One foundation model's tile encoder: standardise -> dropout -> FC -> ReLU -> dropout."""

        def __init__(self, embed_dim: int, proj_dim: int, dropout_in: float, dropout: float):
            super().__init__()
            self.norm = FeatureStandardizer(embed_dim)
            self.proj = nn.Sequential(
                nn.Dropout(dropout_in), nn.Linear(embed_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.proj(self.norm(x))

    class MultiStreamABMIL(nn.Module):
        """Gated-attention MIL over one or more FM feature streams of the SAME tiles.

        fusion (only matters with >1 stream):
          - ``tile_concat``: project each FM per tile, concatenate per tile, ONE shared
            attention over the fused tile vectors (intermediate / early-at-tile-level fusion).
          - ``gated``: one attention per FM -> one slide vector per FM -> a learned,
            per-slide softmax gate over FMs -> weighted sum (attention over model streams).
          - ``late``: one full ABMIL (attention + head) per FM, logits averaged (late fusion,
            trained jointly).
        stream_dropout: probability of dropping a whole FM stream for a training bag
        (modality dropout) so the fused model cannot lean on a single FM.
        meta_dim: metadata (age, sex) is concatenated to the slide vector before the head.
        """

        def __init__(
            self,
            embed_dims: dict[str, int],
            n_classes: int,
            proj_dim: int = 512,
            attn_dim: int = 384,
            dropout_in: float = 0.1,
            dropout: float = 0.25,
            fusion: str = "tile_concat",
            meta_dim: int = 0,
            stream_dropout: float = 0.0,
        ):
            super().__init__()
            if fusion not in FUSION_MODES:
                raise ValueError(f"fusion must be one of {FUSION_MODES}, got {fusion!r}")
            self.stream_names = list(embed_dims.keys())
            self.fusion = fusion if len(self.stream_names) > 1 else "tile_concat"
            self.meta_dim = meta_dim
            self.stream_dropout = stream_dropout
            self.streams = nn.ModuleDict(
                {m: _Stream(d, proj_dim, dropout_in, dropout) for m, d in embed_dims.items()}
            )
            n = len(self.stream_names)
            if self.fusion == "tile_concat":
                self.attn = GatedAttention(proj_dim * n, attn_dim, dropout)
                self.head = nn.Linear(proj_dim * n + meta_dim, n_classes)
            elif self.fusion == "gated":
                self.attns = nn.ModuleDict({m: GatedAttention(proj_dim, attn_dim, dropout) for m in self.stream_names})
                self.gate = nn.Sequential(nn.Linear(proj_dim * n, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n))
                self.head = nn.Linear(proj_dim + meta_dim, n_classes)
            else:  # late
                self.attns = nn.ModuleDict({m: GatedAttention(proj_dim, attn_dim, dropout) for m in self.stream_names})
                self.heads = nn.ModuleDict({m: nn.Linear(proj_dim + meta_dim, n_classes) for m in self.stream_names})

        # -- helpers -------------------------------------------------------
        def stream_parameters(self, m: str):
            """Parameters that belong to FM ``m`` only (for per-FM learning rates)."""
            mods = [self.streams[m]]
            if self.fusion in ("gated", "late"):
                mods.append(self.attns[m])
            if self.fusion == "late":
                mods.append(self.heads[m])
            for mod in mods:
                yield from mod.parameters()

        def _keep_mask(self) -> list[bool]:
            n = len(self.stream_names)
            if not self.training or self.stream_dropout <= 0 or n == 1:
                return [True] * n
            keep = (torch.rand(n) >= self.stream_dropout).tolist()
            if not any(keep):
                keep[int(torch.randint(n, (1,)))] = True
            return keep

        def _with_meta(self, z: "torch.Tensor", meta: "torch.Tensor | None") -> "torch.Tensor":
            if self.meta_dim == 0:
                return z
            if meta is None:
                raise ValueError("model was built with meta_dim > 0 but no metadata was passed")
            return torch.cat([z, meta.to(z.dtype)], dim=-1)

        def forward(self, bags: dict, meta: "torch.Tensor | None" = None) -> dict:
            """bags: {fm_name: (N_tiles, D_fm)} for ONE slide (same tiles, same order)."""
            keep = self._keep_mask()
            h = {}
            for m, k in zip(self.stream_names, keep):
                hm = self.streams[m](bags[m])
                h[m] = hm if k else torch.zeros_like(hm)
            out: dict = {"keep": keep}
            if self.fusion == "tile_concat":
                hcat = torch.cat([h[m] for m in self.stream_names], dim=-1)
                w = self.attn(hcat)
                z = (w.unsqueeze(-1) * hcat).sum(0)
                out.update(logits=self.head(self._with_meta(z, meta)), weights={m: w for m in self.stream_names}, slide=z)
            elif self.fusion == "gated":
                ws = {m: self.attns[m](h[m]) for m in self.stream_names}
                s = torch.stack([(ws[m].unsqueeze(-1) * h[m]).sum(0) for m in self.stream_names])  # (n, proj)
                g_logits = self.gate(s.flatten())
                g_logits = g_logits.masked_fill(~torch.tensor(keep, device=s.device), float("-inf"))
                g = torch.softmax(g_logits, dim=0)
                z = (g.unsqueeze(-1) * s).sum(0)
                out.update(logits=self.head(self._with_meta(z, meta)), weights=ws, gate=g, slide=z)
            else:
                ws = {m: self.attns[m](h[m]) for m in self.stream_names}
                logits = [
                    self.heads[m](self._with_meta((ws[m].unsqueeze(-1) * h[m]).sum(0), meta))
                    for m, k in zip(self.stream_names, keep) if k
                ]
                out.update(logits=torch.stack(logits).mean(0), weights=ws)
            return out

    def ABMILClassifier(embed_dim: int, n_classes: int, **kwargs) -> "MultiStreamABMIL":
        """Single-FM ABMIL (stream name ``"x"``) — kept for backward compatibility."""
        return MultiStreamABMIL({"x": embed_dim}, n_classes, **kwargs)
