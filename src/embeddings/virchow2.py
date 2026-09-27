"""Virchow2 tile encoder (Paige AI), 2560-dim.

Virchow2 outputs a (B, 261, 1280) token sequence: 1 class token, 4 register
tokens (discarded), and 256 patch tokens. Per the official recipe we
concatenate the class token with the mean-pooled patch tokens to get the
final 2560-dim embedding -- this is the model's documented standard usage,
distinct from the "class-token-only, 1280-dim" variant Prism2 expects.
"""
from __future__ import annotations

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

from src.embeddings.base import TileEncoder

N_REGISTER_TOKENS = 4


class Virchow2Encoder(TileEncoder):
    name = "virchow2"
    embed_dim = 2560

    def _load_model(self):
        return timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    def _build_transform(self):
        return create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model(x)  # (B, 261, 1280)
        class_token = tokens[:, 0]
        patch_tokens = tokens[:, 1 + N_REGISTER_TOKENS :]  # skip cls + 4 register tokens
        return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)  # (B, 2560)
