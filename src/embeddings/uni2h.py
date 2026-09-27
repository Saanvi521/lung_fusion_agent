"""UNI2-h tile encoder (Mahmood Lab), 1536-dim.

Loading exactly follows the official model card's ``timm_kwargs`` recipe --
these values (depth, heads, SwiGLU MLP, 8 register tokens, ...) define the
custom ViT-H/14 architecture and must match or the checkpoint won't load
correctly.
"""
from __future__ import annotations

import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from src.embeddings.base import TileEncoder

_TIMM_KWARGS = dict(
    img_size=224,
    patch_size=14,
    depth=24,
    num_heads=24,
    init_values=1e-5,
    embed_dim=1536,
    mlp_ratio=2.66667 * 2,
    num_classes=0,
    no_embed_class=True,
    mlp_layer=timm.layers.SwiGLUPacked,
    act_layer=torch.nn.SiLU,
    reg_tokens=8,
    dynamic_img_size=True,
)


class UNI2HEncoder(TileEncoder):
    name = "uni2h"
    embed_dim = 1536

    def _load_model(self):
        # Requires `huggingface_hub.login()` (token with UNI2-h access
        # granted) to have been called once in the process beforehand.
        return timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **_TIMM_KWARGS)

    def _build_transform(self):
        return create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # already pooled: (B, 1536), num_classes=0
