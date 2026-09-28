"""Shared interface for tile-level foundation-model encoders.

Each concrete encoder (uni2h.py, virchow2.py, genbio_pathfm.py) wraps a
different pretrained model but exposes the same things, so the extraction
script never needs to know which model it's talking to:

- ``.name``       short id used in cache paths / report tables
- ``.embed_dim``  output vector size
- ``.embed_tiles(tiles)`` -> (N, embed_dim) float32 array
- ``.unload()``   frees the model's memory -- ALWAYS call this before
                  constructing the next encoder. Only one foundation model
                  should ever be resident in RAM/VRAM at a time: UNI2-h +
                  Virchow2 + GenBio-PathFM together are ~10GB of weights
                  plus real loading overhead, which reliably exceeds free
                  Colab's ~12-13GB RAM and was the direct cause of earlier
                  OOM crashes in this project.
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image


class TileEncoder(ABC):
    name: str
    embed_dim: int

    @abstractmethod
    def _load_model(self):
        """Load and return the underlying model, in eval mode, on device."""

    @abstractmethod
    def _build_transform(self):
        """Return this model's own preprocessing transform."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = self._load_model().to(device).eval()
        self.transform = self._build_transform()

    @torch.inference_mode()
    def embed_tiles(self, tiles: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Encode a list of (224, 224, 3) uint8 tile arrays -> (N, D) float32."""
        out = []
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i : i + batch_size]
            x = torch.stack([self.transform(Image.fromarray(t)) for t in batch]).to(self.device)
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self._forward(x)
            else:
                feats = self._forward(x)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.embed_dim), dtype=np.float32)

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model-specific forward pass -> (B, embed_dim) tensor."""

    def unload(self) -> None:
        """Free this model's memory (GPU + CPU)."""
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
