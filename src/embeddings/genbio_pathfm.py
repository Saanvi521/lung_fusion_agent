"""GenBio-PathFM tile encoder, 4608-dim.

Uses the model's own mean/std (fit on its pretraining data), NOT ImageNet
stats -- this is exactly the "don't share one transform blindly" case the
brief warns about.
"""
from __future__ import annotations

import torch
from torchvision import transforms
from transformers import AutoModel

from src.embeddings.base import TileEncoder

_MEAN = (0.697, 0.575, 0.728)
_STD = (0.188, 0.240, 0.187)


class GenBioPathFMEncoder(TileEncoder):
    name = "genbio_pathfm"
    embed_dim = 4608

    def _load_model(self):
        return AutoModel.from_pretrained("genbio-ai/genbio-pathfm", trust_remote_code=True)

    def _build_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN, std=_STD),
            ]
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B, 4608) CLS features, per model card
