"""Load a WSI, downsample 80x -> 20x, and cut it into tissue tiles.

We treat the JPGs as plain large images (not pyramidal formats like SVS), so
"downsampling from 80x to 20x" is just a 4x linear resize done once up front,
before tiling -- there is no separate pyramid level to read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.tissue_mask import tile_tissue_fraction, tissue_mask

Image.MAX_IMAGE_PIXELS = None  # these are huge; we trust our own inputs

DOWNSAMPLE_FACTOR = 4  # 80x -> 20x
TILE_SIZE = 224
STRIDE = 224  # non-overlapping; set < TILE_SIZE for overlap
MIN_TISSUE_FRACTION = 0.5
MAX_TILES_PER_SLIDE = 1000  # compute cap; document this choice in the report


@dataclass
class Tile:
    x: int  # top-left coord, in 20x pixel space
    y: int
    array: np.ndarray  # (TILE_SIZE, TILE_SIZE, 3) uint8


def load_and_downsample(image_path: Path) -> np.ndarray:
    """Load a slide JPG and resize 80x -> 20x (4x linear downsample).

    Uses PIL's JPEG "draft" mode: this tells the JPEG decoder to decode
    directly at a reduced resolution (JPEG's format natively supports this,
    in power-of-2 steps), instead of fully decoding the image at full
    resolution and THEN shrinking it. A naive Image.open(...).convert("RGB")
    on an 80x gigapixel slide can balloon to 8-15GB in decoded memory before
    we ever get to resize it -- easily enough to OOM a free Colab session.
    draft() avoids ever materializing that full-size array at all.
    """
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    target_w, target_h = orig_w // DOWNSAMPLE_FACTOR, orig_h // DOWNSAMPLE_FACTOR

    # Ask the decoder to get close to our target for free, during decode
    # itself (nearest power-of-2 downscale JPEG supports natively).
    img.draft("RGB", (target_w, target_h))
    img = img.convert("RGB")  # decodes NOW, but only at the small drafted size

    # draft() only gets us to the NEAREST power-of-2 factor (1/2, 1/4, 1/8...),
    # which may not be exactly our target. One final precise resize gets us
    # to the exact target size -- but this resize is now cheap, since the
    # image is already small going in.
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    return np.array(img)


def make_tiles(
    slide_rgb: np.ndarray,
    tile_size: int = TILE_SIZE,
    stride: int = STRIDE,
    min_tissue_fraction: float = MIN_TISSUE_FRACTION,
    max_tiles: int = MAX_TILES_PER_SLIDE,
    seed: int = 42,
) -> list[Tile]:
    """Grid-tile the (already downsampled) slide, keeping tissue tiles only.

    If more than ``max_tiles`` tiles pass the tissue filter, subsample
    deterministically (seeded) rather than always taking the first N in
    raster order, which would bias toward the top-left of the slide.
    """
    h, w = slide_rgb.shape[:2]
    mask = tissue_mask(slide_rgb)

    candidates: list[Tile] = []
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            frac = tile_tissue_fraction(mask, x, y, tile_size)
            if frac >= min_tissue_fraction:
                candidates.append(Tile(x=x, y=y, array=slide_rgb[y : y + tile_size, x : x + tile_size]))

    if len(candidates) > max_tiles:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=max_tiles, replace=False)
        candidates = [candidates[i] for i in sorted(idx)]

    return candidates


def tile_slide(image_path: Path, **kwargs) -> list[Tile]:
    """Convenience wrapper: load, downsample, tile in one call."""
    slide = load_and_downsample(image_path)
    return make_tiles(slide, **kwargs)