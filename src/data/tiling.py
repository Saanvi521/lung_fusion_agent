"""Load a slide JPG, bring it from 80x to 20x, and cut tissue tiles.

Three memory tricks keep this inside free-Colab RAM (each was a real
problem earlier in this project):

1. ``Image.draft``: the JPEG decoder produces the ~4x smaller image
   directly, so the full-resolution 80x image is never fully decoded.
   A naive ``Image.open(...).convert("RGB")`` on an 80x gigapixel slide
   can balloon to 8-15GB in decoded memory before it's ever resized.
2. The tissue mask is computed on an 8x-smaller THUMBNAIL of the 20x
   image, not on the full 20x image itself.
3. Tiles are cropped straight from the PIL image. The whole 20x slide is
   never converted to one big numpy array (that would be a second,
   redundant ~1GB copy sitting in memory).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.tissue_mask import tissue_mask

Image.MAX_IMAGE_PIXELS = None  # slides are legitimately huge; we trust our own data

DOWNSAMPLE_FACTOR = 4       # 80x -> 20x
TILE_SIZE = 224             # required input size of all three foundation models
STRIDE = 224                # non-overlapping tiles
MIN_TISSUE_FRACTION = 0.5   # keep a tile only if >= 50% of it is tissue
MAX_TILES_PER_SLIDE = 600   # compute/disk cap; a seeded random subset if a slide has more
MASK_FACTOR = 8             # tissue mask is computed at 1/8 of the 20x image


@dataclass
class Tile:
    x: int                  # top-left corner, in 20x pixel coordinates
    y: int
    array: np.ndarray       # (TILE_SIZE, TILE_SIZE, 3) uint8


def load_slide_20x(image_path: Path) -> Image.Image:
    """Open a slide JPG and return it as an RGB PIL image at 20x (1/4 size).

    ``draft`` asks the JPEG decoder to decode at a reduced scale (it can
    only do 1/2, 1/4, 1/8), so the giant 80x image is never fully decoded.
    A final exact resize corrects any rounding to hit precisely
    original_size // 4.
    """
    img = Image.open(image_path)
    target = (img.width // DOWNSAMPLE_FACTOR, img.height // DOWNSAMPLE_FACTOR)
    img.draft("RGB", target)
    img = img.convert("RGB")
    if img.size != target:
        img = img.resize(target, Image.LANCZOS)
    return img


def _tile_coords(
    slide: Image.Image,
    tile_size: int,
    stride: int,
    min_tissue_fraction: float,
    factor: int,
) -> list[tuple[int, int]]:
    """Grid positions (x, y) whose tile is mostly tissue, judged on a thumbnail."""
    assert tile_size % factor == 0 and stride % factor == 0
    thumb = np.asarray(slide.reduce(factor))       # box-filtered 1/factor thumbnail
    mask = tissue_mask(thumb)
    t = tile_size // factor                         # tile size in mask pixels

    coords = []
    w, h = slide.size
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patch = mask[y // factor : y // factor + t, x // factor : x // factor + t]
            if patch.size and patch.mean() >= min_tissue_fraction:
                coords.append((x, y))
    return coords


def make_tiles(
    slide: Image.Image,
    tile_size: int = TILE_SIZE,
    stride: int = STRIDE,
    min_tissue_fraction: float = MIN_TISSUE_FRACTION,
    max_tiles: int = MAX_TILES_PER_SLIDE,
    seed: int = 42,
    slide_name: str = "",
) -> list[Tile]:
    """Cut tissue tiles from a 20x PIL image.

    If more than ``max_tiles`` tiles pass the tissue test, keep a seeded
    random subset (not the first N in raster order, which would bias
    toward the top-left of the slide). The seed is mixed with the slide
    name so every slide gets its own subset, but it's fully reproducible.
    """
    coords = _tile_coords(slide, tile_size, stride, min_tissue_fraction, MASK_FACTOR)

    if len(coords) > max_tiles:
        rng = np.random.default_rng(seed + zlib.crc32(slide_name.encode()))
        keep = np.sort(rng.choice(len(coords), size=max_tiles, replace=False))
        coords = [coords[i] for i in keep]

    return [
        Tile(x=x, y=y, array=np.asarray(slide.crop((x, y, x + tile_size, y + tile_size))))
        for (x, y) in coords
    ]


def tile_slide(image_path: Path, **kwargs) -> list[Tile]:
    """Load + downsample + tile in one call."""
    slide = load_slide_20x(image_path)
    try:
        return make_tiles(slide, slide_name=Path(image_path).stem, **kwargs)
    finally:
        slide.close()
