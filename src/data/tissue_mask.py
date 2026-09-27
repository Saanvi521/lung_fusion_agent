"""Tissue vs. background segmentation for a downsampled WSI.

H&E slides are mostly white/pale glass background around the tissue. We use a
classic, cheap heuristic rather than a learned segmenter: tissue is more
saturated (colorful) than glass, so an Otsu threshold on the HSV saturation
channel separates them well. This runs on the whole (already 4x-downsampled)
slide in one shot, then a tile is kept only if enough of its area is tissue.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu


def tissue_mask(slide_rgb: np.ndarray, min_object_px: int = 64) -> np.ndarray:
    """Return a boolean tissue mask, same H x W as ``slide_rgb``.

    Parameters
    ----------
    slide_rgb : (H, W, 3) uint8 array, the downsampled slide.
    min_object_px : drop connected components smaller than this (removes
        specks of dust / marker ink so they don't masquerade as tissue).
    """
    hsv = np.array(Image.fromarray(slide_rgb).convert("HSV"))
    sat = hsv[..., 1]

    # Otsu fails (raises) on a constant image, e.g. an all-white synthetic
    # test tile; treat that as "no tissue" instead of crashing.
    if sat.max() == sat.min():
        return np.zeros(sat.shape, dtype=bool)

    thresh = threshold_otsu(sat)
    mask = sat > thresh

    # Fill small holes inside tissue and drop small isolated specks.
    mask = ndimage.binary_fill_holes(mask)
    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_object_px
        mask = keep[labeled]
    return mask


def tile_tissue_fraction(mask: np.ndarray, x: int, y: int, tile: int) -> float:
    """Fraction of a tile window that is tissue, from the full-slide mask."""
    patch = mask[y : y + tile, x : x + tile]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())
