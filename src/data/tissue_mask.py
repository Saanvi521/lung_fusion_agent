"""Tissue vs. background segmentation.

H&E slides are mostly pale glass with tissue somewhere on it. Tissue is
more *colorful* (saturated) than glass, so an Otsu threshold on the HSV
saturation channel separates the two without any hand-picked number.

IMPORTANT (memory): this runs on a small THUMBNAIL of the slide (see
``MASK_FACTOR`` in tiling.py), never on the full 20x image. Running Otsu,
hole-filling and connected-component labelling on a ~300-megapixel array
needs several GB of RAM, which is one of the things that crashed free Colab.
On a thumbnail it needs a few MB and gives the same tissue outline.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu


def tissue_mask(thumb_rgb: np.ndarray, min_object_px: int = 16) -> np.ndarray:
    """Boolean tissue mask, same H x W as ``thumb_rgb``.

    thumb_rgb     : (H, W, 3) uint8 thumbnail of the slide.
    min_object_px : drop connected blobs smaller than this many *thumbnail*
                    pixels (dust / pen specks).
    """
    hsv = np.asarray(Image.fromarray(thumb_rgb).convert("HSV"))
    sat = hsv[..., 1]

    if sat.max() == sat.min():
        return np.zeros(sat.shape, dtype=bool)

    mask = sat > threshold_otsu(sat)
    mask = ndimage.binary_fill_holes(mask)

    labeled, n = ndimage.label(mask)
    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= min_object_px
        mask = keep[labeled]
    return mask
