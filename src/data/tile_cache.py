"""On-disk cache of each patient's tiles (one small zip per patient).

Why this exists
----------------
Downloading a ~1GB slide is the slowest step, and loading all three
foundation models into RAM at once is what crashed free Colab. Splitting
extraction into two stages fixes both, without re-downloading anything:

    Stage A (02a_tile_slides.py,  no model, CPU only):
        download slide once -> tile it -> save a small zip of tiles
        -> delete the raw slide image

    Stage B (02b_extract_embeddings.py, ONE model at a time):
        read a patient's cached tile zip -> embed with the one loaded
        model -> save the resulting vector -> move to the next patient
        (run once per model: uni2h, then virchow2, then genbio_pathfm)

So every slide is downloaded exactly once (Stage A), and only one
foundation model is ever resident in memory (Stage B, run 3 times).

Tiles are stored as high-quality JPEGs (q95, no chroma subsampling) inside
an uncompressed zip, so ~600 tiles are roughly 20-30MB instead of ~90MB
raw. This is one extra (very mild) JPEG re-encode on top of the source
JPEG -- worth a sentence in the report, not a real quality concern for
frozen feature extraction.

Every write is atomic (write to a temp file, then rename), so a crash or
Colab disconnect mid-write can never leave a half-written file that a
later run mistakes for a finished one.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 95


def tiles_path(cache_dir: Path, patient_id) -> Path:
    return Path(cache_dir) / f"{patient_id}.zip"


def write_tile_zip(tiles, path: Path, jpeg_quality: int = JPEG_QUALITY) -> int:
    """Write a list of ``Tile`` (from tiling.py) to ``path`` atomically.
    Returns the number of bytes written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for t in tiles:
            buf = io.BytesIO()
            Image.fromarray(t.array).save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
            zf.writestr(f"{t.x}_{t.y}.jpg", buf.getvalue())
    size = tmp.stat().st_size
    os.replace(tmp, path)
    return size


def read_tile_zip(path: Path) -> list[np.ndarray]:
    """Read every tile back out as a list of (H, W, 3) uint8 arrays."""
    out = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            with Image.open(io.BytesIO(zf.read(name))) as im:
                out.append(np.asarray(im.convert("RGB")))
    return out


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """np.save that can never leave a truncated file behind on a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)
