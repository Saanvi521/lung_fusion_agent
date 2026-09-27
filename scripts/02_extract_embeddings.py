"""Stage 2 entry point: for every patient in the split, tile their slide and
extract + cache a mean-pooled slide-level embedding from each of the three
tile encoders.

Run once. Everything downstream (baselines, agent, eval) only ever reads
the .npy files this writes -- the foundation models are never touched again
after this script finishes (see "cache embeddings ... never re-run a
foundation model inside a hyperparameter loop" in the brief).

Usage
-----
    python -m scripts.02_extract_embeddings \
        --split artifacts/splits/split_seed42.csv \
        --image-dir /path/to/slide/jpgs \
        --out-dir artifacts/embeddings

Assumes each slide's image file is named ``<wsi_id>.jpg`` (e.g.
``WSI-35.jpg``) inside --image-dir. Adjust `_image_path` below if the
dataset's actual filenames differ once you inspect the downloaded files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.tiling import tile_slide
from src.embeddings.aggregate import mean_pool
from src.embeddings.base import TileEncoder


def _image_path(image_dir: Path, wsi_id: str) -> Path:
    # Adjust to match the real file naming once you've inspected the
    # downloaded dataset (e.g. it may be "<wsi_id>.jpg" or nested by patient).
    return image_dir / f"{wsi_id}.jpg"


def _build_encoders() -> list[TileEncoder]:
    # Imported lazily so this file can be inspected / unit-tested for its
    # control flow without the heavy model dependencies installed.
    from src.embeddings.genbio_pathfm import GenBioPathFMEncoder
    from src.embeddings.uni2h import UNI2HEncoder
    from src.embeddings.virchow2 import Virchow2Encoder

    return [UNI2HEncoder(), Virchow2Encoder(), GenBioPathFMEncoder()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/embeddings"))
    args = ap.parse_args()

    patients = pd.read_csv(args.split)
    encoders = _build_encoders()

    stats = {e.name: {"embed_dim": e.embed_dim, "wall_clock_sec": 0.0, "n_slides": 0} for e in encoders}

    for enc in encoders:
        cache_dir = args.out_dir / enc.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        for _, row in patients.iterrows():
            out_path = cache_dir / f"{row.patient_id}.npy"
            if out_path.exists():
                continue  # already cached — never recompute

            img_path = _image_path(args.image_dir, row.wsi_id)
            tiles = tile_slide(img_path)
            if not tiles:
                print(f"[warn] {enc.name}: 0 tissue tiles for patient {row.patient_id} ({row.wsi_id})")
                continue

            tile_arrays = [t.array for t in tiles]
            tile_embeds = enc.embed_tiles(tile_arrays)
            slide_embed = mean_pool(tile_embeds)
            np.save(out_path, slide_embed.astype(np.float32))
            stats[enc.name]["n_slides"] += 1

        stats[enc.name]["wall_clock_sec"] = round(time.time() - t0, 1)
        stats[enc.name]["cache_dir"] = str(cache_dir)
        print(f"{enc.name}: {stats[enc.name]}")

    (args.out_dir / "extraction_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {args.out_dir / 'extraction_stats.json'}")


if __name__ == "__main__":
    main()
