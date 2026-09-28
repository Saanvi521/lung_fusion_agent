"""Stage 2b: embed every patient's cached tiles with ONE foundation model.

Run this three times, once per --model, in separate cells/sessions if you
like -- that's the whole point. Only one model is ever loaded into memory
(the previous script, 02a, already did all the downloading and tiling, so
this step never touches the network for slide images and never needs more
than ~1 model's worth of RAM/VRAM).

Point --out-dir at the SAME mounted Drive path used for 02a. Each patient's
embedding is written atomically the instant it's computed, so a disconnect
loses at most the one in-flight patient -- rerun the exact same command and
it picks up where it left off.

Usage (run 3 times total, once per model)
------------------------------------------
    python -m scripts.02b_extract_embeddings --model uni2h \
        --split artifacts/splits/split_seed42.csv \
        --out-dir /content/drive/MyDrive/lung_fusion_data

    python -m scripts.02b_extract_embeddings --model virchow2 \
        --split artifacts/splits/split_seed42.csv \
        --out-dir /content/drive/MyDrive/lung_fusion_data

    python -m scripts.02b_extract_embeddings --model genbio_pathfm \
        --split artifacts/splits/split_seed42.csv \
        --out-dir /content/drive/MyDrive/lung_fusion_data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.tile_cache import read_tile_zip, save_npy_atomic, tiles_path
from src.embeddings.aggregate import mean_pool
from src.embeddings.base import TileEncoder

MODEL_NAMES = ["uni2h", "virchow2", "genbio_pathfm"]


def _build_encoder(name: str) -> TileEncoder:
    if name == "uni2h":
        from src.embeddings.uni2h import UNI2HEncoder
        return UNI2HEncoder()
    if name == "virchow2":
        from src.embeddings.virchow2 import Virchow2Encoder
        return Virchow2Encoder()
    if name == "genbio_pathfm":
        from src.embeddings.genbio_pathfm import GenBioPathFMEncoder
        return GenBioPathFMEncoder()
    raise ValueError(f"unknown model: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Same mounted Drive path used for 02a_tile_slides.py")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    patients = pd.read_csv(args.split)
    tiles_dir = args.out_dir / "tiles"
    emb_dir = args.out_dir / "embeddings" / args.model
    emb_dir.mkdir(parents=True, exist_ok=True)

    remaining = [row for _, row in patients.iterrows() if not (emb_dir / f"{row.patient_id}.npy").exists()]
    if not remaining:
        print(f"{args.model}: every patient already has an embedding. Nothing to do.")
        return

    missing_tiles = [row for row in remaining if not tiles_path(tiles_dir, row.patient_id).exists()]
    if missing_tiles:
        print(f"[warn] {len(missing_tiles)} patients have no cached tiles yet -- "
              f"run 02a_tile_slides.py first (or let it finish). Skipping those for now.")
        remaining = [row for row in remaining if tiles_path(tiles_dir, row.patient_id).exists()]

    print(f"Loading {args.model} ({len(remaining)} patients remaining)...")
    enc = _build_encoder(args.model)

    t0 = time.time()
    n_done = 0
    try:
        for row in remaining:
            tiles = read_tile_zip(tiles_path(tiles_dir, row.patient_id))
            if not tiles:
                continue
            tile_embeds = enc.embed_tiles(tiles, batch_size=args.batch_size)
            slide_embed = mean_pool(tile_embeds).astype(np.float32)
            save_npy_atomic(emb_dir / f"{row.patient_id}.npy", slide_embed)
            n_done += 1
            print(f"[{args.model}] {n_done}/{len(remaining)} patient {row.patient_id} done")
    finally:
        enc.unload()  # always free memory, even if the loop raises partway through

    stats_path = args.out_dir / "embeddings" / "extraction_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats[args.model] = {
        "embed_dim": enc.embed_dim,
        "wall_clock_sec_this_run": round(time.time() - t0, 1),
        "n_slides_this_run": n_done,
        "n_slides_total": len(list(emb_dir.glob("*.npy"))),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\n{args.model} done: {stats[args.model]}")


if __name__ == "__main__":
    main()
