"""Stage 2 entry point, disk- AND memory-conscious version.

Two things changed from earlier drafts, both learned from real failures:

1. Disk: never keep more than one slide's raw image on disk (download,
   process, delete -- unchanged from before).
2. Memory: never keep more than one FOUNDATION MODEL loaded at once.
   Loading UNI2-h + Virchow2 + GenBio-PathFM simultaneously (~10GB of
   weights plus loading overhead) was crashing free-tier Colab's RAM
   silently, mid-run. So the loop is now MODEL-outer, PATIENT-inner: for
   each model, load it, process every patient, unload it, move to the
   next model. The trade-off is each slide gets downloaded up to 3 times
   (once per model) instead of once -- acceptable given how fast this
   dataset's Xet-accelerated downloads are, and much safer than crashing.

Usage
-----
    python -m scripts.02_extract_embeddings \
        --split artifacts/splits/split_seed42.csv \
        --out-dir artifacts/embeddings \
        --repo-id kmmuleelab/Lung_Pathology_Image_JPG
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from src.data.tiling import tile_slide
from src.embeddings.aggregate import mean_pool
from src.embeddings.base import TileEncoder


def _resolve_wsi_to_repo_path(repo_id: str) -> dict[str, str]:
    """Map each WSI_ID (e.g. 'WSI-35') to its actual path inside the HF
    repo. We don't assume a fixed naming convention -- some datasets nest
    files in subfolders -- so we list the repo once and match by substring.
    """
    files = list_repo_files(repo_id, repo_type="dataset")
    jpgs = [f for f in files if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")]

    mapping: dict[str, str] = {}
    for path in jpgs:
        stem = Path(path).stem
        mapping[stem] = path
    return mapping


def _make_encoder(name: str) -> TileEncoder:
    """Build exactly one encoder, by name. Imported lazily per-call so we
    never accidentally hold more than one model class's heavy import
    dependencies resident at once."""
    if name == "uni2h":
        from src.embeddings.uni2h import UNI2HEncoder
        return UNI2HEncoder()
    if name == "virchow2":
        from src.embeddings.virchow2 import Virchow2Encoder
        return Virchow2Encoder()
    if name == "genbio_pathfm":
        from src.embeddings.genbio_pathfm import GenBioPathFMEncoder
        return GenBioPathFMEncoder()
    raise ValueError(f"unknown encoder name: {name}")


MODEL_NAMES = ["uni2h", "virchow2", "genbio_pathfm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--repo-id", default="kmmuleelab/Lung_Pathology_Image_JPG")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/embeddings"))
    ap.add_argument("--tmp-dir", type=Path, default=Path("/content/tmp_slides"))
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    args = ap.parse_args()

    patients = pd.read_csv(args.split)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Listing repo files to resolve WSI_ID -> actual path (one-time)...")
    wsi_to_path = _resolve_wsi_to_repo_path(args.repo_id)
    print(f"Found {len(wsi_to_path)} jpgs in repo.")

    stats: dict = {}
    t_total_start = time.time()

    for model_name in args.models:
        cache_dir = args.out_dir / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Skip loading this model entirely if every patient is already done.
        remaining = [
            row for _, row in patients.iterrows()
            if not (cache_dir / f"{row.patient_id}.npy").exists()
        ]
        if not remaining:
            print(f"{model_name}: all patients already cached, skipping model load.")
            continue

        print(f"\n=== Loading {model_name} ({len(remaining)} patients remaining) ===")
        enc = _make_encoder(model_name)
        t0 = time.time()
        n_done = 0

        for row in remaining:
            out_path = cache_dir / f"{row.patient_id}.npy"
            repo_path = wsi_to_path.get(row.wsi_id)
            if repo_path is None:
                print(f"[warn] no repo file matching WSI_ID={row.wsi_id!r} — skipping patient {row.patient_id}")
                continue

            local_path = hf_hub_download(
                args.repo_id, repo_path, repo_type="dataset", local_dir=str(args.tmp_dir)
            )
            tiles = tile_slide(Path(local_path))
            if not tiles:
                print(f"[warn] 0 tissue tiles for patient {row.patient_id} ({row.wsi_id}) — skipping")
                os.remove(local_path)
                continue

            tile_embeds = enc.embed_tiles([t.array for t in tiles])
            slide_embed = mean_pool(tile_embeds)
            np.save(out_path, slide_embed.astype(np.float32))
            os.remove(local_path)  # free disk immediately

            n_done += 1
            print(f"  [{model_name}] {n_done}/{len(remaining)} patient {row.patient_id} done")

        enc.unload()  # free memory before the next model loads
        stats[model_name] = {
            "embed_dim": enc.embed_dim,
            "wall_clock_sec": round(time.time() - t0, 1),
            "n_slides_this_run": n_done,
            "cache_dir": str(cache_dir),
        }
        print(f"{model_name} done: {stats[model_name]}")

    stats["total_wall_clock_sec"] = round(time.time() - t_total_start, 1)
    (args.out_dir / "extraction_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n" + json.dumps(stats, indent=2))
    print(f"\nWrote {args.out_dir / 'extraction_stats.json'}")


if __name__ == "__main__":
    main()