"""Stage 2 entry point, disk-conscious version.

The dataset is far too large to download in full on free Colab (~100GB for
all 408 slides). Instead, for each patient we need, we:

    1. download ONLY that one slide's JPG
    2. tile it ONCE (shared across all 3 models -- tiling doesn't depend
       on which foundation model will look at the tiles)
    3. run all 3 models on those tiles, mean-pool, cache each .npy
    4. delete the downloaded JPG before moving to the next patient

At any moment, at most one slide's raw image is on disk. This is the
correct pattern regardless of disk limits -- it also fixes the earlier
version's redundant re-tiling (it tiled the same slide 3 separate times,
once per model's loop).

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
        stem = Path(path).stem  # filename without extension
        mapping[stem] = path  # assumes stem IS the WSI_ID, e.g. "WSI-35.jpg" -> "WSI-35"
    return mapping


def _build_encoders() -> list[TileEncoder]:
    from src.embeddings.genbio_pathfm import GenBioPathFMEncoder
    from src.embeddings.uni2h import UNI2HEncoder
    from src.embeddings.virchow2 import Virchow2Encoder

    return [UNI2HEncoder(), Virchow2Encoder(), GenBioPathFMEncoder()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--repo-id", default="kmmuleelab/Lung_Pathology_Image_JPG")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/embeddings"))
    ap.add_argument("--tmp-dir", type=Path, default=Path("/content/tmp_slides"))
    args = ap.parse_args()

    patients = pd.read_csv(args.split)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Listing repo files to resolve WSI_ID -> actual path (one-time)...")
    wsi_to_path = _resolve_wsi_to_repo_path(args.repo_id)
    print(f"Found {len(wsi_to_path)} jpgs in repo.")

    encoders = _build_encoders()
    for enc in encoders:
        (args.out_dir / enc.name).mkdir(parents=True, exist_ok=True)

    stats = {e.name: {"embed_dim": e.embed_dim, "wall_clock_sec": 0.0, "n_slides": 0} for e in encoders}
    t_total_start = time.time()

    for _, row in patients.iterrows():
        out_paths = {enc.name: args.out_dir / enc.name / f"{row.patient_id}.npy" for enc in encoders}
        if all(p.exists() for p in out_paths.values()):
            continue  # this patient is already fully done for all 3 models

        repo_path = wsi_to_path.get(row.wsi_id)
        if repo_path is None:
            print(f"[warn] could not find a repo file matching WSI_ID={row.wsi_id!r} — skipping patient {row.patient_id}")
            continue

        # 1. Download just this one slide.
        local_path = hf_hub_download(
            args.repo_id, repo_path, repo_type="dataset", local_dir=str(args.tmp_dir)
        )

        # 2. Tile it once.
        tiles = tile_slide(Path(local_path))
        if not tiles:
            print(f"[warn] 0 tissue tiles for patient {row.patient_id} ({row.wsi_id}) — skipping")
            os.remove(local_path)
            continue
        tile_arrays = [t.array for t in tiles]

        # 3. Embed with all 3 models, reusing the same tiles.
        for enc in encoders:
            out_path = out_paths[enc.name]
            if out_path.exists():
                continue
            t0 = time.time()
            tile_embeds = enc.embed_tiles(tile_arrays)
            slide_embed = mean_pool(tile_embeds)
            np.save(out_path, slide_embed.astype(np.float32))
            stats[enc.name]["wall_clock_sec"] += time.time() - t0
            stats[enc.name]["n_slides"] += 1

        # 4. Delete the raw image — this is what keeps disk usage flat.
        os.remove(local_path)

        done = sum(p.exists() for p in (args.out_dir / "uni2h").glob("*.npy"))
        print(f"[{done}/{len(patients)}] patient {row.patient_id} done, image deleted")

    for enc in encoders:
        stats[enc.name]["wall_clock_sec"] = round(stats[enc.name]["wall_clock_sec"], 1)
        stats[enc.name]["cache_dir"] = str(args.out_dir / enc.name)

    stats["total_wall_clock_sec"] = round(time.time() - t_total_start, 1)
    (args.out_dir / "extraction_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {args.out_dir / 'extraction_stats.json'}")


if __name__ == "__main__":
    main()