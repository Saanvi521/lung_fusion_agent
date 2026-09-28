"""Stage 2a: download each slide ONCE, tile it, cache the tiles, delete the
raw image. No foundation model is loaded here at all -- this step is pure
CPU + disk I/O, so it can't run out of GPU/model memory.

This is what makes it safe to later load the three foundation models one
at a time (02b) WITHOUT re-downloading each ~1GB slide three times: the
tiles are already sitting in the cache, read from disk in a fraction of a
second.

Point --out-dir at a MOUNTED GOOGLE DRIVE PATH. Every patient's tile zip is
written atomically the moment it's ready, so a Colab disconnect at patient
150 loses at most the one patient that was mid-flight -- rerunning this
exact command resumes from patient 151 automatically.

Usage
-----
    from google.colab import drive
    drive.mount('/content/drive')

    python -m scripts.02a_tile_slides \
        --split artifacts/splits/split_seed42.csv \
        --out-dir /content/drive/MyDrive/lung_fusion_data \
        --repo-id kmmuleelab/Lung_Pathology_Image_JPG
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

from src.data.tile_cache import tiles_path, write_tile_zip
from src.data.tiling import tile_slide


def _resolve_wsi_to_repo_path(repo_id: str) -> dict[str, str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    jpgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
    return {Path(p).stem: p for p in jpgs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--repo-id", default="kmmuleelab/Lung_Pathology_Image_JPG")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Mounted Drive path. Tile cache is written to <out-dir>/tiles/.")
    ap.add_argument("--tmp-dir", type=Path, default=Path("/content/tmp_slides"))
    args = ap.parse_args()

    patients = pd.read_csv(args.split)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = args.out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    print("Listing repo files to resolve WSI_ID -> actual path (one-time)...")
    wsi_to_path = _resolve_wsi_to_repo_path(args.repo_id)
    print(f"Found {len(wsi_to_path)} jpgs in repo.")

    n_done = n_skipped = 0
    t0 = time.time()

    for _, row in patients.iterrows():
        out_zip = tiles_path(tiles_dir, row.patient_id)
        if out_zip.exists():
            continue  # already tiled -- Drive persistence means this survives disconnects

        repo_path = wsi_to_path.get(row.wsi_id)
        if repo_path is None:
            print(f"[warn] no repo file matching WSI_ID={row.wsi_id!r} — skipping patient {row.patient_id}")
            n_skipped += 1
            continue

        local_path = hf_hub_download(args.repo_id, repo_path, repo_type="dataset", local_dir=str(args.tmp_dir))
        tiles = tile_slide(Path(local_path))
        os.remove(local_path)  # raw slide is never needed again once tiled

        if not tiles:
            print(f"[warn] 0 tissue tiles for patient {row.patient_id} ({row.wsi_id}) — skipping")
            n_skipped += 1
            continue

        write_tile_zip(tiles, out_zip)
        n_done += 1
        elapsed = time.time() - t0
        rate = elapsed / n_done if n_done else 0
        print(f"[{n_done+n_skipped}/{len(patients)}] patient {row.patient_id}: {len(tiles)} tiles cached "
              f"({rate:.0f}s/patient avg)")

    print(f"\nDone. {n_done} patients tiled, {n_skipped} skipped, cache at {tiles_dir}")


if __name__ == "__main__":
    main()
