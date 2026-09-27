"""Stage 2 entry point -- final version.

History of what changed and why (kept here so it's not lost to chat history):

  v1: patient-outer, all 3 models loaded at once, downloaded whole JPGs
      naively. CRASHED: Image.open(...).convert("RGB") on an 80x slide
      decodes the FULL image before we ever downsample it -- an 8-15GB
      spike in RAM per slide, easily enough to OOM free Colab.
  v2: switched to model-outer (one model loaded at a time) to dodge the
      RAM spike. This "worked" but re-downloads every slide up to 3x --
      with ~1.1GB average slide size x ~209 patients x 3 passes, that's
      several hundred GB of redundant network transfer, ballooning total
      runtime toward ~18 hours.
  v3 (this file): fixed the ACTUAL bug -- src/data/tiling.py now uses
      PIL's JPEG draft() mode, decoding directly at reduced resolution
      (measured ~5x lower peak memory). That removes the reason v2 was
      needed, so we go back to patient-outer: download each slide ONCE,
      reuse the same tiles across all 3 already-loaded models. This cuts
      total download volume back down to ~1x, which is where most of the
      wall-clock time actually goes (downloads dominate; GPU compute per
      slide is comparatively fast).

Also: --out-dir should point at a mounted Google Drive path (see the
Colab cell below), so every single embedding is durably saved the
instant it's written -- a disconnect can no longer lose finished work,
only the one patient that was mid-flight.

Usage
-----
    from google.colab import drive
    drive.mount('/content/drive')

    python -m scripts.02_extract_embeddings \
        --split artifacts/splits/split_seed42.csv \
        --out-dir /content/drive/MyDrive/lung_fusion_embeddings \
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
    files = list_repo_files(repo_id, repo_type="dataset")
    jpgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
    return {Path(p).stem: p for p in jpgs}


def _build_encoders() -> list[TileEncoder]:
    from src.embeddings.genbio_pathfm import GenBioPathFMEncoder
    from src.embeddings.uni2h import UNI2HEncoder
    from src.embeddings.virchow2 import Virchow2Encoder

    return [UNI2HEncoder(), Virchow2Encoder(), GenBioPathFMEncoder()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--repo-id", default="kmmuleelab/Lung_Pathology_Image_JPG")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="Point this at a mounted Drive path so progress survives disconnects.")
    ap.add_argument("--tmp-dir", type=Path, default=Path("/content/tmp_slides"))
    ap.add_argument("--batch-size", type=int, default=32,
                     help="Tile batch size per model forward pass. Lowered from 64 as extra "
                          "headroom now that all 3 models share GPU memory again.")
    args = ap.parse_args()

    patients = pd.read_csv(args.split)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Listing repo files to resolve WSI_ID -> actual path (one-time)...")
    wsi_to_path = _resolve_wsi_to_repo_path(args.repo_id)
    print(f"Found {len(wsi_to_path)} jpgs in repo.")

    print("Loading all 3 models (one-time; ~10GB total, should fit a T4 comfortably "
          "now that image decoding no longer spikes memory)...")
    encoders = _build_encoders()
    for enc in encoders:
        (args.out_dir / enc.name).mkdir(parents=True, exist_ok=True)

    stats = {e.name: {"embed_dim": e.embed_dim, "wall_clock_sec": 0.0, "n_slides": 0} for e in encoders}
    t_total_start = time.time()
    n_done = 0

    for _, row in patients.iterrows():
        out_paths = {enc.name: args.out_dir / enc.name / f"{row.patient_id}.npy" for enc in encoders}
        if all(p.exists() for p in out_paths.values()):
            continue  # fully done already -- Drive persistence means this survives disconnects

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
        tile_arrays = [t.array for t in tiles]

        for enc in encoders:
            out_path = out_paths[enc.name]
            if out_path.exists():
                continue
            t0 = time.time()
            tile_embeds = enc.embed_tiles(tile_arrays, batch_size=args.batch_size)
            slide_embed = mean_pool(tile_embeds)
            np.save(out_path, slide_embed.astype(np.float32))  # -> Drive, persisted immediately
            stats[enc.name]["wall_clock_sec"] += time.time() - t0
            stats[enc.name]["n_slides"] += 1

        os.remove(local_path)  # only the raw image is temporary; embeddings are already saved

        n_done += 1
        print(f"[{n_done}/{len(patients)}] patient {row.patient_id} done (all 3 models, image deleted)")

    for enc in encoders:
        stats[enc.name]["wall_clock_sec"] = round(stats[enc.name]["wall_clock_sec"], 1)
        stats[enc.name]["cache_dir"] = str(args.out_dir / enc.name)
    stats["total_wall_clock_sec"] = round(time.time() - t_total_start, 1)
    (args.out_dir / "extraction_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()