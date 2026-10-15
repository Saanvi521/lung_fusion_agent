"""Loading cached tile-embedding bags (written by 02b ``--save-tile-embeds``).

A *bag* is one patient's slide: ``{fm_name: (N_tiles, D_fm) array}``. All FMs embed the
same tiles in the same order (02b reads each patient's tile zip in sorted file-name
order), so the rows line up across FMs; ``load_bags`` checks this when it can.
Bags are kept in RAM as float16 to halve memory (all three FMs for ~200 patients x 600
tiles is ~2 GB in float16) and are cast to float32 on the device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COL = "subtype"


def load_bags(
    tile_emb_root: Path,
    model_names: list[str],
    patient_ids: list[str],
    dtype=np.float16,
) -> dict[str, dict[str, np.ndarray]]:
    """Returns {pid: {fm: (N, D)}} for patients that have a non-empty bag for EVERY fm.

    ``tile_emb_root`` is the folder that contains one sub-folder per FM
    (``<root>/<fm>/<pid>.npy``), i.e. ``<out-dir>/tile_embeddings`` from 02b.
    """
    tile_emb_root = Path(tile_emb_root)
    bags: dict[str, dict[str, np.ndarray]] = {}
    n_mismatch = 0
    for pid in patient_ids:
        per_fm = {}
        for m in model_names:
            p = tile_emb_root / m / f"{pid}.npy"
            if not p.exists():
                break
            arr = np.load(p)
            if arr.ndim != 2 or arr.shape[0] == 0:
                break
            per_fm[m] = arr.astype(dtype, copy=False)
        if len(per_fm) != len(model_names):
            continue
        counts = {a.shape[0] for a in per_fm.values()}
        if len(counts) > 1:
            n_mismatch += 1
            continue  # tiles don't line up across FMs -> can't fuse at tile level
        bags[pid] = per_fm
    if n_mismatch:
        print(f"[bags] skipped {n_mismatch} patients whose tile counts differ across FMs")
    print(f"[bags] loaded {len(bags)} bags for {model_names} from {tile_emb_root}")
    return bags


def load_split_table(split_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """Split CSV with string patient ids, plus the sorted class list (same order as src/fusion/data.py)."""
    df = pd.read_csv(split_csv)
    df["patient_id"] = df["patient_id"].astype(str)
    classes = sorted(df[LABEL_COL].unique().tolist())
    df["y"] = df[LABEL_COL].map({c: i for i, c in enumerate(classes)})
    df["sex01"] = (df["sex"].astype(str).str.lower() == "male").astype(np.float32)
    return df, classes


def raw_meta(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """{pid: [age, sex01]} (age NOT standardised — the trainer fits that on train patients)."""
    return {
        pid: np.array([age, sex], dtype=np.float32)
        for pid, age, sex in zip(df["patient_id"], df["age"].astype(np.float32), df["sex01"])
    }
