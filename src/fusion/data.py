"""Loads cached per-model embeddings + patient metadata into plain arrays.

Everything downstream (fusion strategies, the agent) works off of one
``FusionDataset`` built once per run -- nothing here touches images or
foundation models. This is exactly the boundary the brief asks for:
"never re-run a foundation model inside a hyperparameter loop."
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAMES = ["uni2h", "virchow2", "genbio_pathfm"]
LABEL_COL = "subtype"  # real split CSV column name (NOT "label")


@dataclass
class FusionDataset:
    patient_ids: list[str]
    embeddings: dict[str, np.ndarray]  # model_name -> (n_patients, embed_dim)
    age: np.ndarray  # (n_patients,) float, standardized later per-fold
    sex: np.ndarray  # (n_patients,) 0/1
    y: np.ndarray  # (n_patients,) int class index
    classes: list[str]
    split: np.ndarray  # (n_patients,) "train" / "val" / "test"


def load_fusion_dataset(
    split_csv: Path,
    embeddings_dir: Path,
    model_names: list[str] = MODEL_NAMES,
) -> FusionDataset:
    df = pd.read_csv(split_csv)
    classes = sorted(df[LABEL_COL].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_patient_ids = df["patient_id"].astype(str).tolist()
    keep_mask = np.ones(len(df), dtype=bool)

    for i, pid in enumerate(all_patient_ids):
        for m in model_names:
            if not (embeddings_dir / m / f"{pid}.npy").exists():
                keep_mask[i] = False
                break

    # Drop any patient missing an embedding for ANY model -- fusion needs
    # all three streams present. Report how many were dropped so it's
    # visible in the report, not a silent shrink of the dataset.
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        print(f"[data] dropping {n_dropped} patients missing at least one model's embedding")

    df = df[keep_mask].reset_index(drop=True)
    patient_ids = df["patient_id"].astype(str).tolist()

    emb_arrays: dict[str, np.ndarray] = {}
    for m in model_names:
        vecs = [np.load(embeddings_dir / m / f"{pid}.npy") for pid in patient_ids]
        emb_arrays[m] = np.stack(vecs).astype(np.float32)

    age = df["age"].astype(np.float32).to_numpy()
    sex = (df["sex"].astype(str).str.lower() == "male").astype(np.float32).to_numpy()
    y = df[LABEL_COL].map(class_to_idx).to_numpy()
    split = df["split"].astype(str).to_numpy()

    return FusionDataset(
        patient_ids=patient_ids,
        embeddings=emb_arrays,
        age=age,
        sex=sex,
        y=y,
        classes=classes,
        split=split,
    )
