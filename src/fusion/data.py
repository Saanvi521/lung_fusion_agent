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
    bags: dict | None = None  # {pid: {fm: (N_tiles, D)}} -- only when tile bags were loaded (mil family)
    # {fm: (n_patients, n_classes)} cross-fitted ABMIL class probabilities saved by 02d
    # (<fm>_abmil_probs.npz): out-of-fold for train+val patients, mean of the 5 fold networks for
    # test patients. Only loaded with --abmil-probs-dir; used by the ``abmil_late`` family.
    abmil_probs: dict[str, np.ndarray] | None = None


def load_fusion_dataset(
    split_csv: Path,
    embeddings_dir: Path,
    model_names: list[str] = MODEL_NAMES,
    tile_emb_root: Path | None = None,
    abmil_probs_dir: Path | None = None,
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

    bags = None
    if tile_emb_root is not None:
        from src.mil.bags import load_bags

        bags = load_bags(Path(tile_emb_root), model_names, df["patient_id"].astype(str).tolist())
        has_bag = df["patient_id"].astype(str).isin(bags)
        if (~has_bag).any():
            print(f"[data] dropping {int((~has_bag).sum())} more patients without tile bags for every model")
        df = df[has_bag].reset_index(drop=True)
    patient_ids = df["patient_id"].astype(str).tolist()

    emb_arrays: dict[str, np.ndarray] = {}
    for m in model_names:
        vecs = [np.load(embeddings_dir / m / f"{pid}.npy") for pid in patient_ids]
        emb_arrays[m] = np.stack(vecs).astype(np.float32)

    age = df["age"].astype(np.float32).to_numpy()
    sex = (df["sex"].astype(str).str.lower() == "male").astype(np.float32).to_numpy()
    y = df[LABEL_COL].map(class_to_idx).to_numpy()
    split = df["split"].astype(str).to_numpy()

    abmil_probs = None
    if abmil_probs_dir is not None:
        abmil_probs = load_abmil_probs(Path(abmil_probs_dir), model_names, patient_ids, classes)

    return FusionDataset(
        patient_ids=patient_ids,
        embeddings=emb_arrays,
        age=age,
        sex=sex,
        y=y,
        classes=classes,
        split=split,
        bags=bags,
        abmil_probs=abmil_probs,
    )


def load_abmil_probs(
    probs_dir: Path, model_names: list[str], patient_ids: list[str], classes: list[str]
) -> dict[str, np.ndarray]:
    """Read ``<probs_dir>/<fm>_abmil_probs.npz`` (written by 02d_crossfit_abmil) for every FM and
    align the rows to ``patient_ids``. Fails loudly if a patient or the class order doesn't match,
    because a silent misalignment would mix up whose prediction is whose."""
    out: dict[str, np.ndarray] = {}
    for m in model_names:
        f = probs_dir / f"{m}_abmil_probs.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found -- run 02d_crossfit_abmil for {m} first")
        z = np.load(f, allow_pickle=False)
        npz_classes = [str(c) for c in z["classes"]]
        if npz_classes != classes:
            raise ValueError(f"{f}: class order {npz_classes} != dataset class order {classes}")
        row = {str(p): i for i, p in enumerate(z["pids"])}
        missing = [p for p in patient_ids if p not in row]
        if missing:
            raise ValueError(f"{f} has no prediction for {len(missing)} patients, e.g. {missing[:3]}")
        out[m] = z["probs"][[row[p] for p in patient_ids]].astype(np.float64)
    print(f"[data] loaded cross-fitted ABMIL probabilities for {model_names} from {probs_dir}")
    return out
