"""Build the reduced-protocol subset and the fixed train/val/test split.

Pipeline
--------
1. Read CLWD.csv (one row per slide).
2. Group by ``SampleNumber`` (this MERGES the duplicated ID 8377886 into one patient).
3. Keep one slide per patient: the lowest numeric WSI_ID (deterministic rule).
4. Patient label = the selected slide's ``Tumor Subtype``. Patients whose slides
   disagree on subtype are flagged (``label_conflict``) so a sensitivity run can
   exclude them.
5. Stratified 70/10/20 split by subtype. Because there is exactly one row per
   patient after step 3, a plain stratified split is also a grouped split; this
   is asserted, not assumed.
6. Save the assignment to a CSV plus a JSON sidecar (seed, counts, input hash).

Usage
-----
    python -m src.data.make_split --csv path/to/CLWD.csv --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_FRAC = 0.20
VAL_FRAC = 0.10  # fraction of the WHOLE set
LABEL_COL = "Benchmark_Label_7class"
GROUP_COL = "SampleNumber"


def _wsi_number(wsi_id: str) -> int:
    """'WSI-35' -> 35 (numeric, so WSI-9 < WSI-35 < WSI-103)."""
    return int(str(wsi_id).split("-")[-1])


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse slide-level rows to one row per patient (group)."""
    df = df.copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
    df["wsi_num"] = df["WSI_ID"].map(_wsi_number)

    # Quirk 2: detect patients with conflicting labels across their slides.
    n_labels = df.groupby(GROUP_COL)[LABEL_COL].nunique()
    conflict_ids = set(n_labels[n_labels > 1].index)
    n_slides = df.groupby(GROUP_COL).size().rename("n_slides")

    # Deterministic selection: lowest numeric WSI_ID per patient.
    sel = df.sort_values(["wsi_num"]).groupby(GROUP_COL, as_index=False).head(1)
    sel = sel.merge(n_slides, left_on=GROUP_COL, right_index=True)

    out = pd.DataFrame(
        {
            "patient_id": sel[GROUP_COL].values,
            "wsi_id": sel["WSI_ID"].values,
            "subtype": sel[LABEL_COL].values,
            "age": sel["Age"].values,
            "sex": sel["Sex"].values,
            "diagnosis": sel.get("Pathological_Diagnosis", pd.Series(index=sel.index)).values,
            "n_slides_available": sel["n_slides"].values,
            "label_conflict": sel[GROUP_COL].isin(conflict_ids).values,
        }
    )
    return out.sort_values("patient_id").reset_index(drop=True)


def selection_bias_report(df: pd.DataFrame, selected: pd.DataFrame) -> dict:
    """Cheap, metadata-only check on the 'lowest WSI_ID' selection rule.

    For every patient with >1 slide, compares the label on the SELECTED
    slide against the majority label across ALL of that patient's slides.
    A low agreement rate would mean the selection rule is systematically
    picking non-representative slides -- worth reporting either way, not
    just assumed away.
    """
    df = df.copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
    multi = df.groupby(GROUP_COL).filter(lambda g: len(g) > 1)
    if multi.empty:
        return {"n_multi_slide_patients": 0, "agreement_rate": None}

    majority = multi.groupby(GROUP_COL)[LABEL_COL].agg(lambda s: s.value_counts().idxmax())
    sel_by_patient = selected.set_index("patient_id")["subtype"]

    matches = 0
    total = 0
    for pid in majority.index:
        if pid in sel_by_patient.index:
            total += 1
            matches += int(sel_by_patient.loc[pid] == majority.loc[pid])

    return {
        "n_multi_slide_patients": int(total),
        "n_selected_matches_majority": int(matches),
        "agreement_rate": round(matches / total, 3) if total else None,
    }


def assign_splits(patients: pd.DataFrame, seed: int) -> pd.DataFrame:
    assert patients["patient_id"].is_unique, "need exactly one row per patient"
    y = patients["subtype"]

    trainval_idx, test_idx = train_test_split(
        patients.index, test_size=TEST_FRAC, stratify=y, random_state=seed
    )
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)  # 0.125 of the remaining 80%
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_rel,
        stratify=y.loc[trainval_idx],
        random_state=seed,
    )
    patients = patients.copy()
    patients["split"] = "train"
    patients.loc[val_idx, "split"] = "val"
    patients.loc[test_idx, "split"] = "test"
    return patients


def sanity_checks(p: pd.DataFrame) -> None:
    assert p["patient_id"].is_unique
    assert set(p["split"]) == {"train", "val", "test"}
    classes = set(p["subtype"])
    for name, g in p.groupby("split"):
        missing = classes - set(g["subtype"])
        assert not missing, f"split '{name}' is missing classes: {missing}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/splits"))
    args = ap.parse_args()

    raw = pd.read_csv(args.csv)
    patients = build_patient_table(raw)
    bias_report = selection_bias_report(raw, patients)
    patients = assign_splits(patients, args.seed)
    sanity_checks(patients)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"split_seed{args.seed}.csv"
    patients.to_csv(out_csv, index=False)

    meta = {
        "seed": args.seed,
        "n_slides_raw": int(len(raw)),
        "n_patients": int(len(patients)),
        "split_counts": patients["split"].value_counts().to_dict(),
        "subtype_by_split": pd.crosstab(patients["subtype"], patients["split"]).to_dict(),
        "n_label_conflict_patients": int(patients["label_conflict"].sum()),
        "input_csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "selection_rule": "lowest numeric WSI_ID per SampleNumber",
        "duplicate_id_policy": "grouped by SampleNumber string, so repeated IDs merge into one patient",
        "label_policy": "label of the selected slide; conflicts flagged in label_conflict",
        "selection_bias_check": bias_report,
    }
    (args.out_dir / f"split_seed{args.seed}.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out_csv}")
    print(patients["split"].value_counts().to_string())
    print(pd.crosstab(patients["subtype"], patients["split"]).to_string())
    print(f"Label-conflict patients: {meta['n_label_conflict_patients']}")
    print(f"Selection-bias check: {bias_report}")


if __name__ == "__main__":
    main()