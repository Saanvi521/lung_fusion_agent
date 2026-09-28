"""Metrics, bootstrap CIs, and the final §5.3 comparison table.

Kept separate from ``scripts/03_run_agent.py`` so evaluation logic is reusable
(e.g. from a notebook or a plotting script) without re-running the agent.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from src.fusion.data import FusionDataset
from src.fusion.strategies import LateFusionModel, build_model


def bootstrap_ci(y_true, proba, preds, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap CIs over test patients. With a 7-class, small-n test set
    (Cribriform alone has ~3 test patients), a fair number of resamples
    will simply be missing a rare class -- macro OvR AUROC is undefined for
    those resamples, so we skip them for the AUROC average (not an error,
    just an unusable resample) rather than let sklearn's warning fire
    ~1000 times per run."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_classes = proba.shape[1]
    aurocs, bal_accs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        bal_accs.append(balanced_accuracy_score(yt, preds[idx]))
        if len(set(yt.tolist())) < n_classes:
            continue  # a rare class didn't appear in this resample; skip AUROC only
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                aurocs.append(
                    roc_auc_score(yt, proba[idx], multi_class="ovr", average="macro", labels=list(range(n_classes)))
                )
            except ValueError:
                pass
    return {
        "auroc_ci": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))] if aurocs else None,
        "bal_acc_ci": [float(np.percentile(bal_accs, 2.5)), float(np.percentile(bal_accs, 97.5))],
    }


def eval_on_test(config: dict, ds: FusionDataset, row_name: str) -> dict:
    """Fit ``config`` on ALL of train+val, then score once on the held-out test set."""
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    test_idx = np.where(ds.split == "test")[0]

    model = build_model(config)
    model.fit(ds, trainval_idx)  # refit on ALL of train+val, test touched only now
    proba = model.predict_proba(ds, test_idx)
    preds = proba.argmax(axis=1)
    y_true = ds.y[test_idx]
    n_classes = len(ds.classes)

    auroc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    bal_acc = balanced_accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds, labels=list(range(n_classes)))
    ci = bootstrap_ci(y_true, proba, preds)

    result = {
        "row": row_name,
        "macro_auroc": float(auroc),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm.tolist(),
        "classes": ds.classes,
        **ci,
    }
    print(f"[test] {row_name}: AUROC={auroc:.4f} balAcc={bal_acc:.4f}")
    return result


def eval_single_model(model_name: str, ds: FusionDataset) -> dict:
    """Isolate one foundation model's signal: a late-fusion model restricted
    to a single embedding stream reduces to 'that model + metadata'."""
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    test_idx = np.where(ds.split == "test")[0]

    single = LateFusionModel(config={"combine": "mean", "C": 1.0})
    single._fit_one_model(ds, model_name, trainval_idx)
    proba = single._proba_one_model(ds, model_name, test_idx)
    preds = proba.argmax(axis=1)
    y_true = ds.y[test_idx]
    n_classes = len(ds.classes)

    auroc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    bal_acc = balanced_accuracy_score(y_true, preds)
    row = {"row": f"{model_name} + metadata", "macro_auroc": float(auroc), "balanced_accuracy": float(bal_acc)}
    print(f"[test] {model_name} + metadata: AUROC={auroc:.4f} balAcc={bal_acc:.4f}")
    return row


def build_comparison_table(agent_best_config: dict, ds: FusionDataset, model_names: list[str]) -> list[dict]:
    """The §5.3 table: each single model + metadata, then the agent-selected
    fused model, all evaluated on the held-out test set exactly once."""
    rows = [eval_single_model(m, ds) for m in model_names]
    best_single = max(rows, key=lambda r: r["macro_auroc"])
    print(f"[test] best single model: {best_single['row']}")

    fused_result = eval_on_test(agent_best_config, ds, "fused (agent-selected) + metadata")
    rows.append(fused_result)

    diff = fused_result["macro_auroc"] - best_single["macro_auroc"]
    print(f"\n[test] fused - best_single AUROC delta: {diff:+.4f}")
    return rows
