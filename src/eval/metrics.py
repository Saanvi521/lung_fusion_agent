"""Metrics, bootstrap CIs, and the final §5.3 comparison table.

Kept separate from ``scripts/03_run_agent.py`` so evaluation logic is reusable
(e.g. from a notebook or a plotting script) without re-running the agent.

The brief's required §5.3 table has 6 rows: metadata only, each foundation
model ALONE (no metadata), best single model + metadata, and the fused
model + metadata. This file produces all 6, plus (as extra context, not a
replacement for them) each model's own "+ metadata" score, so the report can
still discuss how much metadata helps each model individually.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.fusion.data import FusionDataset
from src.fusion.strategies import LateFusionModel, build_model


def _per_class_auroc(y_true, proba, classes: list[str]) -> dict:
    """Per-class one-vs-rest AUROC, required alongside the macro average.
    Returns {} if a class is entirely missing from y_true (can't be scored)."""
    n_classes = len(classes)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_class = roc_auc_score(y_true, proba, multi_class="ovr", average=None, labels=list(range(n_classes)))
        return {c: float(v) for c, v in zip(classes, per_class)}
    except ValueError:
        return {}


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
        "per_class_auroc": _per_class_auroc(y_true, proba, ds.classes),
        "confusion_matrix": cm.tolist(),
        "classes": ds.classes,
        **ci,
    }
    print(f"[test] {row_name}: AUROC={auroc:.4f} balAcc={bal_acc:.4f}")
    return result


def eval_single_model(model_name: str, ds: FusionDataset) -> dict:
    """One foundation model's late-fusion-style classifier, WITH metadata
    concatenated in (this is 'that model + metadata', not the model alone --
    see eval_model_alone for the no-metadata version the brief also wants)."""
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
    row = {
        "row": f"{model_name} + metadata",
        "macro_auroc": float(auroc),
        "balanced_accuracy": float(bal_acc),
        "per_class_auroc": _per_class_auroc(y_true, proba, ds.classes),
    }
    print(f"[test] {model_name} + metadata: AUROC={auroc:.4f} balAcc={bal_acc:.4f}")
    return row


def eval_model_alone(model_name: str, ds: FusionDataset) -> dict:
    """Required by the brief's §5.3 table: one foundation model's embedding
    ALONE, no age/sex mixed in at all. Same classifier family/budget
    (LogisticRegression, C=1.0, class_weight='balanced') as every other row,
    so the comparison isolates the fusion/metadata effect, not tuning effort."""
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    test_idx = np.where(ds.split == "test")[0]

    scaler = StandardScaler().fit(ds.embeddings[model_name][trainval_idx])
    X_train = scaler.transform(ds.embeddings[model_name][trainval_idx])
    X_test = scaler.transform(ds.embeddings[model_name][test_idx])

    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X_train, ds.y[trainval_idx])
    proba = clf.predict_proba(X_test)
    preds = proba.argmax(axis=1)
    y_true = ds.y[test_idx]
    n_classes = len(ds.classes)

    auroc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    bal_acc = balanced_accuracy_score(y_true, preds)
    row = {
        "row": f"{model_name} alone",
        "macro_auroc": float(auroc),
        "balanced_accuracy": float(bal_acc),
        "per_class_auroc": _per_class_auroc(y_true, proba, ds.classes),
    }
    print(f"[test] {model_name} alone: AUROC={auroc:.4f} balAcc={bal_acc:.4f}")
    return row


def eval_metadata_only(ds: FusionDataset) -> dict:
    """Required by the brief's §5.3 table: age + sex only, no embeddings
    at all. Same classifier family/budget as every other row."""
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    test_idx = np.where(ds.split == "test")[0]

    age_scaler = StandardScaler().fit(ds.age[trainval_idx].reshape(-1, 1))

    def _features(idx: np.ndarray) -> np.ndarray:
        age_std = age_scaler.transform(ds.age[idx].reshape(-1, 1))
        sex = ds.sex[idx].reshape(-1, 1)
        return np.concatenate([age_std, sex], axis=1)

    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(_features(trainval_idx), ds.y[trainval_idx])
    proba = clf.predict_proba(_features(test_idx))
    preds = proba.argmax(axis=1)
    y_true = ds.y[test_idx]
    n_classes = len(ds.classes)

    auroc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
    bal_acc = balanced_accuracy_score(y_true, preds)
    row = {
        "row": "metadata only (age + sex)",
        "macro_auroc": float(auroc),
        "balanced_accuracy": float(bal_acc),
        "per_class_auroc": _per_class_auroc(y_true, proba, ds.classes),
    }
    print(f"[test] metadata only: AUROC={auroc:.4f} balAcc={bal_acc:.4f}")
    return row


def build_comparison_table(agent_best_config: dict, ds: FusionDataset, model_names: list[str]) -> list[dict]:
    """The required §5.3 table (metadata-only, each model alone, best single
    + metadata, fused + metadata), plus each model's own +metadata score as
    extra context for the report. All of this is computed on the SAME held-
    out test set, touched exactly once per row, immediately before this
    function returns -- nothing here is re-used from the agent's CV search."""
    rows = []

    rows.append(eval_metadata_only(ds))
    rows += [eval_model_alone(m, ds) for m in model_names]

    # Extra context beyond the brief's minimum table: how much does metadata
    # help each model individually? Not one of the 6 required rows, but the
    # brief asks for "quality of failure analysis" -- this is cheap evidence
    # worth having for that discussion.
    single_rows = [eval_single_model(m, ds) for m in model_names]
    rows += single_rows

    best_single = max(single_rows, key=lambda r: r["macro_auroc"])
    rows.append({**best_single, "row": f"best single model + metadata ({best_single['row']})"})
    print(f"[test] best single model: {best_single['row']}")

    fused_result = eval_on_test(agent_best_config, ds, "fused (agent-selected) + metadata")
    rows.append(fused_result)

    diff = fused_result["macro_auroc"] - best_single["macro_auroc"]
    print(f"\n[test] fused - best_single AUROC delta: {diff:+.4f}")
    return rows