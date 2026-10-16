"""Metrics, bootstrap CIs, and the final §5.3 comparison table.

Changes vs. the first version
-----------------------------
* **Identical tuning budget across rows.** Every logistic-regression row (metadata only,
  each FM alone, each FM + metadata, fused) now uses the same classifier with C chosen
  by the same inner CV (``C="auto"``). Before, the single-FM rows were fixed at C=1
  while the fused row was tuned by the agent.
* **ABMIL rows** (when tile bags are loaded): each FM alone / + metadata through the
  same attention-MIL family and hyper-parameters as the fused MIL model.
* **Uncertainty for every row**: bootstrap CIs over test patients, and for MIL rows
  mean ± sd over ``final_seeds`` independently trained models (plus their ensemble).
* **Paired bootstrap of (fused − best single)** on the same resampled patients: the
  right test for "does fusion beat the best single model", instead of eyeballing whether
  one model's point estimate sits inside the other's CI.
"""
from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.fusion.data import FusionDataset
from src.fusion.strategies import LateFusionModel, build_model, make_logreg


def _macro_auroc(y_true, proba, n_classes) -> float:
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes))))


def _per_class_auroc(y_true, proba, classes: list[str]) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_class = roc_auc_score(y_true, proba, multi_class="ovr", average=None, labels=list(range(len(classes))))
        return {c: float(v) for c, v in zip(classes, per_class)}
    except ValueError:
        return {}


def _boot_indices(n: int, y: np.ndarray, n_classes: int, n_boot: int, seed: int):
    """Resamples of test patients; resamples missing a class are kept for balanced
    accuracy but flagged so AUROC can skip them."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yield idx, len(set(y[idx].tolist())) == n_classes


def bootstrap_ci(y_true, proba, preds, n_boot: int = 2000, seed: int = 42) -> dict:
    n_classes = proba.shape[1]
    aurocs, bal_accs = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, full in _boot_indices(len(y_true), y_true, n_classes, n_boot, seed):
            bal_accs.append(balanced_accuracy_score(y_true[idx], preds[idx]))
            if full:
                aurocs.append(_macro_auroc(y_true[idx], proba[idx], n_classes))
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None  # noqa: E731
    return {"auroc_ci": pct(aurocs), "bal_acc_ci": pct(bal_accs)}


def paired_bootstrap_delta(y_true, proba_a, proba_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap distribution of metric(A) - metric(B) on the SAME resampled patients."""
    n_classes = proba_a.shape[1]
    d_auc, d_bacc = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, full in _boot_indices(len(y_true), y_true, n_classes, n_boot, seed):
            yt = y_true[idx]
            d_bacc.append(balanced_accuracy_score(yt, proba_a[idx].argmax(1)) - balanced_accuracy_score(yt, proba_b[idx].argmax(1)))
            if full:
                d_auc.append(_macro_auroc(yt, proba_a[idx], n_classes) - _macro_auroc(yt, proba_b[idx], n_classes))
    d_auc, d_bacc = np.array(d_auc), np.array(d_bacc)
    return {
        "delta_auroc_mean": float(d_auc.mean()), "delta_auroc_ci": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))],
        "p_fused_not_better_auroc": float((d_auc <= 0).mean()),
        "delta_bacc_mean": float(d_bacc.mean()), "delta_bacc_ci": [float(np.percentile(d_bacc, 2.5)), float(np.percentile(d_bacc, 97.5))],
        "p_fused_not_better_bacc": float((d_bacc <= 0).mean()),
    }


def _row(name: str, ds: FusionDataset, proba: np.ndarray, extra: dict | None = None) -> dict:
    test_idx = np.where(ds.split == "test")[0]
    y_true = ds.y[test_idx]
    preds = proba.argmax(axis=1)
    n_classes = len(ds.classes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auroc = _macro_auroc(y_true, proba, n_classes)
        bal_acc = balanced_accuracy_score(y_true, preds)
    row = {
        "row": name,
        "macro_auroc": float(auroc),
        "balanced_accuracy": float(bal_acc),
        "per_class_auroc": _per_class_auroc(y_true, proba, ds.classes),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=list(range(n_classes))).tolist(),
        "classes": ds.classes,
        **bootstrap_ci(y_true, proba, preds),
        **(extra or {}),
        "_proba": proba,  # stripped before writing JSON
    }
    sd = f" (seeds: AUROC {extra['seed_auroc_mean']:.4f}±{extra['seed_auroc_sd']:.4f})" if extra and "seed_auroc_mean" in extra else ""
    print(f"[test] {name}: AUROC={auroc:.4f} balAcc={bal_acc:.4f} CI={row['auroc_ci']}{sd}")
    return row


def eval_config(config: dict, ds: FusionDataset, row_name: str, final_seeds: int = 1) -> dict:
    """Fit ``config`` on ALL of train+val, then score once on the held-out test set.
    MIL configs are trained ``final_seeds`` times independently: we report each seed's
    metrics (mean ± sd) and score the seed-ensemble as the row's headline number."""
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    test_idx = np.where(ds.split == "test")[0]
    if config["family"] != "mil" or final_seeds <= 1:
        model = build_model(config)
        model.fit(ds, trainval_idx)
        return _row(row_name, ds, model.predict_proba(ds, test_idx))

    y_true = ds.y[test_idx]
    probs, aucs, baccs, folds = [], [], [], []
    for s in range(final_seeds):
        cfg = {**copy.deepcopy(config), "seed": 1000 + s, "n_seeds": 1}
        model = build_model(cfg)
        model.fit(ds, trainval_idx)
        p = model.predict_proba(ds, test_idx)
        r = model.results[0]  # training history of this seed's network (train vs its early-stopping set)
        folds.append({"fold": s, "best_epoch": r.best_epoch, "stopped_epoch": r.stopped_epoch, "history": r.history})
        probs.append(p)
        aucs.append(_macro_auroc(y_true, p, len(ds.classes)))
        baccs.append(balanced_accuracy_score(y_true, p.argmax(1)))
    extra = {"seed_aurocs": aucs, "seed_auroc_mean": float(np.mean(aucs)), "seed_auroc_sd": float(np.std(aucs, ddof=1)),
             "seed_baccs": baccs, "seed_bacc_mean": float(np.mean(baccs)), "seed_bacc_sd": float(np.std(baccs, ddof=1)),
             "headline": f"ensemble of {final_seeds} seeds",
             "_curves": {"model": row_name, "hparams": model.results[0].hparams, "n_folds": final_seeds,
                         "kind": "final model, one curve per seed", "folds": folds}}
    return _row(row_name, ds, np.mean(probs, axis=0), extra)


def train_curves_only(config: dict, ds: FusionDataset, seeds: int, name: str) -> dict:
    """Train an attention-MIL config on train+val and return its per-epoch training curves.
    The test set is NOT touched -- this is only to show how a non-selected MIL model trains."""
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    folds = []
    for s in range(seeds):
        model = build_model({**copy.deepcopy(config), "seed": 1000 + s, "n_seeds": 1})
        model.fit(ds, trainval_idx)
        r = model.results[0]
        folds.append({"fold": s, "best_epoch": r.best_epoch, "stopped_epoch": r.stopped_epoch, "history": r.history})
    return {"model": name, "hparams": model.results[0].hparams, "n_folds": seeds,
            "kind": "best attention-MIL candidate (not selected; test not used)", "config": config, "folds": folds}


def eval_lr_single(model_name: str, ds: FusionDataset, with_meta: bool) -> dict:
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    test_idx = np.where(ds.split == "test")[0]
    if with_meta:
        single = LateFusionModel(config={"combine": "mean", "C": "auto"})
        single._fit_one_model(ds, model_name, trainval_idx)
        return _row(f"{model_name} + metadata [LR]", ds, single._proba_one_model(ds, model_name, test_idx))
    scaler = StandardScaler().fit(ds.embeddings[model_name][trainval_idx])
    clf = make_logreg("auto").fit(scaler.transform(ds.embeddings[model_name][trainval_idx]), ds.y[trainval_idx])
    return _row(f"{model_name} alone [LR]", ds, clf.predict_proba(scaler.transform(ds.embeddings[model_name][test_idx])))


def eval_metadata_only(ds: FusionDataset) -> dict:
    trainval_idx = np.where(np.isin(ds.split, ["train", "val"]))[0]
    test_idx = np.where(ds.split == "test")[0]
    age_scaler = StandardScaler().fit(ds.age[trainval_idx].reshape(-1, 1))

    def _features(idx: np.ndarray) -> np.ndarray:
        return np.concatenate([age_scaler.transform(ds.age[idx].reshape(-1, 1)), ds.sex[idx].reshape(-1, 1)], axis=1)

    clf = make_logreg("auto").fit(_features(trainval_idx), ds.y[trainval_idx])
    return _row("metadata only (age + sex)", ds, clf.predict_proba(_features(test_idx)))


class _RowCache:
    """Each finished test row is saved to <out>/rows/ so a crashed run resumes without retraining."""

    def __init__(self, folder: Path | None):
        self.folder = Path(folder) if folder else None
        if self.folder:
            self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, name: str, fn, signature=None):
        if self.folder is None:
            return fn()
        sig = json.dumps(signature, sort_keys=True, default=str)
        f = self.folder / (name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "") + ".json")
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("_sig", sig) == sig:  # same config as the cached row -> reuse
                r["_proba"] = np.array(r["_proba"])
                print(f"[test] {r['row']}: loaded from {f.name} (AUROC={r['macro_auroc']:.4f})")
                return r
            print(f"[test] {name}: cached row was made with a different config -- recomputing")
        r = fn()
        f.write_text(json.dumps({**r, "_proba": r["_proba"].tolist(), "_sig": sig}, default=str))
        return r


def build_comparison_table(agent_best_config: dict, ds: FusionDataset, model_names: list[str],
                           final_seeds: int = 3, cache_dir: Path | None = None) -> list[dict]:
    """Required §5.3 rows (+ context rows), all on the same test set, each touched once."""
    cached = _RowCache(cache_dir)
    rows = [cached("metadata_only", lambda: eval_metadata_only(ds))]
    rows += [cached(f"{m}_alone_LR", lambda m=m: eval_lr_single(m, ds, with_meta=False)) for m in model_names]
    single_meta = [cached(f"{m}_meta_LR", lambda m=m: eval_lr_single(m, ds, with_meta=True)) for m in model_names]
    rows += single_meta

    if ds.bags is not None:
        # same ABMIL family as the fused MIL model, one FM at a time, each with ITS OWN tuned
        # settings from 02g (per-FM search) -- the single-model counterpart of the agent's search
        for m in model_names:
            cfg = {"family": "mil", "models": [m], "tuned_single": True}
            from src.mil.config import resolve_hparams
            sig = [resolve_hparams(m)[0].to_dict(), final_seeds]
            rows.append(cached(f"{m}_alone_ABMIL", lambda cfg=cfg, m=m: eval_config(
                {**cfg, "use_meta": False}, ds, f"{m} alone [ABMIL]", final_seeds), sig))
            r = cached(f"{m}_meta_ABMIL", lambda cfg=cfg, m=m: eval_config(
                {**cfg, "use_meta": True}, ds, f"{m} + metadata [ABMIL]", final_seeds), sig)
            rows.append(r)
            single_meta.append(r)

    # identical classifier family: "best single" is chosen among single-FM rows of the SAME
    # family as the fused winner (ABMIL rows if it is attention-MIL, LR rows otherwise);
    # the other family's rows stay in the table as context
    tag = "[ABMIL]" if agent_best_config.get("family") == "mil" else "[LR]"
    pool = [r for r in single_meta if r["row"].endswith(tag)]
    best_single = max(pool or single_meta, key=lambda r: r["macro_auroc"])
    rows.append({**best_single, "row": f"best single model + metadata ({best_single['row']})"})
    fused = cached("fused", lambda: eval_config(agent_best_config, ds, "fused (agent-selected) + metadata", final_seeds),
                   [agent_best_config, final_seeds])
    test_idx = np.where(ds.split == "test")[0]
    fused["vs_best_single"] = paired_bootstrap_delta(ds.y[test_idx], fused["_proba"], best_single["_proba"])
    rows.append(fused)
    d = fused["vs_best_single"]
    print(f"\n[test] fused - best single: dAUROC {d['delta_auroc_mean']:+.4f} CI {d['delta_auroc_ci']} "
          f"(P[fused<=single]={d['p_fused_not_better_auroc']:.2f}); dBalAcc {d['delta_bacc_mean']:+.4f} CI {d['delta_bacc_ci']}")
    return rows  # caller strips "_proba"/"_curves" before writing JSON
