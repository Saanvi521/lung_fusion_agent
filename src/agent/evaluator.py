"""Scores a single fusion config.

This is the piece that answers "how good is this config" for the agent.
Critically: it does NOT use the single 21-patient validation split alone --
that's the exact failure mode the brief warns about (a 200-trial agent will
overfit a 21-patient val set). Instead it pools train+val (~167 patients)
and runs repeated stratified k-fold CV, so every config is scored against
many different train/val partitions and we get a genuine mean +- spread,
not one lucky/unlucky number.

The committed test split is never touched here -- this file only ever sees
patients whose ds.split is "train" or "val".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.fusion.data import FusionDataset
from src.fusion.strategies import build_model


@dataclass
class EvalResult:
    mean_auroc: float
    std_auroc: float
    mean_bal_acc: float
    std_bal_acc: float
    fold_aurocs: list[float]
    fold_bal_accs: list[float]
    n_folds: int


def evaluate_config(
    config: dict,
    ds: FusionDataset,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> EvalResult:
    trainval_mask = np.isin(ds.split, ["train", "val"])
    trainval_idx = np.where(trainval_mask)[0]
    y_trainval = ds.y[trainval_idx]

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    fold_aurocs, fold_bal_accs = [], []
    n_classes = len(ds.classes)

    for fold_train_local, fold_val_local in rskf.split(trainval_idx, y_trainval):
        fold_train_idx = trainval_idx[fold_train_local]
        fold_val_idx = trainval_idx[fold_val_local]

        model = build_model(config)
        model.fit(ds, fold_train_idx)
        proba = model.predict_proba(ds, fold_val_idx)
        y_true = ds.y[fold_val_idx]

        # macro one-vs-rest AUROC needs every class represented in y_true;
        # with 7 classes and small folds this can occasionally fail -- skip
        # that fold's AUROC rather than crash the whole search.
        try:
            auroc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(n_classes)))
            fold_aurocs.append(auroc)
        except ValueError:
            pass

        preds = proba.argmax(axis=1)
        fold_bal_accs.append(balanced_accuracy_score(y_true, preds))

    return EvalResult(
        mean_auroc=float(np.mean(fold_aurocs)) if fold_aurocs else float("nan"),
        std_auroc=float(np.std(fold_aurocs)) if fold_aurocs else float("nan"),
        mean_bal_acc=float(np.mean(fold_bal_accs)),
        std_bal_acc=float(np.std(fold_bal_accs)),
        fold_aurocs=fold_aurocs,
        fold_bal_accs=fold_bal_accs,
        n_folds=len(fold_bal_accs),
    )
