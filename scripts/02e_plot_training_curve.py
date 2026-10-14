"""Stage 2e: plot a training/validation loss curve for the report.

This reproduces fold 1 of 02d_crossfit_abmil.py's actual 5-fold
cross-fitting
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.embeddings.aggregate import ABMILClassifier

LABEL_COL = "subtype"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--tile-emb-dir", required=True, type=Path)
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/figures"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold-to-plot", type=int, default=0, help="which of the n-folds cross-fit folds to log (0-indexed)")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- same loading as 02d ----
    df = pd.read_csv(args.split)
    df["patient_id"] = df["patient_id"].astype(str)
    classes = sorted(df[LABEL_COL].unique().tolist())
    n_classes = len(classes)
    y_by_pid = dict(zip(df["patient_id"], df[LABEL_COL].map({c: i for i, c in enumerate(classes)})))

    bags = {}
    for pid in df["patient_id"]:
        path = args.tile_emb_dir / f"{pid}.npy"
        if path.exists():
            arr = np.load(path)
            if arr.shape[0] > 0:
                bags[pid] = arr
    df = df[df["patient_id"].isin(bags)]

    trainval = df[df["split"].isin(["train", "val"])]["patient_id"].to_numpy()
    trainval_y = np.array([y_by_pid[p] for p in trainval])

    # ---- same fold split as 02d; isolate just the one fold we're logging ----
    folds = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_list = list(folds.split(trainval, trainval_y))
    fit_idx, out_idx = fold_list[args.fold_to_plot]
    fit_pids_all = list(trainval[fit_idx])
    k = args.fold_to_plot
    print(f"[{args.model}] logging fold {k + 1}/{args.n_folds} ({len(fit_pids_all)} patients available to this fold)")

    # ---- same train_abmil recipe as 02d, with per-epoch logging added ----
    torch.manual_seed(args.seed + k)
    rng = np.random.RandomState(args.seed + k)
    labels = [y_by_pid[p] for p in fit_pids_all]
    fit_pids, stop_pids = train_test_split(fit_pids_all, test_size=0.15, random_state=args.seed + k, stratify=labels)

    model = ABMILClassifier(args.embed_dim, n_classes, args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    counts = np.bincount([y_by_pid[p] for p in fit_pids], minlength=n_classes)
    class_weights = torch.tensor(np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0), dtype=torch.float32)
    train_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="sum")
    stop_loss_fn = nn.CrossEntropyLoss()

    def run_epoch(epoch_pids, train):
        model.train(train)
        order = list(epoch_pids)
        if train:
            rng.shuffle(order)
        total = 0.0
        for pid in order:
            x = torch.from_numpy(bags[pid]).float().to(device)
            y = torch.tensor([y_by_pid[pid]], device=device)
            with torch.set_grad_enabled(train):
                logits, _, _ = model(x)
                loss = (train_loss_fn if train else stop_loss_fn)(logits.unsqueeze(0), y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item()
        return total / len(order)

    train_losses, val_losses = [], []
    best_loss, bad_epochs, stopped_at = float("inf"), 0, None
    for epoch in range(args.max_epochs):
        train_losses.append(run_epoch(fit_pids, train=True))
        val_losses.append(run_epoch(stop_pids, train=False))
        if val_losses[-1] < best_loss - 1e-4:
            best_loss, bad_epochs = val_losses[-1], 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            stopped_at = epoch
            break
    if stopped_at is None:
        stopped_at = args.max_epochs - 1

    (args.out_dir / f"{args.model}_training_curve.json").write_text(json.dumps({
        "fold_logged": k, "n_folds": args.n_folds,
        "train_loss": train_losses, "val_loss": val_losses, "stopped_at_epoch": stopped_at,
    }, indent=2))

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.axvline(stopped_at, color="gray", linestyle="--", label=f"Early stop (epoch {stopped_at})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"ABMIL training curve — {args.model} (cross-fit fold {k + 1}/{args.n_folds})")
    plt.legend()
    plt.tight_layout()
    out_png = args.out_dir / f"{args.model}_training_curve.png"
    plt.savefig(out_png, dpi=150)
    print(f"[{args.model}] saved {out_png} (stopped at epoch {stopped_at})")


if __name__ == "__main__":
    main()