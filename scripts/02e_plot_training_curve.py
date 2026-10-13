"""Stage 2e: plot ABMIL's per-epoch train/validation loss curve for the report.

This does NOT change any production embeddings or results -- it's a
diagnostic-only run. It retrains one ABMIL network (same recipe as 02c/02d's
train_abmil) on one foundation model, logging train loss and held-out
("stop") loss every epoch, then saves a PNG plot plus the raw numbers.

By default it reproduces 02c's original setup (one network trained on all
146 train patients, early-stopped on the 21 val patients) since that's the
simplest, most interpretable curve for a report figure. Run once per model
if you want all three, or just one (e.g. uni2h) if you only need to show
the method is learning and stopping sensibly.

Usage
-----
    python -m scripts.02e_plot_training_curve --model uni2h \
        --split artifacts/splits/split_seed42.csv \
        --tile-emb-dir /content/drive/MyDrive/lung_fusion_data/tile_embeddings/uni2h \
        --embed-dim 1536 \
        --out-dir artifacts/figures
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
from sklearn.model_selection import train_test_split

from src.embeddings.aggregate import ABMILClassifier

LABEL_COL = "subtype"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--tile-emb-dir", required=True, type=Path)
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/figures"))
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    # Reproduces 02c: one network trained on train, early-stopped on val.
    fit_pids = df[(df["split"] == "train") & (df["patient_id"].isin(bags))]["patient_id"].tolist()
    stop_pids = df[(df["split"] == "val") & (df["patient_id"].isin(bags))]["patient_id"].tolist()
    print(f"[{args.model}] training on {len(fit_pids)} patients, validating on {len(stop_pids)}")

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
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

    # Save the raw numbers too, not just the picture.
    (args.out_dir / f"{args.model}_training_curve.json").write_text(json.dumps({
        "train_loss": train_losses, "val_loss": val_losses, "stopped_at_epoch": stopped_at,
    }, indent=2))

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.axvline(stopped_at, color="gray", linestyle="--", label=f"Early stop (epoch {stopped_at})")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"ABMIL training curve — {args.model}")
    plt.legend()
    plt.tight_layout()
    out_png = args.out_dir / f"{args.model}_training_curve.png"
    plt.savefig(out_png, dpi=150)
    print(f"[{args.model}] saved {out_png} (stopped at epoch {stopped_at})")


if __name__ == "__main__":
    main()
