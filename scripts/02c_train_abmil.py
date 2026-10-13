"""Stage 2c: trains ABMILClassifier (src/embeddings/aggregate.py), writes attention-pooled (D,) embeddings to --emb-out-dir. Needs 02b_extract_embeddings.py --save-tile-embeds run first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.embeddings.aggregate import ABMILClassifier
from src.data.tile_cache import save_npy_atomic

LABEL_COL = "subtype"


def _load_bags(tile_emb_dir: Path, patient_ids: list[str]) -> dict[str, np.ndarray]:
    bags = {}
    for pid in patient_ids:
        p = tile_emb_dir / f"{pid}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.shape[0] == 0:
                print(f"[warn] {pid} has a 0-tile embedding cache -- skipping")
                continue
            bags[pid] = arr
    return bags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--tile-emb-dir", required=True, type=Path)
    ap.add_argument("--emb-out-dir", required=True, type=Path)
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.split)
    classes = sorted(df[LABEL_COL].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["y"] = df[LABEL_COL].map(class_to_idx)

    all_pids = df["patient_id"].astype(str).tolist()
    bags = _load_bags(args.tile_emb_dir, all_pids)
    missing = [p for p in all_pids if p not in bags]
    if missing:
        print(f"[warn] {len(missing)} patients have no tile-embedding cache yet -- skipping them")

    df = df[df["patient_id"].astype(str).isin(bags.keys())].reset_index(drop=True)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    if len(train_df) == 0:
        raise ValueError("no train patients have a tile-embedding cache -- check --tile-emb-dir and --split")
    if len(val_df) == 0:
        raise ValueError("no val patients have a tile-embedding cache -- check --tile-emb-dir and --split")

    n_classes = len(classes)
    model = ABMILClassifier(args.embed_dim, n_classes, args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # reduction="sum" (not "mean"): loss is per-patient, one bag at a time, and "mean"
    # would divide by that one sample's own class weight -- cancelling the weighting
    # out, and giving 0/0=NaN for any class absent from train.
    class_counts = train_df["y"].value_counts().reindex(range(n_classes), fill_value=0).to_numpy()
    class_weights = torch.tensor(
        np.where(class_counts > 0, 1.0 / np.maximum(class_counts, 1), 0.0), dtype=torch.float32
    ).to(device)
    train_loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
    val_loss_fn = nn.CrossEntropyLoss()  # unweighted: early-stopping signal, not training

    def _epoch_loss(rows, train: bool) -> float:
        model.train(train)
        total, n = 0.0, 0
        rows = rows.sample(frac=1, random_state=args.seed) if train else rows
        for _, row in rows.iterrows():
            pid = str(row.patient_id)
            x = torch.from_numpy(bags[pid]).float().to(device)
            y = torch.tensor([row.y], dtype=torch.long).to(device)
            if train:
                opt.zero_grad()
            logits, _, _ = model(x)
            loss = train_loss_fn(logits.unsqueeze(0), y) if train else val_loss_fn(logits.unsqueeze(0), y)
            if train:
                loss.backward()
                opt.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    t0 = time.time()
    for epoch in range(args.max_epochs):
        train_loss = _epoch_loss(train_df, train=True)
        val_loss = _epoch_loss(val_df, train=False)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch % 10 == 0 or epochs_without_improvement >= args.patience:
            print(f"[{args.model}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"best_val={best_val:.4f}")
        if epochs_without_improvement >= args.patience:
            print(f"[{args.model}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[{args.model}] training done in {time.time() - t0:.1f}s, best val loss {best_val:.4f}")

    model.eval()
    args.emb_out_dir.mkdir(parents=True, exist_ok=True)
    attn_summary = {}
    with torch.inference_mode():
        for pid, bag in bags.items():
            x = torch.from_numpy(bag).float().to(device)
            _, pooled, weights = model(x)
            save_npy_atomic(args.emb_out_dir / f"{pid}.npy", pooled.cpu().numpy().astype(np.float32))
            w = weights.cpu().numpy()
            attn_summary[pid] = {"n_tiles": int(len(w)), "max_weight": float(w.max()), "entropy": float(
                -(w * np.log(w + 1e-12)).sum()
            )}

    (args.emb_out_dir.parent / f"{args.model}_attention_summary.json").write_text(
        json.dumps(attn_summary, indent=2)
    )
    print(f"[{args.model}] wrote {len(bags)} attention-pooled embeddings to {args.emb_out_dir}")


if __name__ == "__main__":
    main()
