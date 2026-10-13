"""Stage 2d: cross-fitted ABMIL embeddings (replaces 02c for the ABMIL comparison).

The problem with 02c
--------------------
02c trains ONE ABMIL network on the train patients, then uses that same network
to pool every patient. So the train patients' vectors come from a network that
already saw their labels (too clean), while the test patients' vectors don't
(realistic). The fusion classifier learns on the too-clean vectors and is then
tested on realistic ones, so it underperforms.

The fix (cross-fitting)
-----------------------
Split train+val into K groups. For each group k:
    train a FRESH ABMIL network on the other K-1 groups,
    use it to pool ONLY group k's patients.
Every train/val patient's vector now comes from a network that never saw them,
so it is exactly as "realistic" as a test patient's vector.

Test patients: pooled by all K networks (none of which saw any test patient),
and the K vectors are averaged.

This works because ABMIL's pooled vector is a weighted average of the raw
foundation-model tile features, so vectors from different networks all live in
the same feature space and can be mixed and averaged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.tile_cache import save_npy_atomic
from src.embeddings.aggregate import ABMILClassifier

LABEL_COL = "subtype"


def train_abmil(bags, y_by_pid, pids, n_classes, args, seed, device):
    """Train one fresh ABMILClassifier on `pids` only.

    Same recipe as 02c (Adam, class-weighted loss, one slide per step, early
    stopping on validation loss). The only difference: the early-stopping
    patients are a 15% carve-out of `pids` itself, so the network never sees
    anyone outside `pids` in any way.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    labels = [y_by_pid[p] for p in pids]
    fit_pids, stop_pids = train_test_split(pids, test_size=0.15, random_state=seed, stratify=labels)

    model = ABMILClassifier(args.embed_dim, n_classes, args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    counts = np.bincount([y_by_pid[p] for p in fit_pids], minlength=n_classes)
    class_weights = torch.tensor(np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0), dtype=torch.float32)
    train_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="sum")  # same as 02c
    stop_loss_fn = nn.CrossEntropyLoss()

    def run_epoch(epoch_pids, train):
        model.train(train)
        order = list(epoch_pids)
        if train:
            rng.shuffle(order)  # new order every epoch (02c reused the same order every epoch)
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

    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(args.max_epochs):
        run_epoch(fit_pids, train=True)
        stop_loss = run_epoch(stop_pids, train=False)
        if stop_loss < best_loss - 1e-4:
            best_loss, bad_epochs = stop_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is None:  # only happens if the loss was NaN from the very first epoch
        raise RuntimeError("ABMIL training never produced a finite early-stopping loss -- check the tile embeddings")
    model.load_state_dict(best_state)
    model.eval()
    print(f"  trained on {len(fit_pids)} patients, early-stopped on {len(stop_pids)}, "
          f"stopped at epoch {epoch}, best stop-loss {best_loss:.4f}")
    return model


@torch.inference_mode()
def pool(model, bags, pids, device):
    """Returns {pid: (pooled_vector, class_probabilities)}."""
    out = {}
    for pid in pids:
        logits, pooled, _ = model(torch.from_numpy(bags[pid]).float().to(device))
        out[pid] = (pooled.cpu().numpy(), torch.softmax(logits, dim=0).cpu().numpy())
    return out


def macro_auroc(y, probs, n_classes):
    return float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(n_classes))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--tile-emb-dir", required=True, type=Path)
    ap.add_argument("--emb-out-dir", required=True, type=Path)
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    # same defaults as 02c, so the ONLY change vs. the old run is the cross-fitting
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load labels + tile bags ----
    df = pd.read_csv(args.split)
    df["patient_id"] = df["patient_id"].astype(str)
    classes = sorted(df[LABEL_COL].unique().tolist())  # same class order as src/fusion/data.py
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
    print(f"[{args.model}] {len(bags)} patients have tile embeddings")

    trainval = df[df["split"].isin(["train", "val"])]["patient_id"].to_numpy()
    test = df[df["split"] == "test"]["patient_id"].to_numpy()
    trainval_y = np.array([y_by_pid[p] for p in trainval])

    # ---- cross-fitting ----
    results = {}                         # pid -> (vector, probs) for train+val patients
    test_results = {p: [] for p in test}  # pid -> list of (vector, probs), one per fold network
    folds = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    for k, (fit_idx, out_idx) in enumerate(folds.split(trainval, trainval_y)):
        print(f"[{args.model}] fold {k + 1}/{args.n_folds}")
        net = train_abmil(bags, y_by_pid, list(trainval[fit_idx]), n_classes, args, seed=args.seed + k, device=device)
        results.update(pool(net, bags, trainval[out_idx], device))  # only patients this net never saw
        for pid, r in pool(net, bags, test, device).items():
            test_results[pid].append(r)

    for pid, rs in test_results.items():  # average the K networks' outputs for each test patient
        results[pid] = (np.mean([r[0] for r in rs], axis=0), np.mean([r[1] for r in rs], axis=0))

    # ---- save one pooled vector per patient (same format as 02c) ----
    args.emb_out_dir.mkdir(parents=True, exist_ok=True)
    for pid, (vec, _) in results.items():
        save_npy_atomic(args.emb_out_dir / f"{pid}.npy", vec.astype(np.float32))
    print(f"[{args.model}] wrote {len(results)} cross-fitted embeddings to {args.emb_out_dir}")

    # ---- check: ABMIL's own classifier head, no logistic regression on top ----
    scores = {
        "abmil_head_out_of_fold_auroc_trainval": macro_auroc(
            trainval_y, np.stack([results[p][1] for p in trainval]), n_classes),
        "abmil_head_test_auroc": macro_auroc(
            np.array([y_by_pid[p] for p in test]), np.stack([results[p][1] for p in test]), n_classes),
    }
    (args.emb_out_dir.parent / f"{args.model}_abmil_head_scores.json").write_text(json.dumps(scores, indent=2))
    print(f"[{args.model}] ABMIL own-head scores: {json.dumps(scores)}")


if __name__ == "__main__":
    main()
