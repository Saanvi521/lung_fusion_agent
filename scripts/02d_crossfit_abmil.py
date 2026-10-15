"""Stage 2d: cross-fitted ABMIL for ONE foundation model, with full training curves.

Cross-fitting (unchanged idea, see report §4): split train+val into K folds; for each
fold train a fresh ABMIL on the other K-1 folds and use it only on the held-out fold.
Every train/val patient's outputs then come from a network that never saw them; test
patients get the average over the K networks.

What is new
-----------
* Uses the shared trainer (src/mil/trainer.py): projection + dropout + standardised
  features, AdamW + warm-up + cosine, gradient accumulation, tile sub-sampling.
* Per-FM hyper-parameters from conf/abmil.yaml (+ conf/tuned/abmil_<fm>.json from 02g).
* Logs, for EVERY fold and EVERY epoch: train/val cross-entropy, macro AUROC and
  balanced accuracy (computed identically for train and val) -> <out>/<fm>_history.json.
  Plot with 02e. The curves are the curves of the networks actually used.
* Saves the ABMIL head's cross-fitted class probabilities (<fm>_abmil_probs.npz) so
  they can be late-fused, not just the pooled vectors.
* Does NOT print test scores unless --report-test is passed: hyper-parameters must be
  chosen with 02g (train+val only), never by looking at these numbers.

Outputs (in --emb-out-dir and its parent):
    <emb-out-dir>/<pid>.npy                 attention-weighted mean of the RAW tile features
                                            (same space for every fold network, so test vectors
                                            can be averaged; consumed by 03_run_agent)
    <parent>/<fm>_abmil_probs.npz           pids, split, y, probs (OOF for train/val; K-avg for test)
    <parent>/<fm>_history.json              hparams + per-fold per-epoch metrics
    <parent>/<fm>_abmil_head_scores.json    OOF train+val metrics (+ test only with --report-test)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.tile_cache import save_npy_atomic
from src.mil.bags import load_bags, load_split_table, raw_meta
from src.mil.config import DEFAULT_CONF, resolve_hparams
from src.mil.trainer import summarize, train_mil


def add_hparam_overrides(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("hyper-parameter overrides (default: from conf/abmil.yaml + tuned json)")
    for name, typ in [("lr", float), ("weight_decay", float), ("dropout", float), ("dropout_in", float),
                      ("proj_dim", int), ("attn_dim", int), ("max_epochs", int), ("min_epochs", int),
                      ("patience", int), ("accum_steps", int), ("max_train_tiles", int),
                      ("label_smoothing", float), ("monitor", str), ("class_weighting", str)]:
        g.add_argument(f"--{name.replace('_', '-')}", dest=name, type=typ, default=None)


def overrides_from(args) -> dict:
    keys = ["lr", "weight_decay", "dropout", "dropout_in", "proj_dim", "attn_dim", "max_epochs", "min_epochs",
            "patience", "accum_steps", "max_train_tiles", "label_smoothing", "monitor", "class_weighting"]
    return {k: getattr(args, k) for k in keys}


@torch.inference_mode()
def raw_pooled(result, pids: list[str], fm: str) -> dict[str, np.ndarray]:
    """Attention-weighted average of the RAW tile features (keeps one shared feature space)."""
    model, batcher = result.model, result.batcher
    model.eval()
    out = {}
    for pid in pids:
        x, m = batcher(pid)
        w = model(x, m)["weights"][fm]
        out[pid] = (w.unsqueeze(-1) * x[fm]).sum(0).float().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="FM name, e.g. uni2h / virchow2 / genbio_pathfm")
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--tile-emb-root", type=Path, default=None,
                    help="folder containing <fm>/<pid>.npy tile bags (02b --save-tile-embeds output)")
    ap.add_argument("--tile-emb-dir", type=Path, default=None,
                    help="(old flag) <root>/<fm> directly; --tile-emb-root is preferred")
    ap.add_argument("--emb-out-dir", required=True, type=Path)
    ap.add_argument("--embed-dim", type=int, default=None, help="ignored (read from the data); kept for old commands")
    ap.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    ap.add_argument("--no-tuned", action="store_true", help="ignore conf/tuned/abmil_<fm>.json")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--es-frac", type=float, default=0.15, help="early-stopping carve-out of each fold's training part")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-test", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every epoch")
    add_hparam_overrides(ap)
    args = ap.parse_args()

    root = args.tile_emb_root or (args.tile_emb_dir.parent if args.tile_emb_dir else None)
    if root is None:
        ap.error("pass --tile-emb-root (or the old --tile-emb-dir)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp, sources = resolve_hparams(args.model, args.conf, use_tuned=not args.no_tuned, cli_overrides=overrides_from(args))
    print(f"[{args.model}] hyper-parameters from {sources}:\n  {json.dumps(hp.to_dict())}")

    df, classes = load_split_table(args.split)
    n_classes = len(classes)
    bags = load_bags(root, [args.model], df["patient_id"].tolist())
    df = df[df["patient_id"].isin(bags)]
    y_by_pid = dict(zip(df["patient_id"], df["y"].astype(int)))
    meta = raw_meta(df)

    trainval = df[df["split"].isin(["train", "val"])]["patient_id"].to_numpy()
    test = df[df["split"] == "test"]["patient_id"].tolist()
    y_tv = np.array([y_by_pid[p] for p in trainval])

    probs, vecs = {}, {}
    test_probs = {p: [] for p in test}
    test_vecs = {p: [] for p in test}
    folds_log = []
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    for k, (fit_idx, out_idx) in enumerate(skf.split(trainval, y_tv)):
        fit_all = list(trainval[fit_idx])
        fit_pids, es_pids = train_test_split(
            fit_all, test_size=args.es_frac, random_state=args.seed + k, stratify=[y_by_pid[p] for p in fit_all]
        )
        hp.seed = args.seed + k
        res = train_mil(bags, y_by_pid, fit_pids, es_pids, n_classes, hp, fms=[args.model], meta=meta,
                        device=device, verbose=args.verbose, tag=f"[{args.model} f{k + 1}] ")
        out_pids = list(trainval[out_idx])
        p_out = res.predict(out_pids)
        held = summarize(np.array([y_by_pid[p] for p in out_pids]), p_out, n_classes)
        b = res.history[res.best_epoch]
        print(f"[{args.model}] fold {k + 1}/{args.n_folds}: fit={len(fit_pids)} es={len(es_pids)} "
              f"held-out={len(out_pids)} | best epoch {res.best_epoch} (stopped {res.stopped_epoch}) | "
              f"es-loss {b['val_loss']:.3f} es-AUROC {b['val_auroc']:.3f} | "
              f"held-out loss {held['loss']:.3f} AUROC {held['auroc']:.3f} bAcc {held['bacc']:.3f}")
        folds_log.append({"fold": k, "n_fit": len(fit_pids), "n_es": len(es_pids), "n_heldout": len(out_pids),
                          "best_epoch": res.best_epoch, "stopped_epoch": res.stopped_epoch,
                          "heldout": held, "history": res.history})
        probs.update(zip(out_pids, p_out))
        vecs.update(raw_pooled(res, out_pids, args.model))
        for pid, p in zip(test, res.predict(test)):
            test_probs[pid].append(p)
        for pid, v in raw_pooled(res, test, args.model).items():
            test_vecs[pid].append(v)

    for pid in test:
        probs[pid] = np.mean(test_probs[pid], axis=0)
        vecs[pid] = np.mean(test_vecs[pid], axis=0)

    # ---- write outputs ----
    args.emb_out_dir.mkdir(parents=True, exist_ok=True)
    for pid, v in vecs.items():
        save_npy_atomic(args.emb_out_dir / f"{pid}.npy", v.astype(np.float32))
    parent = args.emb_out_dir.parent
    pids = list(trainval) + test
    np.savez(parent / f"{args.model}_abmil_probs.npz",
             pids=np.array(pids), split=np.array(["trainval"] * len(trainval) + ["test"] * len(test)),
             y=np.array([y_by_pid[p] for p in pids]), probs=np.stack([probs[p] for p in pids]),
             classes=np.array(classes))
    (parent / f"{args.model}_history.json").write_text(json.dumps({
        "model": args.model, "hparams": {**hp.to_dict(), "seed": args.seed}, "hparam_sources": sources,
        "n_folds": args.n_folds, "classes": classes, "folds": folds_log}, indent=1))

    oof = summarize(y_tv, np.stack([probs[p] for p in trainval]), n_classes)
    scores = {"oof_trainval": oof}
    msg = f"[{args.model}] OOF train+val: loss {oof['loss']:.3f} AUROC {oof['auroc']:.4f} bAcc {oof['bacc']:.4f}"
    if args.report_test:
        te = summarize(np.array([y_by_pid[p] for p in test]), np.stack([probs[p] for p in test]), n_classes)
        scores["test"] = te
        msg += f" | TEST AUROC {te['auroc']:.4f} bAcc {te['bacc']:.4f}"
    (parent / f"{args.model}_abmil_head_scores.json").write_text(json.dumps(scores, indent=2))
    print(msg)
    print(f"[{args.model}] wrote {len(vecs)} pooled vectors to {args.emb_out_dir} and history to "
          f"{parent / (args.model + '_history.json')}")


if __name__ == "__main__":
    main()
