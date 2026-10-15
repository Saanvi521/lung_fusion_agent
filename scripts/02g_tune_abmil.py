"""Stage 2g: per-foundation-model ABMIL hyper-parameter search (train+val ONLY).

Different FMs need different hyper-parameters: their features differ in dimension
(1536 / 2560 / 4608), scale and how linearly separable the subtypes are. This script
picks them in a budgeted, auditable way:

* candidates = the prior in conf/abmil.yaml + `tuning.budget` configs drawn at random
  (seeded) from `tuning.grid`;
* each candidate is scored by K-fold stratified CV on train+val (each fold trains on
  K-1 folds with its own early-stopping carve-out, and is scored on the held-out fold);
* the winner minimises mean held-out cross-entropy (or `--select-by`), and is written
  to conf/tuned/abmil_<fm>.json, which 02d and the `mil` fusion family pick up.

The test split is never loaded. Run once per FM, before 02d.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.mil.bags import load_bags, load_split_table, raw_meta
from src.mil.config import DEFAULT_CONF, TUNED_DIR, load_conf, resolve_hparams, tuned_path
from src.mil.trainer import merge_hparams, summarize, train_mil


def cv_score(bags, y_by_pid, meta, pids, n_classes, hp, fm, n_folds, seed, device, es_frac=0.15):
    y = np.array([y_by_pid[p] for p in pids])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows, epochs = [], []
    for k, (a, b) in enumerate(skf.split(pids, y)):
        fit_all = [pids[i] for i in a]
        fit, es = train_test_split(fit_all, test_size=es_frac, random_state=seed + k,
                                   stratify=[y_by_pid[p] for p in fit_all])
        hp.seed = seed + k
        res = train_mil(bags, y_by_pid, fit, es, n_classes, hp, fms=[fm], meta=meta, device=device)
        held = [pids[i] for i in b]
        rows.append(summarize(y[b], res.predict(held), n_classes))
        epochs.append(res.best_epoch)
    agg = {}
    for key in ("loss", "auroc", "bacc"):
        v = np.array([r[key] for r in rows], dtype=float)
        agg[f"val_{key}"] = float(np.nanmean(v))
        agg[f"val_{key}_se"] = float(np.nanstd(v, ddof=1) / np.sqrt(len(v)))
    agg["fold_scores"] = rows
    agg["best_epochs"] = epochs
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--tile-emb-root", required=True, type=Path)
    ap.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--select-by", choices=["val_loss", "val_auroc", "val_bacc"], default=None)
    ap.add_argument("--max-epochs", type=int, default=None, help="override for quick runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-dir", type=Path, default=Path("artifacts/tuning"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = load_conf(args.conf)
    tcfg = conf.get("tuning", {})
    budget = args.budget if args.budget is not None else tcfg.get("budget", 12)
    n_folds = args.n_folds or tcfg.get("n_folds", 5)
    select_by = args.select_by or tcfg.get("select_by", "val_loss")
    grid = tcfg.get("grid", {})

    prior, _ = resolve_hparams(args.model, args.conf, use_tuned=False,
                               cli_overrides={"max_epochs": args.max_epochs})
    keys = list(grid)
    all_combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    random.Random(args.seed).shuffle(all_combos)
    candidates = [{}] + all_combos[:budget]   # {} = the prior itself

    df, classes = load_split_table(args.split)
    df = df[df["split"].isin(["train", "val"])]          # test never loaded
    bags = load_bags(args.tile_emb_root, [args.model], df["patient_id"].tolist())
    df = df[df["patient_id"].isin(bags)]
    y_by_pid = dict(zip(df["patient_id"], df["y"].astype(int)))
    pids = df["patient_id"].tolist()
    meta = raw_meta(df)

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.model}_tuning.jsonl"
    done = {}  # resume: trials already finished in an earlier (interrupted) run
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[json.dumps(r["delta_from_prior"], sort_keys=True)] = r
        if done:
            print(f"[{args.model}] resuming: {len(done)} finished trials in {log_path}")
    results = []
    maximize = select_by != "val_loss"
    for i, delta in enumerate(candidates):
        hp = merge_hparams(prior.to_dict(), delta)
        prev = done.get(json.dumps(delta, sort_keys=True))
        if prev is not None:
            results.append(prev)
            print(f"[{args.model}] {i:2d}/{len(candidates) - 1} (cached) loss {prev['val_loss']:.3f} AUROC {prev['val_auroc']:.3f}")
            continue
        t0 = time.time()
        r = cv_score(bags, y_by_pid, meta, pids, len(classes), hp, args.model, n_folds, args.seed, device)
        rec = {"trial": i, "delta_from_prior": delta, "hparams": hp.to_dict(), **r, "sec": round(time.time() - t0, 1)}
        results.append(rec)
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{args.model}] {i:2d}/{len(candidates) - 1} {json.dumps(delta) if delta else 'prior':<95} "
              f"loss {r['val_loss']:.3f}±{r['val_loss_se']:.3f}  AUROC {r['val_auroc']:.3f}±{r['val_auroc_se']:.3f}  "
              f"bAcc {r['val_bacc']:.3f}  ({rec['sec']}s)")

    key = (lambda r: r[select_by]) if maximize else (lambda r: -r[select_by])
    best = max(results, key=key)
    # how many candidates are within one SE of the winner? (selection-noise check for the report)
    se = best[f"{select_by}_se"]
    within = [r["trial"] for r in results if abs(r[select_by] - best[select_by]) <= se]
    # store only the keys the search changed; everything else keeps coming from conf/abmil.yaml
    # (so a quick run with --max-epochs doesn't silently persist a short schedule)
    best_hp = dict(best["delta_from_prior"])
    full_hp = dict(best["hparams"])
    full_hp.pop("seed", None)
    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    tuned_path(args.model).write_text(json.dumps({
        "model": args.model, "select_by": select_by, "n_folds": n_folds, "budget": budget, "seed": args.seed,
        "best_trial": best["trial"], "best_score": {k: best[k] for k in best if k.startswith("val_")},
        "trials_within_1se_of_best": within, "best_hparams": best_hp, "full_hparams_used": full_hp,
        "note": "chosen by CV on train+val only; test split never loaded",
    }, indent=2))
    print(f"\n[{args.model}] best = trial {best['trial']} ({json.dumps(best['delta_from_prior']) or 'prior'}) "
          f"{select_by}={best[select_by]:.4f}±{se:.4f}; {len(within)} candidates within 1 SE -> {within}")
    print(f"[{args.model}] wrote {tuned_path(args.model)} and {log_path}")


if __name__ == "__main__":
    main()
