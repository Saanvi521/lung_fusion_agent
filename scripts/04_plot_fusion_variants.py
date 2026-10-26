"""Stage 4: visual comparison of the fusion strategies the agent searched.

Two figures, no test-set use:

1. ``fusion_variants_curves.png`` -- training curves (validation loss, macro AUROC,
   balanced accuracy per epoch) of the three attention-MIL fusion variants
   (tile_concat / gated / late) overlaid. For each variant we take its best-scoring
   configuration from the agent's decision log (for the SELECTED variant: the config the agent
   actually selected) and train it on train+val with the
   usual 15% early-stopping carve-out, ``--seeds`` times. Lines = mean over seeds,
   bands = +/- sd. Logistic-regression families have no epochs, so no curves.

2. ``agent_search.png`` -- every agent trial's cross-validated AUROC (+/- sd over
   folds), coloured by family (early / late / learned / mil), accepted trials filled,
   with the incumbent ("best so far") as a step line. This is the only way to show
   the logistic-regression families visually.

Resumable: each variant's curves are saved to <run>/curves_variants/<variant>.json
and skipped on re-run.

Usage:
    python -m scripts.04_plot_fusion_variants --split artifacts/splits/split_seed42.csv \
        --embeddings-dir $D/embeddings --tile-emb-root $D/tile_embeddings \
        --run-dir $D/agent_run_all --out-dir $D/figures/fusion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.eval.metrics import train_curves_only  # noqa: E402
from src.fusion.data import load_fusion_dataset  # noqa: E402

VARIANT_COLORS = {"tile_concat": "#2a6fdb", "gated": "#e8743b", "late": "#3a9e6f"}
FAMILY_COLORS = {"early": "#8c8c8c", "late": "#b07aa1", "learned": "#d4a72c", "mil": "#2a6fdb"}


def load_trials(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (run_dir / "decision_log.jsonl").read_text().splitlines() if l.strip()]


def best_per_variant(trials: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for t in trials:
        c = t["config"]
        if c.get("family") != "mil" or t["mean_auroc"] != t["mean_auroc"]:
            continue
        v = c.get("fusion", "tile_concat")
        if v not in best or t["mean_auroc"] > best[v]["mean_auroc"]:
            best[v] = t
    return best


def mean_band(folds: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    L = max(len(f["history"]) for f in folds)
    m = np.full((len(folds), L), np.nan)
    for i, f in enumerate(folds):
        v = [h[key] for h in f["history"]]
        m[i, : len(v)] = v
        # a run that early-stopped is frozen from then on, so it is held at its last logged
        # value until the longest run ends. Averaging only the runs still going would make the
        # mean jump whenever one stops; cutting the mean at the shortest run would hide the rest.
        m[i, len(v):] = v[-1]
    with np.errstate(all="ignore"):
        return np.nanmean(m, 0), np.nanstd(m, 0)


def plot_variants(curves: dict[str, dict], winner: str | None, out_png: Path) -> None:
    panels = [("loss", "Validation cross-entropy"), ("auroc", "Validation macro AUROC"), ("bacc", "Validation balanced accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (metric, title) in zip(axes, panels):
        for v, c in curves.items():
            col = VARIANT_COLORS.get(v, "black")
            lw = 2.8 if v == winner else 1.6
            mu, sd = mean_band(c["folds"], f"val_{metric}")
            x = np.arange(len(mu))
            lab = f"{v}, trial {c.get('trial')} (CV AUROC {c['cv_auroc']:.3f}){'  <- selected' if v == winner else ''}"
            ax.plot(x, mu, color=col, lw=lw, label=lab)
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.12)
            tr, _ = mean_band(c["folds"], f"train_{metric}")
            ax.plot(np.arange(len(tr)), tr, color=col, lw=1, ls=":", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        if metric != "loss":
            ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Attention-MIL fusion variants — solid: validation (mean ± sd over seeds), dotted: train  "
                 "| trained on train+val, test not used | seeds that stop early are held at their last value", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_search(trials: list[dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    best_so_far, inc = [], -np.inf
    for t in trials:
        fam = t["config"]["family"]
        y, e = t["mean_auroc"], t["std_auroc"]
        if y != y:
            continue
        col = FAMILY_COLORS.get(fam, "black")
        ax.errorbar(t["trial"], y, yerr=e, fmt="o", color=col, mfc=col if t["accepted"] else "white",
                    ms=8, capsize=3, lw=1)
        c = t["config"]
        variant = c.get("fusion") or c.get("combine") or (
            ("PCA" if c.get("pca_dim") else ("L2" if c.get("l2_normalize") else "no L2")) if fam == "early" else "")
        if variant:
            ax.annotate(variant, (t["trial"], y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7)
        if t["accepted"]:
            inc = y
        best_so_far.append((t["trial"], inc))
    xs, ys = zip(*best_so_far)
    ax.step(xs, ys, where="post", color="black", lw=1, alpha=0.6, label="incumbent (accepted best)")
    for fam, col in FAMILY_COLORS.items():
        ax.plot([], [], "o", color=col, label=fam)
    ax.plot([], [], "o", mfc="white", mec="black", label="rejected (hollow)")
    ax.set_xlabel("Agent trial")
    ax.set_ylabel("CV macro AUROC (train+val, ± sd over folds)")
    ax.set_title("Agent search: every fusion candidate scored by cross-validation (test not used)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--embeddings-dir", required=True, type=Path)
    ap.add_argument("--tile-emb-root", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path, help="the 03_run_agent --out-dir")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(args.run_dir)
    plot_search(trials, args.out_dir / "agent_search.png")
    print(f"saved {args.out_dir / 'agent_search.png'}")

    best = best_per_variant(trials)
    summary = json.loads((args.run_dir / "agent_summary.json").read_text())
    winner = summary["best_config"].get("fusion") if summary["best_config"].get("family") == "mil" else None
    if winner:
        # the selected variant is shown with the config the agent actually SELECTED (its last
        # accepted trial), not with that variant's highest-scoring trial, which may have been
        # rejected by the 1-SE rule (v3: trial 11 scored 0.900 but trial 1 was selected)
        accepted = [t for t in trials if t["accepted"] and t["config"].get("family") == "mil"
                    and t["config"].get("fusion") == winner]
        if accepted:
            best[winner] = accepted[-1]
    cache = args.run_dir / "curves_variants"
    cache.mkdir(exist_ok=True)
    ds = None
    curves = {}
    fused_hist = args.run_dir / "curves" / "fused_history.json"
    for v, t in sorted(best.items()):
        if v == winner and fused_hist.exists():
            # the selected config was already trained on train+val with the same 3 seeds for the
            # final evaluation (03_run_agent): reuse exactly those curves -- no retraining
            c = json.loads(fused_hist.read_text())
            c["cv_auroc"], c["trial"] = t["mean_auroc"], t["trial"]
            curves[v] = c
            print(f"[{v}] selected config (trial {t['trial']}): using the final fused model's curves {fused_hist}")
            continue
        f = cache / f"{v}.json"
        if f.exists():
            cached = json.loads(f.read_text())
            if cached.get("trial") == t["trial"]:
                curves[v] = cached
                print(f"[{v}] loaded {f}")
                continue
            print(f"[{v}] cached curves are for trial {cached.get('trial')}, need trial {t['trial']}: retraining")
        if ds is None:
            ds = load_fusion_dataset(args.split, args.embeddings_dir, tile_emb_root=args.tile_emb_root)
        print(f"[{v}] training best {v} config (trial {t['trial']}, CV AUROC {t['mean_auroc']:.4f}) "
              f"x{args.seeds} seeds on train+val -- test not used")
        c = train_curves_only(t["config"], ds, args.seeds, v)
        c["cv_auroc"] = t["mean_auroc"]
        c["trial"] = t["trial"]
        f.write_text(json.dumps(c, indent=1))
        curves[v] = c
    if curves:
        plot_variants(curves, winner, args.out_dir / "fusion_variants_curves.png")
        print(f"saved {args.out_dir / 'fusion_variants_curves.png'}")


if __name__ == "__main__":
    main()
