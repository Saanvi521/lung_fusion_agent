"""Stage 2e: plot ABMIL training curves from the history 02d saved (no re-training).

For each FM: a 1x3 figure — cross-entropy loss, macro AUROC, balanced accuracy — with
train (blue) and validation / early-stopping set (orange). Thin lines are the K
cross-fit folds; the thick line is the mean over the folds still training at that
epoch; dots mark each fold's restored (best) epoch. Train and validation are computed
the same way (eval mode, full bags, unweighted cross-entropy), so they are comparable.

Usage:
    python -m scripts.02e_plot_training_curve \
        --history artifacts/abmil_cf/uni2h_history.json artifacts/abmil_cf/virchow2_history.json \
        --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [("loss", "Cross-entropy (unweighted)"), ("auroc", "Macro AUROC (OvR)"), ("bacc", "Balanced accuracy")]
COLORS = {"train": "#2a6fdb", "val": "#e8743b"}


def _mean_curve(folds: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    L = max(len(f["history"]) for f in folds)
    mat = np.full((len(folds), L), np.nan)
    for i, f in enumerate(folds):
        v = [h[key] for h in f["history"]]
        mat[i, : len(v)] = v
        # a run that early-stopped is frozen from then on, so it is held at its last logged
        # value until the longest run ends. Averaging only the runs still going would make the
        # mean jump whenever one stops; cutting the mean at the shortest run would hide the rest.
        mat[i, len(v):] = v[-1]
    with np.errstate(all="ignore"):
        return np.nanmean(mat, 0), np.nanstd(mat, 0)


def plot_history(hist: dict, out_png: Path) -> dict:
    folds, fm = hist["folds"], hist["model"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    summary = {}
    for ax, (metric, label) in zip(axes, PANELS):
        for split in ("train", "val"):
            key = f"{split}_{metric}"
            last_ep = max(len(f["history"]) for f in folds) - 1
            for f in folds:
                ep = [h["epoch"] for h in f["history"]]
                ys = [h[key] for h in f["history"]]
                ax.plot(ep, ys, color=COLORS[split], alpha=0.22, lw=1)
                if ep[-1] < last_ep:  # stopped early: dotted = held at its last value (what the mean uses)
                    ax.plot([ep[-1], last_ep], [ys[-1], ys[-1]], color=COLORS[split], alpha=0.22, lw=1, ls=":")
                b = f["best_epoch"]
                ax.scatter([b], [f["history"][b][key]], color=COLORS[split], s=14, zorder=3)
            mu, sd = _mean_curve(folds, key)
            x = np.arange(len(mu))
            name = "train" if split == "train" else "validation (early-stop set)"
            ax.plot(x, mu, color=COLORS[split], lw=2.4, label=f"{name}, mean of {len(folds)} runs*")
            ax.fill_between(x, mu - sd, mu + sd, color=COLORS[split], alpha=0.10)
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        if metric != "loss":
            ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8, loc="upper right")
    best = [f["best_epoch"] for f in folds]
    held = [f["heldout"] for f in folds if "heldout" in f]
    summary = {"best_epochs": best}
    if held:
        summary.update({
            "heldout_auroc_mean": float(np.nanmean([h["auroc"] for h in held])),
            "heldout_auroc_sd": float(np.nanstd([h["auroc"] for h in held])),
            "heldout_bacc_mean": float(np.mean([h["bacc"] for h in held])),
            "heldout_bacc_sd": float(np.std([h["bacc"] for h in held])),
            "heldout_loss_mean": float(np.mean([h["loss"] for h in held])),
        })
        tail = (f"held-out AUROC {summary['heldout_auroc_mean']:.3f} ± {summary['heldout_auroc_sd']:.3f}, "
                f"bAcc {summary['heldout_bacc_mean']:.3f} ± {summary['heldout_bacc_sd']:.3f}")
    else:
        tail = hist.get("kind", "")
    hp = hist["hparams"]
    fig.suptitle(
        f"ABMIL — {fm}  |  lr {hp['lr']:.0e}, wd {hp['weight_decay']:.0e}, proj {hp['proj_dim']}, "
        f"dropout {hp['dropout_in']}/{hp['dropout']}, accum {hp['accum_steps']}  |  {tail}",
        fontsize=10,
    )
    fig.text(0.01, 0.01, "* thin lines = individual runs; a run that early-stopped is held at its last value (dotted) "
             "so the mean covers every run until the longest one ends", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    for h in args.history:
        hist = json.loads(h.read_text())
        out = args.out_dir / f"{hist['model']}_training_curve.png"
        s = plot_history(hist, out)
        (args.out_dir / f"{hist['model']}_training_curve_summary.json").write_text(json.dumps(s, indent=2))
        print(f"[{hist['model']}] saved {out}  {json.dumps(s)}")


if __name__ == "__main__":
    main()
