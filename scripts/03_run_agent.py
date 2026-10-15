"""Stage 3: run the fusion agent, then a single final test-set evaluation.

Usage
-----
    python -m scripts.03_run_agent \
        --split artifacts/splits/split_seed42.csv \
        --embeddings-dir artifacts/embeddings \
        --out-dir artifacts/agent_run \
        --max-trials 35

    # with attention-MIL fusion on tile bags (needs 02b --save-tile-embeds; GPU recommended)
    python -m scripts.03_run_agent \
        --split artifacts/splits/split_seed42.csv \
        --embeddings-dir <drive>/embeddings \
        --tile-emb-root <drive>/tile_embeddings \
        --families early late learned mil \
        --out-dir artifacts/agent_run_mil
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.loop import AgentConfig, run_agent
from src.eval.metrics import build_comparison_table, train_curves_only
from src.fusion.data import MODEL_NAMES, load_fusion_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--embeddings-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/agent_run"))
    ap.add_argument("--max-trials", type=int, default=35)
    ap.add_argument("--no-improve-patience", type=int, default=12)
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile-emb-root", type=Path, default=None,
                    help="<out-dir>/tile_embeddings from 02b; enables the 'mil' family and ABMIL rows")
    ap.add_argument("--families", nargs="+", default=None, choices=["early", "late", "learned", "mil"],
                    help="families the agent may search (default: early late learned [+ mil if bags given])")
    ap.add_argument("--mil-cv-repeats", type=int, default=1)
    ap.add_argument("--final-seeds", type=int, default=3, help="independent trainings per MIL row on test")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Safe to re-run the exact same command after a crash: finished agent trials (trial_cache.jsonl)
    # and finished test rows (rows/) are reloaded, everything else continues where it stopped.
    ds = load_fusion_dataset(args.split, args.embeddings_dir, tile_emb_root=args.tile_emb_root)
    families = args.families or (["early", "late", "learned"] + (["mil"] if ds.bags is not None else []))
    if "mil" in families and ds.bags is None:
        ap.error("--families mil needs --tile-emb-root")
    print(f"[data] {len(ds.patient_ids)} patients loaded, classes={ds.classes}")

    agent_cfg = AgentConfig(
        max_trials=args.max_trials,
        no_improve_patience=args.no_improve_patience,
        use_llm=args.use_llm,
        seed=args.seed,
        families=tuple(families),
        mil_cv_repeats=args.mil_cv_repeats,
    )
    agent_out = run_agent(ds, args.out_dir / "decision_log.jsonl", agent_cfg)
    print(f"\n[agent] done after {agent_out['n_trials_run']} trials via {agent_out['proposer']} proposer")
    print(f"[agent] best config: {json.dumps(agent_out['best_config'], indent=2)}")

    # §5.3 comparison table, touching the test set exactly once
    rows = build_comparison_table(agent_out["best_config"], ds, MODEL_NAMES, final_seeds=args.final_seeds,
                                  cache_dir=args.out_dir / "rows")

    curves_dir = args.out_dir / "curves"
    for r in rows:
        if "_curves" in r and not r["row"].startswith("best single"):
            curves_dir.mkdir(exist_ok=True)
            name = "fused" if r["row"].startswith("fused") else \
                r["row"].split(" ")[0] + ("_meta" if "+ metadata" in r["row"] else "_alone")
            (curves_dir / f"{name}_history.json").write_text(json.dumps({**r["_curves"], "model": name}, indent=1))
    # Always show how attention-MIL trains: if the winner is NOT attention-MIL, train the
    # best-scoring MIL candidate from the search on train+val and save its curves (no test use).
    if "mil" in families and agent_out["best_config"].get("family") != "mil":
        trials = [json.loads(l) for l in (args.out_dir / "decision_log.jsonl").read_text().splitlines() if l.strip()]
        mil = [t for t in trials if t["config"].get("family") == "mil" and t["mean_auroc"] == t["mean_auroc"]]
        out = curves_dir / "best_mil_candidate_history.json"
        if mil and not out.exists():
            best = max(mil, key=lambda t: t["mean_auroc"])
            print(f"[curves] winner is not attention-MIL; training best MIL candidate (trial {best['trial']}, "
                  f"CV AUROC {best['mean_auroc']:.4f}) on train+val for its curves -- test not used")
            c = train_curves_only(best["config"], ds, args.final_seeds, "best_mil_candidate")
            c["cv_auroc"] = best["mean_auroc"]
            curves_dir.mkdir(exist_ok=True)
            out.write_text(json.dumps(c, indent=1))
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (args.out_dir / "comparison_table.json").write_text(json.dumps(rows, indent=2))
    if curves_dir.exists():
        print(f"Wrote training curves of the final models to {curves_dir} (plot with 02e)")
    (args.out_dir / "agent_summary.json").write_text(
        json.dumps({"best_config": agent_out["best_config"], "n_trials_run": agent_out["n_trials_run"],
                    "families": families, "seed": args.seed}, indent=2)
    )
    print(f"\nWrote {args.out_dir / 'comparison_table.json'}")
    print(f"Wrote {args.out_dir / 'decision_log.jsonl'}")


if __name__ == "__main__":
    main()
