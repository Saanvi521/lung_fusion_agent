"""Stage 3: run the fusion agent, then a single final test-set evaluation.

Usage
-----
    python -m scripts.03_run_agent \
        --split artifacts/splits/split_seed42.csv \
        --embeddings-dir artifacts/embeddings \
        --out-dir artifacts/agent_run \
        --max-trials 35
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent.loop import AgentConfig, run_agent
from src.eval.metrics import build_comparison_table
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
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_fusion_dataset(args.split, args.embeddings_dir)
    print(f"[data] {len(ds.patient_ids)} patients loaded, classes={ds.classes}")

    agent_cfg = AgentConfig(
        max_trials=args.max_trials,
        no_improve_patience=args.no_improve_patience,
        use_llm=args.use_llm,
        seed=args.seed,
    )
    agent_out = run_agent(ds, args.out_dir / "decision_log.jsonl", agent_cfg)
    print(f"\n[agent] done after {agent_out['n_trials_run']} trials via {agent_out['proposer']} proposer")
    print(f"[agent] best config: {json.dumps(agent_out['best_config'], indent=2)}")

    # §5.3 comparison table, touching the test set exactly once
    rows = build_comparison_table(agent_out["best_config"], ds, MODEL_NAMES)

    (args.out_dir / "comparison_table.json").write_text(json.dumps(rows, indent=2))
    (args.out_dir / "agent_summary.json").write_text(
        json.dumps({"best_config": agent_out["best_config"], "n_trials_run": agent_out["n_trials_run"]}, indent=2)
    )
    print(f"\nWrote {args.out_dir / 'comparison_table.json'}")
    print(f"Wrote {args.out_dir / 'decision_log.jsonl'}")


if __name__ == "__main__":
    main()
