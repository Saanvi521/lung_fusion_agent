"""Append-only JSONL decision log.

One JSON object per line, one line per trial. JSONL (not a single JSON
array) is deliberate: it means the log is valid and readable even if the
run is interrupted mid-search, and it's trivially appendable without
re-reading/re-writing the whole file each trial -- matters given Colab
disconnects have already been a recurring problem in this project.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class DecisionLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_trial(
        self,
        trial_idx: int,
        proposer_name: str,
        config: dict[str, Any],
        rationale: str,
        mean_auroc: float,
        std_auroc: float,
        mean_bal_acc: float,
        accepted: bool,
        accept_reason: str,
        incumbent_score: float,
    ) -> None:
        record = {
            "trial": trial_idx,
            "timestamp": time.time(),
            "proposer": proposer_name,
            "config": config,
            "rationale": rationale,
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "mean_bal_acc": mean_bal_acc,
            "accepted": accepted,
            "accept_reason": accept_reason,
            "incumbent_score_after": incumbent_score,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]
