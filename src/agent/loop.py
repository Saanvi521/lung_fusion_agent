"""The agent loop.

propose -> evaluate -> log -> accept/reject -> propose again, until either
the trial budget is spent or the no-improvement patience is hit. Both are
hard, explicit numbers (not vibes) per the brief's requirement.

The guardrail that matters most is ``_is_significant_improvement``: a new
config only replaces the incumbent ("best so far") if its CV mean beats the
incumbent by more than one pooled standard error. Without this, with 21
validation patients (or even the ~167 pooled train+val patients used here),
a long enough search will eventually find a config that scores higher by
pure luck -- accepting only "beats noise" improvements is what keeps a
30-40 trial budget from just overfitting the CV folds.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.agent.decision_log import DecisionLog
from src.agent.evaluator import EvalResult, evaluate_config
from src.agent.proposer import build_proposer
from src.fusion.data import FusionDataset


@dataclass
class AgentConfig:
    max_trials: int = 35
    no_improve_patience: int = 12
    use_llm: bool = False
    seed: int = 42
    n_cv_splits: int = 5
    n_cv_repeats: int = 3
    families: tuple = ("early", "late", "learned")
    mil_cv_repeats: int = 1  # MIL trials are ~100x costlier; the first repeat uses the SAME 5 folds,
                             # so paired comparisons against other families stay valid (on those folds)


def _standard_error(result: EvalResult) -> float:
    n = max(result.n_folds, 1)
    return result.std_auroc / math.sqrt(n) if result.std_auroc == result.std_auroc else float("inf")  # NaN-safe


def _is_significant_improvement(candidate: EvalResult, incumbent: EvalResult) -> bool:
    """Accept only if the candidate beats the incumbent by more than one
    standard error -- not just a higher point estimate. This is the
    brief's "validation-variance-aware acceptance rule."

    Fix: every trial's CV folds come from ``RepeatedStratifiedKFold`` with
    the SAME ``seed`` (see evaluator.py), so candidate and incumbent are
    scored on the identical 15 folds every time -- this is a PAIRED
    comparison (same patients re-scored by a different model), not two
    independent experiments. The old version compared them with
    ``sqrt(se_candidate**2 + se_incumbent**2)``, the formula for two
    independent samples, which overstates the true uncertainty of the
    difference whenever the two configs' per-fold errors are correlated
    (likely here, since related fusion configs share per-model
    classifiers) -- inflating the SE and rejecting real improvements as
    noise. Comparing the per-fold differences directly (a paired test) is
    the statistically correct way to compare two models evaluated on
    identical folds, and gives a tighter, more honest SE.
    """
    if incumbent is None:
        return True
    n = min(len(candidate.fold_aurocs), len(incumbent.fold_aurocs))
    if n < 2:
        # Folds didn't line up 1:1 (one config dropped more folds than the
        # other, e.g. a rare class missing from a fold) -- fall back to the
        # conservative independent-SE test rather than guess.
        pooled_se = math.sqrt(_standard_error(candidate) ** 2 + _standard_error(incumbent) ** 2)
        return (candidate.mean_auroc - incumbent.mean_auroc) > pooled_se
    diffs = np.array(candidate.fold_aurocs[:n]) - np.array(incumbent.fold_aurocs[:n])
    mean_diff = float(diffs.mean())
    se_diff = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("inf")
    return mean_diff > se_diff


def describe(config: dict) -> str:
    """One-line human-readable name of a candidate, e.g. 'mil:tile_concat lr=1e-04 proj=128 ...'."""
    f = config.get("family")
    if f == "mil":
        return (f"mil:{config.get('fusion', 'tile_concat')} lr={config.get('lr', 0):.0e} proj={config.get('proj_dim')} "
                f"drop={config.get('dropout')} stream_drop={config.get('stream_dropout')} wd={config.get('weight_decay', 0):.0e} "
                f"ls={config.get('label_smoothing', 0)} tiles={config.get('max_train_tiles', 512)}")
    if f == "late":
        w = f" w={config.get('weights')}" if config.get("combine") == "weighted" else ""
        return f"late:{config.get('combine')}{w} C={config.get('C')}"
    if f == "early":
        return (f"early:{'L2' if config.get('l2_normalize') else 'noL2'}"
                f"{' PCA=' + str(config['pca_dim']) if config.get('pca_dim') else ''} C={config.get('C')}")
    if f == "learned":
        return (f"learned hidden={config.get('hidden_dim')} drop={config.get('dropout')} "
                f"wd={config.get('weight_decay')} epochs={config.get('epochs')}")
    return str(f)


class _TrialCache:
    """Every finished trial's CV result is appended to ``trial_cache.jsonl`` next to the decision
    log. If the run dies (Colab disconnect), rerunning the same command replays the search: the
    deterministic proposer proposes the same configs in the same order, finished ones are read
    from the cache instantly, and the search continues where it stopped."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.store: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.store[rec["key"]] = rec["result"]

    @staticmethod
    def key(config: dict, reps: int) -> str:
        return json.dumps({"config": config, "reps": reps}, sort_keys=True)

    def get(self, key: str) -> EvalResult | None:
        r = self.store.get(key)
        return EvalResult(**r) if r else None

    def put(self, key: str, result: EvalResult) -> None:
        self.store[key] = asdict(result)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "result": asdict(result)}) + "\n")


def run_agent(
    ds: FusionDataset,
    log_path,
    agent_config: AgentConfig = AgentConfig(),
) -> dict:
    proposer = build_proposer(use_llm=agent_config.use_llm, seed=agent_config.seed,
                              families=list(agent_config.families))
    log_path = Path(log_path)
    cache = _TrialCache(log_path.parent / "trial_cache.jsonl")
    if cache.store:
        print(f"[agent] resuming: {len(cache.store)} finished trials found in {cache.path}")
        log_path.unlink(missing_ok=True)  # the replay rewrites the decision log identically
    log = DecisionLog(log_path)

    history: list[dict] = []
    incumbent_config = None
    incumbent_result: EvalResult | None = None
    trials_since_improvement = 0

    for trial_idx in range(1, agent_config.max_trials + 1):
        config, rationale = proposer.propose(history)

        try:
            reps = agent_config.mil_cv_repeats if config.get("family") == "mil" else agent_config.n_cv_repeats
            key = cache.key(config, reps)
            result = cache.get(key)
            if result is None:
                result = evaluate_config(
                    config, ds, n_splits=agent_config.n_cv_splits, n_repeats=reps, seed=agent_config.seed
                )
                cache.put(key, result)
        except Exception as e:  # noqa: BLE001 -- one bad config (e.g. torch missing for
            # "learned") should not end a 35-trial search; log it and move on.
            print(f"[trial {trial_idx}/{agent_config.max_trials}] {describe(config)} FAILED: {e} -- skipping")
            log.log_trial(
                trial_idx=trial_idx,
                proposer_name=proposer.name,
                config=config,
                rationale=rationale,
                mean_auroc=float("nan"),
                std_auroc=float("nan"),
                mean_bal_acc=float("nan"),
                accepted=False,
                accept_reason=f"trial errored: {e}",
                incumbent_score=incumbent_result.mean_auroc if incumbent_result else float("nan"),
            )
            continue

        accepted = _is_significant_improvement(result, incumbent_result)
        if accepted:
            accept_reason = (
                "first trial, becomes incumbent"
                if incumbent_result is None
                else f"beat incumbent by more than 1 SE ({result.mean_auroc:.4f} vs {incumbent_result.mean_auroc:.4f})"
            )
            incumbent_config, incumbent_result = config, result
            trials_since_improvement = 0
        else:
            accept_reason = (
                f"did not beat incumbent ({incumbent_result.mean_auroc:.4f}) by more than 1 SE "
                f"-- treated as noise, incumbent unchanged"
            )
            # the no-improvement patience only starts once every seed config (one per family /
            # variant) has been tried -- otherwise a family listed late in the seed batch (e.g. mil)
            # can be cut before it is ever evaluated
            n_seed = len(getattr(getattr(proposer, "fallback", proposer), "seed_configs", []))
            if trial_idx > n_seed:
                trials_since_improvement += 1

        history.append({"config": config, "score": result.mean_auroc, "accepted": accepted})

        log.log_trial(
            trial_idx=trial_idx,
            proposer_name=proposer.name,
            config=config,
            rationale=rationale,
            mean_auroc=result.mean_auroc,
            std_auroc=result.std_auroc,
            mean_bal_acc=result.mean_bal_acc,
            accepted=accepted,
            accept_reason=accept_reason,
            incumbent_score=incumbent_result.mean_auroc,
        )

        print(
            f"[trial {trial_idx:2d}/{agent_config.max_trials}] {describe(config):<70} "
            f"CV AUROC={result.mean_auroc:.4f}+-{result.std_auroc:.4f} bAcc={result.mean_bal_acc:.3f} "
            f"{'ACCEPTED' if accepted else 'rejected'} (incumbent={incumbent_result.mean_auroc:.4f})"
        )

        if trials_since_improvement >= agent_config.no_improve_patience:
            print(f"[agent] stopping: {agent_config.no_improve_patience} trials with no improvement")
            break

    return {
        "best_config": incumbent_config,
        "best_result": incumbent_result,
        "n_trials_run": len(history),
        "proposer": proposer.name,
    }