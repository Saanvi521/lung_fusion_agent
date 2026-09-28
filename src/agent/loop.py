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

import math
from dataclasses import dataclass

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


def _standard_error(result: EvalResult) -> float:
    n = max(result.n_folds, 1)
    return result.std_auroc / math.sqrt(n) if result.std_auroc == result.std_auroc else float("inf")  # NaN-safe


def _is_significant_improvement(candidate: EvalResult, incumbent: EvalResult) -> bool:
    """Accept only if the candidate beats the incumbent by more than one
    pooled standard error -- not just a higher point estimate. This is the
    brief's "validation-variance-aware acceptance rule."""
    if incumbent is None:
        return True
    pooled_se = math.sqrt(_standard_error(candidate) ** 2 + _standard_error(incumbent) ** 2)
    return (candidate.mean_auroc - incumbent.mean_auroc) > pooled_se


def run_agent(
    ds: FusionDataset,
    log_path,
    agent_config: AgentConfig = AgentConfig(),
) -> dict:
    proposer = build_proposer(use_llm=agent_config.use_llm, seed=agent_config.seed)
    log = DecisionLog(log_path)

    history: list[dict] = []
    incumbent_config = None
    incumbent_result: EvalResult | None = None
    trials_since_improvement = 0

    for trial_idx in range(1, agent_config.max_trials + 1):
        config, rationale = proposer.propose(history)

        try:
            result = evaluate_config(
                config, ds, n_splits=agent_config.n_cv_splits, n_repeats=agent_config.n_cv_repeats, seed=agent_config.seed
            )
        except Exception as e:  # noqa: BLE001 -- one bad config (e.g. torch missing for
            # "learned") should not end a 35-trial search; log it and move on.
            print(f"[trial {trial_idx}/{agent_config.max_trials}] {config.get('family')} FAILED: {e} -- skipping")
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
            f"[trial {trial_idx}/{agent_config.max_trials}] {config['family']} "
            f"auroc={result.mean_auroc:.4f}+-{result.std_auroc:.4f} "
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
