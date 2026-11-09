"""Fitted late-fusion weights: pick one weight per member model on held-out predictions.

The ``late`` family originally combined its per-model predictions with equal weights, a
hand-typed weight vector, or a stacking classifier (~170 learned numbers). ``fitted`` sits
between those: it learns only the mixing weights (one per member, summing to 1) and picks
them by macro AUROC on OUT-OF-FOLD predictions, i.e. predictions for patients the member
models never trained on.

The grid includes every one-hot vector ("100% one member"), so on the patients the
weights are fitted on, the chosen mix scores at least as well as the best single member.
That guarantee does NOT extend to new patients -- the agent's CV score and the test set
are what show whether the mix generalises.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def simplex_grid(n_members: int, step: float) -> np.ndarray:
    """All weight vectors with ``n_members`` entries in multiples of ``step`` that sum to 1.

    ``simplex_grid(3, 0.1)`` has 66 rows, including (1, 0, 0), (0, 1, 0), (0, 0, 1).
    """
    k = round(1 / step)
    if not np.isclose(k * step, 1.0):
        raise ValueError(f"step must divide 1 exactly, got {step}")
    rows = [c for c in itertools.product(range(k + 1), repeat=n_members) if sum(c) == k]
    return np.array(rows, dtype=np.float64) / k


def _macro_auroc(y: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(n_classes))))


def fit_simplex_weights(
    member_probs: list[np.ndarray],
    y: np.ndarray,
    n_classes: int,
    step: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Choose mixing weights for ``member_probs`` (each ``(n_patients, n_classes)``,
    out-of-fold) by macro AUROC on ``y``.

    Ties (common: AUROC is rank-based and many mixes rank patients identically) are broken by
    lower log-loss, then by the earliest grid row, so the result is deterministic.

    Returns ``(weights, auroc_of_chosen_mix)``.
    """
    stack = np.stack(member_probs)  # (n_members, n_patients, n_classes)
    grid = simplex_grid(len(member_probs), step)
    best_key, best_w, best_auc = None, None, float("nan")
    for w in grid:
        mix = np.tensordot(w, stack, axes=1)
        auc = _macro_auroc(y, mix, n_classes)
        clipped = np.clip(mix, 1e-7, 1)
        ll = log_loss(y, clipped / clipped.sum(1, keepdims=True), labels=list(range(n_classes)))
        key = (round(auc, 10), -ll)
        if best_key is None or key > best_key:
            best_key, best_w, best_auc = key, w, auc
    return best_w, best_auc


def apply_weights(member_probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted average of the members' class probabilities (rows still sum to 1)."""
    return np.tensordot(np.asarray(weights, dtype=np.float64), np.stack(member_probs), axes=1)
