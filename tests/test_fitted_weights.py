"""Synthetic checks for fitted-weight late fusion (``late: fitted`` and ``abmil_late``).

Run with:  python -m pytest tests/ -q
No real data, GPU or torch needed.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.agent.evaluator import evaluate_config
from src.eval.metrics import eval_abmil_late
from src.fusion.data import FusionDataset, load_abmil_probs
from src.fusion.strategies import ABMILLateFusionModel, LateFusionModel, default_config
from src.fusion.weights import apply_weights, fit_simplex_weights, simplex_grid

MODELS = ["uni2h", "virchow2", "genbio_pathfm"]
CLASSES = ["A", "B", "C", "D", "E", "F", "G"]
K = len(CLASSES)


def _auroc(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(K)))


def _random_probs(rng, n):
    p = rng.random((n, K))
    return p / p.sum(1, keepdims=True)


def _informative_probs(rng, y, strength):
    logits = rng.normal(size=(len(y), K))
    logits[np.arange(len(y)), y] += strength
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _synthetic_ds(seed=0, n=175, informative="uni2h") -> FusionDataset:
    """Only ``informative`` carries label signal; the other two FMs (and their ABMIL
    probabilities) are pure noise."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(K), n // K + 1)[:n]
    rng.shuffle(y)
    emb, abmil = {}, {}
    for m in MODELS:
        X = rng.normal(size=(n, 16)).astype(np.float32)
        if m == informative:
            X[np.arange(n), y] += 2.5
        emb[m] = X
        abmil[m] = _informative_probs(rng, y, 2.0) if m == informative else _random_probs(rng, n)
    split = np.array(["train"] * 120 + ["val"] * 20 + ["test"] * (n - 140))
    return FusionDataset(patient_ids=[f"p{i}" for i in range(n)], embeddings=emb,
                         age=rng.normal(55, 10, n).astype(np.float32), sex=rng.integers(0, 2, n).astype(np.float32),
                         y=y, classes=CLASSES, split=split, abmil_probs=abmil)


# ---------------------------------------------------------------- weights.py

def test_simplex_grid_has_all_one_hot_rows_and_sums_to_one():
    g = simplex_grid(3, 0.1)
    assert g.shape == (66, 3)
    assert np.allclose(g.sum(1), 1)
    for i in range(3):
        assert any(np.allclose(r, np.eye(3)[i]) for r in g)
    assert simplex_grid(3, 0.2).shape == (21, 3)
    with pytest.raises(ValueError):
        simplex_grid(3, 0.3)


def test_noise_members_get_no_weight():
    rng = np.random.default_rng(1)
    y = rng.integers(0, K, 300)
    members = [_informative_probs(rng, y, 2.0), _random_probs(rng, 300), _random_probs(rng, 300)]
    w, _ = fit_simplex_weights(members, y, K)
    assert w[0] >= 0.8, w


def test_fitted_mix_never_below_best_member_on_fitting_data():
    rng = np.random.default_rng(2)
    for _ in range(5):
        y = rng.integers(0, K, 150)
        members = [_informative_probs(rng, y, s) for s in rng.uniform(0.3, 1.5, 3)]
        _, auc = fit_simplex_weights(members, y, K)
        assert auc >= max(_auroc(y, p) for p in members) - 1e-12


def test_apply_weights_is_a_probability_average():
    rng = np.random.default_rng(3)
    members = [_random_probs(rng, 10) for _ in range(3)]
    out = apply_weights(members, [0.5, 0.5, 0.0])
    assert np.allclose(out, (members[0] + members[1]) / 2)
    assert np.allclose(out.sum(1), 1)


# ---------------------------------------------------------------- late: fitted (mean-pooled)

def test_late_fitted_puts_weight_on_the_informative_fm():
    ds = _synthetic_ds(informative="virchow2")
    trainval = np.where(ds.split != "test")[0]
    model = LateFusionModel(config={**default_config("late"), "combine": "fitted", "C": 1.0})
    model.fit(ds, trainval)
    assert model.fitted_weights[MODELS.index("virchow2")] >= 0.7, model.fitted_weights
    p = model.predict_proba(ds, np.where(ds.split == "test")[0])
    assert np.allclose(p.sum(1), 1)


def test_evaluator_records_per_fold_weights_for_fitted():
    ds = _synthetic_ds()
    res = evaluate_config({**default_config("late"), "combine": "fitted", "C": 1.0}, ds, n_splits=5, n_repeats=1)
    assert res.fold_weights is not None and len(res.fold_weights) == 5
    assert all(abs(sum(w) - 1) < 1e-6 for w in res.fold_weights)
    # other combine options are unaffected
    res2 = evaluate_config({**default_config("late"), "combine": "mean", "C": 1.0}, ds, n_splits=5, n_repeats=1)
    assert res2.fold_weights is None


# ---------------------------------------------------------------- abmil_late

def test_abmil_late_refuses_test_patients_and_weights_informative_model():
    ds = _synthetic_ds(informative="genbio_pathfm")
    model = ABMILLateFusionModel(config=default_config("abmil_late"))
    with pytest.raises(ValueError):
        model.fit(ds, np.arange(len(ds.y)))  # includes test patients
    model.fit(ds, np.where(ds.split != "test")[0])
    assert model.fitted_weights[MODELS.index("genbio_pathfm")] >= 0.8, model.fitted_weights


def test_eval_abmil_late_applies_trainval_weights_to_the_meta_rows():
    """Option B: weights come from train+val only and are applied to the + metadata [ABMIL] rows."""
    ds = _synthetic_ds(informative="uni2h")
    test_idx = np.where(ds.split == "test")[0]
    rng = np.random.default_rng(4)
    rows = [{"row": f"{m} + metadata [ABMIL]", "_proba": _random_probs(rng, len(test_idx))} for m in MODELS]
    r1 = eval_abmil_late(default_config("abmil_late"), ds, MODELS, rows)
    w = np.array([r1["weights"][m] for m in MODELS])
    assert np.allclose(r1["_proba"], apply_weights([r["_proba"] for r in rows], w))

    # changing TEST labels must not change the fitted weights
    ds.y = ds.y.copy()
    ds.y[test_idx] = rng.integers(0, K, len(test_idx))
    r2 = eval_abmil_late(default_config("abmil_late"), ds, MODELS, rows)
    assert r1["weights"] == r2["weights"]

    with pytest.raises(ValueError):
        eval_abmil_late(default_config("abmil_late"), ds, MODELS, rows[:2])


# ---------------------------------------------------------------- loading 02d's npz files

def test_load_abmil_probs_aligns_by_patient_and_checks_classes(tmp_path):
    rng = np.random.default_rng(5)
    pids = [f"p{i}" for i in range(20)]
    for m in MODELS:
        order = rng.permutation(20)
        probs = np.zeros((20, K))
        probs[np.arange(20), np.arange(20) % K] = 1  # row i "belongs" to patient p{i}
        np.savez(tmp_path / f"{m}_abmil_probs.npz", pids=np.array(pids)[order], probs=probs[order],
                 classes=np.array(CLASSES), split=np.array(["trainval"] * 20), y=np.zeros(20))
    out = load_abmil_probs(tmp_path, MODELS, pids, CLASSES)
    for m in MODELS:
        assert np.array_equal(out[m].argmax(1), np.arange(20) % K)
    with pytest.raises(ValueError):
        load_abmil_probs(tmp_path, MODELS, pids, CLASSES[::-1])
    with pytest.raises(ValueError):
        load_abmil_probs(tmp_path, MODELS, pids + ["nobody"], CLASSES)
