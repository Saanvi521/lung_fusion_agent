"""Fusion strategy registry.

A "config" is a plain JSON-serializable dict describing one way to combine
the three model embeddings (+ age/sex) into a prediction. This is exactly
what the agent searches over: it proposes configs, and this file is what
actually fits/evaluates one.

Three families, matching the brief's minimum viable scope:

- early:   concatenate (optionally L2-normalized / PCA-reduced) embeddings
           + metadata, then one classifier on the combined vector.
- late:    train a separate classifier per model (+ metadata on each), then
           combine their predicted probabilities (mean / weighted / a
           small stacking meta-classifier / ``fitted`` weights chosen on
           out-of-fold predictions, see src/fusion/weights.py).
- abmil_late: ``fitted`` weights over the three single-FM ABMIL networks'
           predictions (cross-fitted by 02d). Needs --abmil-probs-dir.
- learned: a small gated-fusion network that learns, per patient, how much
           to trust each model's stream before classifying.
- mil:     attention-MIL fusion on the TILE bags of all FMs (src/mil): the FMs are
           fused inside one ABMIL (tile-level concat, gated attention over FM
           streams, or jointly-trained late fusion), with age/sex fed to the head
           and per-FM learning rates from conf/tuned. Only available when the
           dataset was loaded with tile bags (03_run_agent --tile-emb-root).

Regularisation note: ``C`` may be a number or ``"auto"``. With 1.5k-4.6k
standardised features and ~130 training patients, logistic regression at C=1 is
essentially unregularised (the classes are linearly separable), which gives
over-confident probabilities -- bad for late fusion and for balanced accuracy.
``"auto"`` picks C per model by inner CV log-loss (LogisticRegressionCV), i.e.
each FM gets its own regularisation strength.

Every family fits ONLY on the given training indices -- callers (the CV
evaluator) are responsible for passing train/val splits so nothing here
can leak.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

from src.fusion.data import FusionDataset
from src.fusion.weights import apply_weights, fit_simplex_weights

# sklearn >=1.8 emits FutureWarnings for LogisticRegressionCV's default arguments on every fit
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

MODEL_NAMES = ["uni2h", "virchow2", "genbio_pathfm"]
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


def make_logreg(C) -> LogisticRegression:
    """Logistic regression with either a fixed C or C chosen by inner 3-fold CV log-loss."""
    if C == "auto":
        return LogisticRegressionCV(
            Cs=C_GRID, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="neg_log_loss",
            class_weight="balanced", max_iter=3000, n_jobs=-1,
        )
    return LogisticRegression(C=float(C), max_iter=3000, class_weight="balanced")


# ---------------------------------------------------------------------------
# Config schema (plain dicts, validated loosely here -- the agent's proposer
# is responsible for only emitting configs shaped like this)
# ---------------------------------------------------------------------------

def default_config(family: str) -> dict[str, Any]:
    if family == "early":
        return {
            "family": "early",
            "l2_normalize": True,
            "pca_dim": None,  # e.g. 128, or None to skip
            "C": "auto",
        }
    if family == "late":
        return {
            "family": "late",
            "combine": "mean",  # "mean" | "logmean" | "weighted" | "stacking" | "fitted"
            "weights": None,  # only used if combine == "weighted"
            "weight_step": 0.1,  # only used if combine == "fitted": grid spacing of the weights
            "C": "auto",
        }
    if family == "abmil_late":
        return {
            "family": "abmil_late",
            "weight_step": 0.1,
        }
    if family == "learned":
        return {
            "family": "learned",
            "hidden_dim": 64,
            "dropout": 0.5,
            "weight_decay": 1e-3,
            "epochs": 30,
            "lr": 1e-3,
            "n_seeds": 3,  # average logits over this many seeds
        }
    if family == "mil":
        return {
            "family": "mil",
            "models": list(MODEL_NAMES),
            "fusion": "tile_concat",   # "tile_concat" | "gated" | "late"
            # v2 defaults: the first run's fused model over-fitted after ~4 epochs (validation loss
            # rose while train loss -> 0). Three FMs x 256-d projections is ~2.8M parameters, so the
            # fused default uses a 128-d projection per FM, the learning rate the per-FM search chose
            # (1e-4, not 2e-4), and modality dropout. Chosen from TRAINING curves only (no test use).
            "proj_dim": 128,
            "dropout": 0.25,
            "stream_dropout": 0.2,
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.1,     # curbs the over-confidence that makes validation loss climb
            "max_train_tiles": 256,    # stronger bag augmentation: a random 256 of ~600 tiles per step
            "use_meta": True,
            "per_fm_lr": True,         # each FM's layers use its tuned lr (conf/tuned/abmil_<fm>.json)
            "n_seeds": 1,
        }
    raise ValueError(f"unknown family: {family}")


def _metadata_matrix(ds: FusionDataset, idx: np.ndarray, age_scaler: StandardScaler | None = None, fit: bool = False):
    age = ds.age[idx].reshape(-1, 1)
    if fit:
        age_scaler = StandardScaler().fit(age)
    age_std = age_scaler.transform(age)
    sex = ds.sex[idx].reshape(-1, 1)
    return np.concatenate([age_std, sex], axis=1), age_scaler


# ---------------------------------------------------------------------------
# Early fusion
# ---------------------------------------------------------------------------

@dataclass
class EarlyFusionModel:
    config: dict[str, Any]
    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    pcas: dict[str, PCA] = field(default_factory=dict)
    age_scaler: StandardScaler | None = None
    clf: LogisticRegression | None = None

    def _build_features(self, ds: FusionDataset, idx: np.ndarray, fit: bool) -> np.ndarray:
        blocks = []
        for m in MODEL_NAMES:
            X = ds.embeddings[m][idx]
            if fit:
                self.scalers[m] = StandardScaler().fit(X)
            X = self.scalers[m].transform(X)
            if self.config.get("l2_normalize"):
                X = normalize(X)
            pca_dim = self.config.get("pca_dim")
            if pca_dim:
                if fit:
                    self.pcas[m] = PCA(n_components=min(pca_dim, X.shape[0], X.shape[1]), random_state=42).fit(X)
                X = self.pcas[m].transform(X)
            blocks.append(X)
        meta, self.age_scaler = _metadata_matrix(ds, idx, self.age_scaler, fit=fit)
        blocks.append(meta)
        return np.concatenate(blocks, axis=1)

    def fit(self, ds: FusionDataset, train_idx: np.ndarray) -> None:
        X = self._build_features(ds, train_idx, fit=True)
        y = ds.y[train_idx]
        self.clf = make_logreg(self.config.get("C", "auto")).fit(X, y)

    def predict_proba(self, ds: FusionDataset, idx: np.ndarray) -> np.ndarray:
        X = self._build_features(ds, idx, fit=False)
        return self.clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Late fusion
# ---------------------------------------------------------------------------

@dataclass
class LateFusionModel:
    config: dict[str, Any]
    per_model_clf: dict[str, LogisticRegression] = field(default_factory=dict)
    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    age_scaler: StandardScaler | None = None
    stacker: LogisticRegression | None = None
    fitted_weights: np.ndarray | None = None  # set by combine == "fitted"

    def _fit_one_model(self, ds: FusionDataset, m: str, train_idx: np.ndarray) -> None:
        X = ds.embeddings[m][train_idx]
        self.scalers[m] = StandardScaler().fit(X)
        X = self.scalers[m].transform(X)
        meta, self.age_scaler = _metadata_matrix(ds, train_idx, self.age_scaler, fit=(self.age_scaler is None))
        X = np.concatenate([X, meta], axis=1)
        y = ds.y[train_idx]
        self.per_model_clf[m] = make_logreg(self.config.get("C", "auto")).fit(X, y)

    def _proba_one_model(self, ds: FusionDataset, m: str, idx: np.ndarray) -> np.ndarray:
        X = ds.embeddings[m][idx]
        X = self.scalers[m].transform(X)
        meta, _ = _metadata_matrix(ds, idx, self.age_scaler, fit=False)
        X = np.concatenate([X, meta], axis=1)
        return self.per_model_clf[m].predict_proba(X)

    def _oof_probs(self, ds: FusionDataset, train_idx: np.ndarray) -> dict[str, np.ndarray]:
        """Out-of-fold class probabilities of each per-model classifier on ``train_idx``.

        Fix (stacking): the combiner's training inputs must come from OUT-OF-FOLD predictions,
        not in-sample ones. The per-model classifiers are fit on ALL of train_idx, so asking
        them to predict on that same train_idx gives artificially confident/accurate
        probabilities -- nothing like what they produce on unseen patients at test time. A
        combiner trained on those learns a rule for inputs it will never see again (why
        "stacking" scored worst in the first real run). Inner 5-fold cross-validated
        predictions match train-time inputs to test-time conditions. Used by both
        ``stacking`` and ``fitted``.
        """
        y_train = ds.y[train_idx]
        n_classes = len(ds.classes)
        oof = {m: np.zeros((len(train_idx), n_classes)) for m in MODEL_NAMES}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_tr_local, fold_val_local in skf.split(train_idx, y_train):
            fold_tr_idx = train_idx[fold_tr_local]
            fold_val_idx = train_idx[fold_val_local]
            for m in MODEL_NAMES:
                fold_model = LateFusionModel(config=self.config)
                fold_model._fit_one_model(ds, m, fold_tr_idx)
                oof[m][fold_val_local] = fold_model._proba_one_model(ds, m, fold_val_idx)
        return oof

    def fit(self, ds: FusionDataset, train_idx: np.ndarray) -> None:
        for m in MODEL_NAMES:
            self._fit_one_model(ds, m, train_idx)

        combine = self.config.get("combine")
        if combine in ("stacking", "fitted"):
            oof = self._oof_probs(ds, train_idx)
            y_train = ds.y[train_idx]
        if combine == "stacking":
            meta, _ = _metadata_matrix(ds, train_idx, self.age_scaler, fit=False)
            X_stack = np.concatenate([oof[m] for m in MODEL_NAMES] + [meta], axis=1)
            self.stacker = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_stack, y_train)
        if combine == "fitted":
            # one weight per FM, chosen on the out-of-fold predictions only (age/sex are already
            # inside every member classifier, so the mix uses them whatever the weights)
            self.fitted_weights, _ = fit_simplex_weights(
                [oof[m] for m in MODEL_NAMES], y_train, len(ds.classes), step=float(self.config.get("weight_step", 0.1))
            )

    def predict_proba(self, ds: FusionDataset, idx: np.ndarray) -> np.ndarray:
        probs = [self._proba_one_model(ds, m, idx) for m in MODEL_NAMES]
        combine = self.config.get("combine", "mean")

        if combine == "mean":
            return np.mean(probs, axis=0)
        if combine == "logmean":
            # geometric mean of the probabilities (= averaging log-probabilities, i.e. logits up to a
            # constant; a "product of experts"). Favours classes that ALL models find plausible: a class
            # any one model gives ~0 is effectively vetoed. Different behaviour from the arithmetic mean,
            # not uniformly better -- the agent decides between them by CV.
            lp = np.mean([np.log(np.clip(p, 1e-7, 1)) for p in probs], axis=0)
            e = np.exp(lp - lp.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        if combine == "weighted":
            w = self.config.get("weights") or [1 / 3, 1 / 3, 1 / 3]
            w = np.array(w) / sum(w)
            return sum(p * wi for p, wi in zip(probs, w))
        if combine == "stacking":
            meta, _ = _metadata_matrix(ds, idx, self.age_scaler, fit=False)
            X_stack = np.concatenate(probs + [meta], axis=1)
            return self.stacker.predict_proba(X_stack)
        if combine == "fitted":
            return apply_weights(probs, self.fitted_weights)
        raise ValueError(f"unknown combine: {combine}")


# ---------------------------------------------------------------------------
# ABMIL late fusion with fitted weights
# ---------------------------------------------------------------------------

@dataclass
class ABMILLateFusionModel:
    """Weighted average of the three single-FM ABMIL networks' predictions, with the weights
    chosen by ``fit_simplex_weights``.

    Nothing is trained here: the member predictions are the cross-fitted ABMIL probabilities
    from 02d (``ds.abmil_probs``), which are out-of-fold for every train+val patient. So the
    weights for a CV fold are fitted on its training patients' out-of-fold predictions and
    scored on its held-out patients' out-of-fold predictions.

    Leakage note: the 02d folds differ from the agent's CV folds, so a training patient's
    prediction can come from a network that saw some of this fold's held-out patients. Only
    ~2 free numbers are fitted on those predictions, so the effect on the CV score is small,
    and the test set is unaffected.

    Final test model (option B, see src/eval/metrics.py): the weights fitted on all train+val
    are applied to the "+ metadata [ABMIL]" rows' test predictions, i.e. the same networks the
    single-model baselines in the table use. ``predict_proba`` on test patients here returns
    the 02d-network version instead (used only as a context row).
    """

    config: dict[str, Any]
    fitted_weights: np.ndarray | None = None

    def _members(self, ds: FusionDataset, idx: np.ndarray) -> list[np.ndarray]:
        if ds.abmil_probs is None:
            raise ValueError("abmil_late needs 02d's ABMIL probabilities: pass --abmil-probs-dir")
        return [ds.abmil_probs[m][idx] for m in MODEL_NAMES]

    def fit(self, ds: FusionDataset, train_idx: np.ndarray) -> None:
        if np.any(ds.split[train_idx] == "test"):
            raise ValueError("abmil_late must only be fitted on train/val patients")
        self.fitted_weights, _ = fit_simplex_weights(
            self._members(ds, train_idx), ds.y[train_idx], len(ds.classes),
            step=float(self.config.get("weight_step", 0.1)),
        )

    def predict_proba(self, ds: FusionDataset, idx: np.ndarray) -> np.ndarray:
        return apply_weights(self._members(ds, idx), self.fitted_weights)


# ---------------------------------------------------------------------------
# Learned (gated) fusion
#
# torch is imported lazily, only when this family is actually built --
# early and late fusion (and the whole agent loop) work fine without torch
# installed at all. If torch is missing, LearnedFusionModel raises a clear
# error at construction time instead of the whole module failing to import.
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class _GatedFusionNet(nn.Module if _TORCH_AVAILABLE else object):
    """Projects each model's embedding to a shared width, learns a softmax
    gate over the three streams per-patient, then classifies the gated sum.
    Deliberately tiny: at ~146 training patients, a bigger net overfits."""

    def __init__(self, in_dims: dict[str, int], hidden_dim: int, n_classes: int, meta_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(d, hidden_dim) for m, d in in_dims.items()})
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * len(in_dims), hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, len(in_dims))
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + meta_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes)
        )
        self.model_names = list(in_dims.keys())

    def forward(self, x: dict[str, torch.Tensor], meta: torch.Tensor) -> torch.Tensor:
        projected = [torch.tanh(self.proj[m](x[m])) for m in self.model_names]
        cat = torch.cat(projected, dim=-1)
        gate_logits = self.gate(cat)
        gate_w = torch.softmax(gate_logits, dim=-1)  # (B, n_models)
        stacked = torch.stack(projected, dim=1)  # (B, n_models, hidden)
        fused = (stacked * gate_w.unsqueeze(-1)).sum(dim=1)  # (B, hidden)
        fused = self.drop(fused)
        return self.head(torch.cat([fused, meta], dim=-1))


@dataclass
class LearnedFusionModel:
    config: dict[str, Any]
    scalers: dict[str, StandardScaler] = field(default_factory=dict)
    age_scaler: StandardScaler | None = None
    nets: list = field(default_factory=list)  # one per seed, averaged at inference

    def __post_init__(self):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "learned fusion requires torch, which is not installed. "
                "Early and late fusion do not need torch and are unaffected."
            )

    def _features(self, ds: FusionDataset, idx: np.ndarray, fit: bool):
        x = {}
        for m in MODEL_NAMES:
            X = ds.embeddings[m][idx]
            if fit:
                self.scalers[m] = StandardScaler().fit(X)
            x[m] = torch.tensor(self.scalers[m].transform(X), dtype=torch.float32)
        meta, self.age_scaler = _metadata_matrix(ds, idx, self.age_scaler, fit=fit)
        return x, torch.tensor(meta, dtype=torch.float32)

    def fit(self, ds: FusionDataset, train_idx: np.ndarray) -> None:
        x, meta = self._features(ds, train_idx, fit=True)
        y = torch.tensor(ds.y[train_idx], dtype=torch.long)
        in_dims = {m: ds.embeddings[m].shape[1] for m in MODEL_NAMES}
        n_classes = len(ds.classes)

        self.nets = []
        for seed in range(self.config.get("n_seeds", 3)):
            torch.manual_seed(seed)
            net = _GatedFusionNet(
                in_dims, self.config["hidden_dim"], n_classes, meta.shape[1], self.config["dropout"]
            )
            opt = torch.optim.AdamW(net.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
            loss_fn = nn.CrossEntropyLoss()
            net.train()
            for _ in range(self.config["epochs"]):
                opt.zero_grad()
                out = net(x, meta)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
            net.eval()
            self.nets.append(net)

    def predict_proba(self, ds: FusionDataset, idx: np.ndarray) -> np.ndarray:
        x, meta = self._features(ds, idx, fit=False)
        probs = []
        with torch.no_grad():
            for net in self.nets:
                logits = net(x, meta)
                probs.append(torch.softmax(logits, dim=-1).numpy())
        return np.mean(probs, axis=0)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class MILFusionModel:
    """Attention-MIL fusion on tile bags (see src/mil). Trains on ``train_idx`` with its own
    15% stratified early-stopping carve-out, so it fits the same fit/predict contract as
    the other families and is scored by the same CV evaluator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.results = []

    def _hp(self, seed: int):
        from src.mil.config import load_conf, resolve_hparams
        from src.mil.trainer import merge_hparams

        c = self.config
        if c.get("tuned_single"):
            # single-FM ABMIL baseline: use that FM's OWN full tuned settings from 02g
            # (its 13-config search is the single model's tuning budget)
            (m,) = c["models"]
            hp = resolve_hparams(m)[0]
            return merge_hparams(hp.to_dict(), {"use_meta": c.get("use_meta", False), "seed": seed})
        keys = ["fusion", "proj_dim", "dropout", "stream_dropout", "lr", "weight_decay", "use_meta",
                "dropout_in", "attn_dim", "max_epochs", "accum_steps", "max_train_tiles", "monitor",
                "label_smoothing"]
        stream_lr = {}
        if c.get("per_fm_lr", True):
            for m in c["models"]:
                stream_lr[m] = resolve_hparams(m)[0].lr
        return merge_hparams(load_conf().get("defaults", {}), {k: c[k] for k in keys if k in c},
                             {"stream_lr": stream_lr, "seed": seed})

    def fit(self, ds: FusionDataset, train_idx: np.ndarray) -> None:
        from sklearn.model_selection import train_test_split

        from src.mil.trainer import train_mil

        if ds.bags is None:
            raise ValueError("mil family needs tile bags: run 03_run_agent with --tile-emb-root")
        pids = [ds.patient_ids[i] for i in train_idx]
        y = ds.y[train_idx]
        fit_p, es_p = train_test_split(pids, test_size=0.15, stratify=y, random_state=int(self.config.get("seed", 42)))
        y_by_pid = dict(zip(ds.patient_ids, ds.y.tolist()))
        meta = {p: np.array([a, s], dtype=np.float32) for p, a, s in zip(ds.patient_ids, ds.age, ds.sex)}
        self.results = [
            train_mil(ds.bags, y_by_pid, fit_p, es_p, len(ds.classes), self._hp(int(self.config.get("seed", 42)) + s),
                      fms=list(self.config["models"]), meta=meta)
            for s in range(int(self.config.get("n_seeds", 1)))
        ]

    def predict_proba(self, ds: FusionDataset, idx: np.ndarray) -> np.ndarray:
        pids = [ds.patient_ids[i] for i in idx]
        return np.mean([r.predict(pids) for r in self.results], axis=0)


def build_model(config: dict[str, Any]):
    family = config["family"]
    if family == "mil":
        return MILFusionModel(config=config)
    if family == "early":
        return EarlyFusionModel(config=config)
    if family == "late":
        return LateFusionModel(config=config)
    if family == "learned":
        return LearnedFusionModel(config=config)
    if family == "abmil_late":
        return ABMILLateFusionModel(config=config)
    raise ValueError(f"unknown family: {family}")