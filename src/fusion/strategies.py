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
           small stacking meta-classifier).
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
            "combine": "mean",  # "mean" | "logmean" | "weighted" | "stacking"
            "weights": None,  # only used if combine == "weighted"
            "C": "auto",
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
            "proj_dim": 256,
            "dropout": 0.25,
            "stream_dropout": 0.0,
            "lr": 2e-4,
            "weight_decay": 1e-4,
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

    def fit(self, ds: FusionDataset, train_idx: np.ndarray) -> None:
        for m in MODEL_NAMES:
            self._fit_one_model(ds, m, train_idx)

        if self.config.get("combine") == "stacking":
            # Fix: the meta-classifier's training features must come from
            # OUT-OF-FOLD predictions, not in-sample ones. The per-model
            # classifiers above were just fit on ALL of train_idx, so asking
            # them to predict on that same train_idx gives artificially
            # confident/accurate probabilities -- nothing like what they'll
            # actually produce on unseen patients at test time. Training the
            # stacker on those in-sample probabilities teaches it a
            # combination rule for inputs it will never see again, which is
            # almost certainly why "stacking" scored worst of all 3 combine
            # options in the real run. Building 5-fold cross-validated
            # out-of-fold predictions instead matches train-time features to
            # test-time conditions, the standard fix for stacking leakage.
            y_train = ds.y[train_idx]
            n_classes = len(ds.classes)
            oof_probs = {m: np.zeros((len(train_idx), n_classes)) for m in MODEL_NAMES}
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold_tr_local, fold_val_local in skf.split(train_idx, y_train):
                fold_tr_idx = train_idx[fold_tr_local]
                fold_val_idx = train_idx[fold_val_local]
                for m in MODEL_NAMES:
                    fold_model = LateFusionModel(config=self.config)
                    fold_model._fit_one_model(ds, m, fold_tr_idx)
                    oof_probs[m][fold_val_local] = fold_model._proba_one_model(ds, m, fold_val_idx)
            meta, _ = _metadata_matrix(ds, train_idx, self.age_scaler, fit=False)
            X_stack = np.concatenate([oof_probs[m] for m in MODEL_NAMES] + [meta], axis=1)
            self.stacker = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_stack, y_train)

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
        raise ValueError(f"unknown combine: {combine}")


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
                "dropout_in", "attn_dim", "max_epochs", "accum_steps", "max_train_tiles", "monitor"]
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
    raise ValueError(f"unknown family: {family}")