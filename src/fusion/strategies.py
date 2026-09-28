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

Every family fits ONLY on the given training indices -- callers (the CV
evaluator) are responsible for passing train/val splits so nothing here
can leak.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

from src.fusion.data import FusionDataset

MODEL_NAMES = ["uni2h", "virchow2", "genbio_pathfm"]


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
            "C": 1.0,
        }
    if family == "late":
        return {
            "family": "late",
            "combine": "mean",  # "mean" | "weighted" | "stacking"
            "weights": None,  # only used if combine == "weighted"
            "C": 1.0,
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
        self.clf = LogisticRegression(
            C=self.config.get("C", 1.0), max_iter=2000, class_weight="balanced"
        ).fit(X, y)

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
        self.per_model_clf[m] = LogisticRegression(
            C=self.config.get("C", 1.0), max_iter=2000, class_weight="balanced"
        ).fit(X, y)

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
            probs = [self._proba_one_model(ds, m, train_idx) for m in MODEL_NAMES]
            meta, _ = _metadata_matrix(ds, train_idx, self.age_scaler, fit=False)
            X_stack = np.concatenate(probs + [meta], axis=1)
            y = ds.y[train_idx]
            self.stacker = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_stack, y)

    def predict_proba(self, ds: FusionDataset, idx: np.ndarray) -> np.ndarray:
        probs = [self._proba_one_model(ds, m, idx) for m in MODEL_NAMES]
        combine = self.config.get("combine", "mean")

        if combine == "mean":
            return np.mean(probs, axis=0)
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

def build_model(config: dict[str, Any]):
    family = config["family"]
    if family == "early":
        return EarlyFusionModel(config=config)
    if family == "late":
        return LateFusionModel(config=config)
    if family == "learned":
        return LearnedFusionModel(config=config)
    raise ValueError(f"unknown family: {family}")
