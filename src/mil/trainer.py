"""One training routine for every attention-MIL model in this repo.

Used by: 02d (cross-fitted single-FM ABMIL), 02g (per-FM hyper-parameter search), and
the ``mil`` fusion family the agent searches over. Having exactly one trainer is what
makes the curves in the report the curves of the models that were actually used.

What changed vs. the first version, and why
-------------------------------------------
* **Comparable train/val losses.** The optimisation loss is class-weighted; the old
  code logged it with weights 1/count and ``reduction="sum"``, i.e. ~15x smaller than
  the unweighted validation loss, so the two curves could not be compared. Now every
  epoch we evaluate BOTH sets the same way (eval mode, full bags, unweighted mean
  cross-entropy) and also log macro AUROC and balanced accuracy for both.
* **Class weights normalised** to mean 1 (w_c = N / (C * n_c)), so the loss is on the
  usual cross-entropy scale.
* **AdamW + linear warm-up + cosine decay**, gradient clipping, gradient accumulation
  (a bag is one slide; accumulating k bags gives an effective batch of k without
  padding), and **random tile sub-sampling** of training bags each step (standard MIL
  augmentation: the model never sees exactly the same bag twice).
* **Early stopping** on a monitored validation quantity with a minimum number of
  epochs and the best checkpoint restored.
* **Per-FM learning rates / weight decay** via optimizer parameter groups.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.embeddings.aggregate import MultiStreamABMIL


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class MILHParams:
    # architecture
    proj_dim: int = 512
    attn_dim: int = 384
    dropout_in: float = 0.1
    dropout: float = 0.25
    fusion: str = "tile_concat"        # only used with >1 FM stream
    stream_dropout: float = 0.0        # modality dropout, only used with >1 FM stream
    use_meta: bool = False             # feed (age, sex) to the head
    standardize: bool = True           # per-dim feature standardisation (train stats)
    # optimisation
    lr: float = 2e-4
    weight_decay: float = 1e-4
    stream_lr: dict = field(default_factory=dict)            # {fm: lr} overrides for that FM's layers
    stream_weight_decay: dict = field(default_factory=dict)  # {fm: wd}
    max_epochs: int = 50
    min_epochs: int = 10
    patience: int = 10
    warmup_epochs: int = 2
    accum_steps: int = 4               # effective batch size in bags
    max_train_tiles: int = 512         # random tiles per training bag per step (0 = all)
    label_smoothing: float = 0.0
    class_weighting: str = "loss"      # "loss" | "sampler" | "none"
    grad_clip: float = 1.0
    monitor: str = "val_loss"          # "val_loss" (min) | "val_auroc" (max) | "val_bacc" (max)
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MILHParams":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown MIL hyper-parameters: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Metrics that don't crash on tiny validation sets
# ---------------------------------------------------------------------------
def safe_macro_auroc(y: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """Macro one-vs-rest AUROC averaged over the classes that have both positives and
    negatives in ``y`` (a 20-patient validation set can miss a class)."""
    y = np.asarray(y)
    aucs = []
    for c in range(n_classes):
        pos = y == c
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos.astype(int), probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def summarize(y: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(probs, 1e-7, 1.0)
    return {
        "loss": float(-np.mean(np.log(p[np.arange(len(y)), y]))),
        "auroc": safe_macro_auroc(y, probs, n_classes),
        "bacc": float(balanced_accuracy_score(y, probs.argmax(1))),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_stats(bags: dict, pids: list[str], fm: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std over ALL tiles of the given (training) patients."""
    s = s2 = None
    n = 0
    for pid in pids:
        x = bags[pid][fm].astype(np.float64)
        s = x.sum(0) if s is None else s + x.sum(0)
        s2 = (x ** 2).sum(0) if s2 is None else s2 + (x ** 2).sum(0)
        n += x.shape[0]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


class MetaScaler:
    """Standardise age with train-patient statistics; sex stays 0/1."""

    def __init__(self, meta: dict[str, np.ndarray], train_pids: list[str]):
        ages = np.array([meta[p][0] for p in train_pids], dtype=np.float32)
        self.mu, self.sd = float(ages.mean()), float(ages.std() + 1e-6)

    def __call__(self, m: np.ndarray) -> np.ndarray:
        return np.array([(m[0] - self.mu) / self.sd, m[1]], dtype=np.float32)


def build_model(embed_dims: dict[str, int], n_classes: int, hp: MILHParams) -> MultiStreamABMIL:
    return MultiStreamABMIL(
        embed_dims,
        n_classes,
        proj_dim=hp.proj_dim,
        attn_dim=hp.attn_dim,
        dropout_in=hp.dropout_in,
        dropout=hp.dropout,
        fusion=hp.fusion,
        meta_dim=2 if hp.use_meta else 0,
        stream_dropout=hp.stream_dropout,
    )


def _param_groups(model: MultiStreamABMIL, hp: MILHParams) -> list[dict]:
    groups, claimed = [], set()
    for m in model.stream_names:
        params = [p for p in model.stream_parameters(m)]
        claimed.update(id(p) for p in params)
        groups.append({
            "params": params,
            "lr": hp.stream_lr.get(m, hp.lr),
            "weight_decay": hp.stream_weight_decay.get(m, hp.weight_decay),
            "name": m,
        })
    shared = [p for p in model.parameters() if id(p) not in claimed]
    if shared:
        groups.append({"params": shared, "lr": hp.lr, "weight_decay": hp.weight_decay, "name": "shared"})
    return groups


class _Batcher:
    """Moves one bag (+ metadata) to the device, optionally sub-sampling tiles."""

    def __init__(self, bags, meta, meta_scaler, fms, device):
        self.bags, self.meta, self.meta_scaler, self.fms, self.device = bags, meta, meta_scaler, fms, device

    def __call__(self, pid: str, max_tiles: int = 0, rng: np.random.RandomState | None = None):
        b = self.bags[pid]
        n = b[self.fms[0]].shape[0]
        idx = None
        if max_tiles and n > max_tiles and rng is not None:
            idx = np.sort(rng.choice(n, size=max_tiles, replace=False))
        x = {
            m: torch.from_numpy(np.ascontiguousarray(b[m] if idx is None else b[m][idx])).to(self.device).float()
            for m in self.fms
        }
        meta = None
        if self.meta_scaler is not None:
            meta = torch.from_numpy(self.meta_scaler(self.meta[pid])).to(self.device)
        return x, meta


@torch.inference_mode()
def predict(model, batcher: _Batcher, pids: list[str]) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(*batcher(p))["logits"], -1).float().cpu().numpy() for p in pids]
    return np.stack(out) if out else np.zeros((0, 0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    model: MultiStreamABMIL
    history: list[dict]
    best_epoch: int
    stopped_epoch: int
    hparams: dict
    batcher: Any = None   # use with predict(...) on new patients (same scalers)

    def predict(self, pids: list[str]) -> np.ndarray:
        return predict(self.model, self.batcher, pids)


def train_mil(
    bags: dict[str, dict[str, np.ndarray]],
    y_by_pid: dict[str, int],
    train_pids: list[str],
    val_pids: list[str],
    n_classes: int,
    hp: MILHParams,
    fms: list[str],
    meta: dict[str, np.ndarray] | None = None,
    device: str | None = None,
    verbose: bool = False,
    tag: str = "",
) -> TrainResult:
    """Train on ``train_pids``, early-stop on ``val_pids``. Nothing else is touched."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(hp.seed)
    rng = np.random.RandomState(hp.seed)

    embed_dims = {m: bags[train_pids[0]][m].shape[1] for m in fms}
    model = build_model(embed_dims, n_classes, hp)
    if hp.standardize:
        for m in fms:
            model.streams[m].norm.set_stats(*feature_stats(bags, train_pids, m))
    model.to(device)

    meta_scaler = MetaScaler(meta, train_pids) if hp.use_meta else None
    batcher = _Batcher(bags, meta, meta_scaler, fms, device)

    y_train = np.array([y_by_pid[p] for p in train_pids])
    y_val = np.array([y_by_pid[p] for p in val_pids])
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(n_classes)
    w[present] = len(y_train) / (present.sum() * counts[present])   # mean weight over samples = 1

    use_loss_w = hp.class_weighting == "loss"
    # With class weights, CrossEntropyLoss(weight=..., reduction="mean") on ONE bag divides by
    # that bag's own weight and cancels it, so we use the unreduced loss and multiply by w_y.
    loss_fn_none = nn.CrossEntropyLoss(reduction="none", label_smoothing=hp.label_smoothing)
    sample_w = torch.tensor(w if use_loss_w else np.ones(n_classes), dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(_param_groups(model, hp))
    steps_per_epoch = math.ceil(len(train_pids) / max(hp.accum_steps, 1))
    total_steps = max(1, steps_per_epoch * hp.max_epochs)
    warmup = steps_per_epoch * hp.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    maximize = hp.monitor in ("val_auroc", "val_bacc")
    best_score = -np.inf if maximize else np.inf
    best_state, best_epoch, bad = None, -1, 0
    history: list[dict] = []

    if hp.class_weighting == "sampler":
        p_sample = (1.0 / counts[y_train]) / (1.0 / counts[y_train]).sum()

    for epoch in range(hp.max_epochs):
        model.train()
        if hp.class_weighting == "sampler":
            order = list(rng.choice(train_pids, size=len(train_pids), replace=True, p=p_sample))
        else:
            order = list(train_pids)
            rng.shuffle(order)
        opt.zero_grad(set_to_none=True)
        running, n_in_step = 0.0, 0
        for i, pid in enumerate(order):
            x, m = batcher(pid, hp.max_train_tiles, rng)
            out = model(x, m)
            yt = torch.tensor([y_by_pid[pid]], device=device)
            loss = (loss_fn_none(out["logits"].unsqueeze(0), yt) * sample_w[yt]).mean()
            (loss / hp.accum_steps).backward()
            running += float(loss.detach())
            n_in_step += 1
            if n_in_step == hp.accum_steps or i == len(order) - 1:
                # rescale if the last step had fewer than accum_steps bags
                if n_in_step != hp.accum_steps:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(hp.accum_steps / n_in_step)
                if hp.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                n_in_step = 0

        # ---- evaluate train AND val identically (eval mode, full bags, unweighted CE) ----
        tr = summarize(y_train, predict(model, batcher, train_pids), n_classes)
        va = summarize(y_val, predict(model, batcher, val_pids), n_classes)
        rec = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_objective": running / len(order),  # weighted, dropout on, sub-sampled bags
            "train_loss": tr["loss"], "val_loss": va["loss"],
            "train_auroc": tr["auroc"], "val_auroc": va["auroc"],
            "train_bacc": tr["bacc"], "val_bacc": va["bacc"],
        }
        history.append(rec)

        score = rec[hp.monitor]
        improved = (score > best_score + 1e-4) if maximize else (score < best_score - 1e-4)
        if not np.isfinite(score):
            improved = False
        if improved:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose:
            print(f"  {tag}ep {epoch:3d} lr={rec['lr']:.2e} | loss tr {tr['loss']:.3f} va {va['loss']:.3f} | "
                  f"AUROC tr {tr['auroc']:.3f} va {va['auroc']:.3f} | bAcc tr {tr['bacc']:.3f} va {va['bacc']:.3f}"
                  f"{'  *' if improved else ''}")
        if epoch + 1 >= hp.min_epochs and bad >= hp.patience:
            break

    if best_state is None:
        raise RuntimeError("validation monitor was never finite — check the inputs (NaNs?)")
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model=model, history=history, best_epoch=best_epoch, stopped_epoch=history[-1]["epoch"],
                       hparams=hp.to_dict(), batcher=batcher)


def merge_hparams(*dicts: dict | None) -> MILHParams:
    """Later dicts override earlier ones (defaults < config file < tuned json < CLI)."""
    merged: dict = {}
    for d in dicts:
        if d:
            merged.update({k: v for k, v in d.items() if v is not None})
    return MILHParams.from_dict(merged)
