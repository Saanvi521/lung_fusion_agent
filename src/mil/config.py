"""Resolve per-foundation-model ABMIL hyper-parameters from conf/abmil.yaml (+ tuned JSON)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from src.mil.trainer import MILHParams, merge_hparams

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONF = Path(os.environ.get("ABMIL_CONF", REPO_ROOT / "conf" / "abmil.yaml"))  # env override for quick tests
TUNED_DIR = Path(os.environ.get("ABMIL_TUNED_DIR", REPO_ROOT / "conf" / "tuned"))  # set to a Drive folder on Colab


def load_conf(path: Path = DEFAULT_CONF) -> dict:
    return yaml.safe_load(Path(path).read_text())


def tuned_path(fm: str) -> Path:
    return TUNED_DIR / f"abmil_{fm}.json"


def resolve_hparams(
    fm: str,
    conf_path: Path = DEFAULT_CONF,
    use_tuned: bool = True,
    cli_overrides: dict | None = None,
) -> tuple[MILHParams, list[str]]:
    """Returns (hparams, sources) where sources says which layers were applied."""
    conf = load_conf(conf_path)
    layers, sources = [conf.get("defaults", {}), conf.get("models", {}).get(fm, {})], ["defaults", f"models.{fm}"]
    tp = tuned_path(fm)
    if use_tuned and tp.exists():
        layers.append(json.loads(tp.read_text())["best_hparams"])
        sources.append(str(tp))
    if cli_overrides:
        clean = {k: v for k, v in cli_overrides.items() if v is not None}
        if clean:
            layers.append(clean)
            sources.append(f"cli:{sorted(clean)}")
    return merge_hparams(*layers), sources
