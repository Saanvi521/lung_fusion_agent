"""Config proposers.

Both proposers share one interface: ``propose(history) -> config``, where
``history`` is the list of every trial tried so far (config + score). This
is what lets the agent "condition its next proposal on what it has seen" --
whichever proposer is active, it can look back at everything tried.

DeterministicProposer is the one that's always available and always
runnable with zero external dependencies -- per the brief, this is not a
fallback bolted on afterward, it's a first-class required mode. It does
explore/exploit: half the time it tries something structurally new (a
different family / hyperparameter combo it hasn't tried), half the time it
takes the current best config and perturbs ONE setting (coordinate search).

LLMProposer wraps an LLM call to choose the next config in natural language
and log its reasoning, and degrades to DeterministicProposer automatically
if no API key is set -- checked once at construction, not per-call, so a
run's proposer choice is stable and reportable.
"""
from __future__ import annotations

import copy
import json
import os
import random
from typing import Any

from src.fusion.strategies import default_config

SEARCH_SPACE = {
    "early": {
        "l2_normalize": [True, False],
        "pca_dim": [None, 64, 128, 256],
        # "auto" = C chosen per fit by inner-CV log-loss. The old grid [0.1, 1, 10] had no
        # strongly-regularised option at all, for 1.5k-8.7k features and ~130 patients.
        "C": ["auto", 1e-3, 1e-2, 0.1, 1.0],
    },
    "late": {
        "combine": ["mean", "logmean", "weighted", "stacking"],
        # Fix: "weighted" previously had no actual weight values to search
        # over, so config.get("weights") was always None and silently fell
        # back to equal thirds -- mathematically identical to "mean". These
        # are real, distinct weight vectors so "weighted" is an actually
        # different strategy once this is picked. [0.6, 0.2, 0.2] etc. lean
        # toward whichever model the trial is currently built from (the
        # perturbation logic keeps the config's model order fixed via
        # MODEL_NAMES = [uni2h, virchow2, genbio_pathfm], so these favor
        # uni2h first since it was the strongest standalone model).
        "weights": [
            [1 / 3, 1 / 3, 1 / 3],
            [0.6, 0.2, 0.2],
            [0.5, 0.25, 0.25],
            [0.6, 0.3, 0.1],
            [0.45, 0.35, 0.2],
        ],
        "C": ["auto", 1e-3, 1e-2, 0.1, 1.0],
    },
    "learned": {
        "hidden_dim": [32, 64, 128],
        "dropout": [0.3, 0.5, 0.7],
        "weight_decay": [1e-4, 1e-3, 1e-2],
        "epochs": [20, 30, 50],
    },
    # attention-MIL fusion on tile bags; only searched when bags are loaded
    "mil": {
        "fusion": ["tile_concat", "gated", "late"],
        "proj_dim": [64, 128, 256],
        "dropout": [0.25, 0.5],
        "stream_dropout": [0.0, 0.2, 0.4],
        "lr": [5e-5, 1e-4, 2e-4],
        "weight_decay": [1e-4, 1e-2, 5e-2],
        "label_smoothing": [0.0, 0.1],
        "max_train_tiles": [256, 512],
    },
}

# A fixed initial batch spanning every family, so the very first few trials
# already cover the brief's minimum viable scope regardless of what the
# search does afterward.
SEED_CONFIGS: list[dict[str, Any]] = [
    {**default_config("early"), "l2_normalize": True, "pca_dim": None},
    {**default_config("early"), "l2_normalize": False, "pca_dim": None},
    {**default_config("early"), "l2_normalize": True, "pca_dim": 128},
    {**default_config("late"), "combine": "mean"},
    # Fix: this used to be {"combine": "weighted"} with weights left at the
    # default None, which is identical to "mean" -- give it real weights so
    # the very first weighted trial is an honest, distinct test.
    {**default_config("late"), "combine": "weighted", "weights": [0.6, 0.2, 0.2]},
    {**default_config("late"), "combine": "stacking"},
    {**default_config("learned")},
]

# Added to the seed batch only when the mil family is enabled: one per fusion mode.
MIL_SEED_CONFIGS: list[dict[str, Any]] = [
    {**default_config("mil"), "fusion": "tile_concat"},
    {**default_config("mil"), "fusion": "gated"},
    {**default_config("mil"), "fusion": "late"},
]


class DeterministicProposer:
    name = "deterministic"

    def __init__(self, seed: int = 42, families: list[str] | None = None):
        self.rng = random.Random(seed)
        self._seed_idx = 0
        self.families = families or ["early", "late", "learned"]
        self.seed_configs = [c for c in SEED_CONFIGS if c["family"] in self.families]
        if "mil" in self.families:
            self.seed_configs += MIL_SEED_CONFIGS

    def propose(self, history: list[dict]) -> tuple[dict, str]:
        """Like ``_propose`` but never re-proposes a config that was already evaluated
        (the first version re-ran identical configs, spending trial budget on known answers)."""
        seen = {json.dumps(h["config"], sort_keys=True) for h in history}
        for _ in range(50):
            config, why = self._propose(history)
            if json.dumps(config, sort_keys=True) not in seen:
                return config, why
        return config, why + " (search space around the incumbent exhausted; repeat)"

    def _propose(self, history: list[dict]) -> tuple[dict, str]:
        # Phase 1: exhaust the fixed seed batch first, so every family gets
        # at least one honest, untuned try before any local search begins.
        if self._seed_idx < len(self.seed_configs):
            config = copy.deepcopy(self.seed_configs[self._seed_idx])
            self._seed_idx += 1
            return config, f"seed config {self._seed_idx}/{len(self.seed_configs)}: cover this family's default first"

        # Phase 2: explore/exploit, 50/50
        if not history or self.rng.random() < 0.5:
            family = self.rng.choice(self.families)
            config = default_config(family)
            for key, options in SEARCH_SPACE[family].items():
                config[key] = self.rng.choice(options)
            return config, f"explore: random {family} config, untried region of the search space"

        # Exploit: perturb ONE hyperparameter of the current best.
        # Fix: must only ever consider ACCEPTED trials here. A trial that
        # scored high but was rejected by the SE-gate (not a real
        # improvement, just noise) should never become the base for future
        # mutation -- doing so silently defeats the whole point of the
        # significance test, chasing exactly the noise it was built to
        # ignore. Falls back to the full history only in the (shouldn't
        # happen after the seed phase) case where nothing has been
        # accepted yet.
        accepted_history = [h for h in history if h.get("accepted")]
        best = max(accepted_history or history, key=lambda h: h["score"])
        config = copy.deepcopy(best["config"])
        family = config["family"]
        key = self.rng.choice(list(SEARCH_SPACE[family].keys()))
        options = SEARCH_SPACE[family][key]
        old_val = config.get(key)
        new_val = self.rng.choice([o for o in options if o != old_val] or options)
        config[key] = new_val
        return config, f"exploit: best so far is {family} (score={best['score']:.4f}); perturbing '{key}' {old_val}->{new_val}"


class LLMProposer:
    name = "llm"

    def __init__(self, seed: int = 42, families: list[str] | None = None):
        self.fallback = DeterministicProposer(seed=seed, families=families)
        self.families = self.fallback.families
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.active = self.api_key is not None
        if not self.active:
            print("[proposer] no ANTHROPIC_API_KEY found -- degrading to deterministic search")

    def propose(self, history: list[dict]) -> tuple[dict, str]:
        if not self.active:
            return self.fallback.propose(history)

        # Kept deliberately small and self-contained: send the trial
        # history as JSON, ask for ONE new config as JSON back. If the
        # model returns anything unparseable, fall back to the
        # deterministic proposer for that trial rather than crashing the
        # whole search -- one bad LLM response should not end the run.
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            trimmed_history = [
                {"config": h["config"], "score": round(h["score"], 4), "accepted": h["accepted"]} for h in history[-10:]
            ]
            prompt = (
                "You are tuning a fusion strategy that combines 3 pathology "
                "foundation-model embeddings + age/sex to predict a 7-class "
                "lung cancer subtype. Search space (families: early/late/learned) "
                f"is: {json.dumps({f: SEARCH_SPACE[f] for f in self.families})}. Seed configs already tried: "
                f"{json.dumps(self.fallback.seed_configs)}. Recent trial history (most recent "
                f"last): {json.dumps(trimmed_history)}. Propose ONE new config as "
                "a JSON object matching the search space schema, plus a one "
                "sentence rationale. Respond with ONLY a JSON object: "
                '{"config": {...}, "rationale": "..."}'
            )
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            return parsed["config"], f"[llm] {parsed['rationale']}"
        except Exception as e:  # noqa: BLE001 -- any failure here should degrade, not crash
            print(f"[proposer] LLM call failed ({e}) -- using deterministic fallback for this trial")
            return self.fallback.propose(history)


def build_proposer(use_llm: bool, seed: int = 42, families: list[str] | None = None):
    if use_llm:
        return LLMProposer(seed=seed, families=families)
    return DeterministicProposer(seed=seed, families=families)