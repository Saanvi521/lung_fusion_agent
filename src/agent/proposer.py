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
        "C": [0.1, 1.0, 10.0],
    },
    "late": {
        "combine": ["mean", "weighted", "stacking"],
        "C": [0.1, 1.0, 10.0],
    },
    "learned": {
        "hidden_dim": [32, 64, 128],
        "dropout": [0.3, 0.5, 0.7],
        "weight_decay": [1e-4, 1e-3, 1e-2],
        "epochs": [20, 30, 50],
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
    {**default_config("late"), "combine": "weighted"},
    {**default_config("late"), "combine": "stacking"},
    {**default_config("learned")},
]


class DeterministicProposer:
    name = "deterministic"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._seed_idx = 0

    def propose(self, history: list[dict]) -> tuple[dict, str]:
        # Phase 1: exhaust the fixed seed batch first, so every family gets
        # at least one honest, untuned try before any local search begins.
        if self._seed_idx < len(SEED_CONFIGS):
            config = copy.deepcopy(SEED_CONFIGS[self._seed_idx])
            self._seed_idx += 1
            return config, f"seed config {self._seed_idx}/{len(SEED_CONFIGS)}: cover this family's default first"

        # Phase 2: explore/exploit, 50/50
        if not history or self.rng.random() < 0.5:
            family = self.rng.choice(list(SEARCH_SPACE.keys()))
            config = default_config(family)
            for key, options in SEARCH_SPACE[family].items():
                config[key] = self.rng.choice(options)
            return config, f"explore: random {family} config, untried region of the search space"

        # Exploit: perturb ONE hyperparameter of the current best
        best = max(history, key=lambda h: h["score"])
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

    def __init__(self, seed: int = 42):
        self.fallback = DeterministicProposer(seed=seed)
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
                f"is: {json.dumps(SEARCH_SPACE)}. Seed configs already tried: "
                f"{json.dumps(SEED_CONFIGS)}. Recent trial history (most recent "
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


def build_proposer(use_llm: bool, seed: int = 42):
    if use_llm:
        return LLMProposer(seed=seed)
    return DeterministicProposer(seed=seed)
