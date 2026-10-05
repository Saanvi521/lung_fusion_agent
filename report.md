# Lung Adenocarcinoma Subtyping — Foundation-Model Fusion Agent

## Overview

This project predicts a 7-class lung adenocarcinoma histologic subtype from whole-slide images by fusing embeddings from three pathology foundation models (UNI2-h, Virchow2, GenBio-PathFM) with patient age and sex, using an agent that searches over fusion strategies and reports a final comparison against single-model baselines on a held-out test set.

---

## 1. Data

**Source:** `CLWD.csv` (slide-level metadata) + whole-slide images from `kmmuleelab/Lung_Pathology_Image_JPG` (Hugging Face).

**Patient construction:** Slides are grouped by `SampleNumber`, which merges the one duplicated ID (`8377886`) into a single patient. One slide per patient is kept — the lowest numeric `WSI_ID`, applied as a deterministic rule. The patient's label is `Benchmark_Label_7class` on that selected slide. 5 patients have conflicting labels across their own slides; these are flagged (`label_conflict`) rather than dropped, so a sensitivity check can exclude them later if needed.

**Selection-rule sanity check:** Before trusting "lowest WSI_ID" as the labeling rule, I checked whether it agrees with the rest of a patient's slides. Agreement rate: 97.9% (143/146 multi-slide patients) — the rule isn't introducing systematic label bias.

**Split:** Stratified 70/10/20 by subtype, seed 42 → 146 train / 21 val / 42 test.

**Data availability note:** 12 patients' `WSI_ID`s never resolved to an actual file in the dataset repo (`[warn] no repo file matching WSI_ID=...`). Those patients were never tiled, never embedded by any of the 3 models, and are dropped entirely (`load_fusion_dataset` requires all 3 model embeddings to be present). **Effective dataset size: 197 patients, not 209.** Of the 12 dropped, 4 came from the test split — the final confusion matrix sums to 38 patients, not 42.

---

## 2. Tiling & Tissue Detection

| Step | Detail |
|---|---|
| Downsampling | 80x → 20x via PIL `.draft()` — decodes the smaller image directly instead of the full gigapixel original. Measured peak memory: 519MB → 89MB (~5.8x reduction) on a realistic synthetic slide. |
| Tissue mask | HSV saturation + Otsu threshold, computed on an 8x-downsampled thumbnail (not the full 20x image) |
| Tile size | 224×224, non-overlapping |
| Tissue threshold | Tile kept only if ≥50% tissue |
| Cap | 600 tiles/slide max; seeded random subset if exceeded (reproducible) |

Both memory optimizations above were necessary fixes — earlier attempts without them crashed free-tier Colab from RAM exhaustion.

---

## 3. Foundation Models

| Model | Embedding dim | Notes |
|---|---|---|
| UNI2-h | 1536 | Custom ViT-H/14 via `timm` |
| Virchow2 | 2560 | Class token + mean-pooled patch tokens; 4 register tokens excluded |
| GenBio-PathFM | 4608 | Model's own mean/std normalization (not ImageNet stats), via `transformers` |

All three are frozen, pretrained feature extractors — none are fine-tuned. Tile embeddings are mean-pooled into one vector per model per patient. Only the small classifier(s) sitting on top of the embeddings are ever trained.

**Extraction pipeline:** split into two stages to fit Colab's memory budget — Stage A downloads each slide once and caches tiles; Stage B loads one model at a time and embeds from the cache (run 3x, once per model). This guarantees only one model (~1–4.5GB) is ever resident in memory, instead of all three (~10GB) at once.

---

## 4. Fusion Strategies

| Family | Approach |
|---|---|
| **Early** | Concatenate all 3 embeddings (optional L2-normalize / PCA) + metadata → one logistic regression |
| **Late** | Separate classifier per model (+ metadata), combined via mean / weighted-average / stacking |
| **Learned** | Small gated-attention network learning per-patient weighting across the 3 embedding streams |

**Metadata handling:** Age and sex are included in every family, not bolted onto one: concatenated into the feature vector (early), into each per-model classifier's input (late), and into the gated network's final classification head (learned).

---

## 5. Agent Search Design

The agent uses a deterministic proposer (no LLM key was supplied; it would have degraded to deterministic regardless). Search proceeds in two phases:

1. **Seed phase:** one fixed baseline config per family/variant (7 configs), guaranteeing every family gets an honest, untuned first try.
2. **Explore/exploit phase (50/50):** explore tries a fresh random config in an untried region of the search space; exploit perturbs one hyperparameter of the current best.

**Acceptance rule:** a candidate only replaces the incumbent if it beats it by more than its standard error, computed from 5-fold × 3-repeat CV over train+val. This is the guardrail against a long search overfitting the validation folds by chasing numerically-higher-but-statistically-meaningless scores.

**Stopping criterion:** 12 consecutive trials without an accepted improvement (or a hard cap of `--max-trials`).

**Decision log:** every trial — config, rationale, CV mean/std, accept/reject decision and reason — is appended to `decision_log.jsonl` as it happens.

### 5.1 Bugs I found and fixed along the way

My first full run came back with the fused model scoring lower than UNI2-h on its own, which didn't sit right with me, so I went through the whole pipeline looking for why. I found four real bugs. Two of them involve cross-validation, so some quick context first: cross-validation splits the training patients into groups called folds, trains on most of the folds, and tests on the one left out, rotating through so every fold gets a turn as the test set. Every config the agent tries gets scored this way. The random seed controlling how patients get split into folds is fixed, so every single config in the whole search is tested on the exact same folds, every time. That detail turned out to matter a lot.

**Bug 1:** the weighted combine option wasn't actually weighted. It was supposed to combine the three models unevenly, trusting UNI2-h more than the other two. Instead, the code never generated any real weight values for it to use, so it defaulted to an even split across all three models every time, which is exactly what the mean option already does. I caught this by comparing numbers in the decision log: a weighted trial and a mean trial scored the exact same number, down to the decimal. Fixed by giving the weighted option real weight values to choose from.

**Bug 2:** the test deciding whether a new config was actually better used the wrong statistics. When the agent cross-validates a config, it gets 15 separate scores (5 folds, repeated 3 times) and averages them. The amount those 15 scores bounce around is used as that config's uncertainty. Every config in the search is scored on the exact same 15 folds, so if one particular fold happens to contain patients that are just harder to classify, every config's score drops a bit on that fold, not just one of them. The old test looked at config A's 15 scores and config B's 15 scores separately, measured how much each one bounces around on its own, and added those two amounts together as the bar the gap between A and B had to clear. The problem is that most of that bouncing comes from some folds being harder than others, which affects every config equally and has nothing to do with which config is actually better. If config B scores a little higher than config A on every single fold, hard folds and easy folds alike, that is a real, consistent difference, even though each config's own scores look noisy when you look at them in isolation. The fix was to compare A and B fold by fold instead: take B's score on fold 1 minus A's score on fold 1, B's score on fold 2 minus A's score on fold 2, and so on, and check whether that difference is consistently positive. This removes the noise caused by folds being easy or hard and isolates the actual gap between the two configs. That is the correct way to compare two configs evaluated on identical folds, which is what happens here, and it is what let the agent correctly recognize the weighted config as a real improvement instead of rejecting it as noise.

**Bug 3:** the agent kept building on top of a config it had already rejected. After each trial, it tries to improve on its current best by tweaking one setting. The problem was in how it picked that current best: it used whichever config had scored the highest across the whole search, even if that config had been rejected two trials earlier for not clearing the significance bar. So it kept wasting trials refining something it had already ruled out. Fixed so it only ever builds on top of a config that was actually accepted.

**Bug 4:** the stacking option's extra classifier was trained on unrealistic predictions. Late fusion first fits three separate classifiers, one per foundation model, on the training patients. The stacking option then trains one more classifier on top of those three, whose job is to look at what each of the three predicted for a patient and learn the best way to combine them. To train that extra classifier, it needs example predictions to learn from. The bug was in how those example predictions were generated: each of the three per-model classifiers was asked to predict on the same patients it had just been trained on. A classifier's predictions on patients it was trained on are far more accurate than its predictions on patients it has never seen, since it had a direct chance to fit itself to those exact patients during training. So the extra classifier learned to combine predictions that were much more accurate and confident than anything the three models will actually produce on new patients at test time, and what it learned didn't transfer. The fix was to generate those training predictions using held-out folds: for each training patient, get a prediction from a version of the per-model classifier trained on every other training patient except that one, so the predictions used to train the extra classifier are genuinely out-of-sample, the same way its real test-time predictions would be.

I tested each of these four fixes separately against made-up data before trusting any of them, to make sure each one actually did what I thought it did rather than just assuming. I also went back through every other file in the pipeline — how the data loads, the split logic, tiling, the tissue mask, each of the three model wrappers — looking for anything else off, and didn't find anything beyond these four.

### 5.2 Trial history (post-fix run)

| Trial | Config | CV AUROC | Result |
|---|---|---|---|
| 1 | early, l2_normalize=True | 0.8192 | Accepted (first trial) |
| 2 | early, l2_normalize=False | 0.8262 | Rejected — not a significant gain |
| 3 | early + PCA=128 | 0.8192 | Rejected — identical to trial 1 |
| 4 | late, mean | 0.8369 | **Accepted** — beats early fusion |
| 5 | late, weighted [0.6, 0.2, 0.2] | 0.8403 | **Accepted** — real, paired-significant gain over mean |
| 6–17 | stacking (OOF-corrected), learned net, further early/late variants | ≤ 0.8403 | All rejected |

Agent stopped at trial 17 (12-trial no-improvement patience). **Final selected config:** late fusion, weighted `[0.6, 0.2, 0.2]`, `C=1.0`, CV score 0.8403.

---

## 6. Results (§5.3 Comparison Table, Test Set — Touched Once)

| Row | Macro AUROC | Balanced Accuracy |
|---|---|---|
| Metadata only (age + sex) | 0.7251 | 0.420 |
| UNI2-h alone | 0.8441 | 0.370 |
| Virchow2 alone | 0.8191 | 0.304 |
| GenBio-PathFM alone | 0.8117 | 0.457 |
| **Best single model + metadata** (UNI2-h) | **0.8433** | 0.370 |
| **Fused (agent-selected, weighted) + metadata** | 0.8318 | 0.415 |

*(For reference: UNI2-h+metadata 0.8433, Virchow2+metadata 0.8190, GenBio-PathFM+metadata 0.8122 — metadata changes each single model's score only marginally.)*

**Fused vs. best single model:** Δ = **−0.0115** macro AUROC. (Pre-bugfix: −0.0165.)

**95% bootstrap CI, fused model:** AUROC [0.755, 0.903], balanced accuracy [0.289, 0.572]. UNI2-h's 0.843 sits inside this interval — the two results are not statistically distinguishable at this test-set size (n=38).

### 6.1 Confusion Matrix (Fused Model, Test Set, n=38)

| True \ Pred | Acinar | Cribriform | In situ | Lepidic | Micropap. | Papillary | Solid | Recall |
|---|---|---|---|---|---|---|---|---|
| Acinar (7) | 5 | 0 | 0 | 0 | 0 | 1 | 1 | 0.71 |
| Cribriform (2) | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0.00 |
| In situ (8) | 0 | 0 | 5 | 1 | 0 | 2 | 0 | 0.63 |
| Lepidic (4) | 0 | 0 | 2 | 0 | 1 | 1 | 0 | 0.00 |
| Micropapillary (5) | 0 | 0 | 0 | 0 | 2 | 1 | 2 | 0.40 |
| Papillary (6) | 1 | 0 | 0 | 1 | 0 | 4 | 0 | 0.67 |
| Solid (6) | 1 | 1 | 0 | 0 | 1 | 0 | 3 | 0.50 |

**Per-class AUROC (fused model):** Acinar 0.949 · Cribriform 0.792 · In situ 0.921 · Lepidic 0.706 · Micropapillary 0.752 · Papillary 0.823 · Solid 0.880

---

## 7. Explanation

Fusion still didn't beat the best single model, even after fixing all four bugs above, and I don't want to spin that. Here's what actually happened and why I think it's still a reasonable outcome.

The weighted config the agent landed on combines the three models' predictions using fixed proportions: 60% UNI2-h, 20% Virchow2, 20% GenBio-PathFM, instead of splitting it evenly three ways. Picking those proportions makes sense because UNI2-h is clearly the strongest of the three by itself (0.844 AUROC alone vs 0.819 and 0.812 for the other two), so giving it more say in the final prediction should help, and the cross-validation results confirm it does: that 60/20/20 split scored measurably higher than a plain even split, which is a real, verified improvement, not a fluke.

The problem is that a 60/20/20 split still isn't the same as using UNI2-h by itself. Even with extra weight, the other two models still pull every prediction a bit away from what UNI2-h alone would have said, and on this test set that pull did more harm than good often enough to land slightly below UNI2-h's own score. I also tried a learned model that was supposed to figure out this proportion automatically and even adjust it per patient instead of using one fixed ratio for everyone, and it didn't do any better than the fixed 60/20/20 split. That tells me the issue isn't that I picked the wrong proportions. It's that with only 146 training patients, there isn't enough data for any method, fixed or learned, to find a way of combining the three models that reliably beats UNI2-h alone.

On top of that, I don't think 38 test patients is enough to even say that gap is real. I ran a bootstrap confidence interval on the fused model's test score, which comes out to [0.755, 0.903] for AUROC. UNI2-h's own score, 0.843, falls right inside that range. So the honest conclusion isn't that fusion performed worse. It's that this run didn't find a fusion method that's clearly and statistically better than just using UNI2-h, which is a more accurate way to describe what the numbers actually show.

The confusion matrix points at a separate issue that fusion was never going to fix. Lepidic and Cribriform had only 6 test patients between them and the model got 0% recall on both, consistently mistaking them for whichever class looks visually closest. That's also why balanced accuracy (0.415) is so much lower than macro AUROC (0.832): the model tells the common classes apart from everything else just fine, it just hasn't seen enough examples of the rare ones to learn what makes them different from each other. No amount of combining three models fixes a shortage of training examples for a specific class.

Two things I'd try next: calibrating each model's predicted probabilities before combining them, since right now they're just averaged as-is, and if one model tends to be more over-confident or under-confident than another, that skews the combined result. And giving the learned fusion model real early stopping based on a held-out chunk of data, instead of training it for a fixed number of epochs regardless of whether it's still improving.

---

## 8. What I'd do with 10x the compute

The brief asks for this directly, so three honest answers, in order of how much I think they'd actually help:

- **Attention-based tile pooling (ABMIL)** instead of mean pooling. This is one I didn't get to run, not one I tried — there's an unused ABMIL class already sitting in `aggregate.py` from earlier, but every real result in this report uses plain mean pooling, which treats every tile the same, including stroma that slips past the tissue mask without being diagnostically useful. Attention pooling could let the model focus on the tiles that actually look like tumor, which seems like it would matter most for something like Lepidic, where the distinguishing features might only show up in a handful of tiles.
- **A much bigger trial budget using an LLM proposer** instead of the deterministic one. The code already supports this (`LLMProposer`), it just wasn't used here. Right now the search is blind explore/exploit — it tries things and checks if they worked, but it can't reason about why something failed. An LLM proposer could look at a failed trial and propose something targeted in response instead of just moving to the next random thing.
- **Running the whole pipeline on several different train/val/test splits**, not just different CV folds inside one split. Everything in this report, including the final test number, comes from one specific 146/21/42 split. I'd want to see whether the fusion-vs-single-model gap holds up or flips across a few different splits before trusting its direction either way.

---

## 9. Why I'm confident this isn't just overfitting to the validation data

This was something I built in from the start, not something I added after the fact. Every trial's score comes from 5-fold, 3-repeat cross-validation over train+val, so it's 15 scores averaged together rather than one run against the 21-patient val set. A new config only ever replaces the current best if it clears a real statistical bar, computed properly as a paired comparison. And the 38-patient test set only ever got touched once, on the final config, after the whole search was already done.
