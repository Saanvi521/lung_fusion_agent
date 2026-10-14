# Lung Adenocarcinoma Subtyping — Foundation-Model Fusion Agent

Predicts a 7-class lung adenocarcinoma histologic subtype from whole-slide images by fusing three pathology foundation models (UNI2-h, Virchow2, GenBio-PathFM) with patient age/sex, via an agent that searches fusion strategies. See `report.md` for the full write-up, results, and analysis.

## Setup

```bash
pip install -r requirements.txt
```

Designed to run on free-tier Google Colab/Kaggle GPU. Point Drive-dependent stages (`02a`, `02b`, `02c`, `02d`) at a mounted Drive path so embeddings persist across Colab sessions — see each script's own docstring for exact mount instructions.

## Pipeline (run in order)

| Stage | Script | What it does |
|---|---|---|
| 1 | `src/data/make_split.py` | Builds the patient-level, stratified 70/10/20 train/val/test split; writes `artifacts/splits/`. |
| 2a | `scripts/02a_tile_slides.py` | Downloads each slide once, builds a tissue mask, tiles at 224×224 (20x), caches tiles to Drive. Run once. |
| 2b | `scripts/02b_extract_embeddings.py` | Embeds cached tiles with one foundation model, mean-pools to one vector/patient. Run 3x — once per `--model` (`uni2h`, `virchow2`, `genbio_pathfm`). |
| 2c | `scripts/02c_train_abmil.py` | Trains one ABMIL attention-pooling network per model (superseded by `02d` for reported results — kept for reference). |
| 2d | `scripts/02d_crossfit_abmil.py` | Cross-fitted ABMIL pooling (fixes a train/test leakage issue in `02c` — see `report.md` §4/§6.1). Produces the ABMIL numbers actually reported. Run 3x, once per model. |
| 2e | `scripts/02e_plot_training_curve.py` | Diagnostic-only: plots ABMIL's per-epoch train/validation loss curve for one representative cross-fit fold, using `02d`'s own split logic and hyperparameters. Writes to `figures/`. Doesn't affect any results. |
| 3 | `scripts/03_run_agent.py` | Runs the fusion-strategy search agent (early/late/learned) and the once-only test evaluation. Run once per `--embeddings-dir` (mean-pooled vs. ABMIL cross-fit). |

Example:
```bash
python -m scripts.03_run_agent \
  --split artifacts/splits/split_seed42.csv \
  --embeddings-dir artifacts/embeddings \
  --out-dir artifacts/agent_run
```

`notebooks/lung_agent.ipynb` has all of the above as runnable Colab cells in order.

## Repository layout

```
conf/               configuration
src/
  data/              subset selection, tiling, splits
  embeddings/        one module per foundation model + pooling (mean, ABMIL)
  fusion/            fusion strategies (early/late/learned), registered by name
  agent/             search loop, proposer, decision logging
  eval/              metrics, bootstrap CIs, comparison table
scripts/             stage entry points (table above)
notebooks/           Colab-runnable version of the full pipeline
artifacts/           splits, agent run outputs, decision logs (cached embeddings are gitignored — too large to commit)
figures/             plots referenced in report.md (ABMIL training curves, agent search trajectories)
report.md            full technical report
```

## Key results folders in `artifacts/`

- `artifacts/` (root) — original run on mean-pooled embeddings.
- `artifacts/agent_run_abmil_cf/` — run on cross-fitted ABMIL embeddings (the ABMIL numbers in the report), plus each model's own-head AUROC (`*_abmil_head_scores.json`).

No API key is required to run the agent.
