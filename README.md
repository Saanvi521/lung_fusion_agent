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
| 1 | `src/data/make_split.py` | Patient-level, stratified 70/10/20 split → `artifacts/splits/`. Pass `--repo-id kmmuleelab/Lung_Pathology_Image_JPG` so each patient's slide is the lowest-numbered one **that exists in the repo** (recovers patients whose lowest slide is missing); `--check-against` asserts the patient→split assignment is unchanged. |
| 2a | `scripts/02a_tile_slides.py` | Downloads each slide once, tissue mask, 224×224 tiles at 20x, caches tiles. Skips patients already tiled. |
| 2b | `scripts/02b_extract_embeddings.py` | Embeds cached tiles with one FM (`--save-tile-embeds` keeps the tile bags needed by 2d/2g/3-mil). Run once per `--model`. |
| 2g | `scripts/02g_tune_abmil.py` | **Per-FM ABMIL hyper-parameter search** (budgeted random grid from `conf/abmil.yaml`, 5-fold CV on train+val only, selects by held-out cross-entropy). Writes `conf/tuned/abmil_<fm>.json`. Run once per FM. |
| 2d | `scripts/02d_crossfit_abmil.py` | Cross-fitted ABMIL per FM using the shared trainer (`src/mil/trainer.py`) and the per-FM hyper-parameters. Saves pooled vectors, cross-fitted head probabilities and the **full per-epoch history (train/val loss, macro AUROC, balanced accuracy) for every fold**. |
| 2e | `scripts/02e_plot_training_curve.py` | Plots 2d's saved history (no re-training): loss / AUROC / balanced accuracy, train vs validation, all folds + mean. |
| 3 | `scripts/03_run_agent.py` | Fusion-strategy agent + one-time test evaluation. With `--tile-emb-root` it also searches the `mil` family (attention-MIL fusion of all FMs on tile bags) and adds ABMIL single-FM rows, multi-seed spread, and a paired bootstrap of fused − best single. |

| 4 | `scripts/04_plot_fusion_variants.py` | Visual comparison of the fusion strategies (no test use): the agent's search (CV AUROC of every trial, by family) and the per-epoch train/val curves of the three attention-MIL fusion variants overlaid. |

Hyper-parameter precedence: `MILHParams` defaults < `conf/abmil.yaml` (`defaults`, then `models.<fm>`) < `conf/tuned/abmil_<fm>.json` < CLI flags.

Example (cluster or Colab, `D` = data folder from 2a/2b):
```bash
python -m src.data.make_split --csv CLWD.csv --seed 42 --repo-id kmmuleelab/Lung_Pathology_Image_JPG \
    --check-against artifacts/splits/split_seed42.csv
for m in uni2h virchow2 genbio_pathfm; do
  python -m scripts.02g_tune_abmil   --model $m --split artifacts/splits/split_seed42.csv --tile-emb-root $D/tile_embeddings
  python -m scripts.02d_crossfit_abmil --model $m --split artifacts/splits/split_seed42.csv --tile-emb-root $D/tile_embeddings \
      --emb-out-dir $D/embeddings_abmil_cf/$m
done
python -m scripts.02e_plot_training_curve --history $D/embeddings_abmil_cf/*_history.json --out-dir figures
python -m scripts.03_run_agent --split artifacts/splits/split_seed42.csv --embeddings-dir $D/embeddings \
    --tile-emb-root $D/tile_embeddings --families early late learned mil --max-trials 25 --no-improve-patience 8 --out-dir artifacts/agent_run_all
```
`scripts/slurm/abmil.sbatch` wraps the same commands for a SLURM cluster.

`notebooks/lung_agent.ipynb` has all of the above as runnable Colab cells in order.

## Repository layout

```
conf/               configuration
src/
  data/              subset selection, tiling, splits
  embeddings/        one module per foundation model + pooling (mean, ABMIL / multi-stream ABMIL)
  mil/               tile-bag loading, shared MIL trainer (curves, early stopping), per-FM config
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
- `artifacts/agent_run_abmil_cf/` — run on cross-fitted ABMIL embeddings from the FIRST ABMIL recipe (raw-feature attention, no projection/dropout; superseded).
- `artifacts/tuning/` — per-FM hyper-parameter search logs (02g); `conf/tuned/` — the selected values.

No API key is required to run the agent.
