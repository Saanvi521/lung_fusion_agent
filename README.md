# Lung Adenocarcinoma Subtyping — Foundation-Model Fusion Agent

Predicts a 7-class lung adenocarcinoma histologic subtype from whole-slide images by fusing three pathology foundation models (UNI2-h, Virchow2, GenBio-PathFM) with patient age/sex, via an agent that searches fusion strategies. See `report.md` for the full write-up, results, and analysis.

## Setup

```bash
pip install -r requirements.txt
```

Designed to run on free-tier Google Colab/Kaggle GPU. Point Drive-dependent stages (`02a`, `02b`, `02g`, `02d`, `03`, `04`) at a mounted Drive path so embeddings persist across Colab sessions — see each script's own docstring for exact mount instructions.

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

Final (v3) run, exactly as in the notebook (`D` = data folder on Drive from 2a/2b, `S` = the split):
```bash
S=artifacts/splits/split_seed42.csv
python -m src.data.make_split --csv CLWD.csv --seed 42 --repo-id kmmuleelab/Lung_Pathology_Image_JPG --check-against $S
export ABMIL_TUNED_DIR=$D/tuned_v3          # where 02g writes / everything else reads the per-FM settings
for m in uni2h virchow2 genbio_pathfm; do
  python -m scripts.02a_tile_slides --split $S --out-dir $D --repo-id kmmuleelab/Lung_Pathology_Image_JPG   # once; skips done slides
  python -m scripts.02b_extract_embeddings --model $m --split $S --out-dir $D --save-tile-embeds
  python -m scripts.02g_tune_abmil --model $m --split $S --tile-emb-root $D/tile_embeddings --log-dir $D/tuning_v3
  python -m scripts.02d_crossfit_abmil --model $m --split $S --tile-emb-root $D/tile_embeddings --emb-out-dir $D/embeddings_abmil_v3/$m
done
python -m scripts.02e_plot_training_curve --history $D/embeddings_abmil_v3/*_history.json --out-dir $D/figures_v3
python -m scripts.03_run_agent --split $S --embeddings-dir $D/embeddings --tile-emb-root $D/tile_embeddings \
    --families mil --max-trials 20 --no-improve-patience 8 --final-seeds 3 --out-dir $D/agent_run_v3
python -m scripts.04_plot_fusion_variants --split $S --embeddings-dir $D/embeddings --tile-emb-root $D/tile_embeddings \
    --run-dir $D/agent_run_v3 --out-dir $D/figures_v3/fusion
python -m scripts.02e_plot_training_curve --history $D/agent_run_v3/curves/*_history.json --out-dir $D/figures_v3/final
```
The earlier full-search run (run 1, `--families early late learned mil --max-trials 25`) covered all four fusion families;
its results are kept in `artifacts/agent_run_v2_first_test_look/`. No API key is needed: without one the agent uses its
deterministic proposer.
`scripts/slurm/abmil.sbatch` wraps the same commands for a SLURM cluster.

`notebooks/lung_agent.ipynb` has all of the above as runnable Colab cells in order (Parts 0–8; Part 8 prints the report numbers and packages the results).

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

## Key results (committed)

| Path | What |
|---|---|
| `artifacts/splits/` | the fixed patient → split assignment (seed 42) |
| `artifacts/extraction_stats.json` | embedding size, wall-clock time and slide count per FM |
| `artifacts/tuning_v3/`, `conf/tuned/` | per-FM ABMIL search logs (02g) and the selected settings |
| `artifacts/abmil_v3/` | per-fold, per-epoch training histories of the per-FM ABMIL models (02d) |
| `artifacts/agent_run_v3/` | **final run**: `decision_log.jsonl` (every trial, score, accept/reject reason), `agent_summary.json`, `comparison_table.json` (test rows with bootstrap CIs, per-class AUROC, confusion matrices, seed spread, paired bootstrap), final-model and fusion-variant curves |
| `artifacts/agent_run_v2_first_test_look/` | run 1 (all four fusion families searched); its test numbers are reported alongside the final run |
| `artifacts/` (root), `artifacts/agent_run_abmil_cf/` | run 0 (mean pooling; first ABMIL recipe); superseded |
| `figures/` | per-FM training curves; `figures/final/` final-model curves; `figures/fusion/` agent search + fusion-variant curves |
| `requirements.lock` | exact package versions of the Colab environment the final run used |

No API key is required to run the agent.
