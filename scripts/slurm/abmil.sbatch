#!/bin/bash
# SLURM template (Digital Research Alliance clusters, e.g. Fir / Trillium-GPU).
# Edit --account and the paths, then:  sbatch scripts/slurm/abmil.sbatch
#SBATCH --account=def-CHANGEME
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=lung-abmil
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load python/3.11 cuda   # adjust to the cluster's module names
source "$HOME/venvs/lung/bin/activate"
cd "$SLURM_SUBMIT_DIR"

D=${D:-$SCRATCH/lung_fusion_data}           # contains tile_embeddings/ and embeddings/ from 02b
SPLIT=artifacts/splits/split_seed42.csv

for m in uni2h virchow2 genbio_pathfm; do
  python -m scripts.02g_tune_abmil --model $m --split $SPLIT --tile-emb-root $D/tile_embeddings
  python -m scripts.02d_crossfit_abmil --model $m --split $SPLIT --tile-emb-root $D/tile_embeddings \
      --emb-out-dir $D/embeddings_abmil_cf/$m
done
python -m scripts.02e_plot_training_curve --history $D/embeddings_abmil_cf/*_history.json --out-dir figures
python -m scripts.03_run_agent --split $SPLIT --embeddings-dir $D/embeddings --tile-emb-root $D/tile_embeddings \
    --families early late learned mil --max-trials 25 --no-improve-patience 8 --out-dir artifacts/agent_run_all --final-seeds 3
