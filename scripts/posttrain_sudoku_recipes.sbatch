#!/bin/bash
#SBATCH --job-name=posttrain-sudoku
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --open-mode=append

set -euo pipefail

# Positional args forwarded to scripts/posttrain_sudoku_recipes.sh:
# 1 checkpoint (required)
# 2 model
# 3 npy_root
# 4 recipe
# 5 output_dir
# 6 batch_size
# 7 lr
# 8 warmup_steps
# 9 max_steps

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO}" ]] || [[ ! -f "${REPO}/scripts/posttrain_sudoku_recipes.sh" ]]; then
  REPO="/project/6086776/hongsunw/mdlm"
fi
if [[ ! -f "${REPO}/scripts/posttrain_sudoku_recipes.sh" ]]; then
  REPO="/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm"
fi

cd "${REPO}"
mkdir -p watch_folder
mkdir -p "${REPO}/wandb_offline"

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  source "${REPO}/.venv/bin/activate"
elif [[ -f "/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm/.venv/bin/activate" ]]; then
  source "/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm/.venv/bin/activate"
fi

export WANDB_MODE=offline
export WANDB_DIR="${REPO}/wandb_offline"
export TOKENIZERS_PARALLELISM=false

bash scripts/posttrain_sudoku_recipes.sh "$@"
