#!/bin/bash
#SBATCH --job-name=sudoku3m5m
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --open-mode=append

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO}" ]] || [[ ! -f "${REPO}/scripts/train_sudoku_3m_only_5m.sh" ]]; then
  REPO="/project/6086776/hongsunw/mdlm"
fi
if [[ ! -f "${REPO}/scripts/train_sudoku_3m_only_5m.sh" ]]; then
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
bash scripts/train_sudoku_3m_only_5m.sh
