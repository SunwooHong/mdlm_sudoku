#!/bin/bash
#SBATCH --job-name=sudoku100m
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpubase_l40s_b5
#SBATCH --qos=high
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO}" ]] || [[ ! -f "${REPO}/scripts/train_sudoku_100m.sh" ]]; then
  REPO="/home/hongsunw/projects/aip-rahulgk/hongsunw/mdlm"
fi
cd "${REPO}"
mkdir -p watch_folder

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO}/.venv/bin/activate"
fi

export TOKENIZERS_PARALLELISM=false
bash scripts/train_sudoku_100m.sh
