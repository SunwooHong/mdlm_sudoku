#!/bin/bash
#SBATCH --job-name=grok39m5m
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO}" ]] || [[ ! -f "${REPO}/scripts/grokking/train_sudoku_3m_9m_5m.sh" ]]; then REPO="/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm"; fi
cd "${REPO}"; mkdir -p watch_folder "${REPO}/wandb_offline"
if [[ -f "${REPO}/.venv/bin/activate" ]]; then source "${REPO}/.venv/bin/activate"; fi
export WANDB_MODE=offline WANDB_DIR="${REPO}/wandb_offline" TOKENIZERS_PARALLELISM=false
bash scripts/grokking/train_sudoku_3m_9m_5m.sh
