#!/bin/bash
#SBATCH --job-name=sudoku10m
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
# Partition omitted; scheduler picks a valid partition (match KLASS/run_llada_math_train.slurm).
# This cluster requires an explicit GPU type in GRES.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${REPO}" ]] || [[ ! -f "${REPO}/scripts/train_sudoku_10m.sh" ]]; then
  REPO="/project/6086776/hongsunw/mdlm"
fi
if [[ ! -f "${REPO}/scripts/train_sudoku_10m.sh" ]]; then
  REPO="/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm"
fi
cd "${REPO}"
mkdir -p watch_folder
mkdir -p "${REPO}/wandb_offline"
echo "Working directory: $(pwd) Host: $(hostname) Job: ${SLURM_JOB_ID:-local}"
# W&B: GPU 노드는 오프라인만. 로그는 wandb_offline/ 아래에 쌓임. 끝난 뒤 통신되는 노드에서:
#   cd "${REPO}" && ls wandb_offline && wandb sync wandb_offline/offline-run-*

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO}/.venv/bin/activate"
elif [[ -f "/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm/.venv/bin/activate"
fi

export WANDB_MODE=offline
export WANDB_DIR="${REPO}/wandb_offline"
export TOKENIZERS_PARALLELISM=false
bash scripts/train_sudoku_10m.sh
echo "W&B offline dir (sync when online): ${REPO}/wandb_offline"
