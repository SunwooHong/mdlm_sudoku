#!/bin/bash
#SBATCH --job-name=sudoku1m-oscar
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

DEFAULT_REPO="/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm"

resolve_repo() {
  local d
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    for d in "${SLURM_SUBMIT_DIR}" "$(cd "${SLURM_SUBMIT_DIR}/.." 2>/dev/null && pwd)"; do
      [[ -z "${d}" ]] && continue
      if d="$(cd "${d}" 2>/dev/null && pwd)" && [[ -f "${d}/main.py" ]]; then
        printf '%s\n' "${d}"
        return 0
      fi
    done
  fi
  printf '%s\n' "$(cd "${DEFAULT_REPO}" && pwd)"
}

REPO="$(resolve_repo)"
cd "${REPO}"
mkdir -p watch_folder
mkdir -p "${REPO}/wandb_offline"

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO}/.venv/bin/activate"
fi

export WANDB_MODE=offline
export WANDB_DIR="${REPO}/wandb_offline"
export TOKENIZERS_PARALLELISM=false

INIT_CKPT="${INIT_CKPT:-${REPO}/outputs/sudoku9-solutions/sudoku-1m/2026.04.25/004255/checkpoints/best.ckpt}"
if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Checkpoint not found: ${INIT_CKPT}" >&2
  echo "Override with: INIT_CKPT=/path/to/best.ckpt sbatch $0" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-default}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${REPO}/oscar_lite_outputs/sudoku_1m/${STAMP}}"
export OUT_DIR
mkdir -p "${OUT_DIR}"
MAX_STEPS="${MAX_STEPS:-100000}"

python main.py \
  mode=oscar \
  model=sudoku_1m \
  data=sudoku9-anchors \
  eval.generate_samples=false \
  oscar.init_checkpoint="${INIT_CKPT}" \
  oscar.num_transforms=4 \
  oscar.lambda_eq=1.0 \
  oscar.lambda_prox=0.05 \
  oscar.lambda_vicreg=0.0 \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval=1000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  'hydra.run.dir=${oc.env:OUT_DIR}'

echo "OSCAR output dir: ${OUT_DIR}"
echo "W&B offline dir: ${WANDB_DIR}"
