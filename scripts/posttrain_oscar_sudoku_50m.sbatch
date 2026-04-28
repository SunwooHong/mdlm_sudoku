#!/bin/bash
#SBATCH --job-name=sudoku50m-oscar
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=20:00:00
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

INIT_CKPT="${INIT_CKPT:-/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm/outputs/sudoku9-solutions/sudoku-50m/2026.04.25/005045/checkpoints/best.ckpt}"
if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Checkpoint not found: ${INIT_CKPT}" >&2
  echo "Override with: INIT_CKPT=/path/to/best.ckpt sbatch $0" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-default}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${REPO}/oscar_lite_outputs/sudoku_50m/${STAMP}}"
export OUT_DIR
mkdir -p "${OUT_DIR}"
GLOBAL_BS="${GLOBAL_BS:-16}"
N_TRANSFORMS="${N_TRANSFORMS:-2}"
T_GRID="${T_GRID:-[0.35,0.6]}"
MAX_STEPS="${MAX_STEPS:-20000}"

python main.py \
  mode=oscar \
  model=sudoku_50m \
  data=sudoku9-anchors \
  eval.generate_samples=false \
  loader.global_batch_size="${GLOBAL_BS}" \
  loader.eval_global_batch_size="${GLOBAL_BS}" \
  oscar.init_checkpoint="${INIT_CKPT}" \
  oscar.num_transforms="${N_TRANSFORMS}" \
  "oscar.t_grid=${T_GRID}" \
  oscar.lambda_eq=1.0 \
  oscar.lambda_prox=0.05 \
  oscar.lambda_vicreg=0.0 \
  trainer.precision=bf16-mixed \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval=1000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000 \
  'hydra.run.dir=${oc.env:OUT_DIR}'

echo "OSCAR output dir: ${OUT_DIR}"
echo "W&B offline dir: ${WANDB_DIR}"
