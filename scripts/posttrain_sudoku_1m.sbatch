#!/bin/bash
#SBATCH --job-name=sudoku1m-post
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
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

RUN_TAG="${RUN_TAG:-tau2p0}"
OUT_DIR="${OUT_DIR:-${REPO}/obrbit_kto_outputs/sudoku_1m/${RUN_TAG}}"
mkdir -p "${OUT_DIR}"

python main.py \
  mode=orbit_kto \
  data=sudoku9-anchors \
  model=sudoku_1m \
  backbone=dit \
  parameterization=subs \
  model.length=81 \
  orbit_kto.init_checkpoint="${INIT_CKPT}" \
  optim.lr=1e-5 \
  orbit_kto.sample_steps=128 \
  orbit_kto.max_steps=5000 \
  orbit_kto.examples_per_refresh=1024 \
  orbit_kto.updates_per_refresh=200 \
  orbit_kto.batch_size=32 \
  orbit_kto.lambda_ac=0.3 \
  orbit_kto.tau=2.0 \
  orbit_kto.mc_samples_per_t=4 \
  orbit_kto.normalize_delta=true \
  orbit_kto.kappa_min=null \
  orbit_kto.save_every=1000 \
  orbit_kto.output_dir="${OUT_DIR}"

echo "Orbit-KTO output dir: ${OUT_DIR}"
echo "W&B offline dir: ${WANDB_DIR}"
