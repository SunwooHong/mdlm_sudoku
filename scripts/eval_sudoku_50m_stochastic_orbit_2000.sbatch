#!/bin/bash
#SBATCH --job-name=sudoku50m-stoch-eval
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=def-rahulgk
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

DEFAULT_REPO="/home/hongsunw/projects/def-rahulgk/hongsunw/mdlm"

# Prefer submit cwd (repo root or scripts/), then fixed clone path.
resolve_repo() {
  local d
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    for d in "${SLURM_SUBMIT_DIR}" "$(cd "${SLURM_SUBMIT_DIR}/.." 2>/dev/null && pwd)"; do
      [[ -z "${d}" ]] && continue
      if d="$(cd "${d}" 2>/dev/null && pwd)" && [[ -f "${d}/scripts/sudoku_solve_eval_stochastic_orbit.py" ]]; then
        printf '%s\n' "${d}"
        return 0
      fi
    done
  fi
  printf '%s\n' "$(cd "${DEFAULT_REPO}" && pwd)"
}

REPO="$(resolve_repo)"
cd "${REPO}"
mkdir -p watch_folder results

if [[ -f "${REPO}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO}/.venv/bin/activate"
fi

export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-${REPO}/outputs/sudoku3m-only/sudoku-50m/2026.04.28/052903/checkpoints/best.ckpt}"
NPUZZLES="${NPUZZLES:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NSTEPS="${NSTEPS:-128}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAMPLES_PER_TRANSFORM="${SAMPLES_PER_TRANSFORM:-16}"
CONTROL_CANONICAL_K="${CONTROL_CANONICAL_K:-16}"
SUCCESS_METRIC="${SUCCESS_METRIC:-exact}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
PSEUDO_BASELINE_GROUPS="${PSEUDO_BASELINE_GROUPS:-8}"
SPLIT="${SPLIT:-test}"
RUN_SITE="${RUN_SITE:-nibi}"
RUN_SIZE="${RUN_SIZE:-50M}"
OUT_JSON="${OUT_JSON:-results/eval_3m_only_${RUN_SIZE}_${SPLIT}_stochK${CONTROL_CANONICAL_K}_n${NPUZZLES}.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Override with: CKPT=/path/to/best.ckpt sbatch $0" >&2
  exit 1
fi

python scripts/sudoku_solve_eval_stochastic_orbit.py \
  --checkpoint "${CKPT}" \
  --model sudoku_50m \
  --npy-root dataset/3m_only_eval_npy \
  --split "${SPLIT}" \
  --task infill \
  --max-puzzles "${NPUZZLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-steps "${NSTEPS}" \
  --temperature "${TEMPERATURE}" \
  --samples-per-transform "${SAMPLES_PER_TRANSFORM}" \
  --control-canonical-k "${CONTROL_CANONICAL_K}" \
  --success-metric "${SUCCESS_METRIC}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --save-npz \
  --pseudo-baseline-groups "${PSEUDO_BASELINE_GROUPS}" \
  --output-json "${OUT_JSON}"
