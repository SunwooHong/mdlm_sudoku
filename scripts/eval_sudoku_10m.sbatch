#!/bin/bash
#SBATCH --job-name=sudoku10m-eval
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

# Prefer submit cwd (repo root or scripts/), then fixed clone path.
resolve_repo() {
  local d
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    for d in "${SLURM_SUBMIT_DIR}" "$(cd "${SLURM_SUBMIT_DIR}/.." 2>/dev/null && pwd)"; do
      [[ -z "${d}" ]] && continue
      if d="$(cd "${d}" 2>/dev/null && pwd)" && [[ -f "${d}/scripts/sudoku_solve_eval.py" ]]; then
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

CKPT="${CKPT:-${REPO}/outputs/sudoku9-solutions/2026.04.23/220826/checkpoints/best.ckpt}"
NPUZZLES="${NPUZZLES:-449855}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NSTEPS="${NSTEPS:-128}"
SPLIT="${SPLIT:-test}"
RUN_SITE="${RUN_SITE:-nibi}"
RUN_SIZE="${RUN_SIZE:-10M}"
OUT_JSON="${OUT_JSON:-results/sudoku_${RUN_SIZE}_${RUN_SITE}_infill_d4_${SPLIT}_n${NPUZZLES}.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Override with: CKPT=/path/to/best.ckpt sbatch $0" >&2
  exit 1
fi

python scripts/sudoku_solve_eval.py \
  --checkpoint "${CKPT}" \
  --model sudoku_10m \
  --npy-root dataset/sudoku_9m_npy \
  --split "${SPLIT}" \
  --task infill \
  --max-puzzles "${NPUZZLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-steps "${NSTEPS}" \
  --output-json "${OUT_JSON}"
