#!/bin/bash
#SBATCH --job-name=sudoku50m-ac-eval
#SBATCH --output=watch_folder/%x_%j.out
#SBATCH --error=watch_folder/%x_%j.err
#SBATCH --account=aip-rahulgk
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --qos=high
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

DEFAULT_REPO="/home/hongsunw/projects/aip-rahulgk/hongsunw/mdlm"

resolve_repo() {
  local d
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    for d in "${SLURM_SUBMIT_DIR}" "$(cd "${SLURM_SUBMIT_DIR}/.." 2>/dev/null && pwd)"; do
      [[ -z "${d}" ]] && continue
      if d="$(cd "${d}" 2>/dev/null && pwd)" && [[ -f "${d}/scripts/sudoku_solve_eval_ac_orbit.py" ]]; then
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

CKPT="${CKPT:-${REPO}/outputs/sudoku9-solutions/temp_nibi/50M_3m_only_best.ckpt}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"     # maps to --max-puzzles
BATCH_SIZE="${BATCH_SIZE:-64}"
NSTEPS="${NSTEPS:-128}"
TEMP="${TEMP:-1.0}"
STOCH_K="${STOCH_K:-16}"               # maps to --samples-per-transform
BOOTSTRAP="${BOOTSTRAP:-1000}"
PSEUDO_GROUPS="${PSEUDO_GROUPS:-8}"
AC_K="${AC_K:-24}"                     # 24 or 32

NPY_ROOT="${NPY_ROOT:-dataset/3m_only_ac${AC_K}_eval_npy}"
TARGET_MASK="${TARGET_MASK:-${NPY_ROOT}/${SPLIT}_target_mask.npy}"
OUT_JSON="${OUT_JSON:-results/eval_3m_only_50M_ac${AC_K}_${SPLIT}_max${MAX_SAMPLES}_stochK${STOCH_K}.json}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Override with: CKPT=/path/to/best.ckpt sbatch $0" >&2
  exit 1
fi

if [[ ! -f "${REPO}/${TARGET_MASK}" && ! -f "${TARGET_MASK}" ]]; then
  echo "Target mask not found: ${TARGET_MASK}" >&2
  echo "Expected e.g. ${REPO}/dataset/3m_only_ac${AC_K}_eval_npy/${SPLIT}_target_mask.npy" >&2
  echo "You can override with: TARGET_MASK=/path/to/${SPLIT}_target_mask.npy sbatch $0" >&2
  exit 1
fi

python scripts/sudoku_solve_eval_ac_orbit.py \
  --checkpoint "${CKPT}" \
  --model sudoku_50m \
  --npy-root "${NPY_ROOT}" \
  --split "${SPLIT}" \
  --task infill \
  --eval-mode ac \
  --target-mask-npy "${TARGET_MASK}" \
  --max-puzzles "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-steps "${NSTEPS}" \
  --temperature "${TEMP}" \
  --samples-per-transform "${STOCH_K}" \
  --control-canonical-k "${STOCH_K}" \
  --success-metric target_exact_and_clue \
  --bootstrap-samples "${BOOTSTRAP}" \
  --save-npz \
  --pseudo-baseline-groups "${PSEUDO_GROUPS}" \
  --output-json "${OUT_JSON}"



