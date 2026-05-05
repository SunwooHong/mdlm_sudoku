#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Usage:
#   bash scripts/posttrain_sudoku_recipes.sh \
#     <checkpoint> <model> <npy_root> <recipe> <output_dir> [batch_size] [lr] [warmup_steps] [max_steps]
#
# Example:
#   bash scripts/posttrain_sudoku_recipes.sh \
#     outputs/sudoku3m-only/sudoku-50m/.../checkpoints/best.ckpt \
#     sudoku_50m dataset/3m_only_posttrain_npy canonical_sft_80k \
#     outputs/posttrain/3m_only_50m/canonical_sft_80k 512 1e-5 150 3000

CHECKPOINT="${1:-}"
MODEL="${2:-sudoku_50m}"
NPY_ROOT="${3:-dataset/3m_only_posttrain_npy}"
RECIPE="${4:-canonical_sft_80k}"
OUTPUT_DIR="${5:-outputs/posttrain/3m_only_50m/canonical_sft_80k}"
BATCH_SIZE="${6:-512}"
LR="${7:-1e-5}"
WARMUP_STEPS="${8:-150}"
MAX_STEPS="${9:-3000}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: missing checkpoint path"
  echo "Usage: bash scripts/posttrain_sudoku_recipes.sh <checkpoint> <model> <npy_root> <recipe> <output_dir> [batch_size] [lr] [warmup_steps] [max_steps]"
  exit 1
fi

export WANDB_MODE=offline
export WANDB_DIR="${ROOT}/wandb_offline"
mkdir -p "${WANDB_DIR}" "${OUTPUT_DIR}"
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"

python scripts/sudoku_posttrain_sft.py \
  --checkpoint "${CHECKPOINT}" \
  --model "${MODEL}" \
  --npy-root "${NPY_ROOT}" \
  --recipe "${RECIPE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --eval-every 500 \
  --save-every 1000 \
  --output-dir "${OUTPUT_DIR}"
