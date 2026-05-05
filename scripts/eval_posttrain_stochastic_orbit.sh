#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Usage:
#   bash scripts/eval_posttrain_stochastic_orbit.sh \
#     <checkpoint_pt> <model> <eval_npy_root> <output_json> [max_puzzles] [batch_size]
#
# Example:
#   bash scripts/eval_posttrain_stochastic_orbit.sh \
#     outputs/posttrain/3m_only_50m/canonical_sft_80k/best.pt \
#     sudoku_50m dataset/3m_only_eval_npy \
#     results/posttrain_3m_only_50m_canonical_sft80k_stochK16.json \
#     500 64

CHECKPOINT="${1:-}"
MODEL="${2:-sudoku_50m}"
NPY_ROOT="${3:-dataset/3m_only_eval_npy}"
OUTPUT_JSON="${4:-results/posttrain_stochastic_orbit.json}"
MAX_PUZZLES="${5:-500}"
BATCH_SIZE="${6:-64}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: missing post-train checkpoint path"
  echo "Usage: bash scripts/eval_posttrain_stochastic_orbit.sh <checkpoint_pt> <model> <eval_npy_root> <output_json> [max_puzzles] [batch_size]"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")"

python scripts/sudoku_solve_eval_stochastic_orbit.py \
  --checkpoint "${CHECKPOINT}" \
  --model "${MODEL}" \
  --npy-root "${NPY_ROOT}" \
  --split test \
  --task infill \
  --max-puzzles "${MAX_PUZZLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-steps 128 \
  --temperature 1.0 \
  --samples-per-transform 16 \
  --control-canonical-k 16 \
  --control-canonical-pseudo-orbit \
  --success-metric exact \
  --bootstrap-samples 1000 \
  --save-npz \
  --output-json "${OUTPUT_JSON}"
