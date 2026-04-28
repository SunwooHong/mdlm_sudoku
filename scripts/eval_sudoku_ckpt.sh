#!/usr/bin/env bash
# Run validation NLL (same as training val) on a saved checkpoint.
#
# Usage:
#   bash scripts/eval_sudoku_ckpt.sh /path/to/model.ckpt [model_config_name]
#
# Examples (paths from your run):
#   bash scripts/eval_sudoku_ckpt.sh \
#     /project/6101781/hongsunw/mdlm/outputs/sudoku9-solutions/2026.04.23/212732/checkpoints/best.ckpt
#   bash scripts/eval_sudoku_ckpt.sh \
#     /project/6101781/hongsunw/mdlm/outputs/sudoku9-solutions/2026.04.23/212732/checkpoints/epoch=0-step=10000.ckpt \
#     sudoku_5m
#
# Single-GPU: set TRAINER_DEVICES=1 and matching batch sizes, e.g.
#   TRAINER_DEVICES=1 GBS=512 bash scripts/eval_sudoku_ckpt.sh /path/to.ckpt sudoku_5m

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${1:?usage: $0 /path/to/checkpoint.ckpt [model_yaml e.g. sudoku_5m]}"
MODEL="${2:-sudoku_5m}"

: "${TRAINER_DEVICES:=2}"
: "${GBS:=512}"

# batch_size * devices * accumulate == global_batch_size (accumulate defaults to 1)
BS=$((GBS / TRAINER_DEVICES))

export WANDB_MODE="${WANDB_MODE:-disabled}"

python main.py \
  mode=ppl_eval \
  "eval.checkpoint_path=$CKPT" \
  model="$MODEL" \
  data=sudoku9-solutions \
  backbone=dit \
  parameterization=subs \
  model.length=81 \
  loader.global_batch_size="$GBS" \
  loader.eval_global_batch_size="$GBS" \
  loader.batch_size="$BS" \
  loader.eval_batch_size="$BS" \
  loader.num_workers="${NUM_WORKERS:-0}" \
  loader.pin_memory=true \
  trainer.devices="$TRAINER_DEVICES" \
  trainer.num_sanity_val_steps=0 \
  checkpointing.resume_from_ckpt=false
