#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export WANDB_MODE=offline
export WANDB_DIR="${ROOT}/wandb_offline"
mkdir -p "${WANDB_DIR}"
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"

export RUN_DATE="$(date +%Y%m%d)"
export RUN_SITE="nibi"
export RUN_DATASET="sudoku-3m-only"
export RUN_MODEL="sudoku-10m"
export RUN_PREFIX="${RUN_DATE}-${RUN_SITE}"
export RUN_NAME="${RUN_PREFIX}-${RUN_DATASET}-${RUN_MODEL}"

python main.py \
  model=sudoku_10m \
  data=sudoku3m-only \
  backbone=dit \
  parameterization=subs \
  model.length=81 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=512 \
  loader.num_workers=4 \
  loader.pin_memory=true \
  optim.lr=3e-4 \
  training.ema=0.9999 \
  trainer.max_steps=646670 \
  trainer.val_check_interval=1.0 \
  +trainer.check_val_every_n_epoch=5 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  trainer.log_every_n_steps=50 \
  +callbacks.phase_split_checkpoint._target_=phase_checkpoint.PhaseSplitCheckpoint \
  +callbacks.phase_split_checkpoint.boundary_step=64667 \
  +callbacks.phase_split_checkpoint.monitor=val/nll \
  +callbacks.phase_split_checkpoint.mode=min \
  '+callbacks.phase_split_checkpoint.dirpath=${checkpointing.save_dir}/checkpoints' \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  checkpointing.resume_from_ckpt=false \
  'hydra.run.dir=${mdlm_root:}/outputs/sudoku3m-only/${oc.env:RUN_MODEL}/${now:%Y.%m.%d}/${now:%H%M%S}' \
  wandb.project=sudoku-mdlm \
  wandb.group=sudoku-3m-only-v0 \
  wandb.name=${RUN_NAME} \
