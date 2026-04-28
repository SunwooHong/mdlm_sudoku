#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Weights & Biases: no network on GPU nodes — log locally, upload later from a login node:
#   cd "$ROOT" && wandb sync wandb_offline/offline-run-<id>
# Or: find wandb_offline -maxdepth 1 -type d -name 'offline-run-*' -exec wandb sync {} \;
export WANDB_MODE=offline
export WANDB_DIR="${ROOT}/wandb_offline"
mkdir -p "${WANDB_DIR}"

# Lightning 2.2 still imports pkg_resources; setuptools 80.x warns until Lightning drops it.
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"

export RUN_DATE="$(date +%Y%m%d)"
export RUN_SITE="nibi"
export RUN_MODEL="sudoku-1m"
export RUN_PREFIX="${RUN_DATE}-${RUN_SITE}"
export RUN_NAME="${RUN_PREFIX}-${RUN_MODEL}"

python main.py \
  model=sudoku_1m \
  data=sudoku9-solutions \
  backbone=dit \
  parameterization=subs \
  model.length=81 \
  loader.global_batch_size=512 \
  loader.eval_global_batch_size=512 \
  loader.num_workers=4 \
  loader.pin_memory=true \
  optim.lr=5e-4 \
  training.ema=0.9999 \
  trainer.max_steps=100000 \
  trainer.val_check_interval=1.0 \
  +trainer.check_val_every_n_epoch=5 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=50000 \
  trainer.log_every_n_steps=50 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  checkpointing.resume_from_ckpt=false \
  'hydra.run.dir=${mdlm_root:}/outputs/sudoku9-solutions/${oc.env:RUN_MODEL}/${now:%Y.%m.%d}/${now:%H%M%S}' \
  wandb.project=sudoku-mdlm \
  wandb.group=sudoku-v0 \
  wandb.name=${RUN_NAME} \
