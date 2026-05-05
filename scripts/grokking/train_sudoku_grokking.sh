#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATASET="${1:-}"
MODEL_SIZE="${2:-}"
if [[ -z "${DATASET}" || -z "${MODEL_SIZE}" ]]; then
  echo "Usage: bash scripts/grokking/train_sudoku_grokking.sh <3m-only|3m9m-mix> <1m|3m|5m|10m|50m|100m>"
  exit 1
fi

export WANDB_MODE=offline
export WANDB_DIR="${ROOT}/wandb_offline"
mkdir -p "${WANDB_DIR}"
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API${PYTHONWARNINGS:+,${PYTHONWARNINGS}}"
export RUN_DATE="$(date +%Y%m%d)"
export RUN_SITE="nibi"

if [[ "${DATASET}" == "3m-only" ]]; then
  DATA_CONFIG="sudoku3m-only"
  OUT_DATASET="sudoku3m-only-grokking"
  WANDB_GROUP="sudoku-3m-only-grokking"
  RUN_DATASET="sudoku-3m-only-grokking"
elif [[ "${DATASET}" == "3m9m-mix" ]]; then
  DATA_CONFIG="sudoku3m9m-mix"
  OUT_DATASET="sudoku3m9m-mix-grokking"
  WANDB_GROUP="sudoku-3m9m-grokking"
  RUN_DATASET="sudoku-3m-9m-grokking"
else
  echo "Unknown dataset: ${DATASET}"
  exit 1
fi

case "${MODEL_SIZE}" in
  1m)
    MODEL_CFG="sudoku_1m"; RUN_MODEL="sudoku-1m"; BATCH=512; LR=5e-4
    if [[ "${DATASET}" == "3m-only" ]]; then MAX_STEPS=323340; BOUNDARY=32334; CKPT_EVERY=10000; else MAX_STEPS=1000000; BOUNDARY=100000; CKPT_EVERY=25000; fi
    ;;
  3m)
    MODEL_CFG="sudoku_3m"; RUN_MODEL="sudoku-3m"; BATCH=512; LR=4e-4
    if [[ "${DATASET}" == "3m-only" ]]; then MAX_STEPS=404170; BOUNDARY=40417; CKPT_EVERY=10000; else MAX_STEPS=1250000; BOUNDARY=125000; CKPT_EVERY=25000; fi
    ;;
  5m)
    MODEL_CFG="sudoku_5m"; RUN_MODEL="sudoku-5m"; BATCH=512; LR=3e-4
    if [[ "${DATASET}" == "3m-only" ]]; then MAX_STEPS=485000; BOUNDARY=48500; CKPT_EVERY=10000; else MAX_STEPS=1500000; BOUNDARY=150000; CKPT_EVERY=25000; fi
    ;;
  10m)
    MODEL_CFG="sudoku_10m"; RUN_MODEL="sudoku-10m"; BATCH=512; LR=3e-4
    if [[ "${DATASET}" == "3m-only" ]]; then MAX_STEPS=646670; BOUNDARY=64667; CKPT_EVERY=10000; else MAX_STEPS=2000000; BOUNDARY=200000; CKPT_EVERY=25000; fi
    ;;
  50m)
    MODEL_CFG="sudoku_50m"; RUN_MODEL="sudoku-50m"; BATCH=512; LR=2e-4
    if [[ "${DATASET}" == "3m-only" ]]; then MAX_STEPS=970000; BOUNDARY=97000; CKPT_EVERY=10000; else MAX_STEPS=3000000; BOUNDARY=300000; CKPT_EVERY=50000; fi
    ;;
  100m)
    MODEL_CFG="sudoku_100m"; RUN_MODEL="sudoku-100m"; BATCH=256; LR=1.5e-4
    if [[ "${DATASET}" == "3m-only" ]]; then MAX_STEPS=1293340; BOUNDARY=129334; CKPT_EVERY=10000; else MAX_STEPS=4000000; BOUNDARY=400000; CKPT_EVERY=50000; fi
    ;;
  *)
    echo "Unknown model_size: ${MODEL_SIZE}"
    exit 1
    ;;
esac

GROK_LR="${GROKKING_LR:-$LR}"
GROK_WD="${GROKKING_WEIGHT_DECAY:-0.1}"
GROK_CLIP="${GROKKING_GRAD_CLIP:-1.0}"
export RUN_PREFIX="${RUN_DATE}-${RUN_SITE}"
export RUN_NAME="${RUN_PREFIX}-${RUN_DATASET}-${RUN_MODEL}"

python main.py \
  model="${MODEL_CFG}" \
  data="${DATA_CONFIG}" \
  backbone=dit \
  parameterization=subs \
  model.length=81 \
  loader.global_batch_size="${BATCH}" \
  loader.eval_global_batch_size="${BATCH}" \
  loader.num_workers=4 \
  loader.pin_memory=true \
  optim.lr="${GROK_LR}" \
  optim.weight_decay="${GROK_WD}" \
  training.ema=0.9999 \
  lr_scheduler=cosine_decay_warmup_2500 \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.gradient_clip_val="${GROK_CLIP}" \
  trainer.val_check_interval=1.0 \
  +trainer.check_val_every_n_epoch=5 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${CKPT_EVERY}" \
  trainer.log_every_n_steps=50 \
  +callbacks.phase_split_checkpoint._target_=phase_checkpoint.PhaseSplitCheckpoint \
  +callbacks.phase_split_checkpoint.boundary_step="${BOUNDARY}" \
  +callbacks.phase_split_checkpoint.monitor=val/nll \
  +callbacks.phase_split_checkpoint.mode=min \
  '+callbacks.phase_split_checkpoint.dirpath=${checkpointing.save_dir}/checkpoints' \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  checkpointing.resume_from_ckpt=false \
  "hydra.run.dir=\${mdlm_root:}/outputs/${OUT_DATASET}/\${oc.env:RUN_MODEL}/\${now:%Y.%m.%d}/\${now:%H%M%S}" \
  wandb.project=sudoku-mdlm \
  wandb.group="${WANDB_GROUP}" \
  wandb.name="${RUN_NAME}"
