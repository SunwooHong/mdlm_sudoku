#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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
  trainer.val_check_interval=5000 \
  trainer.log_every_n_steps=50 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  checkpointing.resume_from_ckpt=false \
  wandb.project=sudoku-mdlm \
  wandb.group=sudoku-v0 \
  wandb.name=sudoku-1m
