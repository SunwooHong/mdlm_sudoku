#!/usr/bin/env bash
set -euo pipefail

python main.py \
  mode=oscar \
  model=sudoku_1m \
  data=sudoku9-anchors \
  loader.num_workers=4 \
  loader.pin_memory=true \
  eval.generate_samples=false \
  oscar.init_checkpoint=/path/to/sft_best.ckpt \
  oscar.num_transforms=8 \
  oscar.lambda_eq=1.0 \
  oscar.lambda_prox=0.05 \
  oscar.lambda_vicreg=0.0 \
  wandb.project=my-sudoku-oscar \
  wandb.name=oscar_sudoku_1m
