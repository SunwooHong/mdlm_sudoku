#!/bin/bash

export HYDRA_FULL_ERROR=1

python main.py \
  mode=orbit_kto \
  data=sudoku9-anchors \
  model=sudoku_1m \
  backbone=dit \
  parameterization=subs \
  model.length=81 \
  eval.checkpoint_path=/path/to/sft_best.ckpt \
  orbit_kto.init_checkpoint=/path/to/sft_best.ckpt \
  orbit_kto.num_transforms=8 \
  orbit_kto.sample_steps=32 \
  orbit_kto.examples_per_refresh=2048 \
  orbit_kto.updates_per_refresh=1000 \
  orbit_kto.max_steps=10000 \
  orbit_kto.batch_size=64 \
  orbit_kto.tau=0.1 \
  orbit_kto.alpha_pos=1.0 \
  orbit_kto.alpha_neg=1.0 \
  orbit_kto.lambda_ac=0.1 \
  orbit_kto.output_dir=outputs/orbit_kto_sudoku1m
