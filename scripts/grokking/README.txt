Grokking pre-training scripts

- Core runner: `train_sudoku_grokking.sh`
- Per-combo wrappers (same naming pattern as baseline scripts):
  - `train_sudoku_3m_9m_{1m,3m,5m,10m,50m,100m}.{sh,sbatch}`
  - `train_sudoku_3m_only_{1m,3m,5m,10m,50m,100m}.{sh,sbatch}`

Defaults in core runner:
- `lr_scheduler=cosine_decay_warmup_2500`
- `optim.weight_decay=0.1`
- `trainer.gradient_clip_val=1.0`

Optional env overrides:
- `GROKKING_LR`
- `GROKKING_WEIGHT_DECAY`
- `GROKKING_GRAD_CLIP`
Grokking pre-training scripts
=============================

These scripts are intentionally separate from existing baseline scripts under `scripts/`.

Files:
- `train_sudoku_grokking.sh`
- `train_sudoku_grokking.sbatch`

Supported combinations:
- dataset: `3m-only`, `3m9m-mix`
- model_size: `1m`, `3m`, `5m`, `10m`, `50m`, `100m`

Default grokking-oriented changes vs baseline:
- `lr_scheduler=cosine_decay_warmup_2500`
- `optim.weight_decay=0.1`
- keep gradient clipping enabled (`trainer.gradient_clip_val=1.0`)

Quick start:

```bash
# interactive
bash scripts/grokking/train_sudoku_grokking.sh 3m-only 50m

# slurm
sbatch scripts/grokking/train_sudoku_grokking.sbatch 3m9m-mix 10m
```

Optional overrides via env:
- `GROKKING_LR` (override model default LR)
- `GROKKING_WEIGHT_DECAY` (default `0.1`)
- `GROKKING_GRAD_CLIP` (default `1.0`)
