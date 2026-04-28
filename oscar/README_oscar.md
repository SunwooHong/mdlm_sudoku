# OSCAR-lite for `mdlm_sudoku`

This bundle implements a stable, LE-directed post-training recipe for the
`SunwooHong/mdlm_sudoku` repository.

## Files
- `oscar_orbit.py`: Lightning-compatible OSCAR trainer built by subclassing `diffusion.Diffusion`
- `sudoku_orbit.py`: exact D4 transforms and inverse transforms for 9x9 Sudoku
- `main_oscar.patch`: minimal patch to route `mode=oscar`
- `oscar_config_snippet.yaml`: config block to append to `configs/config.yaml`
- `sudoku9-anchors.yaml`: data config using the repo's existing `sudoku9-anchors` dataset
- `train_sudoku_oscar.sh`: example launch script

## What the implementation does
OSCAR-lite optimizes
1. the base anchor-aware diffusion loss,
2. an orbit-score variance penalty across transformed views of the same logical state,
3. a proximal KL anchor to a frozen reference backbone,
4. optional logit-space VICReg alignment (disabled by default).

## Important repo assumptions
- `data=sudoku9-anchors` is available through `dataloader.get_dataset(...)`.
- `SudokuNpyDataset` returns `input_ids`, `attention_mask`, and `anchor_mask`.
- `Diffusion._loss(...)` already supports `anchor_mask`.
- Reverse diffusion preserves non-mask tokens, so anchors remain fixed.

## Recommended first run
Start with
- `oscar.lambda_eq = 1.0`
- `oscar.lambda_prox = 0.05`
- `oscar.lambda_vicreg = 0.0`

Then turn on VICReg only after confirming stable loss and non-degenerate orbit metrics.

## Caveat
This implementation targets the stable core of OSCAR. It directly regularizes
cross-transform score spread and uses a frozen-reference proximal term. The
optional VICReg term uses canonicalized log-prob tensors as a practical
representation proxy because the repo does not expose backbone hidden states in
its current public interface.
