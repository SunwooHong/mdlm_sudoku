# Orbit-KTO integration for `mdlm_sudoku`

Included files:
- `orbit_kto.py`: semi-online Orbit-KTO trainer
- `sudoku_orbit.py`: D4 transforms and orbit-aware weighting
- `main_orbit_kto.patch`: minimal `main.py` patch (`mode=orbit_kto`)
- `orbit_kto_config_snippet.yaml`: config block to merge into `configs/config.yaml`
- `sudoku9-anchors.yaml`: new data config for anchored Sudoku
- `train_sudoku_orbit_kto.sh`: example launch command

Integration steps:
1. Copy `orbit_kto.py` and `sudoku_orbit.py` into the repository root.
2. Apply `main_orbit_kto.patch` to `main.py`.
3. Add the Orbit-KTO config block from `orbit_kto_config_snippet.yaml` to `configs/config.yaml`.
4. Add `configs/data/sudoku9-anchors.yaml`.
5. Launch with `mode=orbit_kto` and `data=sudoku9-anchors`.

Notes:
- This implementation is single-process / single-GPU oriented.
- It reuses the repo's `Diffusion` model, anchor-aware loss, and reverse samplers.
- It avoids changing `diffusion.py` by wrapping conditioned sampling and fixed-t scoring externally.
