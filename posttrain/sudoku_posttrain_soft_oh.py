#!/usr/bin/env python3
"""Post-train MDLM Sudoku with SoftOH (smooth worst-orbit) + optional ordinary SFT mix.

Orbit-coupled batches: ``orbit_batch_size = B * 8`` rows share the same base puzzle
per block of 8, with identical corruption (canonical blank mask + noise level ``t``)
propagated by D4. Per-view token-mean CE is computed after pulling logits back to
canonical coordinates, forming ``ell[b, g]``. Group loss::

  mu = mean_g ell
  L_soft = (logsumexp(alpha * ell) - log(8)) / alpha
  L_group = (1 - lambda_eff) * mu + lambda_eff * L_soft

``lambda_eff`` linearly warms from 0 to ``--soft-oh-lambda`` over the first
``--lambda-warmup-frac`` of **optimizer updates** (after ``optimizer.step()``), not
micro-batches. LR cosine schedule uses the same optimizer-update horizon.

``--max-steps`` counts **micro-batches** (one ``backward`` per loop iteration). After
training, a **partial grad-accum buffer is flushed** with one extra ``optimizer.step()``
when ``max_steps % grad_accum_steps != 0``, so optimizer updates =
``ceil(max_steps / grad_accum_steps)``. LR cosine and λ warmup use that horizon.
Prefer ``--grad-accum-steps 1`` unless you intend partial windows.

Use ``--fair-method`` for the controlled ablations:
  1. ``sft80k``: 80k unique canonical SFT, independent random masks.
  2. ``repeat10k_x8``: 10k canonical puzzles repeated 8x, independent random masks.
  3. ``d4_shuffle10k_indmask``: 10k×8 D4 rows shuffled; transform first, then independent random masks (3A).
  4. ``d4_grouped_avg10k``: 10k×8 orbit groups; sample one canonical mask, transform it, grouped average CE.
  5. ``softoh10k``: same grouped orbit states as 4, but SoftOH loss.
  6. ``d4_grouped_different_random``: same orbit grouping as 4, but independent random mask + ``t`` per D4 view.
  7. ``d4_grouped_different_random_softoh``: same corruption as 6, SoftOH loss.

**Checkpoint selection:** ``best.ckpt`` defaults to lowest **canonical** ``val_sft_loss``,
which is not macro-All-8 / LSR. For orbit-aware selection use ``--best-checkpoint-key
orbit_group_loss`` (validation mean ``L_group`` on orbit batches). For paper metrics,
still re-rank ``step_*.ckpt`` with your external orbit evaluator.

Expected npy files under ``--npy-root``:
  train_solution.npy   [N, 81] int ids 0..8
  train_anchor.npy     [N, 81] bool/int; True where clues are visible

Example::

  python posttrain/sudoku_posttrain_soft_oh.py \
    --checkpoint outputs/.../best.ckpt \
    --model sudoku_50m \
    --npy-root dataset/3m_only_posttrain_npy \
    --fair-method softoh10k \
    --batch-size 512 \
    --max-steps 3000 \
    --output-dir outputs/posttrain/softoh10k
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

# Repo imports (run with cwd = mdlm root, or PYTHONPATH=.)
MDLM_ROOT = Path(__file__).resolve().parent.parent
if str(MDLM_ROOT) not in sys.path:
  sys.path.insert(0, str(MDLM_ROOT))

import diffusion as diffusion_mod  # noqa: E402
import dataloader as dataloader_mod  # noqa: E402


TRANSFORMS = [
  'identity',
  'rot90',
  'rot180',
  'rot270',
  'flip_lr',
  'flip_ud',
  'transpose',
  'anti_diagonal',
]

RECIPE_DEFAULTS = {
  # 1. 80k unique canonical SFT rows. Random masking is applied inside
  #    _sft_loss_full_blank on every row.
  'canonical_sft_80k': dict(num_base_examples=80_000, orbit_expand=False, random_transform=False),
  # Extra convenience recipe: 80k unique rows, one random D4 view per row.
  'random_transform_sft_80k': dict(num_base_examples=80_000, orbit_expand=False, random_transform=True),
  # 3A. 10k base puzzles expanded to 8 transformed rows, then globally shuffled.
  #     Random masking happens AFTER the row transform inside _sft_loss_full_blank,
  #     so the 8 views do not share the same masked state.
  'full_orbit_sft_10k': dict(num_base_examples=10_000, orbit_expand=True, random_transform=False),
  # 2. 10k canonical base puzzles repeated 8 times. Each repeat is independently
  #    random-masked by _sft_loss_full_blank.
  'repeat8_sft_10k': dict(
      num_base_examples=10_000,
      orbit_expand=False,
      random_transform=False,
      repeat_factor=8,
  ),
}


FAIR_METHODS = {
  # Use --fair-method to avoid accidentally mixing data construction, masking order,
  # and objective. All 10k variants use the same selected 10k base indices when the
  # same --subset-mode/--subset-seed are used.
  'sft80k': dict(
      recipe='canonical_sft_80k',
      num_base_examples=80_000,
      r_orbit=0.0,
      orbit_only=False,
      masking_policy='80k canonical rows; independent random mask per row',
  ),
  'repeat10k_x8': dict(
      recipe='repeat8_sft_10k',
      num_base_examples=10_000,
      r_orbit=0.0,
      orbit_only=False,
      masking_policy='10k canonical rows repeated 8x; independent random mask per repeat',
  ),
  'd4_shuffle10k_indmask': dict(
      recipe='full_orbit_sft_10k',
      num_base_examples=10_000,
      r_orbit=0.0,
      orbit_only=False,
      masking_policy='D4 transform first, global shuffle, then independent random mask per transformed row (3A)',
  ),
  'd4_grouped_avg10k': dict(
      recipe='full_orbit_sft_10k',
      num_base_examples=10_000,
      r_orbit=1.0,
      orbit_only=True,
      soft_oh_alpha=0.0,
      soft_oh_lambda=0.0,
      orbit_sample_mode='sequential_drop_last',
      masking_policy='one canonical random mask per base state, then D4-transform that same mask; grouped average CE',
  ),
  'softoh10k': dict(
      recipe='full_orbit_sft_10k',
      num_base_examples=10_000,
      r_orbit=1.0,
      orbit_only=True,
      orbit_sample_mode='sequential_drop_last',
      masking_policy='same grouped states as d4_grouped_avg10k; SoftOH loss instead of grouped average CE',
  ),
  'd4_grouped_different_random': dict(
      recipe='full_orbit_sft_10k',
      num_base_examples=10_000,
      r_orbit=1.0,
      orbit_only=True,
      soft_oh_alpha=0.0,
      soft_oh_lambda=0.0,
      orbit_sample_mode='sequential_drop_last',
      orbit_mask_coupling='per_view',
      masking_policy=(
          '10k×8 orbit groups; independent random mask + t per D4 view (still one forward per view); grouped average CE'
      ),
  ),
  'd4_grouped_different_random_softoh': dict(
      recipe='full_orbit_sft_10k',
      num_base_examples=10_000,
      r_orbit=1.0,
      orbit_only=True,
      orbit_sample_mode='sequential_drop_last',
      orbit_mask_coupling='per_view',
      masking_policy='same per-view corruptions as d4_grouped_different_random; SoftOH loss',
  ),
}


def _register_omegaconf_resolvers() -> None:
  OmegaConf.register_new_resolver('cwd', os.getcwd, replace=True)
  OmegaConf.register_new_resolver('mdlm_root', lambda: str(MDLM_ROOT), replace=True)
  OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
  OmegaConf.register_new_resolver('eval', eval, replace=True)
  OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)


def _apply_transform(grid: np.ndarray, transform_name: str) -> np.ndarray:
  """Apply a D4 transform to a [9,9] array."""
  if transform_name == 'identity':
    return grid.copy()
  if transform_name == 'rot90':
    return np.rot90(grid, k=-1).copy()
  if transform_name == 'rot180':
    return np.rot90(grid, k=2).copy()
  if transform_name == 'rot270':
    return np.rot90(grid, k=1).copy()
  if transform_name == 'flip_lr':
    return np.fliplr(grid).copy()
  if transform_name == 'flip_ud':
    return np.flipud(grid).copy()
  if transform_name == 'transpose':
    return np.transpose(grid).copy()
  if transform_name == 'anti_diagonal':
    return np.flip(grid, (0, 1)).T.copy()
  raise ValueError(f'Unknown transform: {transform_name}')


def _build_transform_indices() -> Dict[str, np.ndarray]:
  base = np.arange(81, dtype=np.int64).reshape(9, 9)
  return {name: _apply_transform(base, name).reshape(-1) for name in TRANSFORMS}


def _invert_perm_np(perm: np.ndarray) -> np.ndarray:
  """perm[i] = canonical index feeding transformed cell i → inv[j] = transformed index showing canon j."""
  inv = np.empty(81, dtype=np.int64)
  inv[perm] = np.arange(81, dtype=np.int64)
  return inv


def _load_split_arrays(root: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
  split = {'validation': 'valid', 'val': 'valid'}.get(split, split)
  solution_path = root / f'{split}_solution.npy'
  anchor_path = root / f'{split}_anchor.npy'
  if not solution_path.exists():
    raise FileNotFoundError(f'Missing {solution_path}')
  if not anchor_path.exists():
    raise FileNotFoundError(f'Missing {anchor_path}')
  solutions = np.load(solution_path, mmap_mode='r')
  anchors = np.load(anchor_path, mmap_mode='r')
  if solutions.ndim != 2 or solutions.shape[1] != 81:
    raise ValueError(f'{solution_path} must have shape [N,81], got {solutions.shape}')
  if anchors.shape != solutions.shape:
    raise ValueError(f'{anchor_path} shape {anchors.shape} != solutions shape {anchors.shape}')
  return solutions, anchors


def _select_indices(
    num_total: int,
    num_base_examples: int,
    subset_mode: str,
    seed: int,
) -> np.ndarray:
  if num_base_examples <= 0 or num_base_examples > num_total:
    num_base_examples = num_total
  if subset_mode == 'first':
    return np.arange(num_base_examples, dtype=np.int64)
  if subset_mode == 'random':
    rng = np.random.default_rng(seed)
    return rng.choice(num_total, size=num_base_examples, replace=False).astype(np.int64)
  raise ValueError(f'Unknown subset_mode={subset_mode!r}')


class SudokuFullBlankSFTDataset(Dataset):
  """Single-row clue-conditioned SFT samples (ordinary / validation)."""

  def __init__(
      self,
      solutions: np.ndarray,
      anchors: np.ndarray,
      base_indices: np.ndarray,
      *,
      recipe: str,
      seed: int = 0,
  ) -> None:
    if recipe not in RECIPE_DEFAULTS:
      raise ValueError(f'Unknown recipe {recipe}; choices={list(RECIPE_DEFAULTS)}')
    self.solutions = solutions
    self.anchors = anchors
    self.base_indices = np.asarray(base_indices, dtype=np.int64)
    self.recipe = recipe
    self.random_transform = bool(RECIPE_DEFAULTS[recipe]['random_transform'])
    self.orbit_expand = bool(RECIPE_DEFAULTS[recipe]['orbit_expand'])
    self.transform_indices = _build_transform_indices()
    self.seed = int(seed)

  def __len__(self) -> int:
    if self.orbit_expand:
      return int(len(self.base_indices) * len(TRANSFORMS))
    repeat_factor = int(RECIPE_DEFAULTS[self.recipe].get('repeat_factor', 1))
    if repeat_factor > 1:
      return int(len(self.base_indices) * repeat_factor)
    return int(len(self.base_indices))

  def _choose_transform(self, idx: int) -> str:
    if self.orbit_expand:
      return TRANSFORMS[idx % len(TRANSFORMS)]
    if self.random_transform:
      return TRANSFORMS[int(np.random.randint(0, len(TRANSFORMS)))]
    return 'identity'

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    repeat_factor = int(RECIPE_DEFAULTS[self.recipe].get('repeat_factor', 1))
    if self.orbit_expand:
      base_pos = idx // len(TRANSFORMS)
    elif repeat_factor > 1:
      base_pos = idx % len(self.base_indices)
    else:
      base_pos = idx
    arr_idx = int(self.base_indices[base_pos])
    transform_name = self._choose_transform(idx)
    perm = self.transform_indices[transform_name]

    sol = np.asarray(self.solutions[arr_idx], dtype=np.int64)[perm].copy()
    anchor = np.asarray(self.anchors[arr_idx], dtype=np.bool_)[perm].copy()
    loss_mask = ~anchor

    return {
      'input_ids': torch.from_numpy(sol).long(),
      'anchor_mask': torch.from_numpy(anchor).bool(),
      'loss_mask': torch.from_numpy(loss_mask).bool(),
      'transform_id': torch.tensor(TRANSFORMS.index(transform_name), dtype=torch.long),
      'base_index': torch.tensor(arr_idx, dtype=torch.long),
    }


class SudokuOrbitSoftOHChunkDataset(Dataset):
  """One item = ``B * 8`` rows: ``B`` canonical puzzles × all D4 transforms.

  Corruption coupling is controlled by ``orbit_mask_coupling``:
  - ``shared``: one canonical random blank mask + ``t`` per base puzzle, permuted to all views.
  - ``per_view``: independent random mask + ``t`` for each D4 view (mapped to canonical for loss).
  """

  def __init__(
      self,
      solutions: np.ndarray,
      anchors: np.ndarray,
      base_indices: np.ndarray,
      *,
      orbit_base_size: int,
      seed: int,
      blank_mask_mode: str,
      blank_mask_eps: float,
      sft_time: float,
      sample_mode: str = 'random',
      stochastic_masks: bool = True,
      orbit_mask_coupling: str = 'shared',
  ) -> None:
    self.solutions = solutions
    self.anchors = anchors
    self.base_indices = np.asarray(base_indices, dtype=np.int64)
    self.B = int(orbit_base_size)
    self.seed = int(seed)
    self.blank_mask_mode = blank_mask_mode
    self.blank_mask_eps = float(blank_mask_eps)
    self.sft_time = float(sft_time)
    self.sample_mode = str(sample_mode)
    self.stochastic_masks = bool(stochastic_masks)
    self.orbit_mask_coupling = str(orbit_mask_coupling)
    if self.orbit_mask_coupling not in ('shared', 'per_view'):
      raise ValueError(f'orbit_mask_coupling must be shared or per_view, got {orbit_mask_coupling!r}')
    if self.sample_mode not in ('random', 'sequential', 'sequential_drop_last'):
      raise ValueError(
        f'sample_mode must be random, sequential, or sequential_drop_last, got {sample_mode!r}')
    self.transform_indices = _build_transform_indices()
    self.perms_np = np.stack(
        [self.transform_indices[name] for name in TRANSFORMS],
        axis=0,
    )  # [8, 81]
    self.inv_perms_np = np.stack(
        [_invert_perm_np(self.perms_np[g]) for g in range(len(TRANSFORMS))],
        axis=0,
    )  # [8, 81]
    n = len(self.base_indices)
    if self.sample_mode == 'sequential':
      self._len = max(1, (n + self.B - 1) // max(1, self.B))
    elif self.sample_mode == 'sequential_drop_last':
      self._len = max(1, n // max(1, self.B))
    else:
      self._len = max(1, n // max(1, self.B))

  def __len__(self) -> int:
    return int(self._len)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    # Validation uses deterministic rng(seed, idx). Training defaults to stochastic masks
    # using each worker's numpy RNG, so repeated epochs get fresh corruptions just like
    # ordinary SFT rows do inside _sft_loss_full_blank.
    rng = np.random if self.stochastic_masks else np.random.default_rng(self.seed + int(idx) * 1000003)

    if self.sample_mode in ('sequential', 'sequential_drop_last'):
      start = int(idx) * self.B
      n_pool = len(self.base_indices)
      pick = np.empty(self.B, dtype=np.int64)
      for j in range(self.B):
        # sequential_drop_last never reaches the incomplete chunk unless n_pool < B;
        # modulo is retained only as a safe fallback for tiny debugging datasets.
        pick[j] = int(self.base_indices[(start + j) % n_pool])
    else:
      pool = self.base_indices
      replace = len(pool) < self.B
      pick = rng.choice(pool, size=self.B, replace=replace).astype(np.int64)

    rows_x: List[torch.Tensor] = []
    rows_t: List[float] = []
    canon_sol_t: List[torch.Tensor] = []
    canon_anchor_t: List[torch.Tensor] = []
    mask_blank_canon_t: List[torch.Tensor] = []
    mask_blank_view_rows: List[torch.Tensor] = []
    supervise_per_view: List[torch.Tensor] = []

    for b in range(self.B):
      arr_idx = int(pick[b])
      sol_c = np.asarray(self.solutions[arr_idx], dtype=np.int64).reshape(81)
      anc_c = np.asarray(self.anchors[arr_idx], dtype=np.bool_).reshape(81)
      blank_c = ~anc_c

      if self.orbit_mask_coupling == 'shared':
        if self.blank_mask_mode == 'full':
          t_b = self.sft_time
          mask_blank_c = blank_c.copy()
        else:
          eps = self.blank_mask_eps
          t_b = float((1.0 - eps) * rng.random() + eps)
          rand_c = rng.random(81)
          mask_blank_c = blank_c & (rand_c < t_b)
          if mask_blank_c.sum() == 0 and blank_c.any():
            mask_blank_c = blank_c.copy()

        canon_sol_t.append(torch.from_numpy(sol_c).long())
        canon_anchor_t.append(torch.from_numpy(anc_c).bool())
        mask_blank_canon_t.append(torch.from_numpy(mask_blank_c).bool())

        for g in range(len(TRANSFORMS)):
          perm = self.perms_np[g]
          sol_v = sol_c[perm]
          mask_blank_v = mask_blank_c[perm]
          rows_x.append(torch.from_numpy(sol_v).long())
          rows_t.append(t_b)
          mask_blank_view_rows.append(torch.from_numpy(mask_blank_v).bool())

      else:
        canon_sol_t.append(torch.from_numpy(sol_c).long())
        canon_anchor_t.append(torch.from_numpy(anc_c).bool())
        mask_blank_canon_t.append(torch.zeros(81, dtype=torch.bool))

        for g in range(len(TRANSFORMS)):
          perm = self.perms_np[g]
          inv_g = self.inv_perms_np[g]
          sol_v = sol_c[perm]
          anc_v = anc_c[perm]
          blank_v = ~anc_v

          if self.blank_mask_mode == 'full':
            t_bg = self.sft_time
            mask_blank_v = blank_v.copy()
          else:
            eps = self.blank_mask_eps
            t_bg = float((1.0 - eps) * rng.random() + eps)
            rand_v = rng.random(81)
            mask_blank_v = blank_v & (rand_v < t_bg)
            if mask_blank_v.sum() == 0 and blank_v.any():
              mask_blank_v = blank_v.copy()

          rows_x.append(torch.from_numpy(sol_v).long())
          rows_t.append(t_bg)
          mask_blank_view_rows.append(torch.from_numpy(mask_blank_v.copy()).bool())
          supervise_per_view.append(torch.from_numpy(mask_blank_v[inv_g].copy()).bool())

    x_gold = torch.stack(rows_x, dim=0)  # [B*8, 81]
    anchor_mask = torch.stack([
        canon_anchor_t[b // 8][self.perms_np[b % 8]]
        for b in range(self.B * 8)
    ], dim=0)
    mask_blank_view = torch.stack(mask_blank_view_rows, dim=0)

    out: Dict[str, torch.Tensor] = {
      'x_gold': x_gold,
      'anchor_mask': anchor_mask,
      'mask_blank_view': mask_blank_view.bool(),
      'canonical_sol': torch.stack(canon_sol_t, dim=0),
      'canonical_anchor': torch.stack(canon_anchor_t, dim=0).bool(),
      'supervise_canon': torch.stack(mask_blank_canon_t, dim=0).bool(),
      't_cond': torch.tensor(rows_t, dtype=torch.float32),
      'orbit_base_size': torch.tensor(self.B, dtype=torch.long),
    }
    if self.orbit_mask_coupling == 'per_view':
      gsz = len(TRANSFORMS)
      out['supervise_canon_orbit'] = torch.stack(supervise_per_view, dim=0).view(self.B, gsz, 81).bool()
    return out


def _worker_init_fn(worker_id: int) -> None:
  base_seed = torch.initial_seed() % (2**32)
  np.random.seed(base_seed + worker_id)
  random.seed(base_seed + worker_id)


def _orbit_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
  return batch[0]


def _compose_config(
    *,
    checkpoint: str,
    model: str,
    batch_size: int,
    num_steps: int,
    predictor: str,
    noise_removal: bool,
    extra_overrides: Optional[List[str]] = None,
) -> Any:
  overrides = [
    f'model={model}',
    'data=sudoku9-solutions',
    'backbone=dit',
    'parameterization=subs',
    'model.length=81',
    f'eval.checkpoint_path={checkpoint}',
    f'loader.global_batch_size={batch_size}',
    f'loader.eval_global_batch_size={batch_size}',
    f'loader.batch_size={batch_size}',
    f'loader.eval_batch_size={batch_size}',
    'trainer.devices=1',
    'trainer.num_nodes=1',
    'trainer.accumulate_grad_batches=1',
    f'sampling.steps={num_steps}',
    f'sampling.predictor={predictor}',
    f'sampling.noise_removal={str(noise_removal).lower()}',
    'noise.type=loglinear',
  ]
  if extra_overrides:
    overrides.extend(extra_overrides)
  with initialize_config_dir(version_base=None, config_dir=str(MDLM_ROOT / 'configs')):
    cfg = compose(config_name='config', overrides=overrides)
  OmegaConf.resolve(cfg)
  return cfg


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
  if isinstance(payload, dict) and 'model_state_dict' in payload:
    state_dict = payload['model_state_dict']
  elif isinstance(payload, dict) and 'state_dict' in payload:
    state_dict = payload['state_dict']
  elif isinstance(payload, dict):
    state_dict = payload
  else:
    raise ValueError('Unsupported checkpoint payload type')

  if isinstance(state_dict, dict):
    state_dict = {
      k: v for k, v in state_dict.items()
      if not k.startswith('ref_backbone.')
    }
  return state_dict


def _load_model(cfg: Any, device: torch.device) -> diffusion_mod.Diffusion:
  tok = dataloader_mod.get_tokenizer(cfg)
  payload = torch.load(str(cfg.eval.checkpoint_path), map_location=device)
  state_dict = _extract_state_dict(payload)

  model = diffusion_mod.Diffusion(cfg, tokenizer=tok)
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if unexpected:
    print(f'[posttrain] unexpected state_dict keys (up to 12): {unexpected[:12]}')
  if missing:
    print(f'[posttrain] missing state_dict keys (up to 12): {missing[:12]}')

  model.ema = None
  model.to(device)
  model.train()
  return model


def _make_conditioned_input(
    input_ids: torch.Tensor,
    anchor_mask: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
  mask_tokens = torch.full_like(input_ids, fill_value=int(mask_id))
  return torch.where(anchor_mask.bool(), input_ids, mask_tokens)


def _build_perms_t() -> torch.Tensor:
  """[8, 81] long: perm[g, transformed_cell] = canonical cell under transform g."""
  ti = _build_transform_indices()
  perms = np.stack([ti[name] for name in TRANSFORMS], axis=0)
  return torch.from_numpy(perms).long()

def _build_inv_perms_t() -> torch.Tensor:
  """[8, 81] long: inv[g, c] = transformed flat index for canonical cell c under transform g."""
  ti = _build_transform_indices()
  inv_list = []
  for name in TRANSFORMS:
    inv_list.append(_invert_perm_np(ti[name]))
  inv = np.stack(inv_list, axis=0)
  return torch.from_numpy(inv).long()


def _sft_loss_full_blank(
    model: diffusion_mod.Diffusion,
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    sft_time: float,
    label_smoothing: float = 0.0,
    blank_mask_mode: str = 'random',
    blank_mask_eps: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
  input_ids = batch['input_ids'].to(device=device, dtype=torch.long)
  anchor_mask = batch['anchor_mask'].to(device=device, dtype=torch.bool)
  loss_mask = batch['loss_mask'].to(device=device, dtype=torch.bool)

  B, _ = input_ids.shape
  mid = int(model.mask_index)

  if blank_mask_mode == 'full':
    x_cond = _make_conditioned_input(input_ids, anchor_mask, mid)
    t_cond = torch.full((B,), float(sft_time), device=device, dtype=model.dtype)
    supervise_mask = loss_mask
  elif blank_mask_mode == 'random':
    eps = float(blank_mask_eps)
    t_prob = torch.rand((B,), device=device, dtype=model.dtype)
    t_prob = (1.0 - eps) * t_prob + eps
    rand_pos = torch.rand((B, input_ids.shape[1]), device=device, dtype=model.dtype)
    mask_blank = loss_mask & (rand_pos < t_prob[:, None])
    n_masked_row = mask_blank.sum(dim=1)
    has_blanks = loss_mask.any(dim=1)
    fallback = (n_masked_row == 0) & has_blanks
    mask_blank = torch.where(fallback[:, None], loss_mask, mask_blank)
    mask_tokens = torch.full_like(input_ids, mid)
    x_cond = torch.where(
      anchor_mask,
      input_ids,
      torch.where(mask_blank, mask_tokens, input_ids),
    )
    t_cond = t_prob
    supervise_mask = mask_blank
  else:
    raise ValueError(f'Unknown blank_mask_mode={blank_mask_mode!r}; use full or random')

  sigma, _ = model.noise(t_cond)
  logits = model.forward(x_cond, sigma[:, None])

  if logits.shape[-1] > 9:
    logits_for_ce = logits[..., :9]
  else:
    logits_for_ce = logits

  flat_logits = logits_for_ce.reshape(-1, logits_for_ce.shape[-1])
  flat_targets = input_ids.reshape(-1)
  flat_mask = supervise_mask.reshape(-1)
  if flat_mask.sum().item() == 0:
    raise RuntimeError('Empty supervision mask in batch; check anchor / blank masks.')

  ce = F.cross_entropy(
    flat_logits[flat_mask],
    flat_targets[flat_mask],
    reduction='mean',
    label_smoothing=float(label_smoothing),
  )

  with torch.no_grad():
    pred = flat_logits[flat_mask].argmax(dim=-1)
    acc = (pred == flat_targets[flat_mask]).float().mean().item()
    n_targets = int(flat_mask.sum().item())
  return ce, {'sft_token_acc': acc, 'num_loss_tokens': float(n_targets)}


def _orbit_soft_oh_loss(
    model: diffusion_mod.Diffusion,
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    inv_perms: torch.Tensor,
    label_smoothing: float,
    alpha: float,
    lambda_eff: float,
    perms: Optional[torch.Tensor] = None,
    blank_mask_mode: str = 'random',
    blank_mask_eps: float = 1e-3,
    sft_time: float = 1.0,
    resample_corruption: bool = True,
    orbit_mask_coupling: str = 'shared',
) -> Tuple[torch.Tensor, Dict[str, float]]:
  """Orbit-coupled SoftOH / grouped-average loss.

  If ``resample_corruption`` is True, this function samples fresh corruptions on
  device. With ``orbit_mask_coupling='shared'``, one canonical random mask and
  noise level per base puzzle is propagated to all eight D4 views (methods 4/5).
  With ``orbit_mask_coupling='per_view'``, each view gets an independent random
  mask and ``t`` on that view’s grid; supervision is mapped back to canonical
  coordinates for the pulled-back CE.

  If ``resample_corruption`` is False, masks come from the batch (validation /
  sanity). If ``supervise_canon_orbit`` is present with shape ``[B,8,81]``, it is
  used; otherwise ``supervise_canon`` ``[B,81]`` is broadcast across views.

  ``inv_perms``: [8,81], inv[g, canonical_cell] = transformed cell index.
  ``perms``: [8,81], perm[g, transformed_cell] = canonical cell index.
  """
  x_gold = batch['x_gold'].to(device=device, dtype=torch.long)
  canonical_sol = batch['canonical_sol'].to(device=device, dtype=torch.long)
  canonical_anchor = batch['canonical_anchor'].to(device=device, dtype=torch.bool)
  B = int(batch['orbit_base_size'].item())
  G = len(TRANSFORMS)
  mid = int(model.mask_index)
  coupling = str(orbit_mask_coupling)

  if resample_corruption:
    if perms is None:
      raise ValueError('perms must be provided when resample_corruption=True')
    perms_t = perms.to(device=device)
    inv = inv_perms.to(device=device)
    blank_c = ~canonical_anchor

    if coupling == 'shared':
      if blank_mask_mode == 'full':
        t_base = torch.full((B,), float(sft_time), device=device, dtype=model.dtype)
        supervise_canon = blank_c
      elif blank_mask_mode == 'random':
        eps = float(blank_mask_eps)
        t_base = torch.rand((B,), device=device, dtype=model.dtype)
        t_base = (1.0 - eps) * t_base + eps
        rand_c = torch.rand((B, 81), device=device, dtype=model.dtype)
        supervise_canon = blank_c & (rand_c < t_base[:, None])
        n_masked_row = supervise_canon.sum(dim=1)
        has_blanks = blank_c.any(dim=1)
        fallback = (n_masked_row == 0) & has_blanks
        supervise_canon = torch.where(fallback[:, None], blank_c, supervise_canon)
      else:
        raise ValueError(f'Unknown blank_mask_mode={blank_mask_mode!r}; use full or random')

      anchor_views = []
      mask_views = []
      for g in range(G):
        anchor_views.append(canonical_anchor[:, perms_t[g]])
        mask_views.append(supervise_canon[:, perms_t[g]])
      anchor_mask = torch.stack(anchor_views, dim=1).reshape(B * G, 81)
      mask_blank_view = torch.stack(mask_views, dim=1).reshape(B * G, 81)
      t_cond = t_base[:, None].expand(B, G).reshape(B * G)
      supervise_stack = supervise_canon.float().unsqueeze(1).expand(B, G, -1)

    elif coupling == 'per_view':
      idx_g = perms_t.long().unsqueeze(0).expand(B, G, 81)
      blank_v = torch.gather(blank_c.unsqueeze(1).expand(B, G, 81), 2, idx_g)

      if blank_mask_mode == 'full':
        t_bg = torch.full((B, G), float(sft_time), device=device, dtype=model.dtype)
        mask_bgv = blank_v
      elif blank_mask_mode == 'random':
        eps = float(blank_mask_eps)
        t_bg = torch.rand((B, G), device=device, dtype=model.dtype)
        t_bg = (1.0 - eps) * t_bg + eps
        rand_v = torch.rand((B, G, 81), device=device, dtype=model.dtype)
        mask_bgv = blank_v & (rand_v < t_bg.unsqueeze(-1))
        has_blanks = blank_v.any(dim=-1)
        n_masked = mask_bgv.sum(dim=-1)
        fallback = (n_masked == 0) & has_blanks
        mask_bgv = torch.where(fallback.unsqueeze(-1), blank_v, mask_bgv)
      else:
        raise ValueError(f'Unknown blank_mask_mode={blank_mask_mode!r}; use full or random')

      anchor_views = []
      for g in range(G):
        anchor_views.append(canonical_anchor[:, perms_t[g]])
      anchor_mask = torch.stack(anchor_views, dim=1).reshape(B * G, 81)
      mask_blank_view = mask_bgv.reshape(B * G, 81)
      t_cond = t_bg.reshape(B * G)
      supervise_stack = torch.zeros((B, G, 81), device=device, dtype=torch.float32)
      for g in range(G):
        supervise_stack[:, g, :] = mask_bgv[:, g, inv[g, :]].float()
    else:
      raise ValueError(f'orbit_mask_coupling must be shared or per_view, got {coupling!r}')
  else:
    anchor_mask = batch['anchor_mask'].to(device=device, dtype=torch.bool)
    mask_blank_view = batch['mask_blank_view'].to(device=device, dtype=torch.bool)
    t_cond = batch['t_cond'].to(device=device, dtype=model.dtype)
    if 'supervise_canon_orbit' in batch:
      supervise_stack = batch['supervise_canon_orbit'].to(device=device, dtype=torch.float32)
    else:
      sc = batch['supervise_canon'].to(device=device, dtype=torch.bool)
      supervise_stack = sc.float().unsqueeze(1).expand(B, G, -1)

  mask_tokens = torch.full_like(x_gold, mid)
  x_cond = torch.where(
    anchor_mask,
    x_gold,
    torch.where(mask_blank_view, mask_tokens, x_gold),
  )

  sigma, _ = model.noise(t_cond)
  logits = model.forward(x_cond, sigma[:, None])
  if logits.shape[-1] > 9:
    logits = logits[..., :9]

  logits_b = logits.view(B, G, 81, -1)
  inv = inv_perms.to(device=device)
  ell = torch.empty((B, G), device=device, dtype=torch.float32)
  targets = canonical_sol

  for g in range(G):
    logits_bg = logits_b[:, g, :, :]
    idx = inv[g].view(1, 81, 1).expand(B, 81, logits_bg.shape[-1])
    logits_canon = torch.gather(logits_bg, 1, idx).float()
    sup = supervise_stack[:, g, :]
    ce_flat = F.cross_entropy(
      logits_canon.reshape(-1, logits_canon.shape[-1]),
      targets.reshape(-1),
      reduction='none',
      label_smoothing=float(label_smoothing),
    ).view(B, 81)
    denom = sup.sum(dim=1).clamp(min=1.0)
    ell[:, g] = ((ce_flat * sup).sum(dim=1) / denom).float()

  mu = ell.mean(dim=1)
  a = float(alpha)
  if a <= 1e-6:
    l_soft = mu
  else:
    lse = torch.logsumexp(ell * a, dim=1)
    l_soft = (lse - math.log(float(G))) / a
  l_group = (1.0 - float(lambda_eff)) * mu + float(lambda_eff) * l_soft
  loss = l_group.mean()

  with torch.no_grad():
    pred_all = logits.reshape(B * G, 81, -1).argmax(dim=-1)
    tgt_flat = x_gold
    mflat = mask_blank_view.reshape(-1, 81)
    acc = (pred_all == tgt_flat).float()
    acc_row = (acc * mflat.float()).sum(dim=1) / mflat.float().sum(dim=1).clamp(min=1.0)
    n_tok = float(mflat.sum().item())
    worst_mean = float(ell.max(dim=1).values.mean().item())
    gap_soft_mu = float((l_soft - mu).mean().item())
    orbit_var = float(ell.var(dim=1, unbiased=False).mean().item())
  return loss, {
    'sft_token_acc': float(acc_row.mean().item()),
    'num_loss_tokens': n_tok,
    'train_mu': float(mu.mean().item()),
    'train_l_soft': float(l_soft.mean().item()),
    'train_l_group': float(loss.item()),
    'train_worst': worst_mean,
    'train_gap_soft_mu': gap_soft_mu,
    'train_orbit_var': orbit_var,
    'lambda_eff': float(lambda_eff),
  }


@torch.no_grad()
def _evaluate_sft_loss(
    model: diffusion_mod.Diffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    sft_time: float,
    label_smoothing: float,
    max_batches: int,
) -> Dict[str, float]:
  model.eval()
  total_loss = 0.0
  total_acc = 0.0
  total_tokens = 0.0
  n_batches = 0
  for batch in loader:
    loss, info = _sft_loss_full_blank(
      model,
      batch,
      device=device,
      sft_time=sft_time,
      label_smoothing=label_smoothing,
      blank_mask_mode='full',
    )
    tokens = info['num_loss_tokens']
    total_loss += float(loss.item()) * tokens
    total_acc += float(info['sft_token_acc']) * tokens
    total_tokens += tokens
    n_batches += 1
    if max_batches > 0 and n_batches >= max_batches:
      break
  model.train()
  return {
    'val_sft_loss': total_loss / max(total_tokens, 1.0),
    'val_sft_token_acc': total_acc / max(total_tokens, 1.0),
    'val_batches': float(n_batches),
  }


@torch.no_grad()
def _evaluate_orbit_soft_oh(
    model: diffusion_mod.Diffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    inv_perms: torch.Tensor,
    label_smoothing: float,
    alpha: float,
    lambda_target: float,
    max_batches: int,
    orbit_mask_coupling: str = 'shared',
) -> Dict[str, float]:
  """Mean orbit ``L_group`` on validation chunks (LE-aware proxy; not macro-All-8)."""
  model.eval()
  total_lg = 0.0
  total_mu = 0.0
  n_batches = 0
  for batch in loader:
    loss, info = _orbit_soft_oh_loss(
      model,
      batch,
      device=device,
      inv_perms=inv_perms,
      label_smoothing=label_smoothing,
      alpha=alpha,
      lambda_eff=float(lambda_target),
      resample_corruption=False,
      orbit_mask_coupling=orbit_mask_coupling,
    )
    total_lg += float(info['train_l_group'])
    total_mu += float(info['train_mu'])
    n_batches += 1
    if max_batches > 0 and n_batches >= max_batches:
      break
  model.train()
  denom = max(n_batches, 1)
  return {
    'val_orbit_l_group': total_lg / denom,
    'val_orbit_mu': total_mu / denom,
    'val_orbit_batches': float(n_batches),
  }


def _run_orbit_mask_sanity(orbit_loader: DataLoader, inv_cpu: torch.Tensor) -> None:
  """Assert view-space blank masks match canonical supervision (one batch)."""
  inv = inv_cpu.long()
  batch = next(iter(orbit_loader))
  B = int(batch['orbit_base_size'].item())
  mb = batch['mask_blank_view']
  if 'supervise_canon_orbit' in batch:
    sup_pv = batch['supervise_canon_orbit']
    for b in range(B):
      for g in range(len(TRANSFORMS)):
        row = b * 8 + g
        mask_back = mb[row][inv[g]]
        if not torch.equal(mask_back, sup_pv[b, g]):
          raise AssertionError(f'orbit mask sanity failed (per_view): b={b} g={g}')
  else:
    sup = batch['supervise_canon']
    for b in range(B):
      for g in range(len(TRANSFORMS)):
        row = b * 8 + g
        mask_back = mb[row][inv[g]]
        if not torch.equal(mask_back, sup[b]):
          raise AssertionError(f'orbit mask sanity failed: b={b} g={g}')


def _soft_oh_numeric_sanity() -> None:
  ell = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.0]])
  mu = ell.mean(dim=1)
  for alpha in (1.0, 2.0, 4.0):
    lse = torch.logsumexp(ell * alpha, dim=1)
    l_soft = (lse - math.log(8.0)) / alpha
    if not (l_soft + 1e-5 >= mu).all():
      raise AssertionError('SoftOH sanity: l_soft < mu')
    if not (l_soft - 1e-5 <= ell.max(dim=1).values).all():
      raise AssertionError('SoftOH sanity: l_soft > max')


def _build_lr_lambda(max_steps: int, warmup_steps: int, min_lr_ratio: float):
  max_steps = max(1, int(max_steps))
  warmup_steps = max(0, int(warmup_steps))
  min_lr_ratio = float(min_lr_ratio)

  def lr_lambda(step: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
      return max(1e-8, float(step + 1) / float(warmup_steps))
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

  return lr_lambda


def _lambda_eff_warmup(step: int, max_steps: int, lambda_target: float, warmup_frac: float) -> float:
  if lambda_target <= 0.0:
    return 0.0
  end = max(1, int(math.ceil(float(warmup_frac) * float(max_steps))))
  t = min(1.0, float(step) / float(end))
  return float(lambda_target) * t


def _save_checkpoint(
    path: Path,
    model: diffusion_mod.Diffusion,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    args: argparse.Namespace,
    step: int,
    metrics: Dict[str, Any],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    'step': int(step),
    'args': vars(args),
    'metrics': metrics,
  }
  torch.save(payload, path)


def _write_jsonl(path: Path, row: Dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('a', encoding='utf-8') as f:
    f.write(json.dumps(row, sort_keys=True) + '\n')


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description='Post-train MDLM Sudoku with SoftOH + optional SFT mix.')
  p.add_argument('--checkpoint', type=str, required=True)
  p.add_argument('--model', type=str, default='sudoku_50m')
  p.add_argument('--npy-root', type=str, required=True)
  p.add_argument('--valid-npy-root', type=str, default='')
  p.add_argument(
      '--valid-max-examples',
      type=int,
      default=0,
      help='If > 0, use only the first N rows of valid_*.npy for val loaders (deterministic prefix). '
      '0 means use all rows on disk. Npy files may still hold 2k; this caps in-memory val size.',
  )
  p.add_argument('--recipe', type=str, default='canonical_sft_80k', choices=sorted(RECIPE_DEFAULTS),
                 help='Low-level recipe. Prefer --fair-method for controlled ablations.')
  p.add_argument(
      '--fair-method',
      type=str,
      default='',
      choices=('', *sorted(FAIR_METHODS)),
      help=('Preset fair ablations: sft80k, repeat10k_x8, d4_shuffle10k_indmask, d4_grouped_avg10k, '
            'softoh10k, d4_grouped_different_random, d4_grouped_different_random_softoh. Overrides '
            '--recipe/--num-base-examples/--r-orbit/--orbit-only and grouped loss settings.'),
  )
  p.add_argument(
      '--orbit-mask-coupling',
      type=str,
      default='shared',
      choices=('shared', 'per_view'),
      help='Orbit corruption: shared = one canonical random mask+t per base puzzle for all 8 views; '
           'per_view = independent mask+t per D4 view (still B×8 grouped rows). Fair presets set this.',
  )
  p.add_argument('--output-dir', type=str, required=True)

  p.add_argument('--soft-oh-alpha', type=float, default=2.0,
                 help='SoftOH strictness; 0 means L_soft = μ (grouped average CE).')
  p.add_argument('--soft-oh-lambda', type=float, default=0.5,
                 help='Target λ in L_group = (1-λ)*μ + λ*L_soft; warmed up from 0.')
  p.add_argument('--lambda-warmup-frac', type=float, default=0.1,
                 help='Linear λ warmup over first this fraction of **optimizer updates**.')
  p.add_argument('--r-orbit', type=float, default=0.25,
                 help='Probability of taking an orbit-grouped SoftOH step (else ordinary SFT).')
  p.add_argument('--orbit-batch-size', type=int, default=0,
                 help='Rows per orbit step = B*8. 0 means match --batch-size; must be divisible by 8.')
  p.add_argument('--orbit-only', action='store_true',
                 help='Set r_orbit effectively to 1 (only SoftOH orbit steps).')
  p.add_argument('--orbit-sample-mode', type=str, default='random',
                 choices=('random', 'sequential', 'sequential_drop_last'),
                 help=('Training orbit chunks: random samples base puzzles; sequential covers the selected pool by chunks; '
                       'sequential_drop_last drops the incomplete final chunk, matching DataLoader(drop_last=True).'))
  p.add_argument('--deterministic-orbit-masks', action='store_true',
                 help='Use deterministic orbit corruptions keyed by dataset index. Default: fresh stochastic masks for training.')

  p.add_argument('--num-base-examples', type=int, default=0)
  p.add_argument('--subset-mode', type=str, default='random', choices=('random', 'first'))
  p.add_argument('--subset-seed', type=int, default=0)
  p.add_argument('--seed', type=int, default=0)

  p.add_argument('--batch-size', type=int, default=512)
  p.add_argument('--num-workers', type=int, default=4)
  p.add_argument('--pin-memory', action='store_true', default=True)
  p.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
  p.add_argument('--grad-accum-steps', type=int, default=1,
                 help='Micro-batches per optimizer step; λ warmup and LR cosine use optimizer-update count.')
  p.add_argument('--max-steps', type=int, default=3000,
                 help='Total training micro-steps (one backward per iteration). '
                 'Optimizer updates = ceil(max_steps / grad_accum_steps).')
  p.add_argument('--lr', type=float, default=1e-5)
  p.add_argument('--weight-decay', type=float, default=0.01)
  p.add_argument('--beta1', type=float, default=0.9)
  p.add_argument('--beta2', type=float, default=0.999)
  p.add_argument('--eps', type=float, default=1e-8)
  p.add_argument('--warmup-steps', type=int, default=150)
  p.add_argument('--min-lr-ratio', type=float, default=0.1)
  p.add_argument('--grad-clip', type=float, default=1.0)
  p.add_argument('--label-smoothing', type=float, default=0.0)
  p.add_argument('--blank-mask-mode', type=str, default='random', choices=('random', 'full'))
  p.add_argument('--blank-mask-eps', type=float, default=1e-3)
  p.add_argument('--sft-time', type=float, default=1.0)
  p.add_argument('--train-noise', action='store_true', default=False)

  p.add_argument('--eval-every', type=int, default=500)
  p.add_argument('--eval-max-batches', type=int, default=32)
  p.add_argument('--eval-orbit-max-batches', type=int, default=16,
                 help='Max orbit chunks for val orbit loss (each chunk is orbit_batch_size rows).')
  p.add_argument(
      '--best-checkpoint-key',
      type=str,
      default='sft_loss',
      choices=('sft_loss', 'orbit_group_loss'),
      help='best.ckpt: minimize val_sft_loss (canonical) or val_orbit_l_group (orbit proxy).',
  )
  p.add_argument(
      '--sanity-check',
      action='store_true',
      help='Run orbit mask + SoftOH numeric asserts on one batch then train.',
  )
  p.add_argument('--save-every', type=int, default=1000)
  p.add_argument('--log-every', type=int, default=50)

  p.add_argument('--predictor', type=str, default='ddpm_cache')
  p.add_argument('--num-steps', type=int, default=128)
  p.add_argument('--no-noise-removal', action='store_true')
  p.add_argument('--device', type=str, default='cuda')
  p.add_argument('--extra-override', action='append', default=[])
  return p.parse_args()


def _apply_fair_method(args: argparse.Namespace) -> None:
  if not args.fair_method:
    return
  preset = FAIR_METHODS[args.fair_method]
  args.recipe = str(preset['recipe'])
  args.num_base_examples = int(preset['num_base_examples'])
  args.r_orbit = float(preset['r_orbit'])
  args.orbit_only = bool(preset['orbit_only'])
  if 'soft_oh_alpha' in preset:
    args.soft_oh_alpha = float(preset['soft_oh_alpha'])
  if 'soft_oh_lambda' in preset:
    args.soft_oh_lambda = float(preset['soft_oh_lambda'])
  if 'orbit_sample_mode' in preset:
    args.orbit_sample_mode = str(preset['orbit_sample_mode'])
  if 'orbit_mask_coupling' in preset:
    args.orbit_mask_coupling = str(preset['orbit_mask_coupling'])
  if args.fair_method in (
      'd4_grouped_avg10k',
      'softoh10k',
      'd4_grouped_different_random',
      'd4_grouped_different_random_softoh',
  ):
    # Fair grouped methods should use the same row count per optimizer step as row-level SFT.
    # They also resample canonical masks on repeated epochs, just as row-level SFT resamples
    # masks inside _sft_loss_full_blank. Validation remains deterministic.
    if int(args.orbit_batch_size) <= 0:
      args.orbit_batch_size = int(args.batch_size)
    args.deterministic_orbit_masks = False

def _method_semantics(args: argparse.Namespace) -> Dict[str, Any]:
  if args.fair_method:
    masking_policy = FAIR_METHODS[args.fair_method]['masking_policy']
  elif args.orbit_only or args.r_orbit > 0:
    masking_policy = 'manual mixed setting: ordinary branch masks transformed rows independently; orbit branch masks canonical state then transforms it'
  elif args.recipe == 'full_orbit_sft_10k':
    masking_policy = 'D4 transform first, global shuffle, then independent random mask per transformed row (3A-style)'
  elif args.recipe == 'repeat8_sft_10k':
    masking_policy = '10k canonical rows repeated 8x; independent random mask per repeat'
  else:
    masking_policy = 'ordinary row-level SFT; independent random mask per row'
  if str(getattr(args, 'orbit_mask_coupling', 'shared')) == 'per_view':
    orbit_mask_order = (
      'per base puzzle and per D4 view: sample independent (t, blank mask) on that view’s grid, '
      'map supervision mask back to canonical coords for pulled-back CE'
    )
  else:
    orbit_mask_order = 'sample one canonical random mask + t, then apply D4 to solution/anchors/mask for all 8 views'
  return {
    'fair_method': args.fair_method or None,
    'masking_policy': masking_policy,
    'orbit_mask_coupling': str(getattr(args, 'orbit_mask_coupling', 'shared')),
    'row_sft_mask_order': 'apply recipe transform first, then sample independent random mask in _sft_loss_full_blank',
    'orbit_branch_mask_order': orbit_mask_order,
    'ordinary_branch_active': bool((not args.orbit_only) and float(args.r_orbit) < 1.0),
    'orbit_branch_active': bool(args.orbit_only or float(args.r_orbit) > 0.0),
  }


def main() -> None:
  args = parse_args()
  _apply_fair_method(args)
  _register_omegaconf_resolvers()

  if int(args.orbit_batch_size) <= 0:
    args.orbit_batch_size = int(args.batch_size)

  if args.soft_oh_alpha < 0:
    raise SystemExit('--soft-oh-alpha must be >= 0')
  if not (0.0 <= float(args.soft_oh_lambda) <= 1.0):
    raise SystemExit('--soft-oh-lambda must be in [0, 1]')
  if float(args.lambda_warmup_frac) < 0.0:
    raise SystemExit('--lambda-warmup-frac must be >= 0')
  if int(args.valid_max_examples) < 0:
    raise SystemExit('--valid-max-examples must be >= 0')
  if args.orbit_batch_size % 8 != 0 or args.orbit_batch_size < 8:
    raise SystemExit('--orbit-batch-size must be >= 8 and divisible by 8')
  if int(args.grad_accum_steps) <= 0:
    raise SystemExit('--grad-accum-steps must be >= 1')
  orbit_B = args.orbit_batch_size // 8
  total_opt_steps = max(
      1,
      (int(args.max_steps) + int(args.grad_accum_steps) - 1) // int(args.grad_accum_steps),
  )

  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('CUDA requested but not available')
  device = torch.device(args.device)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)
  if device.type == 'cuda':
    torch.cuda.manual_seed_all(args.seed)

  r_orbit = 1.0 if args.orbit_only else float(args.r_orbit)
  r_orbit = min(max(r_orbit, 0.0), 1.0)

  recipe_cfg = RECIPE_DEFAULTS[args.recipe]
  num_base_examples = int(args.num_base_examples) if args.num_base_examples > 0 else int(recipe_cfg['num_base_examples'])

  npy_root = Path(args.npy_root)
  if not npy_root.is_absolute():
    npy_root = MDLM_ROOT / npy_root
  train_solutions, train_anchors = _load_split_arrays(npy_root, 'train')
  base_indices = _select_indices(
    num_total=int(train_solutions.shape[0]),
    num_base_examples=num_base_examples,
    subset_mode=args.subset_mode,
    seed=args.subset_seed,
  )

  train_ds = SudokuFullBlankSFTDataset(
    train_solutions,
    train_anchors,
    base_indices,
    recipe=args.recipe,
    seed=args.seed,
  )
  orbit_ds = SudokuOrbitSoftOHChunkDataset(
    train_solutions,
    train_anchors,
    base_indices,
    orbit_base_size=orbit_B,
    seed=args.seed,
    blank_mask_mode=args.blank_mask_mode,
    blank_mask_eps=args.blank_mask_eps,
    sft_time=args.sft_time,
    sample_mode=args.orbit_sample_mode,
    stochastic_masks=not args.deterministic_orbit_masks,
    orbit_mask_coupling=args.orbit_mask_coupling,
  )

  generator = torch.Generator()
  generator.manual_seed(args.seed)
  train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    persistent_workers=args.num_workers > 0,
    worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    generator=generator,
  )
  orbit_loader = DataLoader(
    orbit_ds,
    batch_size=1,
    shuffle=True,
    collate_fn=_orbit_collate,
    num_workers=args.num_workers,
    pin_memory=args.pin_memory,
    persistent_workers=args.num_workers > 0,
    worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
  )

  valid_loader: Optional[DataLoader] = None
  valid_orbit_loader: Optional[DataLoader] = None
  n_valid_disk_for_cfg: Optional[int] = None
  n_valid_used_for_cfg: Optional[int] = None
  valid_npy_root = Path(args.valid_npy_root) if args.valid_npy_root else npy_root
  if not valid_npy_root.is_absolute():
    valid_npy_root = MDLM_ROOT / valid_npy_root
  try:
    valid_solutions, valid_anchors = _load_split_arrays(valid_npy_root, 'valid')
    n_valid_disk = int(valid_solutions.shape[0])
    valid_indices = np.arange(n_valid_disk, dtype=np.int64)
    if int(args.valid_max_examples) > 0:
      n_use = min(int(args.valid_max_examples), n_valid_disk)
      valid_indices = valid_indices[:n_use].copy()
    n_valid_disk_for_cfg = n_valid_disk
    n_valid_used_for_cfg = int(len(valid_indices))
    valid_ds = SudokuFullBlankSFTDataset(
      valid_solutions,
      valid_anchors,
      valid_indices,
      recipe='canonical_sft_80k',
      seed=args.seed,
    )
    valid_loader = DataLoader(
      valid_ds,
      batch_size=args.batch_size,
      shuffle=False,
      drop_last=False,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory,
      persistent_workers=args.num_workers > 0,
      worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )
    valid_orbit_ds = SudokuOrbitSoftOHChunkDataset(
      valid_solutions,
      valid_anchors,
      valid_indices,
      orbit_base_size=orbit_B,
      seed=args.seed + 1,
      blank_mask_mode=args.blank_mask_mode,
      blank_mask_eps=args.blank_mask_eps,
      sft_time=args.sft_time,
      sample_mode='sequential',
      stochastic_masks=False,
      orbit_mask_coupling=args.orbit_mask_coupling,
    )
    valid_orbit_loader = DataLoader(
      valid_orbit_ds,
      batch_size=1,
      shuffle=False,
      collate_fn=_orbit_collate,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory,
      persistent_workers=args.num_workers > 0,
      worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )
  except FileNotFoundError:
    valid_loader = None
    valid_orbit_loader = None
    n_valid_disk_for_cfg = None
    n_valid_used_for_cfg = None

  if args.best_checkpoint_key == 'orbit_group_loss' and valid_orbit_loader is None:
    print(
      '[posttrain] warning: --best-checkpoint-key orbit_group_loss but no valid split; '
      'falling back to val_sft_loss for best.ckpt.',
    )

  compose_bs = max(int(args.batch_size), int(args.orbit_batch_size))
  cfg = _compose_config(
    checkpoint=args.checkpoint,
    model=args.model,
    batch_size=compose_bs,
    num_steps=args.num_steps,
    predictor=args.predictor,
    noise_removal=not args.no_noise_removal,
    extra_overrides=args.extra_override,
  )
  model = _load_model(cfg, device)
  inv_perms_cpu = _build_inv_perms_t()
  perms_cpu = _build_perms_t()

  if args.sanity_check:
    _run_orbit_mask_sanity(orbit_loader, inv_perms_cpu)
    _soft_oh_numeric_sanity()
    print('[posttrain] sanity_check passed.')

  if not args.train_noise:
    for p in model.noise.parameters():
      p.requires_grad_(False)

  if args.train_noise:
    params = list(model.backbone.parameters()) + list(model.noise.parameters())
  else:
    params = list(model.backbone.parameters())
  optimizer = torch.optim.AdamW(
    [p for p in params if p.requires_grad],
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
    weight_decay=args.weight_decay,
  )
  scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=_build_lr_lambda(total_opt_steps, args.warmup_steps, args.min_lr_ratio),
  )

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  config_summary = {
    'args': vars(args),
    'recipe_defaults': recipe_cfg,
    'npy_root': str(npy_root),
    'valid_npy_root': str(valid_npy_root),
    'num_train_base_examples': int(len(base_indices)),
    'num_train_views_per_epoch': int(len(train_ds)),
    'num_sft_rows_per_epoch_after_drop_last': int((len(train_ds) // int(args.batch_size)) * int(args.batch_size)),
    'num_orbit_chunks_per_epoch': int(len(orbit_ds)),
    'num_orbit_rows_per_epoch': int(len(orbit_ds) * int(args.orbit_batch_size)),
    'orbit_base_B': orbit_B,
    'orbit_batch_size': int(args.orbit_batch_size),
    'r_orbit_effective': r_orbit,
    'has_valid_split': valid_loader is not None,
    'has_valid_orbit': valid_orbit_loader is not None,
    'num_valid_rows_on_disk': n_valid_disk_for_cfg,
    'num_valid_examples_used': n_valid_used_for_cfg,
    'total_opt_steps': total_opt_steps,
    'total_micro_steps': int(args.max_steps),
    'best_checkpoint_key_effective': (
      args.best_checkpoint_key
      if args.best_checkpoint_key != 'orbit_group_loss' or valid_orbit_loader is not None
      else 'sft_loss'),
    'method_semantics': _method_semantics(args),
  }
  (out_dir / 'posttrain_config.json').write_text(json.dumps(config_summary, indent=2), encoding='utf-8')
  print(json.dumps(config_summary, indent=2))

  step = 0
  optim_step = 0
  best_val_sft = float('inf')
  best_val_orbit = float('inf')
  running_loss = 0.0
  running_acc = 0.0
  running_steps = 0
  log_window_steps = 0
  log_window_orbit = 0
  optimizer.zero_grad(set_to_none=True)

  data_iter = iter(train_loader)
  orbit_iter = iter(orbit_loader)
  orbit_metric_sums: Dict[str, float] = {}
  orbit_metric_count = 0
  _orbit_log_keys = (
      'train_worst', 'train_gap_soft_mu', 'train_orbit_var',
      'train_mu', 'train_l_soft', 'train_l_group',
  )
  lam_used = 0.0

  while step < args.max_steps:
    lam_eff = _lambda_eff_warmup(
      optim_step + 1, total_opt_steps, args.soft_oh_lambda, args.lambda_warmup_frac)
    lam_used = float(lam_eff)
    use_orbit = r_orbit > 0.0 and (r_orbit >= 1.0 or torch.rand(1).item() < r_orbit)
    if use_orbit:
      try:
        ob = next(orbit_iter)
      except StopIteration:
        orbit_iter = iter(orbit_loader)
        ob = next(orbit_iter)
      loss, info = _orbit_soft_oh_loss(
        model,
        ob,
        device=device,
        inv_perms=inv_perms_cpu,
        perms=perms_cpu,
        label_smoothing=args.label_smoothing,
        alpha=args.soft_oh_alpha,
        lambda_eff=lam_eff,
        blank_mask_mode=args.blank_mask_mode,
        blank_mask_eps=args.blank_mask_eps,
        sft_time=args.sft_time,
        resample_corruption=True,
        orbit_mask_coupling=args.orbit_mask_coupling,
      )
      for k in _orbit_log_keys:
        if k in info:
          orbit_metric_sums[k] = orbit_metric_sums.get(k, 0.0) + float(info[k])
      orbit_metric_count += 1
    else:
      try:
        batch = next(data_iter)
      except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
      loss, info = _sft_loss_full_blank(
        model,
        batch,
        device=device,
        sft_time=args.sft_time,
        label_smoothing=args.label_smoothing,
        blank_mask_mode=args.blank_mask_mode,
        blank_mask_eps=args.blank_mask_eps,
      )

    scaled_loss = loss / max(1, args.grad_accum_steps)
    scaled_loss.backward()

    running_loss += float(loss.item())
    running_acc += float(info['sft_token_acc'])
    running_steps += 1
    log_window_steps += 1
    log_window_orbit += 1.0 if use_orbit else 0.0

    if (step + 1) % args.grad_accum_steps == 0:
      if args.grad_clip and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
      optimizer.step()
      scheduler.step()
      optim_step += 1
      optimizer.zero_grad(set_to_none=True)

    step += 1

    if step % args.log_every == 0 or step == 1:
      row = {
        'step': step,
        'optim_step': optim_step,
        'train_loss': running_loss / max(float(running_steps), 1.0),
        'train_sft_token_acc': running_acc / max(float(running_steps), 1.0),
        'train_orbit_frac': log_window_orbit / max(float(log_window_steps), 1.0),
        'lambda_eff': lam_used,
        'lr': optimizer.param_groups[0]['lr'],
        'recipe': args.recipe,
      }
      if orbit_metric_count > 0:
        denom_om = float(orbit_metric_count)
        for k, s in orbit_metric_sums.items():
          row[k] = s / denom_om
      print(json.dumps(row))
      _write_jsonl(out_dir / 'train_log.jsonl', row)
      running_loss = 0.0
      running_acc = 0.0
      running_steps = 0
      log_window_steps = 0
      log_window_orbit = 0.0
      orbit_metric_sums = {}
      orbit_metric_count = 0

    if valid_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
      val_metrics = _evaluate_sft_loss(
        model,
        valid_loader,
        device=device,
        sft_time=args.sft_time,
        label_smoothing=args.label_smoothing,
        max_batches=args.eval_max_batches,
      )
      val_row: Dict[str, Any] = {'step': step, **val_metrics, 'recipe': args.recipe}
      if valid_orbit_loader is not None:
        orb_m = _evaluate_orbit_soft_oh(
          model,
          valid_orbit_loader,
          device=device,
          inv_perms=inv_perms_cpu,
          label_smoothing=args.label_smoothing,
          alpha=args.soft_oh_alpha,
          lambda_target=args.soft_oh_lambda,
          max_batches=args.eval_orbit_max_batches,
          orbit_mask_coupling=args.orbit_mask_coupling,
        )
        val_row.update(orb_m)
      print(json.dumps(val_row))
      _write_jsonl(out_dir / 'valid_log.jsonl', val_row)

      best_key = args.best_checkpoint_key
      if best_key == 'orbit_group_loss' and valid_orbit_loader is None:
        best_key = 'sft_loss'
      improved = False
      if best_key == 'orbit_group_loss':
        v_orb = float(val_row.get('val_orbit_l_group', float('inf')))
        if v_orb < best_val_orbit:
          best_val_orbit = v_orb
          improved = True
      else:
        if val_metrics['val_sft_loss'] < best_val_sft:
          best_val_sft = val_metrics['val_sft_loss']
          improved = True
      if improved:
        _save_checkpoint(
          out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, val_row)
        print(f'[posttrain] saved best.ckpt at step={step} key={best_key}: {out_dir / "best.ckpt"}')

    if args.save_every > 0 and step % args.save_every == 0:
      _save_checkpoint(
        out_dir / f'step_{step}.ckpt', model, optimizer, scheduler, args, step,
        {'step': step, 'recipe': args.recipe})
      print(f'[posttrain] saved checkpoint: {out_dir / f"step_{step}.ckpt"}')

  if int(args.max_steps) > 0 and (int(args.max_steps) % int(args.grad_accum_steps)) != 0:
    if args.grad_clip and args.grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
    optimizer.step()
    scheduler.step()
    optim_step += 1
    optimizer.zero_grad(set_to_none=True)
    print(
      f'[posttrain] flushed partial grad-accum buffer '
      f'(max_steps={args.max_steps} % grad_accum_steps={args.grad_accum_steps} != 0).',
    )

  final_metrics = {
    'step': step,
    'recipe': args.recipe,
    'best_val_sft_loss': best_val_sft,
    'best_val_orbit_l_group': best_val_orbit,
  }
  _save_checkpoint(out_dir / 'last.ckpt', model, optimizer, scheduler, args, step, final_metrics)
  if valid_loader is None:
    _save_checkpoint(out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, final_metrics)
  print(f'[posttrain] done. last={out_dir / "last.ckpt"}; best={out_dir / "best.ckpt"}')


if __name__ == '__main__':
  main()
