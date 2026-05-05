#!/usr/bin/env python3
"""Post-train an MDLM Sudoku checkpoint with clue-conditioned SFT and SFS.

This script implements post-training recipes for 9x9 Sudoku, including
Solver-Frontier Supervision (SFS) when target masks are provided.

New capabilities:
  - Supports `--use-target-masks` to load oracle ``train_target_mask`` rows.
    By default loss is only on the frontier; with ``--dense-sfs-loss`` loss
    is on **all blank** cells with higher weight on frontier cells.
  - ``--online-random-mask-sfs``: train-time random masking (``rand < t`` per
    cell, ``t ~ Uniform(eps,1)`` per sequence) on the gold grid, then run the
    logical oracle's **first wave** frontier for SFS (no ``train_target_mask``).
  - Added ``sfs_80k`` and ``sfs_transform_80k`` recipes.
  - Optional orbit validation (same ``val_orbit_l_group`` as ``sudoku_posttrain_soft_oh``) via
    ``--best-checkpoint-key orbit_group_loss``; best weights are saved as ``best_step_<step>.ckpt``.

Expected npy layout under ``--npy-root``:

**Flat bundle:** ``train_solution.npy``, ``train_anchor.npy``, optional ``train_target_mask.npy``.

**Sharded** (from ``build_random_mask_sfs_npy.py --split-epochs``): subdirs ``epoch_0000``, ``epoch_0001``, ...
each containing the same three filenames; logical row order is epoch-major (all rows of
``epoch_0000``, then ``epoch_0001``, ...). Optional **flat** ``valid_*.npy`` at the same
``--npy-root`` (from ``--valid-base-npy-dir`` when building) are used for evaluation.

Example SFS run:
  python posttrain/sudoku_posttrain_sfs.py \
    --checkpoint outputs/.../checkpoints/best.ckpt \
    --model sudoku_50m \
    --npy-root dataset/3m_only_posttrain_npy \
    --recipe sfs_80k \
    --use-target-masks \
    --batch-size 512 \
    --max-steps 3000 \
    --lr 1e-5 \
    --output-dir outputs/posttrain/sfs_50m
"""

from __future__ import annotations

import argparse
import importlib.util
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
  'canonical_sft_80k': dict(num_base_examples=80_000, orbit_expand=False, random_transform=False),
  'random_transform_sft_80k': dict(num_base_examples=80_000, orbit_expand=False, random_transform=True),
  'full_orbit_sft_10k': dict(num_base_examples=10_000, orbit_expand=True, random_transform=False),
  'repeat8_sft_10k': dict(num_base_examples=10_000, orbit_expand=False, random_transform=False, repeat_factor=8),
  'sfs_80k': dict(num_base_examples=80_000, orbit_expand=False, random_transform=False),
  'sfs_transform_80k': dict(num_base_examples=80_000, orbit_expand=False, random_transform=True),
}


def _register_omegaconf_resolvers() -> None:
  OmegaConf.register_new_resolver('cwd', os.getcwd, replace=True)
  OmegaConf.register_new_resolver('mdlm_root', lambda: str(MDLM_ROOT), replace=True)
  OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
  OmegaConf.register_new_resolver('eval', eval, replace=True)
  OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)


def _apply_transform(grid: np.ndarray, transform_name: str) -> np.ndarray:
  """Apply a D4 transform to a [9,9] array.

  The convention matches the evaluation scripts used in this project.
  """
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
  """perm[i] = canonical index at transformed cell i → inv[j] = transformed cell showing canon j."""
  inv = np.empty(81, dtype=np.int64)
  inv[perm] = np.arange(81, dtype=np.int64)
  return inv


def _build_inv_perms_t() -> torch.Tensor:
  """[8, 81] long: inv[g, c] = transformed flat index for canonical cell c under transform g."""
  ti = _build_transform_indices()
  inv_list = [_invert_perm_np(ti[name]) for name in TRANSFORMS]
  inv = np.stack(inv_list, axis=0)
  return torch.from_numpy(inv).long()


_SOFT_OH_EVAL_MOD = None


def _soft_oh_eval_mod():
  """Lazy import of ``sudoku_posttrain_soft_oh`` for orbit validation only (no training dep)."""
  global _SOFT_OH_EVAL_MOD
  if _SOFT_OH_EVAL_MOD is not None:
    return _SOFT_OH_EVAL_MOD
  import importlib.util

  path = MDLM_ROOT / 'posttrain' / 'sudoku_posttrain_soft_oh.py'
  spec = importlib.util.spec_from_file_location('sudoku_posttrain_soft_oh_sfs_eval', path)
  if spec is None or spec.loader is None:
    raise ImportError(f'Cannot load soft_oh module from {path}')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  _SOFT_OH_EVAL_MOD = mod
  return mod


def _discover_epoch_shard_dirs(root: Path) -> List[Path]:
  if not root.is_dir():
    return []
  out: List[Path] = []
  for p in sorted(root.iterdir()):
    if p.is_dir() and p.name.startswith('epoch_'):
      out.append(p)
  return out


class _ShardedArrayChain:
  """Row-major view over equally-sized [rows_per, ...] memmap shards (epoch_*, ...)."""

  def __init__(self, paths: List[Path], *, label: str) -> None:
    if not paths:
      raise ValueError(f'{label}: no shard npy paths')
    self._parts: List[np.ndarray] = []
    for path in paths:
      if not path.exists():
        raise FileNotFoundError(f'Missing {path}')
      self._parts.append(np.load(path, mmap_mode='r'))
    n_per = int(self._parts[0].shape[0])
    base_tail = tuple(int(x) for x in self._parts[0].shape[1:])
    for i, a in enumerate(self._parts):
      if int(a.shape[0]) != n_per:
        raise ValueError(
            f'{label}: shard {paths[i]} has rows {a.shape[0]}, expected {n_per} like {paths[0]}')
      if tuple(int(x) for x in a.shape[1:]) != base_tail:
        raise ValueError(f'{label}: trailing dims mismatch at {paths[i]}')
    self._rows_per_shard = n_per
    self.shape = (n_per * len(self._parts),) + base_tail
    self.ndim = len(self.shape)
    self.num_shards = int(len(self._parts))

  def __getitem__(self, idx: Any) -> np.ndarray:
    ii = int(idx)
    if ii < 0 or ii >= int(self.shape[0]):
      raise IndexError(ii)
    s = ii // self._rows_per_shard
    r = ii % self._rows_per_shard
    return self._parts[s][r]


def _load_flat_split_arrays(
    root: Path,
    split_name: str,
    *,
    use_target_masks: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
  solution_path = root / f'{split_name}_solution.npy'
  anchor_path = root / f'{split_name}_anchor.npy'
  if not solution_path.exists():
    raise FileNotFoundError(f'Missing {solution_path}')
  if not anchor_path.exists():
    raise FileNotFoundError(f'Missing {anchor_path}')
  solutions = np.load(solution_path, mmap_mode='r')
  anchors = np.load(anchor_path, mmap_mode='r')

  targets: Optional[np.ndarray] = None
  if use_target_masks:
    target_path = root / f'{split_name}_target_mask.npy'
    if not target_path.exists():
      target_path = root / f'{split_name}_target.npy'
    if not target_path.exists():
      raise FileNotFoundError(
          f'Missing target mask for {split_name} (required by --use-target-masks): '
          f'expected {split_name}_target_mask.npy or {split_name}_target.npy')
    targets = np.load(target_path, mmap_mode='r')
    if targets.shape != solutions.shape:
      raise ValueError(f'Target shape {targets.shape} != solutions shape {solutions.shape}')

  if solutions.ndim != 2 or solutions.shape[1] != 81:
    raise ValueError(f'{solution_path} must have shape [N,81], got {solutions.shape}')
  if anchors.shape != solutions.shape:
    raise ValueError(f'{anchor_path} shape {anchors.shape} != solutions shape {solutions.shape}')
  return solutions, anchors, targets


def _load_sharded_split_arrays(
    shard_dirs: List[Path],
    split_name: str,
    *,
    use_target_masks: bool,
) -> Tuple[_ShardedArrayChain, _ShardedArrayChain, Optional[_ShardedArrayChain]]:
  sol_paths = [d / f'{split_name}_solution.npy' for d in shard_dirs]
  anc_paths = [d / f'{split_name}_anchor.npy' for d in shard_dirs]
  solutions = _ShardedArrayChain(sol_paths, label=f'{split_name}_solution')
  anchors = _ShardedArrayChain(anc_paths, label=f'{split_name}_anchor')
  if solutions.shape != anchors.shape:
    raise ValueError(f'Sharded anchor shape {anchors.shape} != solutions {solutions.shape}')
  targets: Optional[_ShardedArrayChain] = None
  if use_target_masks:
    tgt_paths: List[Path] = []
    for d in shard_dirs:
      tp = d / f'{split_name}_target_mask.npy'
      if not tp.exists():
        tp = d / f'{split_name}_target.npy'
      tgt_paths.append(tp)
    targets = _ShardedArrayChain(tgt_paths, label=f'{split_name}_target_mask')
    if targets.shape != solutions.shape:
      raise ValueError(f'Sharded target shape {targets.shape} != solutions {solutions.shape}')
  return solutions, anchors, targets


def _load_split_arrays(
    root: Path,
    split: str,
    *,
    use_target_masks: bool = False,
) -> Tuple[Any, Any, Optional[Any]]:
  split_name = {'validation': 'valid', 'val': 'valid'}.get(split, split)
  flat_sol = root / f'{split_name}_solution.npy'
  if flat_sol.exists():
    return _load_flat_split_arrays(root, split_name, use_target_masks=use_target_masks)
  shard_dirs = _discover_epoch_shard_dirs(root)
  if shard_dirs:
    probe = shard_dirs[0] / f'{split_name}_solution.npy'
    if probe.exists():
      return _load_sharded_split_arrays(
          shard_dirs, split_name, use_target_masks=use_target_masks)
  raise FileNotFoundError(
      f'Missing flat {flat_sol} and no sharded "{split_name}_solution.npy" under epoch_* in {root}')


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


class SudokuSFTDataset(Dataset):
  """Clue-conditioned Sudoku SFT and SFS dataset.

  Returns:
    input_ids: gold solution ids [81], values 0..8
    anchor_mask: bool [81], True for clues that remain visible
    loss_mask: bool [81], True for tokens where loss is computed.
               If targets are provided (SFS) and not ``dense_sfs_loss``,
               ``loss_mask = targets & ~anchor``.
               Otherwise (full-blank SFT), ``loss_mask = ~anchor``.
    loss_weights: optional float [81]; when present, weighted CE over
      positive-weight positions (used for dense SFS: all blanks + frontier boost).

  The training loop builds ``x_cond = where(anchor_mask, input_ids, mask_id)``.
  """

  def __init__(
      self,
      solutions: np.ndarray,
      anchors: np.ndarray,
      targets: Optional[np.ndarray],
      base_indices: np.ndarray,
      *,
      recipe: str,
      seed: int = 0,
      dense_sfs_loss: bool = False,
      frontier_weight_mult: float = 2.0,
  ) -> None:
    if recipe not in RECIPE_DEFAULTS:
      raise ValueError(f'Unknown recipe {recipe}; choices={list(RECIPE_DEFAULTS)}')
    self.solutions = solutions
    self.anchors = anchors
    self.targets = targets
    self.base_indices = np.asarray(base_indices, dtype=np.int64)
    self.recipe = recipe
    self.random_transform = bool(RECIPE_DEFAULTS[recipe]['random_transform'])
    self.orbit_expand = bool(RECIPE_DEFAULTS[recipe]['orbit_expand'])
    self.transform_indices = _build_transform_indices()
    self.seed = int(seed)
    self.dense_sfs_loss = bool(dense_sfs_loss)
    self.frontier_weight_mult = float(frontier_weight_mult)

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
      # Use numpy's global RNG inside each worker. worker_init_fn seeds it.
      return TRANSFORMS[int(np.random.randint(0, len(TRANSFORMS)))]
    return 'identity'

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    repeat_factor = int(RECIPE_DEFAULTS[self.recipe].get('repeat_factor', 1))
    if self.orbit_expand:
      base_pos = idx // len(TRANSFORMS)
    elif repeat_factor > 1:
      # Repeat each selected base sample a fixed number of times for deterministic
      # exposure budgeting (e.g., exact 10k x 8 views).
      base_pos = idx % len(self.base_indices)
    else:
      base_pos = idx
    arr_idx = int(self.base_indices[base_pos])
    transform_name = self._choose_transform(idx)
    perm = self.transform_indices[transform_name]

    sol = np.asarray(self.solutions[arr_idx], dtype=np.int64)[perm].copy()
    anchor = np.asarray(self.anchors[arr_idx], dtype=np.bool_)[perm].copy()
    out: Dict[str, torch.Tensor] = {
      'input_ids': torch.from_numpy(sol).long(),
      'anchor_mask': torch.from_numpy(anchor).bool(),
      'transform_id': torch.tensor(TRANSFORMS.index(transform_name), dtype=torch.long),
      'base_index': torch.tensor(arr_idx, dtype=torch.long),
    }
    if self.targets is not None:
      tgt = np.asarray(self.targets[arr_idx], dtype=np.bool_)[perm].copy()
      if self.dense_sfs_loss:
        frontier = tgt & (~anchor)
        w = np.zeros(81, dtype=np.float32)
        w[~anchor] = 1.0
        w[frontier] = np.float32(self.frontier_weight_mult)
        out['loss_weights'] = torch.from_numpy(w)
        out['loss_mask'] = torch.from_numpy(~anchor).bool()
      else:
        out['loss_mask'] = torch.from_numpy(tgt & (~anchor)).bool()
    else:
      out['loss_mask'] = torch.from_numpy(~anchor).bool()

    return out


def _worker_init_fn(worker_id: int) -> None:
  # Deterministic but distinct RNG streams per worker.
  base_seed = torch.initial_seed() % (2**32)
  np.random.seed(base_seed + worker_id)
  random.seed(base_seed + worker_id)


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
    # Strip optional reference/backbone copies from previous experiments.
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

  # For SFT post-training we train the current weights directly and do not keep
  # Lightning EMA state inside the module. This also makes saving/loading simple.
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


_SFS_ORACLE_MOD = None


def _sfs_oracle_module():
  """Load scripts/build_sfs_frontier_npy.py (logical oracle for online SFS)."""
  global _SFS_ORACLE_MOD
  if _SFS_ORACLE_MOD is None:
    path = MDLM_ROOT / 'scripts' / 'build_sfs_frontier_npy.py'
    spec = importlib.util.spec_from_file_location('build_sfs_frontier_npy', path)
    if spec is None or spec.loader is None:
      raise ImportError(f'Cannot load SFS oracle from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _SFS_ORACLE_MOD = mod
  return _SFS_ORACLE_MOD


def _random_hide_mask(batch_size: int, seq_len: int, device: torch.device, eps: float) -> torch.Tensor:
  """Bernoulli hide mask: True = replace with [MASK] (matches forward_process recipe)."""
  t = torch.rand((batch_size,), device=device, dtype=torch.float32)
  t = (1.0 - float(eps)) * t + float(eps)
  t = t[:, None].expand(batch_size, seq_len)
  return torch.rand((batch_size, seq_len), device=device) < t


def _online_first_wave_loss_mask(
    input_ids: torch.Tensor,
    anchor_mask: torch.Tensor,
    *,
    oracle: str,
) -> torch.Tensor:
  """First-wave frontier ∩ blank cells (CPU oracle loop per batch row)."""
  mod = _sfs_oracle_module()
  B = int(input_ids.shape[0])
  sol_np = input_ids.detach().cpu().numpy().astype(np.int64)
  anc_np = anchor_mask.detach().cpu().numpy().astype(np.bool_)
  masks: List[np.ndarray] = []
  for b in range(B):
    try:
      waves = mod.logical_frontier_waves_from_sol_anchor(sol_np[b], anc_np[b], oracle=oracle, max_waves=1)
    except (ValueError, RuntimeError):
      waves = []
    if waves:
      m = np.asarray(waves[0], dtype=np.bool_) & (~anc_np[b])
    else:
      m = np.zeros(81, dtype=np.bool_)
    masks.append(m)
  stacked = np.stack(masks, axis=0)
  return torch.from_numpy(stacked).to(device=input_ids.device, dtype=torch.bool)


def _sft_loss_full_blank(
    model: diffusion_mod.Diffusion,
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    sft_time: float,
    label_smoothing: float = 0.0,
    online_random_mask_sfs: bool = False,
    random_mask_eps: float = 1e-3,
    sfs_oracle: str = 'strong',
    dense_sfs_loss: bool = False,
    sfs_frontier_weight_mult: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
  input_ids = batch['input_ids'].to(device=device, dtype=torch.long)
  loss_weights: Optional[torch.Tensor] = None
  if online_random_mask_sfs:
    B, N = input_ids.shape
    hide = _random_hide_mask(B, N, device, random_mask_eps)
    clue = batch['anchor_mask'].to(device=device, dtype=torch.bool)
    anchor_mask = clue.clone()
    blanks = ~clue
    anchor_mask[blanks] = (~hide)[blanks]
    frontier_mask = _online_first_wave_loss_mask(input_ids, anchor_mask, oracle=sfs_oracle)
    if dense_sfs_loss:
      w = torch.zeros((B, N), device=device, dtype=torch.float32)
      blank = ~anchor_mask
      w[blank] = 1.0
      w[frontier_mask] = float(sfs_frontier_weight_mult)
      loss_weights = w
    else:
      loss_mask = frontier_mask
  else:
    anchor_mask = batch['anchor_mask'].to(device=device, dtype=torch.bool)
    if 'loss_weights' in batch:
      loss_weights = batch['loss_weights'].to(device=device, dtype=torch.float32)
    loss_mask = batch['loss_mask'].to(device=device, dtype=torch.bool)
  x_cond = _make_conditioned_input(input_ids, anchor_mask, model.mask_index)

  # Use the same time-conditioning path as diffusion training.  With t=1 and
  # clue-only input, this matches the full-blank infill state used at sampling start.
  t = torch.full((input_ids.shape[0],), float(sft_time), device=device, dtype=model.dtype)
  sigma, _ = model.noise(t)
  logits = model.forward(x_cond, sigma[:, None])  # log-probs for SUBS parameterization

  # Exclude mask token from the supervised Sudoku digit loss.
  if logits.shape[-1] > 9:
    logits_for_ce = logits[..., :9]
  else:
    logits_for_ce = logits

  flat_logits = logits_for_ce.reshape(-1, logits_for_ce.shape[-1])
  flat_targets = input_ids.reshape(-1)

  if loss_weights is not None:
    flat_w = loss_weights.reshape(-1)
    denom = float(flat_w.sum().item())
    if denom <= 0.0:
      ce = flat_logits.sum() * 0.0
      return ce, {'sft_token_acc': 0.0, 'num_loss_tokens': 0.0}
    ce_n = F.cross_entropy(
      flat_logits,
      flat_targets,
      reduction='none',
      label_smoothing=float(label_smoothing),
    )
    ce = (ce_n * flat_w).sum() / flat_w.sum().clamp_min(1e-12)
    with torch.no_grad():
      pred = flat_logits.argmax(dim=-1)
      correct = (pred == flat_targets).float()
      acc = float((correct * flat_w).sum().item() / denom)
    return ce, {'sft_token_acc': acc, 'num_loss_tokens': denom}

  flat_mask = loss_mask.reshape(-1)
  if flat_mask.sum().item() == 0:
    ce = flat_logits.sum() * 0.0
    return ce, {'sft_token_acc': 0.0, 'num_loss_tokens': 0.0}

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


@torch.no_grad()
def _evaluate_sft_loss(
    model: diffusion_mod.Diffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    sft_time: float,
    label_smoothing: float,
    max_batches: int,
    online_random_mask_sfs: bool = False,
    random_mask_eps: float = 1e-3,
    sfs_oracle: str = 'strong',
    dense_sfs_loss: bool = False,
    sfs_frontier_weight_mult: float = 2.0,
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
      online_random_mask_sfs=online_random_mask_sfs,
      random_mask_eps=random_mask_eps,
      sfs_oracle=sfs_oracle,
      dense_sfs_loss=dense_sfs_loss,
      sfs_frontier_weight_mult=sfs_frontier_weight_mult,
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
  p = argparse.ArgumentParser(description='Post-train MDLM Sudoku with SFT and SFS.')
  p.add_argument('--checkpoint', type=str, required=True)
  p.add_argument('--model', type=str, default='sudoku_50m')
  p.add_argument(
      '--npy-root', type=str, required=True,
      help='Flat dir with train_*.npy, or sharded parent with epoch_*/train_*.npy per epoch.',
  )
  p.add_argument('--valid-npy-root', type=str, default='',
                 help='Optional directory with valid_solution.npy/valid_anchor.npy. '
                 'If empty, validation files are loaded from --npy-root.')
  p.add_argument(
      '--valid-max-examples',
      type=int,
      default=0,
      help='If > 0, use only the first N validation rows (prefix of valid_*.npy). 0 = all.',
  )
  p.add_argument('--recipe', type=str, required=True, choices=sorted(RECIPE_DEFAULTS))
  p.add_argument('--output-dir', type=str, required=True)

  p.add_argument(
      '--use-target-masks',
      action='store_true',
      default=False,
      help='Load train_*_target_mask.npy for SFS (default: CE only on frontier).',
  )
  p.add_argument(
      '--dense-sfs-loss',
      action='store_true',
      default=False,
      help='With --use-target-masks (or --online-random-mask-sfs): CE on all blank cells; '
      'frontier cells use --sfs-frontier-weight-mult (default 2) higher weight.',
  )
  p.add_argument(
      '--sfs-frontier-weight-mult',
      type=float,
      default=2.0,
      help='Per-token loss multiplier on frontier vs other blanks (dense SFS only). Must be >= 1.',
  )
  p.add_argument(
      '--online-random-mask-sfs',
      action='store_true',
      default=False,
      help='Each step: random t mask on gold grid (forward_process-style), then oracle first-wave SFS targets.',
  )
  p.add_argument('--random-mask-eps', type=float, default=1e-3,
                 help='Epsilon floor on t in random masking (same role as forward_process eps).')
  p.add_argument('--sfs-oracle', type=str, default='strong', choices=('strong', 'singles'),
                 help='Logical oracle for online SFS (scripts/build_sfs_frontier_npy.py).')

  p.add_argument('--num-base-examples', type=int, default=0,
                 help='Override recipe default. <=0 uses recipe default; capped by dataset length.')
  p.add_argument('--subset-mode', type=str, default='random', choices=('random', 'first'))
  p.add_argument('--subset-seed', type=int, default=0)
  p.add_argument('--seed', type=int, default=0)

  p.add_argument('--batch-size', type=int, default=512)
  p.add_argument('--num-workers', type=int, default=4)
  p.add_argument('--pin-memory', action='store_true', default=True)
  p.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
  p.add_argument('--grad-accum-steps', type=int, default=1)
  p.add_argument('--max-steps', type=int, default=3000)
  p.add_argument('--lr', type=float, default=1e-5)
  p.add_argument('--weight-decay', type=float, default=0.01)
  p.add_argument('--beta1', type=float, default=0.9)
  p.add_argument('--beta2', type=float, default=0.999)
  p.add_argument('--eps', type=float, default=1e-8)
  p.add_argument('--warmup-steps', type=int, default=150)
  p.add_argument('--min-lr-ratio', type=float, default=0.1)
  p.add_argument('--grad-clip', type=float, default=1.0)
  p.add_argument('--label-smoothing', type=float, default=0.0)
  p.add_argument('--sft-time', type=float, default=1.0,
                 help='Diffusion time conditioning used for clue-only full-blank SFT.')
  p.add_argument('--train-noise', action='store_true', default=False,
                 help='By default only backbone params are optimized; enable to include noise params.')

  p.add_argument('--eval-every', type=int, default=500)
  p.add_argument('--eval-max-batches', type=int, default=32)
  p.add_argument(
      '--eval-orbit-max-batches',
      type=int,
      default=32,
      help='Max orbit chunks for val_orbit_l_group (same chunking as sudoku_posttrain_soft_oh).',
  )
  p.add_argument(
      '--best-checkpoint-key',
      type=str,
      default='sft_loss',
      choices=('sft_loss', 'orbit_group_loss'),
      help='Which validation metric improves best_step_*.ckpt: val_sft_loss or val_orbit_l_group.',
  )
  p.add_argument(
      '--orbit-batch-size',
      type=int,
      default=0,
      help='Rows per orbit val chunk = B*8. 0 means match --batch-size; must be divisible by 8.',
  )
  p.add_argument(
      '--orbit-mask-coupling',
      type=str,
      default='shared',
      choices=('shared', 'per_view'),
      help='Orbit val corruption coupling (passed to SudokuOrbitSoftOHChunkDataset).',
  )
  p.add_argument(
      '--orbit-val-blank-mask-mode',
      type=str,
      default='random',
      choices=('random', 'full'),
      help='Blank corruption for orbit validation chunks.',
  )
  p.add_argument('--orbit-val-blank-mask-eps', type=float, default=1e-3)
  p.add_argument(
      '--orbit-eval-soft-oh-alpha',
      type=float,
      default=2.0,
      help='SoftOH alpha for orbit val only (see sudoku_posttrain_soft_oh).',
  )
  p.add_argument(
      '--orbit-eval-soft-oh-lambda',
      type=float,
      default=0.5,
      help='SoftOH lambda for orbit val only; must be in [0, 1].',
  )
  p.add_argument('--save-every', type=int, default=1000)
  p.add_argument('--log-every', type=int, default=50)

  p.add_argument('--predictor', type=str, default='ddpm_cache')
  p.add_argument('--num-steps', type=int, default=128)
  p.add_argument('--no-noise-removal', action='store_true')
  p.add_argument('--device', type=str, default='cuda')
  p.add_argument('--extra-override', action='append', default=[],
                 help='Additional Hydra override, repeatable, e.g. optim.lr=1e-5')
  return p.parse_args()


def main() -> None:
  args = parse_args()
  _register_omegaconf_resolvers()

  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('CUDA requested but not available')
  device = torch.device(args.device)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)
  if device.type == 'cuda':
    torch.cuda.manual_seed_all(args.seed)

  recipe_cfg = RECIPE_DEFAULTS[args.recipe]
  num_base_examples = int(args.num_base_examples) if args.num_base_examples > 0 else int(recipe_cfg['num_base_examples'])

  npy_root = Path(args.npy_root)
  if not npy_root.is_absolute():
    npy_root = MDLM_ROOT / npy_root
  if args.online_random_mask_sfs and args.use_target_masks:
    raise SystemExit('Use either --online-random-mask-sfs or --use-target-masks, not both.')
  if args.dense_sfs_loss and not args.use_target_masks and not args.online_random_mask_sfs:
    raise SystemExit('--dense-sfs-loss requires --use-target-masks or --online-random-mask-sfs')
  if float(args.sfs_frontier_weight_mult) < 1.0:
    raise SystemExit('--sfs-frontier-weight-mult must be >= 1')

  orbit_bs = int(args.orbit_batch_size)
  if orbit_bs <= 0:
    orbit_bs = int(args.batch_size)
  if orbit_bs < 8 or orbit_bs % 8 != 0:
    raise SystemExit('--orbit-batch-size (or --batch-size when orbit size is 0) must be >= 8 and divisible by 8')
  orbit_B = orbit_bs // 8
  if int(args.eval_orbit_max_batches) < 0:
    raise SystemExit('--eval-orbit-max-batches must be >= 0')
  if float(args.orbit_eval_soft_oh_lambda) < 0.0 or float(args.orbit_eval_soft_oh_lambda) > 1.0:
    raise SystemExit('--orbit-eval-soft-oh-lambda must be in [0, 1]')
  if float(args.orbit_eval_soft_oh_alpha) < 0.0:
    raise SystemExit('--orbit-eval-soft-oh-alpha must be >= 0')

  train_solutions, train_anchors, train_targets = _load_split_arrays(
      npy_root, 'train', use_target_masks=args.use_target_masks and not args.online_random_mask_sfs)
  base_indices = _select_indices(
    num_total=int(train_solutions.shape[0]),
    num_base_examples=num_base_examples,
    subset_mode=args.subset_mode,
    seed=args.subset_seed,
  )
  train_ds = SudokuSFTDataset(
    train_solutions,
    train_anchors,
    train_targets,
    base_indices,
    recipe=args.recipe,
    seed=args.seed,
    dense_sfs_loss=bool(args.dense_sfs_loss and args.use_target_masks),
    frontier_weight_mult=float(args.sfs_frontier_weight_mult),
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

  valid_loader: Optional[DataLoader] = None
  valid_orbit_loader: Optional[DataLoader] = None
  valid_npy_root = Path(args.valid_npy_root) if args.valid_npy_root else npy_root
  if not valid_npy_root.is_absolute():
    valid_npy_root = MDLM_ROOT / valid_npy_root
  try:
    valid_solutions, valid_anchors, valid_targets = _load_split_arrays(
        valid_npy_root, 'valid',
        use_target_masks=args.use_target_masks and not args.online_random_mask_sfs)
    valid_indices = np.arange(int(valid_solutions.shape[0]), dtype=np.int64)
    if int(args.valid_max_examples) > 0:
      n_cap = min(int(args.valid_max_examples), int(valid_indices.shape[0]))
      valid_indices = valid_indices[:n_cap].copy()
    valid_ds = SudokuSFTDataset(
      valid_solutions,
      valid_anchors,
      valid_targets,
      valid_indices,
      recipe='canonical_sft_80k',
      seed=args.seed,
      dense_sfs_loss=bool(args.dense_sfs_loss and args.use_target_masks and valid_targets is not None),
      frontier_weight_mult=float(args.sfs_frontier_weight_mult),
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
    soh = _soft_oh_eval_mod()
    valid_orbit_ds = soh.SudokuOrbitSoftOHChunkDataset(
      valid_solutions,
      valid_anchors,
      valid_indices,
      orbit_base_size=orbit_B,
      seed=int(args.seed) + 1,
      blank_mask_mode=str(args.orbit_val_blank_mask_mode),
      blank_mask_eps=float(args.orbit_val_blank_mask_eps),
      sft_time=float(args.sft_time),
      sample_mode='sequential',
      stochastic_masks=False,
      orbit_mask_coupling=str(args.orbit_mask_coupling),
    )
    valid_orbit_loader = DataLoader(
      valid_orbit_ds,
      batch_size=1,
      shuffle=False,
      collate_fn=soh._orbit_collate,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory,
      persistent_workers=args.num_workers > 0,
      worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
    )
  except FileNotFoundError:
    valid_loader = None
    valid_orbit_loader = None

  if str(args.best_checkpoint_key) == 'orbit_group_loss' and valid_orbit_loader is None:
    print(
        '[posttrain] warning: --best-checkpoint-key orbit_group_loss but no valid/orbit loader; '
        'falling back to sft_loss for best_step_*.ckpt.',
    )

  compose_bs = (
      max(int(args.batch_size), int(orbit_bs))
      if valid_loader is not None
      else int(args.batch_size))

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

  # Freeze noise by default. We want SFT to adapt the denoiser, not the noise schedule.
  if not args.train_noise:
    for p in model.noise.parameters():
      p.requires_grad_(False)

  params: Iterable[torch.nn.Parameter]
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
    lr_lambda=_build_lr_lambda(args.max_steps, args.warmup_steps, args.min_lr_ratio),
  )

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  _layout = (
      'sharded'
      if isinstance(train_solutions, _ShardedArrayChain)
      else 'flat')
  _nshard = (
      int(train_solutions.num_shards)
      if isinstance(train_solutions, _ShardedArrayChain)
      else None)
  config_summary = {
    'args': vars(args),
    'recipe_defaults': recipe_cfg,
    'npy_root': str(npy_root),
    'valid_npy_root': str(valid_npy_root),
    'npy_layout': _layout,
    'num_epoch_shards': _nshard,
    'num_train_base_examples': int(len(base_indices)),
    'num_train_views_per_epoch': int(len(train_ds)),
    'has_valid_split': valid_loader is not None,
    'has_valid_orbit': valid_orbit_loader is not None,
    'best_checkpoint_key': str(args.best_checkpoint_key),
    'orbit_batch_size': int(orbit_bs),
    'compose_batch_size': int(compose_bs),
    'use_target_masks': args.use_target_masks,
    'dense_sfs_loss': args.dense_sfs_loss,
    'sfs_frontier_weight_mult': float(args.sfs_frontier_weight_mult),
    'online_random_mask_sfs': args.online_random_mask_sfs,
    'random_mask_eps': args.random_mask_eps,
    'sfs_oracle': args.sfs_oracle,
  }
  (out_dir / 'posttrain_config.json').write_text(json.dumps(config_summary, indent=2), encoding='utf-8')

  print(json.dumps(config_summary, indent=2))

  step = 0
  best_val_sft = float('inf')
  best_val_orbit = float('inf')
  best_step_tracked = -1
  running_loss = 0.0
  running_acc = 0.0
  running_tokens = 0.0
  optimizer.zero_grad(set_to_none=True)

  data_iter = iter(train_loader)
  while step < args.max_steps:
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
      online_random_mask_sfs=args.online_random_mask_sfs,
      random_mask_eps=args.random_mask_eps,
      sfs_oracle=args.sfs_oracle,
      dense_sfs_loss=args.dense_sfs_loss,
      sfs_frontier_weight_mult=float(args.sfs_frontier_weight_mult),
    )
    scaled_loss = loss / max(1, args.grad_accum_steps)
    scaled_loss.backward()

    running_loss += float(loss.item()) * info['num_loss_tokens']
    running_acc += float(info['sft_token_acc']) * info['num_loss_tokens']
    running_tokens += float(info['num_loss_tokens'])

    if (step + 1) % args.grad_accum_steps == 0:
      if args.grad_clip and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad(set_to_none=True)

    step += 1

    if step % args.log_every == 0 or step == 1:
      row = {
        'step': step,
        'train_sft_loss': (
            running_loss / max(running_tokens, 1.0) if running_tokens > 0 else 0.0),
        'train_sft_token_acc': (
            running_acc / max(running_tokens, 1.0) if running_tokens > 0 else 0.0),
        'lr': optimizer.param_groups[0]['lr'],
        'recipe': args.recipe,
      }
      print(json.dumps(row))
      _write_jsonl(out_dir / 'train_log.jsonl', row)
      running_loss = 0.0
      running_acc = 0.0
      running_tokens = 0.0

    if valid_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
      val_metrics = _evaluate_sft_loss(
        model,
        valid_loader,
        device=device,
        sft_time=args.sft_time,
        label_smoothing=args.label_smoothing,
        max_batches=args.eval_max_batches,
        online_random_mask_sfs=args.online_random_mask_sfs,
        random_mask_eps=args.random_mask_eps,
        sfs_oracle=args.sfs_oracle,
        dense_sfs_loss=args.dense_sfs_loss,
        sfs_frontier_weight_mult=float(args.sfs_frontier_weight_mult),
      )
      val_row: Dict[str, Any] = {'step': step, **val_metrics, 'recipe': args.recipe}
      if valid_orbit_loader is not None:
        soh_ev = _soft_oh_eval_mod()
        orb_m = soh_ev._evaluate_orbit_soft_oh(
            model,
            valid_orbit_loader,
            device=device,
            inv_perms=inv_perms_cpu,
            label_smoothing=float(args.label_smoothing),
            alpha=float(args.orbit_eval_soft_oh_alpha),
            lambda_target=float(args.orbit_eval_soft_oh_lambda),
            max_batches=int(args.eval_orbit_max_batches),
            orbit_mask_coupling=str(args.orbit_mask_coupling),
        )
        val_row.update(orb_m)
      print(json.dumps(val_row))
      _write_jsonl(out_dir / 'valid_log.jsonl', val_row)

      best_key = str(args.best_checkpoint_key)
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
        best_step_tracked = int(step)
        best_path = out_dir / f'best_step_{step}.ckpt'
        _save_checkpoint(
            best_path, model, optimizer, scheduler, args, step, val_row)
        print(f'[posttrain] saved best checkpoint at step={step}: {best_path} (key={best_key})')

    if args.save_every > 0 and step % args.save_every == 0:
      _save_checkpoint(
        out_dir / f'step_{step}.ckpt', model, optimizer, scheduler, args, step,
        {'step': step, 'recipe': args.recipe})
      print(f'[posttrain] saved checkpoint: {out_dir / f"step_{step}.ckpt"}')

  final_metrics = {
      'step': step,
      'recipe': args.recipe,
      'best_val_sft_loss': best_val_sft,
      'best_val_orbit_l_group': best_val_orbit,
      'best_step': int(best_step_tracked),
  }
  _save_checkpoint(out_dir / 'last.ckpt', model, optimizer, scheduler, args, step, final_metrics)
  if valid_loader is None:
    _save_checkpoint(out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, final_metrics)
    print(f'[posttrain] done. last={out_dir / "last.ckpt"}; best={out_dir / "best.ckpt"} (no valid split)')
  else:
    print(
        f'[posttrain] done. last={out_dir / "last.ckpt"}; '
        f'best checkpoints under {out_dir} as best_step_<step>.ckpt when validation improved.',
    )


if __name__ == '__main__':
  main()
