#!/usr/bin/env python3
"""Post-train an MDLM Sudoku checkpoint with clue-conditioned SFT.

This script is designed for the SunwooHong/mdlm_sudoku fork.  It implements
three baseline post-training recipes for 9x9 Sudoku:

  1. canonical_sft_80k
     - use canonical layouts only
     - default base examples: 80,000

  2. random_transform_sft_80k
     - use the same base examples, but apply a random D4 transform to each
       item every time it is drawn
     - default base examples: 80,000

  3. full_orbit_sft_10k
     - use 10,000 base examples and expand each into all 8 D4 transforms
     - default training views per epoch: 80,000

  4. repeat8_sft_10k
     - use 10,000 base examples and repeat each sample exactly 8 times
     - no transform augmentation; default training views per epoch: 80,000

The default SFT objective (--blank-mask-mode random) keeps anchors visible, draws a
scalar mask probability t ~ Uniform per sequence on blank cells only (Bernoulli
masking per blank position), and applies CE only on masked blanks; sigma/time
conditioning uses the same t per row.  Use --blank-mask-mode full to mask every
blank and supervise all blanks at fixed --sft-time (legacy baseline).

Expected npy files under --npy-root:
  train_solution.npy   [N, 81] int ids 0..8
  train_anchor.npy     [N, 81] bool/int; True where given clues are visible
Optional validation files:
  valid_solution.npy, valid_anchor.npy

Example:
  python scripts/sudoku_posttrain_sft.py \
    --checkpoint outputs/.../checkpoints/best.ckpt \
    --model sudoku_50m \
    --npy-root dataset/3m_only_posttrain_npy \
    --recipe canonical_sft_80k \
    --batch-size 512 \
    --max-steps 3000 \
    --lr 1e-5 \
    --output-dir outputs/posttrain/canonical_50m
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
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
    raise ValueError(f'{anchor_path} shape {anchors.shape} != solutions shape {solutions.shape}')
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
  """Clue-conditioned Sudoku SFT dataset with M=B.

  Returns:
    input_ids: gold solution ids [81], values 0..8
    anchor_mask: bool [81], True for clues that remain visible
    loss_mask: bool [81], True for original blanks; loss is computed here

  The model input is not stored; the training loop constructs it as
    x_cond = where(anchor_mask, input_ids, mask_id).
  """

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
    loss_mask = ~anchor

    return {
      'input_ids': torch.from_numpy(sol).long(),
      'anchor_mask': torch.from_numpy(anchor).bool(),
      'loss_mask': torch.from_numpy(loss_mask).bool(),
      'transform_id': torch.tensor(TRANSFORMS.index(transform_name), dtype=torch.long),
      'base_index': torch.tensor(arr_idx, dtype=torch.long),
    }


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
  """Compute clue-conditioned SFT loss on blank cells.

  blank_mask_mode:
    - 'full': mask every blank; supervise every blank; time = sft_time (broadcast).
    - 'random': per-row t ~ Uniform[eps,1]; blank positions masked independently
      with probability t; CE only on masked blanks; sigma uses same t per row.
  """
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
    # Same convention as MDLM-style forward_process: t in (eps, 1] after affine clamp.
    t_prob = torch.rand((B,), device=device, dtype=model.dtype)
    t_prob = (1.0 - eps) * t_prob + eps
    rand_pos = torch.rand((B, input_ids.shape[1]), device=device, dtype=model.dtype)
    mask_blank = loss_mask & (rand_pos < t_prob[:, None])
    # Avoid empty supervision rows when blanks exist ( rare when t is tiny ).
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
  logits = model.forward(x_cond, sigma[:, None])  # log-probs for SUBS parameterization

  # Exclude mask token from the supervised Sudoku digit loss.
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
  p = argparse.ArgumentParser(description='Post-train MDLM Sudoku with full-blank SFT baselines.')
  p.add_argument('--checkpoint', type=str, required=True)
  p.add_argument('--model', type=str, default='sudoku_50m')
  p.add_argument('--npy-root', type=str, required=True, help='Directory with train_solution.npy/train_anchor.npy')
  p.add_argument('--recipe', type=str, required=True, choices=sorted(RECIPE_DEFAULTS))
  p.add_argument('--output-dir', type=str, required=True)

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
  p.add_argument('--blank-mask-mode', type=str, default='random', choices=('random', 'full'),
                 help='random: Bernoulli-mask blanks per sequence with sampled t; '
                 'full: mask all blanks (legacy). Validation always uses full.')
  p.add_argument('--blank-mask-eps', type=float, default=1e-3,
                 help='Lower bound scaling for sampled t in random mode (matches MDLM-style eps).')
  p.add_argument('--sft-time', type=float, default=1.0,
                 help='Diffusion time conditioning when blank-mask-mode is full (broadcast to batch).')
  p.add_argument('--train-noise', action='store_true', default=False,
                 help='By default only backbone params are optimized; enable to include noise params.')

  p.add_argument('--eval-every', type=int, default=500)
  p.add_argument('--eval-max-batches', type=int, default=32)
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
  try:
    valid_solutions, valid_anchors = _load_split_arrays(npy_root, 'valid')
    valid_indices = np.arange(int(valid_solutions.shape[0]), dtype=np.int64)
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
  except FileNotFoundError:
    valid_loader = None

  cfg = _compose_config(
    checkpoint=args.checkpoint,
    model=args.model,
    batch_size=args.batch_size,
    num_steps=args.num_steps,
    predictor=args.predictor,
    noise_removal=not args.no_noise_removal,
    extra_overrides=args.extra_override,
  )
  model = _load_model(cfg, device)

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
  config_summary = {
    'args': vars(args),
    'recipe_defaults': recipe_cfg,
    'npy_root': str(npy_root),
    'num_train_base_examples': int(len(base_indices)),
    'num_train_views_per_epoch': int(len(train_ds)),
    'has_valid_split': valid_loader is not None,
  }
  (out_dir / 'posttrain_config.json').write_text(json.dumps(config_summary, indent=2), encoding='utf-8')

  print(json.dumps(config_summary, indent=2))

  step = 0
  best_val = float('inf')
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
      blank_mask_mode=args.blank_mask_mode,
      blank_mask_eps=args.blank_mask_eps,
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
        'train_sft_loss': running_loss / max(running_tokens, 1.0),
        'train_sft_token_acc': running_acc / max(running_tokens, 1.0),
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
      )
      val_row = {'step': step, **val_metrics, 'recipe': args.recipe}
      print(json.dumps(val_row))
      _write_jsonl(out_dir / 'valid_log.jsonl', val_row)
      if val_metrics['val_sft_loss'] < best_val:
        best_val = val_metrics['val_sft_loss']
        _save_checkpoint(
          out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, val_row)
        print(f'[posttrain] saved best checkpoint at step={step}: {out_dir / "best.ckpt"}')

    if args.save_every > 0 and step % args.save_every == 0:
      _save_checkpoint(
        out_dir / f'step_{step}.ckpt', model, optimizer, scheduler, args, step,
        {'step': step, 'recipe': args.recipe})
      print(f'[posttrain] saved checkpoint: {out_dir / f"step_{step}.ckpt"}')

  final_metrics = {'step': step, 'recipe': args.recipe, 'best_val_sft_loss': best_val}
  _save_checkpoint(out_dir / 'last.ckpt', model, optimizer, scheduler, args, step, final_metrics)
  if valid_loader is None:
    _save_checkpoint(out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, final_metrics)
  print(f'[posttrain] done. last={out_dir / "last.ckpt"}; best={out_dir / "best.ckpt"}')


if __name__ == '__main__':
  main()
