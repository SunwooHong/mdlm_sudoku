#!/usr/bin/env python3
"""Post-train MDLM Sudoku with **latent orbit equivalence (LOC)** on the DiT pre-output stream.

For each base puzzle we feed two distinct D4 views sharing the same canonical blank mask and
noise level ``t`` (same construction as ``sudoku_posttrain_pair_orbit.py``). Let ``h`` be the
last transformer block output (before the vocab head), shape ``[B, 81, d]``. We permute each
view's sequence axis into **canonical** cell order with ``g^{-1}`` (``inv[g]`` gather), then::

  L_LOC = mean_{supervised cells} rho( align(h_0), align(h_1) )

where ``rho`` is squared L2 per cell (mean over ``d``) or ``1 - cosine`` per cell **on ``h``**.

**SimCLR-style projection head (optional, ``--loc-projection``):** a small 2-layer MLP
``z = W_2 \\mathrm{ReLU}(W_1 h)`` maps per-token ``h`` to ``z``; LOC uses only ``z`` with
``\\mathcal{L}_{LOC} = 1 - \\cos(z, \\tilde{z})`` per supervised cell (cosine only in this mode).
The backbone ``h`` stays free for the vocab head / CE; equivariance pressure lives in ``z``.

Total objective::

  L = L_SFT + lambda_eff * L_LOC

``L_SFT`` is the mean canonical masked CE averaged over the two views (same supervision mask as
the pair trainer). ``lambda_eff`` can warm up linearly over the first ``--loc-lambda-warmup-frac``
of **optimizer updates**.

Requires ``backbone=dit`` (``return_backbone_hidden`` path in ``diffusion.Diffusion.forward``).

Example::

  python posttrain/sudoku_posttrain_loc_equiv.py \\
    --checkpoint outputs/.../best.ckpt \\
    --model sudoku_50m \\
    --npy-root dataset/3m_posttrain_10k \\
    --valid-npy-root dataset/3m_only_val_npy \\
    --output-dir outputs/posttrain/loc10k \\
    --fair-method pair10k \\
    --pair-base-size 4 \\
    --loc-lambda 0.05 \\
    --loc-lambda-warmup-frac 0.1 \\
    --loc-distance cosine

  # With projection head (LOC cosine on ``z`` only)::

  python posttrain/sudoku_posttrain_loc_equiv.py ... \\
    --loc-projection \\
    --loc-proj-dim 128 \\
    --loc-proj-hidden 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_POST = Path(__file__).resolve().parent
MDLM_ROOT = _POST.parent
if str(MDLM_ROOT) not in sys.path:
  sys.path.insert(0, str(MDLM_ROOT))

_spec_soh = importlib.util.spec_from_file_location(
    'sudoku_posttrain_soft_oh_mod',
    str(_POST / 'sudoku_posttrain_soft_oh.py'),
)
if _spec_soh is None or _spec_soh.loader is None:
  raise RuntimeError('Cannot load sudoku_posttrain_soft_oh.py')
soh = importlib.util.module_from_spec(_spec_soh)
_spec_soh.loader.exec_module(soh)

_spec_pop = importlib.util.spec_from_file_location(
    'sudoku_posttrain_pair_orbit_mod',
    str(_POST / 'sudoku_posttrain_pair_orbit.py'),
)
if _spec_pop is None or _spec_pop.loader is None:
  raise RuntimeError('Cannot load sudoku_posttrain_pair_orbit.py')
pop = importlib.util.module_from_spec(_spec_pop)
_spec_pop.loader.exec_module(pop)

SudokuPairOrbitChunkDataset = pop.SudokuPairOrbitChunkDataset
_pair_collate = pop._pair_collate
PAIR_FAIR_METHODS = pop.PAIR_FAIR_METHODS


class LocProjectionHead(nn.Module):
  """Two-layer MLP on per-token backbone features: ``z = W2 ReLU(W1 h)`` (SimCLR-style)."""

  def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.ReLU(inplace=False),
        nn.Linear(hidden_dim, out_dim, bias=True),
    )

  def forward(self, h: torch.Tensor) -> torch.Tensor:
    """``h`` [..., in_dim] → ``z`` [..., out_dim]."""
    return self.net(h)


def _loc_lambda_eff(
    *,
    lambda_target: float,
    optim_step: int,
    total_opt_steps: int,
    warmup_frac: float,
) -> float:
  lam = float(lambda_target)
  if lam <= 0.0:
    return 0.0
  total_opt_steps = max(1, int(total_opt_steps))
  wu = max(0.0, min(1.0, float(warmup_frac)))
  end = max(1, int(round(wu * total_opt_steps)))
  gs = int(optim_step)
  t = min(1.0, float(gs) / float(end))
  return lam * max(0.0, min(1.0, t))


def _gather_hidden_to_canon(
    h_view: torch.Tensor,
    g: int,
    inv_perms: torch.Tensor,
) -> torch.Tensor:
  """Map view-space sequence [..., 81, D] to canonical cell order for transform index g."""
  inv = inv_perms.to(device=h_view.device, dtype=torch.long)
  D = int(h_view.shape[-1])
  idx = inv[g].view(1, 81, 1).expand(h_view.shape[0], 81, D)
  return torch.gather(h_view, 1, idx)


def _save_loc_checkpoint(
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    step: int,
    metrics: Dict[str, Any],
    loc_proj: Optional[nn.Module],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  payload: Dict[str, Any] = {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
      'step': int(step),
      'args': vars(args),
      'metrics': metrics,
      'loc_proj_state_dict': loc_proj.state_dict() if loc_proj is not None else None,
  }
  torch.save(payload, path)


def _pair_loc_equiv_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    inv_perms: torch.Tensor,
    label_smoothing: float,
    lambda_eff: float,
    loc_distance: str,
    blank_mask_mode: str,
    blank_mask_eps: float,
    sft_time: float,
    resample_corruption: bool,
    loc_proj: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
  """Canonical CE on both views + latent alignment in canonical coordinates (``h`` or ``z``)."""
  x_gold = batch['x_gold'].to(device=device, dtype=torch.long)
  canonical_sol = batch['canonical_sol'].to(device=device, dtype=torch.long)
  P = int(batch['pair_base_size'].item())
  pair_view = batch['pair_view_idx'].to(device=device, dtype=torch.long)
  mid = int(model.mask_index)

  if resample_corruption:
    perms_t = soh._build_perms_t().to(device=device)
    inv = inv_perms.to(device=device, dtype=torch.long)
    canonical_anchor = batch['canonical_anchor'].to(device=device).bool()
    blank_c = ~canonical_anchor

    if blank_mask_mode == 'full':
      t_base = torch.full((P,), float(sft_time), device=device, dtype=model.dtype)
      supervise_canon = blank_c
    elif blank_mask_mode == 'random':
      eps = float(blank_mask_eps)
      t_base = torch.rand((P,), device=device, dtype=model.dtype)
      t_base = (1.0 - eps) * t_base + eps
      rand_c = torch.rand((P, 81), device=device, dtype=model.dtype)
      supervise_canon = blank_c & (rand_c < t_base[:, None])
      n_masked_row = supervise_canon.sum(dim=1)
      has_blanks = blank_c.any(dim=1)
      fallback = (n_masked_row == 0) & has_blanks
      supervise_canon = torch.where(fallback[:, None], blank_c, supervise_canon)
    else:
      raise ValueError(f'Unknown blank_mask_mode={blank_mask_mode!r}')

    anchor_views = []
    mask_views = []
    Gpair = 2
    for b in range(P):
      for j in range(Gpair):
        row = b * Gpair + j
        g = int(pair_view[row].item())
        anchor_views.append(canonical_anchor[b, perms_t[g]])
        mask_views.append(supervise_canon[b, perms_t[g]])
    anchor_mask = torch.stack(anchor_views, dim=0).reshape(P * Gpair, 81)
    mask_blank_view = torch.stack(mask_views, dim=0).reshape(P * Gpair, 81)
    t_cond = t_base[:, None].expand(P, Gpair).reshape(P * Gpair)
    supervise_f = supervise_canon.float()
  else:
    anchor_mask = batch['anchor_mask'].to(device=device, dtype=torch.bool)
    mask_blank_view = batch['mask_blank_view'].to(device=device, dtype=torch.bool)
    t_cond = batch['t_cond'].to(device=device, dtype=model.dtype)
    supervise_f = batch['supervise_canon'].to(device=device).float()

  inv = inv_perms.to(device=device, dtype=torch.long)

  mask_tokens = torch.full_like(x_gold, mid)
  x_cond = torch.where(
      anchor_mask,
      x_gold,
      torch.where(mask_blank_view, mask_tokens, x_gold),
  )
  sigma, _ = model.noise(t_cond)
  logits, h_pre = model.forward(
      x_cond,
      sigma[:, None],
      return_backbone_hidden=True,
  )
  if logits.shape[-1] > 9:
    logits = logits[..., :9]

  feat_pre = loc_proj(h_pre) if loc_proj is not None else h_pre

  targets = canonical_sol
  ce_list: List[torch.Tensor] = []
  loc_list: List[torch.Tensor] = []

  for b in range(P):
    denom = supervise_f[b].sum().clamp(min=1.0)
    ce_arm: List[torch.Tensor] = []
    rep_canon: List[torch.Tensor] = []
    for j in range(2):
      row = b * 2 + j
      g = int(pair_view[row].item())
      logits_row = logits[row : row + 1]
      idx = inv[g].view(1, 81, 1).expand(1, 81, logits_row.shape[-1])
      logits_canon = torch.gather(logits_row, 1, idx).squeeze(0).float()
      sup = supervise_f[b]
      ce_flat = F.cross_entropy(
          logits_canon,
          targets[b],
          reduction='none',
          label_smoothing=float(label_smoothing),
      )
      ce_arm.append((ce_flat * sup).sum() / denom)

      tok_row = feat_pre[row : row + 1]
      rep_canon.append(_gather_hidden_to_canon(tok_row, g, inv).squeeze(0))

    L_ce = 0.5 * (ce_arm[0] + ce_arm[1])
    ce_list.append(L_ce)

    d0, d1 = rep_canon[0], rep_canon[1]
    sup = supervise_f[b]
    if loc_proj is not None:
      d0n = F.normalize(d0, dim=-1, eps=1e-8)
      d1n = F.normalize(d1, dim=-1, eps=1e-8)
      cos = (d0n * d1n).sum(dim=-1)
      per_cell = 1.0 - cos
    elif loc_distance == 'l2':
      per_cell = (d0 - d1).pow(2).mean(dim=-1)
    elif loc_distance == 'cosine':
      d0n = F.normalize(d0, dim=-1, eps=1e-8)
      d1n = F.normalize(d1, dim=-1, eps=1e-8)
      cos = (d0n * d1n).sum(dim=-1)
      per_cell = 1.0 - cos
    else:
      raise ValueError(f'Unknown loc_distance={loc_distance!r}')

    loc_b = (per_cell * sup).sum() / denom
    loc_list.append(loc_b)

  L_sft = torch.stack(ce_list).mean()
  L_loc = torch.stack(loc_list).mean()
  loss = L_sft + float(lambda_eff) * L_loc

  with torch.no_grad():
    pred_all = logits.reshape(P * 2, 81, -1).argmax(dim=-1)
    mflat = mask_blank_view.reshape(-1, 81).float()
    acc = (pred_all == x_gold).float()
    acc_row = (acc * mflat).sum(dim=1) / mflat.sum(dim=1).clamp(min=1.0)
    n_tok = float(mflat.sum().item())

  return loss, {
      'sft_token_acc': float(acc_row.mean().item()),
      'num_loss_tokens': n_tok,
      'train_loc_sft_ce': float(L_sft.item()),
      'train_loc_latent': float(L_loc.item()),
      'train_loc_total': float(loss.item()),
      'lambda_eff': float(lambda_eff),
  }


def _apply_pair_fair_method(args: argparse.Namespace) -> None:
  if not getattr(args, 'fair_method', ''):
    return
  preset = PAIR_FAIR_METHODS[str(args.fair_method)]
  args.num_base_examples = int(preset['num_base_examples'])
  args.recipe = str(preset['recipe'])


@torch.no_grad()
def _evaluate_loc_loss(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    inv_perms: torch.Tensor,
    label_smoothing: float,
    lambda_target: float,
    max_batches: int,
    loc_distance: str,
    blank_mask_mode: str,
    blank_mask_eps: float,
    sft_time: float,
    loc_proj: Optional[nn.Module] = None,
) -> Dict[str, float]:
  model.eval()
  if loc_proj is not None:
    loc_proj.eval()
  total = 0.0
  n_batches = 0
  for batch in loader:
    loss, _info = _pair_loc_equiv_loss(
        model,
        batch,
        device=device,
        inv_perms=inv_perms,
        label_smoothing=label_smoothing,
        lambda_eff=float(lambda_target),
        loc_distance=loc_distance,
        blank_mask_mode=blank_mask_mode,
        blank_mask_eps=blank_mask_eps,
        sft_time=sft_time,
        resample_corruption=False,
        loc_proj=loc_proj,
    )
    total += float(loss.item())
    n_batches += 1
    if max_batches > 0 and n_batches >= max_batches:
      break
  model.train()
  if loc_proj is not None:
    loc_proj.train()
  denom = max(n_batches, 1)
  return {'val_loc_total': total / denom, 'val_loc_batches': float(n_batches)}


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description='Post-train Sudoku MDLM with latent orbit equivalence (LOC).')
  p.add_argument('--checkpoint', type=str, required=True)
  p.add_argument('--model', type=str, default='sudoku_50m')
  p.add_argument('--npy-root', type=str, required=True)
  p.add_argument('--valid-npy-root', type=str, default='')
  p.add_argument('--valid-max-examples', type=int, default=0)
  p.add_argument(
      '--fair-method',
      type=str,
      default='',
      choices=('', *sorted(PAIR_FAIR_METHODS)),
      help='pair10k / pair80k presets.',
  )
  p.add_argument('--recipe', type=str, default='canonical_sft_80k', choices=sorted(soh.RECIPE_DEFAULTS))
  p.add_argument('--output-dir', type=str, required=True)

  p.add_argument('--pair-base-size', type=int, default=4)
  p.add_argument('--loc-lambda', type=float, default=0.05, help='Weight on L_LOC.')
  p.add_argument(
      '--loc-lambda-warmup-frac',
      type=float,
      default=0.1,
      help='Linear warmup of lambda_eff over this fraction of optimizer updates.',
  )
  p.add_argument(
      '--loc-distance',
      type=str,
      default='l2',
      choices=('l2', 'cosine'),
      help='Per-cell latent distance on ``h`` (ignored when ``--loc-projection``: always cosine on ``z``).',
  )
  p.add_argument(
      '--loc-projection',
      action='store_true',
      default=False,
      help='SimCLR-style 2-layer MLP z=g(h); LOC is 1-cos(z) in canonical coords only (CE still on logits).',
  )
  p.add_argument(
      '--loc-proj-dim',
      type=int,
      default=128,
      help='Output dimension of the projection head.',
  )
  p.add_argument(
      '--loc-proj-hidden',
      type=int,
      default=0,
      help='Hidden width; 0 = use backbone hidden size (same as SimCLR default scale).',
  )
  p.add_argument('--r-sft', type=float, default=0.0,
                 help='Probability of an ordinary row-SFT micro-step (else pair chunk).')

  p.add_argument('--num-base-examples', type=int, default=0)
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
  p.add_argument('--blank-mask-mode', type=str, default='random', choices=('random', 'full'))
  p.add_argument('--blank-mask-eps', type=float, default=1e-3)
  p.add_argument('--sft-time', type=float, default=1.0)
  p.add_argument('--train-noise', action='store_true', default=False)
  p.add_argument('--pair-sample-mode', type=str, default='random',
                 choices=('random', 'sequential', 'sequential_drop_last'))
  p.add_argument('--deterministic-pair-masks', action='store_true')

  p.add_argument('--eval-every', type=int, default=500)
  p.add_argument('--eval-max-batches', type=int, default=32)
  p.add_argument('--eval-pair-max-batches', type=int, default=16)
  p.add_argument(
      '--best-checkpoint-key',
      type=str,
      default='loc_total',
      choices=('sft_loss', 'loc_total'),
  )
  p.add_argument('--sanity-check', action='store_true')
  p.add_argument('--save-every', type=int, default=1000)
  p.add_argument('--log-every', type=int, default=50)

  p.add_argument('--predictor', type=str, default='ddpm_cache')
  p.add_argument('--num-steps', type=int, default=128)
  p.add_argument('--no-noise-removal', action='store_true')
  p.add_argument('--device', type=str, default='cuda')
  p.add_argument('--extra-override', action='append', default=[])
  return p.parse_args()


def main() -> None:
  args = parse_args()
  _apply_pair_fair_method(args)
  soh._register_omegaconf_resolvers()

  if int(args.pair_base_size) < 1:
    raise SystemExit('--pair-base-size must be >= 1')
  if float(args.loc_lambda) < 0.0:
    raise SystemExit('--loc-lambda must be >= 0')
  if int(args.loc_proj_dim) < 1:
    raise SystemExit('--loc-proj-dim must be >= 1')
  if int(args.loc_proj_hidden) < 0:
    raise SystemExit('--loc-proj-hidden must be >= 0 (0 = use backbone hidden size)')
  if not (0.0 <= float(args.r_sft) <= 1.0):
    raise SystemExit('--r-sft must be in [0, 1]')
  if int(args.grad_accum_steps) <= 0:
    raise SystemExit('--grad-accum-steps must be >= 1')
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

  recipe_cfg = soh.RECIPE_DEFAULTS[args.recipe]
  num_base_examples = int(args.num_base_examples) if args.num_base_examples > 0 else int(recipe_cfg['num_base_examples'])

  npy_root = Path(args.npy_root)
  if not npy_root.is_absolute():
    npy_root = MDLM_ROOT / npy_root
  train_solutions, train_anchors = soh._load_split_arrays(npy_root, 'train')
  base_indices = soh._select_indices(
      num_total=int(train_solutions.shape[0]),
      num_base_examples=num_base_examples,
      subset_mode=args.subset_mode,
      seed=args.subset_seed,
  )

  pair_ds = SudokuPairOrbitChunkDataset(
      train_solutions,
      train_anchors,
      base_indices,
      pair_base_size=int(args.pair_base_size),
      seed=args.seed,
      blank_mask_mode=args.blank_mask_mode,
      blank_mask_eps=args.blank_mask_eps,
      sft_time=args.sft_time,
      sample_mode=args.pair_sample_mode,
      stochastic_masks=not args.deterministic_pair_masks,
  )
  pair_loader = torch.utils.data.DataLoader(
      pair_ds,
      batch_size=1,
      shuffle=True,
      collate_fn=_pair_collate,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory,
      persistent_workers=args.num_workers > 0,
      worker_init_fn=soh._worker_init_fn if args.num_workers > 0 else None,
  )

  train_loader: Optional[torch.utils.data.DataLoader] = None
  if float(args.r_sft) > 0.0:
    train_ds = soh.SudokuFullBlankSFTDataset(
        train_solutions,
        train_anchors,
        base_indices,
        recipe=args.recipe,
        seed=args.seed,
    )
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=soh._worker_init_fn if args.num_workers > 0 else None,
        generator=gen,
    )

  valid_loader: Optional[torch.utils.data.DataLoader] = None
  valid_pair_loader: Optional[torch.utils.data.DataLoader] = None
  n_valid_disk_for_cfg: Optional[int] = None
  n_valid_used_for_cfg: Optional[int] = None
  valid_npy_root = Path(args.valid_npy_root) if args.valid_npy_root else npy_root
  if not valid_npy_root.is_absolute():
    valid_npy_root = MDLM_ROOT / valid_npy_root
  try:
    valid_solutions, valid_anchors = soh._load_split_arrays(valid_npy_root, 'valid')
    n_valid_disk = int(valid_solutions.shape[0])
    valid_indices = np.arange(n_valid_disk, dtype=np.int64)
    if int(args.valid_max_examples) > 0:
      n_use = min(int(args.valid_max_examples), n_valid_disk)
      valid_indices = valid_indices[:n_use].copy()
    n_valid_disk_for_cfg = n_valid_disk
    n_valid_used_for_cfg = int(len(valid_indices))
    valid_ds = soh.SudokuFullBlankSFTDataset(
        valid_solutions,
        valid_anchors,
        valid_indices,
        recipe='canonical_sft_80k',
        seed=args.seed,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=soh._worker_init_fn if args.num_workers > 0 else None,
    )
    valid_pair_ds = SudokuPairOrbitChunkDataset(
        valid_solutions,
        valid_anchors,
        valid_indices,
        pair_base_size=int(args.pair_base_size),
        seed=args.seed + 1,
        blank_mask_mode=args.blank_mask_mode,
        blank_mask_eps=args.blank_mask_eps,
        sft_time=args.sft_time,
        sample_mode='sequential',
        stochastic_masks=False,
    )
    valid_pair_loader = torch.utils.data.DataLoader(
        valid_pair_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=_pair_collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=soh._worker_init_fn if args.num_workers > 0 else None,
    )
  except FileNotFoundError:
    valid_loader = None
    valid_pair_loader = None

  if args.best_checkpoint_key == 'loc_total' and valid_pair_loader is None:
    print('[posttrain-loc] warning: --best-checkpoint-key loc_total but no valid pair split; using sft_loss.')

  compose_bs = max(int(args.batch_size), 2 * int(args.pair_base_size))
  cfg = soh._compose_config(
      checkpoint=args.checkpoint,
      model=args.model,
      batch_size=compose_bs,
      num_steps=args.num_steps,
      predictor=args.predictor,
      noise_removal=not args.no_noise_removal,
      extra_overrides=args.extra_override,
  )
  if str(cfg.backbone) != 'dit':
    raise SystemExit(
        f'LOC post-training requires backbone=dit (got {cfg.backbone!r}). '
        'Use the default Sudoku DiT config.',
    )
  model = soh._load_model(cfg, device)
  inv_perms_cpu = soh._build_inv_perms_t()

  in_dim = int(cfg.model.hidden_size)
  proj_hidden = int(args.loc_proj_hidden) if int(args.loc_proj_hidden) > 0 else in_dim
  loc_proj: Optional[nn.Module] = None
  if bool(args.loc_projection):
    loc_proj = LocProjectionHead(in_dim, proj_hidden, int(args.loc_proj_dim)).to(device)
    loc_proj.train()
    print(
        f'[posttrain-loc] projection head: in={in_dim} hidden={proj_hidden} out={int(args.loc_proj_dim)} '
        f'(LOC cosine on z only).',
    )

  if args.sanity_check:
    pop._run_pair_mask_sanity(pair_loader, inv_perms_cpu)
    print('[posttrain-loc] pair mask sanity_check passed.')

  if not args.train_noise:
    for param in model.noise.parameters():
      param.requires_grad_(False)
  params: List[torch.nn.Parameter] = (
      list(model.backbone.parameters()) + list(model.noise.parameters())
      if args.train_noise
      else list(model.backbone.parameters()))
  if loc_proj is not None:
    params.extend(list(loc_proj.parameters()))
  optimizer = torch.optim.AdamW(
      [p for p in params if p.requires_grad],
      lr=args.lr,
      betas=(args.beta1, args.beta2),
      eps=args.eps,
      weight_decay=args.weight_decay,
  )
  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=soh._build_lr_lambda(total_opt_steps, args.warmup_steps, args.min_lr_ratio),
  )

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  config_summary: Dict[str, Any] = {
      'args': vars(args),
      'recipe_defaults': recipe_cfg,
      'npy_root': str(npy_root),
      'valid_npy_root': str(valid_npy_root),
      'num_train_base_examples': int(len(base_indices)),
      'pair_chunk_rows': 2 * int(args.pair_base_size),
      'compose_batch_size': compose_bs,
      'total_opt_steps': total_opt_steps,
      'total_micro_steps': int(args.max_steps),
      'has_valid_split': valid_loader is not None,
      'has_valid_pair': valid_pair_loader is not None,
      'num_valid_rows_on_disk': n_valid_disk_for_cfg,
      'num_valid_examples_used': n_valid_used_for_cfg,
      'method': 'latent_orbit_equivalence_loc',
      'loc_projection': bool(args.loc_projection),
      'loc_proj_in_dim': in_dim,
      'loc_proj_hidden': proj_hidden if loc_proj is not None else None,
      'loc_proj_out_dim': int(args.loc_proj_dim) if loc_proj is not None else None,
  }
  (out_dir / 'posttrain_config.json').write_text(json.dumps(config_summary, indent=2), encoding='utf-8')
  print(json.dumps(config_summary, indent=2))

  step = 0
  optim_step = 0
  best_val_sft = float('inf')
  best_val_loc = float('inf')
  running_loss = 0.0
  running_acc = 0.0
  running_steps = 0
  log_window_pair = 0.0
  log_window_steps = 0
  loc_metric_sums: Dict[str, float] = {}
  loc_metric_count = 0
  optimizer.zero_grad(set_to_none=True)

  pair_iter = iter(pair_loader)
  sft_iter = iter(train_loader) if train_loader is not None else None

  while step < args.max_steps:
    lambda_eff = _loc_lambda_eff(
        lambda_target=float(args.loc_lambda),
        optim_step=optim_step + 1,
        total_opt_steps=total_opt_steps,
        warmup_frac=float(args.loc_lambda_warmup_frac),
    )
    use_sft = float(args.r_sft) > 0.0 and (float(args.r_sft) >= 1.0 or torch.rand(1).item() < float(args.r_sft))
    if use_sft and sft_iter is not None:
      try:
        batch = next(sft_iter)
      except StopIteration:
        sft_iter = iter(train_loader)
        batch = next(sft_iter)
      loss, info = soh._sft_loss_full_blank(
          model,
          batch,
          device=device,
          sft_time=args.sft_time,
          label_smoothing=args.label_smoothing,
          blank_mask_mode=args.blank_mask_mode,
          blank_mask_eps=args.blank_mask_eps,
      )
    else:
      try:
        ob = next(pair_iter)
      except StopIteration:
        pair_iter = iter(pair_loader)
        ob = next(pair_iter)
      loss, info = _pair_loc_equiv_loss(
          model,
          ob,
          device=device,
          inv_perms=inv_perms_cpu,
          label_smoothing=args.label_smoothing,
          lambda_eff=lambda_eff,
          loc_distance=str(args.loc_distance),
          blank_mask_mode=args.blank_mask_mode,
          blank_mask_eps=args.blank_mask_eps,
          sft_time=args.sft_time,
          resample_corruption=True,
          loc_proj=loc_proj,
      )
      for k in ('train_loc_sft_ce', 'train_loc_latent', 'train_loc_total', 'lambda_eff'):
        if k in info:
          loc_metric_sums[k] = loc_metric_sums.get(k, 0.0) + float(info[k])
      loc_metric_count += 1

    scaled_loss = loss / max(1, args.grad_accum_steps)
    scaled_loss.backward()

    running_loss += float(loss.item())
    running_acc += float(info['sft_token_acc'])
    running_steps += 1
    log_window_steps += 1
    log_window_pair += 0.0 if use_sft else 1.0

    if (step + 1) % args.grad_accum_steps == 0:
      if args.grad_clip and args.grad_clip > 0:
        clip_params = list(model.parameters())
        if loc_proj is not None:
          clip_params += list(loc_proj.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, float(args.grad_clip))
      optimizer.step()
      scheduler.step()
      optim_step += 1
      optimizer.zero_grad(set_to_none=True)

    step += 1

    if step % args.log_every == 0 or step == 1:
      row: Dict[str, Any] = {
          'step': step,
          'optim_step': optim_step,
          'train_loss': running_loss / max(float(running_steps), 1.0),
          'train_sft_token_acc': running_acc / max(float(running_steps), 1.0),
          'train_pair_frac': log_window_pair / max(float(log_window_steps), 1.0),
          'loc_lambda_eff': lambda_eff,
          'lr': optimizer.param_groups[0]['lr'],
          'recipe': args.recipe,
      }
      if loc_metric_count > 0:
        denom_om = float(loc_metric_count)
        for k, s in loc_metric_sums.items():
          row[k] = s / denom_om
      print(json.dumps(row))
      soh._write_jsonl(out_dir / 'train_log.jsonl', row)
      running_loss = 0.0
      running_acc = 0.0
      running_steps = 0
      log_window_pair = 0.0
      log_window_steps = 0
      loc_metric_sums = {}
      loc_metric_count = 0

    if valid_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
      val_metrics = soh._evaluate_sft_loss(
          model,
          valid_loader,
          device=device,
          sft_time=args.sft_time,
          label_smoothing=args.label_smoothing,
          max_batches=args.eval_max_batches,
      )
      val_row: Dict[str, Any] = {'step': step, **val_metrics, 'recipe': args.recipe}
      if valid_pair_loader is not None:
        lam_full = float(args.loc_lambda)
        pm = _evaluate_loc_loss(
            model,
            valid_pair_loader,
            device=device,
            inv_perms=inv_perms_cpu,
            label_smoothing=args.label_smoothing,
            lambda_target=lam_full,
            max_batches=args.eval_pair_max_batches,
            loc_distance=str(args.loc_distance),
            blank_mask_mode=args.blank_mask_mode,
            blank_mask_eps=args.blank_mask_eps,
            sft_time=args.sft_time,
            loc_proj=loc_proj,
        )
        val_row.update(pm)
      print(json.dumps(val_row))
      soh._write_jsonl(out_dir / 'valid_log.jsonl', val_row)

      best_key = args.best_checkpoint_key
      if best_key == 'loc_total' and valid_pair_loader is None:
        best_key = 'sft_loss'
      improved = False
      if best_key == 'loc_total':
        v_loc = float(val_row.get('val_loc_total', float('inf')))
        if v_loc < best_val_loc:
          best_val_loc = v_loc
          improved = True
      else:
        if val_metrics['val_sft_loss'] < best_val_sft:
          best_val_sft = val_metrics['val_sft_loss']
          improved = True
      if improved:
        _save_loc_checkpoint(
            out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, val_row, loc_proj)
        print(f'[posttrain-loc] saved best.ckpt at step={step} key={best_key}: {out_dir / "best.ckpt"}')

    if args.save_every > 0 and step % args.save_every == 0:
      _save_loc_checkpoint(
          out_dir / f'step_{step}.ckpt', model, optimizer, scheduler, args, step,
          {'step': step, 'recipe': args.recipe}, loc_proj)
      print(f'[posttrain-loc] saved checkpoint: {out_dir / f"step_{step}.ckpt"}')

  if int(args.max_steps) > 0 and (int(args.max_steps) % int(args.grad_accum_steps)) != 0:
    if args.grad_clip and args.grad_clip > 0:
      _clip = list(model.parameters())
      if loc_proj is not None:
        _clip += list(loc_proj.parameters())
      torch.nn.utils.clip_grad_norm_(_clip, float(args.grad_clip))
    optimizer.step()
    scheduler.step()
    optim_step += 1
    optimizer.zero_grad(set_to_none=True)
    print(
        f'[posttrain-loc] flushed partial grad-accum buffer '
        f'(max_steps={args.max_steps} % grad_accum_steps={args.grad_accum_steps} != 0).',
    )

  final_metrics = {
      'step': step,
      'recipe': args.recipe,
      'best_val_sft_loss': best_val_sft,
      'best_val_loc_total': best_val_loc,
  }
  _save_loc_checkpoint(out_dir / 'last.ckpt', model, optimizer, scheduler, args, step, final_metrics, loc_proj)
  if valid_loader is None:
    _save_loc_checkpoint(out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, final_metrics, loc_proj)
  print(f'[posttrain-loc] done. last={out_dir / "last.ckpt"}; best={out_dir / "best.ckpt"}')


if __name__ == '__main__':
  main()
