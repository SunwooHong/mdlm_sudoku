#!/usr/bin/env python3
"""Post-train MDLM Sudoku with **pairwise D4 orbit consistency** (same format as ``sudoku_posttrain_soft_oh.py``).

Each step (unless ``--r-sft`` mixes in row-SFT) samples ``--pair-base-size`` canonical puzzles; for each,
two **distinct** D4 views share the same canonical blank mask and noise level ``t``. Logits are pulled to
canonical coordinates. Loss::

  L = (1/2)[CE(g) + CE(h)] / t + γ_eff · mean_{i ∈ supervise} JS(p_g(i), p_h(i))

``γ_eff`` follows ``--pair-gamma-warmup-frac`` / ``--pair-gamma-ramp-frac`` over **optimizer updates**
(aligned with LR cosine horizon), mirroring the HF pair trainer behavior.

Run from repo root::

  python posttrain/sudoku_posttrain_pair_orbit.py \\
    --checkpoint outputs/.../best.ckpt \\
    --model sudoku_50m \\
    --npy-root dataset/3m_posttrain_10k \\
    --valid-npy-root dataset/3m_only_val_npy \\
    --output-dir outputs/posttrain/pair10k \\
    --fair-method pair10k \\
    --pair-base-size 4 \\
    --pair-gamma 0.05 \\
    --pair-gamma-warmup-frac 0.15 \\
    --pair-gamma-ramp-frac 0.10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_POST = Path(__file__).resolve().parent
MDLM_ROOT = _POST.parent
if str(MDLM_ROOT) not in sys.path:
  sys.path.insert(0, str(MDLM_ROOT))

_spec = importlib.util.spec_from_file_location(
    'sudoku_posttrain_soft_oh_mod',
    str(_POST / 'sudoku_posttrain_soft_oh.py'),
)
if _spec is None or _spec.loader is None:
  raise RuntimeError('Cannot load sudoku_posttrain_soft_oh.py')
soh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(soh)

TRANSFORMS = soh.TRANSFORMS


def _js_divergence_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """JS(p,q) per cell; p,q are probabilities [L, 9]."""
  m = 0.5 * (p + q)
  lp = p.clamp(min=eps).log()
  lq = q.clamp(min=eps).log()
  lm = m.clamp(min=eps).log()
  kl_p = (p * (lp - lm)).sum(dim=-1)
  kl_q = (q * (lq - lm)).sum(dim=-1)
  return 0.5 * (kl_p + kl_q)


def _pair_gamma_eff(
    *,
    gamma: float,
    optim_step: int,
    total_opt_steps: int,
    warmup_frac: float,
    ramp_frac: float,
) -> float:
  gamma = float(gamma)
  if gamma <= 0.0:
    return 0.0
  total_opt_steps = max(1, int(total_opt_steps))
  wu = max(0.0, min(1.0, float(warmup_frac)))
  ramp = max(0.0, min(1.0, float(ramp_frac)))
  wu_end = int(round(wu * total_opt_steps))
  ramp_end = int(round((wu + ramp) * total_opt_steps))
  gs = int(optim_step)
  if gs < wu_end:
    return 0.0
  if ramp <= 0.0 or ramp_end <= wu_end or gs >= ramp_end:
    return gamma
  t = float(gs - wu_end) / float(max(ramp_end - wu_end, 1))
  return gamma * max(0.0, min(1.0, t))


class SudokuPairOrbitChunkDataset(Dataset):
  """One chunk = ``P`` puzzles × 2 D4 views; shared canonical mask + ``t`` per puzzle."""

  def __init__(
      self,
      solutions: np.ndarray,
      anchors: np.ndarray,
      base_indices: np.ndarray,
      *,
      pair_base_size: int,
      seed: int,
      blank_mask_mode: str,
      blank_mask_eps: float,
      sft_time: float,
      sample_mode: str = 'random',
      stochastic_masks: bool = True,
  ) -> None:
    self.solutions = solutions
    self.anchors = anchors
    self.base_indices = np.asarray(base_indices, dtype=np.int64)
    self.P = int(pair_base_size)
    self.seed = int(seed)
    self.blank_mask_mode = str(blank_mask_mode)
    self.blank_mask_eps = float(blank_mask_eps)
    self.sft_time = float(sft_time)
    self.sample_mode = str(sample_mode)
    self.stochastic_masks = bool(stochastic_masks)
    if self.sample_mode not in ('random', 'sequential', 'sequential_drop_last'):
      raise ValueError(f'sample_mode must be random, sequential, or sequential_drop_last, got {sample_mode!r}')
    self.transform_indices = soh._build_transform_indices()
    self.perms_np = np.stack(
        [self.transform_indices[name] for name in TRANSFORMS],
        axis=0,
    )
    self.inv_perms_np = np.stack(
        [soh._invert_perm_np(self.perms_np[g]) for g in range(len(TRANSFORMS))],
        axis=0,
    )
    n = len(self.base_indices)
    if self.sample_mode == 'sequential':
      self._len = max(1, (n + self.P - 1) // max(1, self.P))
    elif self.sample_mode == 'sequential_drop_last':
      self._len = max(1, n // max(1, self.P))
    else:
      self._len = max(1, n // max(1, self.P))

  def __len__(self) -> int:
    return int(self._len)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    rng = np.random if self.stochastic_masks else np.random.default_rng(self.seed + int(idx) * 1000003)

    if self.sample_mode in ('sequential', 'sequential_drop_last'):
      start = int(idx) * self.P
      n_pool = len(self.base_indices)
      pick = np.empty(self.P, dtype=np.int64)
      for j in range(self.P):
        pick[j] = int(self.base_indices[(start + j) % n_pool])
    else:
      pool = self.base_indices
      replace = len(pool) < self.P
      pick = rng.choice(pool, size=self.P, replace=replace).astype(np.int64)

    rows_x: List[torch.Tensor] = []
    rows_t: List[float] = []
    canon_sol_t: List[torch.Tensor] = []
    canon_anchor_t: List[torch.Tensor] = []
    mask_blank_canon_t: List[torch.Tensor] = []
    mask_blank_view_rows: List[torch.Tensor] = []
    anchor_view_rows: List[torch.Tensor] = []
    view_idx_rows: List[int] = []

    for b in range(self.P):
      arr_idx = int(pick[b])
      sol_c = np.asarray(self.solutions[arr_idx], dtype=np.int64).reshape(81)
      anc_c = np.asarray(self.anchors[arr_idx], dtype=np.int64).reshape(81).astype(np.bool_)
      blank_c = ~anc_c

      if self.blank_mask_mode == 'full':
        t_b = self.sft_time
        mask_blank_c = blank_c.copy()
      elif self.blank_mask_mode == 'random':
        eps = self.blank_mask_eps
        t_b = float((1.0 - eps) * rng.random() + eps)
        rand_c = rng.random(81)
        mask_blank_c = blank_c & (rand_c < t_b)
        if mask_blank_c.sum() == 0 and blank_c.any():
          mask_blank_c = blank_c.copy()
      else:
        raise ValueError(f'Unknown blank_mask_mode={self.blank_mask_mode!r}')

      anc_t = torch.from_numpy(anc_c).bool()
      tf_g, tf_h = rng.choice(len(TRANSFORMS), size=2, replace=False).tolist()
      for tf in (int(tf_g), int(tf_h)):
        perm = self.perms_np[tf]
        sol_v = sol_c[perm]
        mask_blank_v = mask_blank_c[perm]
        rows_x.append(torch.from_numpy(sol_v).long())
        rows_t.append(t_b)
        mask_blank_view_rows.append(torch.from_numpy(mask_blank_v).bool())
        anchor_view_rows.append(anc_t[perm].clone())
        view_idx_rows.append(tf)

      canon_sol_t.append(torch.from_numpy(sol_c).long())
      canon_anchor_t.append(anc_t.clone())
      mask_blank_canon_t.append(torch.from_numpy(mask_blank_c).bool())

    x_gold = torch.stack(rows_x, dim=0)
    anchor_mask = torch.stack(anchor_view_rows, dim=0)
    mask_blank_view = torch.stack(mask_blank_view_rows, dim=0)

    return {
        'x_gold': x_gold,
        'anchor_mask': anchor_mask,
        'mask_blank_view': mask_blank_view.bool(),
        'canonical_sol': torch.stack(canon_sol_t, dim=0),
        'canonical_anchor': torch.stack(canon_anchor_t, dim=0).bool(),
        'supervise_canon': torch.stack(mask_blank_canon_t, dim=0).bool(),
        't_cond': torch.tensor(rows_t, dtype=torch.float32),
        'pair_base_size': torch.tensor(self.P, dtype=torch.long),
        'pair_view_idx': torch.tensor(view_idx_rows, dtype=torch.long),
    }


def _pair_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
  return batch[0]


def _pair_orbit_consistency_loss(
    model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    inv_perms: torch.Tensor,
    label_smoothing: float,
    gamma_eff: float,
    blank_mask_mode: str,
    blank_mask_eps: float,
    sft_time: float,
    t_min: float,
    resample_corruption: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
  """Pairwise orbit JS + gold CE (canonical), with CE scaled by ``1/t`` like row-SFT."""
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
  logits = model.forward(x_cond, sigma[:, None])
  if logits.shape[-1] > 9:
    logits = logits[..., :9]

  targets = canonical_sol
  loss_terms: List[torch.Tensor] = []
  L_ce_list: List[torch.Tensor] = []
  L_js_list: List[torch.Tensor] = []

  for b in range(P):
    denom = supervise_f[b].sum().clamp(min=1.0)
    logits_canon_list: List[torch.Tensor] = []
    ce_arm: List[torch.Tensor] = []
    for j in range(2):
      row = b * 2 + j
      g = int(pair_view[row].item())
      logits_row = logits[row : row + 1]
      idx = inv[g].view(1, 81, 1).expand(1, 81, logits_row.shape[-1])
      logits_canon = torch.gather(logits_row, 1, idx).squeeze(0).float()
      logits_canon_list.append(logits_canon)
      sup = supervise_f[b]
      ce_flat = F.cross_entropy(
          logits_canon,
          targets[b],
          reduction='none',
          label_smoothing=float(label_smoothing),
      )
      ce_arm.append((ce_flat * sup).sum() / denom)

    L_ce = 0.5 * (ce_arm[0] + ce_arm[1])
    L_ce_list.append(L_ce)
    pg = F.softmax(logits_canon_list[0], dim=-1)
    ph = F.softmax(logits_canon_list[1], dim=-1)
    js_cell = _js_divergence_probs(pg, ph)
    L_js = (js_cell * supervise_f[b]).sum() / denom
    L_js_list.append(L_js)

    t_b = t_cond[b * 2].clamp(min=float(t_min))
    loss_terms.append((L_ce / t_b) + float(gamma_eff) * L_js)

  loss = torch.stack(loss_terms).mean()

  with torch.no_grad():
    pred_all = logits.reshape(P * 2, 81, -1).argmax(dim=-1)
    mflat = mask_blank_view.reshape(-1, 81).float()
    acc = (pred_all == x_gold).float()
    acc_row = (acc * mflat).sum(dim=1) / mflat.sum(dim=1).clamp(min=1.0)
    n_tok = float(mflat.sum().item())
    L_js_m = torch.stack(L_js_list).mean()

  return loss, {
      'sft_token_acc': float(acc_row.mean().item()),
      'num_loss_tokens': n_tok,
      'train_pair_ce_mean': float(torch.stack(L_ce_list).mean().item()),
      'train_pair_js_mean': float(L_js_m.item()),
      'train_pair_total': float(loss.item()),
      'gamma_eff': float(gamma_eff),
  }


PAIR_FAIR_METHODS = {
    'pair10k': dict(num_base_examples=10_000, recipe='canonical_sft_80k'),
    'pair80k': dict(num_base_examples=80_000, recipe='canonical_sft_80k'),
}


def _apply_pair_fair_method(args: argparse.Namespace) -> None:
  if not getattr(args, 'fair_method', ''):
    return
  preset = PAIR_FAIR_METHODS[str(args.fair_method)]
  args.num_base_examples = int(preset['num_base_examples'])
  args.recipe = str(preset['recipe'])


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description='Post-train MDLM Sudoku with pairwise orbit consistency (CE + JS).')
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
      help='pair10k / pair80k presets (num bases + row-SFT recipe when --r-sft>0).',
  )
  p.add_argument('--recipe', type=str, default='canonical_sft_80k', choices=sorted(soh.RECIPE_DEFAULTS))
  p.add_argument('--output-dir', type=str, required=True)

  p.add_argument('--pair-base-size', type=int, default=4,
                 help='Puzzles per chunk; forward rows = 2 * this value.')
  p.add_argument('--pair-gamma', type=float, default=0.05, help='JS weight γ in the objective.')
  p.add_argument('--pair-gamma-warmup-frac', type=float, default=0.0,
                 help='First this fraction of *optimizer updates*: γ_eff = 0 (CE only).')
  p.add_argument('--pair-gamma-ramp-frac', type=float, default=0.0,
                 help='After warmup, linearly ramp γ_eff to --pair-gamma over this optimizer-frac.')
  p.add_argument('--pair-t-min', type=float, default=0.05, help='Clamp t when dividing CE.')
  p.add_argument('--r-sft', type=float, default=0.0,
                 help='Probability of an ordinary row-SFT micro-step (else pair chunk). 0 = pair only.')

  p.add_argument('--num-base-examples', type=int, default=0)
  p.add_argument('--subset-mode', type=str, default='random', choices=('random', 'first'))
  p.add_argument('--subset-seed', type=int, default=0)
  p.add_argument('--seed', type=int, default=0)

  p.add_argument('--batch-size', type=int, default=512, help='Row batch size when using --r-sft > 0.')
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
  p.add_argument('--deterministic-pair-masks', action='store_true',
                 help='Fixed pair corruptions keyed by dataset index (validation-style).')

  p.add_argument('--eval-every', type=int, default=500)
  p.add_argument('--eval-max-batches', type=int, default=32)
  p.add_argument('--eval-pair-max-batches', type=int, default=16)
  p.add_argument(
      '--best-checkpoint-key',
      type=str,
      default='pair_loss',
      choices=('sft_loss', 'pair_loss'),
      help='best.ckpt: minimize val_sft_loss or val_pair_total.',
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


def _run_pair_mask_sanity(pair_loader: DataLoader, inv_cpu: torch.Tensor) -> None:
  inv = inv_cpu.long()
  batch = next(iter(pair_loader))
  P = int(batch['pair_base_size'].item())
  mb = batch['mask_blank_view']
  sup = batch['supervise_canon']
  pv = batch['pair_view_idx']
  for b in range(P):
    for j in range(2):
      row = b * 2 + j
      g = int(pv[row].item())
      mask_back = mb[row][inv[g]]
      if not torch.equal(mask_back, sup[b]):
        raise AssertionError(f'pair mask sanity failed: b={b} j={j} g={g}')


@torch.no_grad()
def _evaluate_pair_loss(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    inv_perms: torch.Tensor,
    label_smoothing: float,
    gamma: float,
    max_batches: int,
    blank_mask_mode: str,
    blank_mask_eps: float,
    sft_time: float,
    t_min: float,
) -> Dict[str, float]:
  model.eval()
  total = 0.0
  n_batches = 0
  for batch in loader:
    loss, _info = _pair_orbit_consistency_loss(
        model,
        batch,
        device=device,
        inv_perms=inv_perms,
        label_smoothing=label_smoothing,
        gamma_eff=float(gamma),
        blank_mask_mode=blank_mask_mode,
        blank_mask_eps=blank_mask_eps,
        sft_time=sft_time,
        t_min=t_min,
        resample_corruption=False,
    )
    total += float(loss.item())
    n_batches += 1
    if max_batches > 0 and n_batches >= max_batches:
      break
  model.train()
  denom = max(n_batches, 1)
  return {'val_pair_total': total / denom, 'val_pair_batches': float(n_batches)}


def main() -> None:
  args = parse_args()
  _apply_pair_fair_method(args)
  soh._register_omegaconf_resolvers()

  if int(args.pair_base_size) < 1:
    raise SystemExit('--pair-base-size must be >= 1')
  if float(args.pair_gamma) < 0.0:
    raise SystemExit('--pair-gamma must be >= 0')
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
  pair_loader = DataLoader(
      pair_ds,
      batch_size=1,
      shuffle=True,
      collate_fn=_pair_collate,
      num_workers=args.num_workers,
      pin_memory=args.pin_memory,
      persistent_workers=args.num_workers > 0,
      worker_init_fn=soh._worker_init_fn if args.num_workers > 0 else None,
  )

  train_loader: Optional[DataLoader] = None
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
    train_loader = DataLoader(
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

  valid_loader: Optional[DataLoader] = None
  valid_pair_loader: Optional[DataLoader] = None
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
    valid_loader = DataLoader(
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
    valid_pair_loader = DataLoader(
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

  if args.best_checkpoint_key == 'pair_loss' and valid_pair_loader is None:
    print('[posttrain-pair] warning: --best-checkpoint-key pair_loss but no valid split; using sft_loss.')

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
  model = soh._load_model(cfg, device)
  inv_perms_cpu = soh._build_inv_perms_t()

  if args.sanity_check:
    _run_pair_mask_sanity(pair_loader, inv_perms_cpu)
    print('[posttrain-pair] sanity_check passed.')

  if not args.train_noise:
    for param in model.noise.parameters():
      param.requires_grad_(False)
  params = list(model.backbone.parameters()) + list(model.noise.parameters()) if args.train_noise else list(model.backbone.parameters())
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
      'method': 'pairwise_orbit_consistency',
  }
  (out_dir / 'posttrain_config.json').write_text(json.dumps(config_summary, indent=2), encoding='utf-8')
  print(json.dumps(config_summary, indent=2))

  step = 0
  optim_step = 0
  best_val_sft = float('inf')
  best_val_pair = float('inf')
  running_loss = 0.0
  running_acc = 0.0
  running_steps = 0
  log_window_pair = 0.0
  log_window_steps = 0
  pair_metric_sums: Dict[str, float] = {}
  pair_metric_count = 0
  optimizer.zero_grad(set_to_none=True)

  pair_iter = iter(pair_loader)
  sft_iter = iter(train_loader) if train_loader is not None else None

  while step < args.max_steps:
    gamma_eff = _pair_gamma_eff(
        gamma=float(args.pair_gamma),
        optim_step=optim_step + 1,
        total_opt_steps=total_opt_steps,
        warmup_frac=float(args.pair_gamma_warmup_frac),
        ramp_frac=float(args.pair_gamma_ramp_frac),
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
      loss, info = _pair_orbit_consistency_loss(
          model,
          ob,
          device=device,
          inv_perms=inv_perms_cpu,
          label_smoothing=args.label_smoothing,
          gamma_eff=gamma_eff,
          blank_mask_mode=args.blank_mask_mode,
          blank_mask_eps=args.blank_mask_eps,
          sft_time=args.sft_time,
          t_min=float(args.pair_t_min),
          resample_corruption=True,
      )
      for k in ('train_pair_ce_mean', 'train_pair_js_mean', 'train_pair_total', 'gamma_eff'):
        if k in info:
          pair_metric_sums[k] = pair_metric_sums.get(k, 0.0) + float(info[k])
      pair_metric_count += 1

    scaled_loss = loss / max(1, args.grad_accum_steps)
    scaled_loss.backward()

    running_loss += float(loss.item())
    running_acc += float(info['sft_token_acc'])
    running_steps += 1
    log_window_steps += 1
    log_window_pair += 0.0 if use_sft else 1.0

    if (step + 1) % args.grad_accum_steps == 0:
      if args.grad_clip and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
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
          'pair_gamma_eff': gamma_eff,
          'lr': optimizer.param_groups[0]['lr'],
          'recipe': args.recipe,
      }
      if pair_metric_count > 0:
        denom_om = float(pair_metric_count)
        for k, s in pair_metric_sums.items():
          row[k] = s / denom_om
      print(json.dumps(row))
      soh._write_jsonl(out_dir / 'train_log.jsonl', row)
      running_loss = 0.0
      running_acc = 0.0
      running_steps = 0
      log_window_pair = 0.0
      log_window_steps = 0
      pair_metric_sums = {}
      pair_metric_count = 0

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
        pm = _evaluate_pair_loss(
            model,
            valid_pair_loader,
            device=device,
            inv_perms=inv_perms_cpu,
            label_smoothing=args.label_smoothing,
            gamma=float(args.pair_gamma),
            max_batches=args.eval_pair_max_batches,
            blank_mask_mode=args.blank_mask_mode,
            blank_mask_eps=args.blank_mask_eps,
            sft_time=args.sft_time,
            t_min=float(args.pair_t_min),
        )
        val_row.update(pm)
      print(json.dumps(val_row))
      soh._write_jsonl(out_dir / 'valid_log.jsonl', val_row)

      best_key = args.best_checkpoint_key
      if best_key == 'pair_loss' and valid_pair_loader is None:
        best_key = 'sft_loss'
      improved = False
      if best_key == 'pair_loss':
        v_pair = float(val_row.get('val_pair_total', float('inf')))
        if v_pair < best_val_pair:
          best_val_pair = v_pair
          improved = True
      else:
        if val_metrics['val_sft_loss'] < best_val_sft:
          best_val_sft = val_metrics['val_sft_loss']
          improved = True
      if improved:
        soh._save_checkpoint(
            out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, val_row)
        print(f'[posttrain-pair] saved best.ckpt at step={step} key={best_key}: {out_dir / "best.ckpt"}')

    if args.save_every > 0 and step % args.save_every == 0:
      soh._save_checkpoint(
          out_dir / f'step_{step}.ckpt', model, optimizer, scheduler, args, step,
          {'step': step, 'recipe': args.recipe})
      print(f'[posttrain-pair] saved checkpoint: {out_dir / f"step_{step}.ckpt"}')

  if int(args.max_steps) > 0 and (int(args.max_steps) % int(args.grad_accum_steps)) != 0:
    if args.grad_clip and args.grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
    optimizer.step()
    scheduler.step()
    optim_step += 1
    optimizer.zero_grad(set_to_none=True)
    print(
        f'[posttrain-pair] flushed partial grad-accum buffer '
        f'(max_steps={args.max_steps} % grad_accum_steps={args.grad_accum_steps} != 0).',
    )

  final_metrics = {
      'step': step,
      'recipe': args.recipe,
      'best_val_sft_loss': best_val_sft,
      'best_val_pair_total': best_val_pair,
  }
  soh._save_checkpoint(out_dir / 'last.ckpt', model, optimizer, scheduler, args, step, final_metrics)
  if valid_loader is None:
    soh._save_checkpoint(out_dir / 'best.ckpt', model, optimizer, scheduler, args, step, final_metrics)
  print(f'[posttrain-pair] done. last={out_dir / "last.ckpt"}; best={out_dir / "best.ckpt"}')


if __name__ == '__main__':
  main()
