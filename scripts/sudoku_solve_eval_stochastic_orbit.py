#!/usr/bin/env python3
"""Evaluate an MDLM Sudoku checkpoint with stochastic orbit metrics.

This script extends the original sudoku_solve_eval.py in three important ways:

1. It supports K stochastic samples per D4 transform via --samples-per-transform.
2. It reports stochastic LE-style metrics based on per-transform success rates:
   mean success, capacity max_T p_T, robust min_T p_T, and SLS=max_T p_T-min_T p_T.
3. It canonicalizes transformed predictions and reports marginal orbit divergence (MOD),
   a distributional equivariance diagnostic over target cells.

The script remains backward compatible for one-sample orbit eval, but for stochastic
MDLMs the recommended setting is K>=16 with temperature=1.0 or a validation-selected
sampling temperature.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Repo imports (run with cwd = mdlm root, or PYTHONPATH=.)
MDLM_ROOT = Path(__file__).resolve().parent.parent
if str(MDLM_ROOT) not in sys.path:
  sys.path.insert(0, str(MDLM_ROOT))

import diffusion as diffusion_mod
import dataloader as dataloader_mod
from sudoku_dataloader import SudokuNpyDataset

# Snapshot before any patch so temperature=1 restores exact repo behavior.
_ORIGINAL_SAMPLE_CATEGORICAL = diffusion_mod._sample_categorical


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


SUCCESS_METRICS = (
  'exact',
  'target_exact',
  'target_exact_and_clue',
  'valid',
  'valid_and_clue',
)


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
  """Explicit RNG for sampling (isolates eval noise from global torch.manual_seed)."""
  if device.type == 'cuda':
    gen = torch.Generator(device=device)
  else:
    gen = torch.Generator()
  gen.manual_seed(int(seed))
  return gen


def _configure_sampling_temperature(temperature: float) -> None:
  """Patch categorical sampling used inside diffusion DDPM updates.

  temperature=1: original diffusion._sample_categorical.
  temperature=0: local argmax; useful as a diagnostic but may collapse for MDLM Sudoku.
  otherwise: scale probabilities as softmax(log(p)/T) before the original-style
  Gumbel-ratio categorical sample.
  """
  orig = _ORIGINAL_SAMPLE_CATEGORICAL
  t = float(temperature)
  if t < 0:
    raise ValueError(f'temperature must be >= 0, got {t}')
  if abs(t - 1.0) < 1e-7:
    diffusion_mod._sample_categorical = orig
    return

  if t <= 1e-12:

    def _greedy_sample(
        categorical_probs: torch.Tensor,
        generator: Optional[torch.Generator] = None) -> torch.Tensor:
      del generator
      return categorical_probs.argmax(dim=-1)

    diffusion_mod._sample_categorical = _greedy_sample
    return

  def _temp_sample(
      categorical_probs: torch.Tensor,
      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    logits = torch.log(categorical_probs.clamp(min=1e-30))
    scaled = F.softmax(logits / t, dim=-1)
    if generator is None:
      rand_src = torch.rand_like(scaled)
    else:
      rand_src = torch.rand(
        scaled.shape,
        device=scaled.device,
        dtype=scaled.dtype,
        generator=generator)
    gumbel_norm = 1e-10 - (rand_src + 1e-10).log()
    return (scaled / gumbel_norm).argmax(dim=-1)

  diffusion_mod._sample_categorical = _temp_sample


def _register_omegaconf_resolvers() -> None:
  OmegaConf.register_new_resolver('cwd', os.getcwd, replace=True)
  OmegaConf.register_new_resolver('mdlm_root', lambda: str(MDLM_ROOT), replace=True)
  OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
  OmegaConf.register_new_resolver('eval', eval, replace=True)
  OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)


def _is_valid_completed_sudoku(grid_0_8: np.ndarray) -> bool:
  """grid_0_8: length-81 ints in 0..8 representing digits 1..9."""
  g = grid_0_8.reshape(9, 9).astype(np.int64) + 1
  if g.shape != (9, 9):
    return False
  if (g < 1).any() or (g > 9).any():
    return False
  digits = frozenset(range(1, 10))
  for r in range(9):
    if frozenset(g[r, :]) != digits:
      return False
  for c in range(9):
    if frozenset(g[:, c]) != digits:
      return False
  for br in range(0, 9, 3):
    for bc in range(0, 9, 3):
      if frozenset(g[br:br + 3, bc:bc + 3].ravel()) != digits:
        return False
  return True


def _apply_transform(grid: np.ndarray, transform_name: str) -> np.ndarray:
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
  """Return idx where transformed_flat[p] = canonical_flat[idx[p]]."""
  base = np.arange(81, dtype=np.int64).reshape(9, 9)
  return {name: _apply_transform(base, name).reshape(-1) for name in TRANSFORMS}


def _canonicalize_from_transform(transformed: np.ndarray, idx: np.ndarray) -> np.ndarray:
  """Map [N,81] predictions from transformed coordinates back to canonical.

  If transformed[:, p] corresponds to canonical position idx[p], then canonical[:, idx[p]]
  receives transformed[:, p].
  """
  canonical = np.empty_like(transformed)
  canonical[:, idx] = transformed
  return canonical


def _compose_config(
    *,
    checkpoint: str,
    model: str,
    num_steps: int,
    batch_size: int,
    predictor: str,
    noise_removal: bool,
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
  with initialize_config_dir(version_base=None, config_dir=str(MDLM_ROOT / 'configs')):
    return compose(config_name='config', overrides=overrides)


def _load_model(cfg: Any, device: torch.device) -> diffusion_mod.Diffusion:
  tok = dataloader_mod.get_tokenizer(cfg)
  ckpt_path = str(cfg.eval.checkpoint_path)
  payload = torch.load(ckpt_path, map_location=device)
  if isinstance(payload, dict) and 'model_state_dict' in payload:
    state_dict = payload['model_state_dict']
  elif isinstance(payload, dict) and 'state_dict' in payload:
    state_dict = payload['state_dict']
  else:
    state_dict = payload

  # Some post-training checkpoints include frozen reference weights.
  if isinstance(state_dict, dict):
    state_dict = {
      k: v for k, v in state_dict.items()
      if not k.startswith('ref_backbone.')
    }

  model = diffusion_mod.Diffusion(cfg, tokenizer=tok)
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if unexpected:
    raise RuntimeError(f'Unexpected state_dict keys: {unexpected[:8]}')

  ema_state = None
  if isinstance(payload, dict):
    ema_state = payload.get('ema_state_dict')
  if ema_state is not None and getattr(model, 'ema', None) is not None:
    model.ema.load_state_dict(ema_state)
  else:
    model.ema = None

  if missing:
    print(f'[eval] load_state_dict missing keys (showing up to 8): {missing[:8]}')
  model = model.to(device)
  model.eval()
  return model


def _safe_numpy_slice(arr: np.ndarray, n: int, dtype: Any) -> np.ndarray:
  """Copy memmap slices to writable arrays to avoid torch.from_numpy warnings."""
  return np.asarray(arr[:n], dtype=dtype).copy()


def _find_split_file(npy_root: Path, split: str, candidates: Sequence[str]) -> Optional[Path]:
  aliases = [split]
  if split == 'valid':
    aliases.append('validation')
  if split == 'validation':
    aliases.append('valid')
  for sp in aliases:
    for pattern in candidates:
      path = npy_root / pattern.format(split=sp)
      if path.exists():
        return path
  return None


def _load_optional_target_masks(
    npy_root: Path,
    split: str,
    n: int,
    explicit_path: str,
) -> Optional[np.ndarray]:
  """Load target masks if available; otherwise return None.

  For legacy sudoku9-anchor datasets, target masks usually do not exist; in that case
  the target set defaults to all non-anchor positions.
  """
  if explicit_path:
    path = Path(explicit_path)
    if not path.is_absolute():
      path = MDLM_ROOT / path
    if not path.exists():
      raise FileNotFoundError(f'--target-mask-npy not found: {path}')
    return np.asarray(np.load(path, mmap_mode='r')[:n], dtype=np.bool_).copy()

  path = _find_split_file(
    npy_root,
    split,
    candidates=(
      '{split}_target_mask.npy',
      '{split}_targets_mask.npy',
      '{split}_target.npy',
      '{split}_targets.npy',
    ),
  )
  if path is None:
    return None
  print(f'[eval] loaded target masks from {path}', file=sys.stderr)
  return np.asarray(np.load(path, mmap_mode='r')[:n], dtype=np.bool_).copy()


def _choose_success_flags(
    *,
    success_metric: str,
    exact: np.ndarray,
    target_exact: np.ndarray,
    clue_ok: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
  if success_metric == 'exact':
    return exact
  if success_metric == 'target_exact':
    return target_exact
  if success_metric == 'target_exact_and_clue':
    return target_exact & clue_ok
  if success_metric == 'valid':
    return valid
  if success_metric == 'valid_and_clue':
    return valid & clue_ok
  raise ValueError(f'Unknown success metric: {success_metric}')


def _score_prediction_batch(
    pred: np.ndarray,
    sol: np.ndarray,
    anchor: np.ndarray,
    target_mask: np.ndarray,
    success_metric: str,
) -> Dict[str, np.ndarray]:
  """Score a batch of completed samples in transformed coordinates."""
  n = pred.shape[0]
  exact = np.all(pred == sol, axis=1)
  clue_ok = np.all((pred == sol) | (~anchor), axis=1)
  target_exact = np.ones(n, dtype=np.bool_)
  target_cell_correct = np.zeros(n, dtype=np.int64)
  target_cell_count = np.maximum(target_mask.sum(axis=1), 1).astype(np.int64)

  for i in range(n):
    m = target_mask[i]
    if m.any():
      correct = pred[i, m] == sol[i, m]
      target_exact[i] = bool(np.all(correct))
      target_cell_correct[i] = int(correct.sum())
    else:
      target_exact[i] = True
      target_cell_correct[i] = 0

  valid = np.asarray([_is_valid_completed_sudoku(p) for p in pred], dtype=np.bool_)
  success = _choose_success_flags(
    success_metric=success_metric,
    exact=exact,
    target_exact=target_exact,
    clue_ok=clue_ok,
    valid=valid,
  )
  return {
    'exact': exact,
    'target_exact': target_exact,
    'clue_ok': clue_ok,
    'valid': valid,
    'valid_and_clue': valid & clue_ok,
    'success': success,
    'target_cell_correct': target_cell_correct,
    'target_cell_count': target_cell_count,
  }


@torch.no_grad()
def _sample_infill_arrays_with_scores(
    *,
    model: diffusion_mod.Diffusion,
    solutions: np.ndarray,
    anchors: np.ndarray,
    target_masks: np.ndarray,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    success_metric: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
  """Sample once per example and return aggregate metrics plus predictions [N,81]."""
  n = solutions.shape[0]
  mask_id = model.mask_index
  pred_all = np.empty((n, 81), dtype=np.int16)

  exact_flags = np.zeros(n, dtype=np.bool_)
  target_exact_flags = np.zeros(n, dtype=np.bool_)
  clue_ok_flags = np.zeros(n, dtype=np.bool_)
  valid_flags = np.zeros(n, dtype=np.bool_)
  valid_and_clue_flags = np.zeros(n, dtype=np.bool_)
  success_flags = np.zeros(n, dtype=np.bool_)
  target_cell_correct_total = 0
  target_cell_count_total = 0
  clue_cells_total = 0

  for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    sol_np = np.asarray(solutions[start:end], dtype=np.int64)
    anc_np = np.asarray(anchors[start:end], dtype=np.bool_)
    tgt_np = np.asarray(target_masks[start:end], dtype=np.bool_)

    sol = torch.from_numpy(sol_np.copy()).to(device=device, dtype=torch.long)
    anchor = torch.from_numpy(anc_np.copy()).to(device=device, dtype=torch.bool)
    x_init = torch.where(anchor, sol, torch.full_like(sol, mask_id))
    pred = model.restore_model_and_sample(num_steps, x_init=x_init, generator=generator)
    pred_cpu = pred.detach().cpu().numpy().astype(np.int16)
    pred_all[start:end] = pred_cpu

    scores = _score_prediction_batch(
      pred=pred_cpu.astype(np.int64),
      sol=sol_np,
      anchor=anc_np,
      target_mask=tgt_np,
      success_metric=success_metric,
    )
    sl = slice(start, end)
    exact_flags[sl] = scores['exact']
    target_exact_flags[sl] = scores['target_exact']
    clue_ok_flags[sl] = scores['clue_ok']
    valid_flags[sl] = scores['valid']
    valid_and_clue_flags[sl] = scores['valid_and_clue']
    success_flags[sl] = scores['success']
    target_cell_correct_total += int(scores['target_cell_correct'].sum())
    target_cell_count_total += int(scores['target_cell_count'].sum())
    clue_cells_total += int(anc_np.sum())

  metrics = {
    'n_puzzles': int(n),
    'exact_match_rate': float(exact_flags.mean()),
    'target_exact_rate': float(target_exact_flags.mean()),
    'clue_consistency_rate': float(clue_ok_flags.mean()),
    'valid_sudoku_rate': float(valid_flags.mean()),
    'valid_and_clue_rate': float(valid_and_clue_flags.mean()),
    'success_rate': float(success_flags.mean()),
    'target_cell_accuracy': float(target_cell_correct_total / max(target_cell_count_total, 1)),
    'avg_clue_cells': float(clue_cells_total / max(n, 1)),
    'flags': {
      'exact': exact_flags,
      'target_exact': target_exact_flags,
      'clue_ok': clue_ok_flags,
      'valid': valid_flags,
      'valid_and_clue': valid_and_clue_flags,
      'success': success_flags,
    },
  }
  return metrics, pred_all


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return float('nan'), float('nan')
  if n_boot <= 0:
    return float('nan'), float('nan')
  rng = np.random.default_rng(int(seed))
  n = values.shape[0]
  means = np.empty(n_boot, dtype=np.float64)
  for b in range(n_boot):
    idx = rng.integers(0, n, size=n)
    means[b] = values[idx].mean()
  lo, hi = np.percentile(means, [2.5, 97.5])
  return float(lo), float(hi)


def _summarize_stochastic_orbit(
    success: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
  """success: bool array [N, T, K]."""
  if success.ndim != 3:
    raise ValueError(f'success must have shape [N,T,K], got {success.shape}')
  p_t = success.mean(axis=2)  # [N,T]

  per_puzzle_mean = p_t.mean(axis=1)
  per_puzzle_cap = p_t.max(axis=1)
  per_puzzle_robust = p_t.min(axis=1)
  per_puzzle_sls = per_puzzle_cap - per_puzzle_robust
  any_success = success.any(axis=(1, 2))
  all_transform_has_success = success.any(axis=2).all(axis=1)

  def _ci_dict(vals: np.ndarray) -> Dict[str, float]:
    lo, hi = _bootstrap_ci(vals, bootstrap_samples, seed)
    return {'mean': float(vals.mean()), 'ci95_low': lo, 'ci95_high': hi}

  return {
    'samples_per_transform': int(success.shape[2]),
    'num_transforms': int(success.shape[1]),
    'mean_success_rate': _ci_dict(per_puzzle_mean),
    'capacity_max_transform_rate': _ci_dict(per_puzzle_cap),
    'robust_min_transform_rate': _ci_dict(per_puzzle_robust),
    'stochastic_layout_sensitivity_rate': _ci_dict(per_puzzle_sls),
    'any_success_over_all_samples_rate': float(any_success.mean()),
    'all_transforms_have_at_least_one_success_rate': float(all_transform_has_success.mean()),
    'per_transform_success_rate': {
      name: float(p_t[:, i].mean()) for i, name in enumerate(TRANSFORMS)
    },
  }


def _compute_mod(
    canonical_preds: np.ndarray,
    target_masks: np.ndarray,
    vocab_size: int,
) -> Tuple[float, np.ndarray]:
  """Compute Marginal Orbit Divergence over target positions.

  canonical_preds: int array [N,T,K,81], already mapped back to canonical coords.
  target_masks: bool array [N,81] in canonical coords.
  vocab_size: categories 0..vocab_size-1; values outside range are clipped into an
              extra-safe range by ignoring them in counts.
  """
  if canonical_preds.ndim != 4:
    raise ValueError(f'canonical_preds must be [N,T,K,81], got {canonical_preds.shape}')
  n, num_t, _k, _length = canonical_preds.shape
  pair_count = num_t * (num_t - 1) // 2
  per_puzzle = np.zeros(n, dtype=np.float64)

  for i in range(n):
    pos = np.where(target_masks[i])[0]
    if pos.size == 0:
      per_puzzle[i] = 0.0
      continue
    pred_i = canonical_preds[i, :, :, pos]  # [T,K,P]
    q = np.zeros((num_t, pos.size, vocab_size), dtype=np.float32)
    for t in range(num_t):
      vals = pred_i[t]  # [K,P]
      for v in range(vocab_size):
        q[t, :, v] = (vals == v).mean(axis=0)
    total = 0.0
    for a in range(num_t):
      for b in range(a + 1, num_t):
        tv_per_pos = 0.5 * np.abs(q[a] - q[b]).sum(axis=1)
        total += float(tv_per_pos.mean())
    per_puzzle[i] = total / max(pair_count, 1)
  return float(per_puzzle.mean()), per_puzzle


@torch.no_grad()
def _run_canonical_control_k(
    *,
    model: diffusion_mod.Diffusion,
    solutions: np.ndarray,
    anchors: np.ndarray,
    target_masks: np.ndarray,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    k: int,
    success_metric: str,
    generator: Optional[torch.Generator] = None,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
  """Canonical layout K independent samples per puzzle."""
  if k <= 0:
    return {}

  success = np.zeros((solutions.shape[0], 1, k), dtype=np.bool_)
  exact_rates: List[float] = []
  target_exact_rates: List[float] = []
  valid_rates: List[float] = []
  clue_rates: List[float] = []

  for sample_i in range(k):
    metrics, _pred = _sample_infill_arrays_with_scores(
      model=model,
      solutions=solutions,
      anchors=anchors,
      target_masks=target_masks,
      device=device,
      num_steps=num_steps,
      batch_size=batch_size,
      success_metric=success_metric,
      generator=generator,
    )
    success[:, 0, sample_i] = metrics['flags']['success']
    exact_rates.append(float(metrics['exact_match_rate']))
    target_exact_rates.append(float(metrics['target_exact_rate']))
    valid_rates.append(float(metrics['valid_sudoku_rate']))
    clue_rates.append(float(metrics['clue_consistency_rate']))

  summary = _summarize_stochastic_orbit(success, bootstrap_samples, seed)
  return {
    'k': int(k),
    'success_metric': success_metric,
    'single_sample_mean_rates': {
      'exact_match_rate': float(np.mean(exact_rates)),
      'target_exact_rate': float(np.mean(target_exact_rates)),
      'valid_sudoku_rate': float(np.mean(valid_rates)),
      'clue_consistency_rate': float(np.mean(clue_rates)),
      'selected_success_rate': float(success.mean()),
    },
    'any_of_k_rate': float(success[:, 0, :].any(axis=1).mean()),
    'all_k_rate': float(success[:, 0, :].all(axis=1).mean()),
    'stochastic_summary': summary,
  }


@torch.no_grad()
def _run_unconditional(
    model: diffusion_mod.Diffusion,
    num_batches: int,
    device: torch.device,
    num_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int]:
  bsz = int(model.config.loader.eval_batch_size)
  del bsz
  n_valid = 0
  n_total = 0
  for _ in range(num_batches):
    pred = model.restore_model_and_sample(num_steps, x_init=None, generator=generator)
    pred_cpu = pred.cpu().numpy()
    for i in range(pred_cpu.shape[0]):
      n_total += 1
      if _is_valid_completed_sudoku(pred_cpu[i]):
        n_valid += 1
  return n_total, n_valid


def _json_sanitize(obj: Any) -> Any:
  """Convert numpy scalars/arrays nested in metrics to JSON-safe objects."""
  if isinstance(obj, dict):
    return {str(k): _json_sanitize(v) for k, v in obj.items() if k != 'flags'}
  if isinstance(obj, list):
    return [_json_sanitize(v) for v in obj]
  if isinstance(obj, tuple):
    return [_json_sanitize(v) for v in obj]
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, (np.integer,)):
    return int(obj)
  if isinstance(obj, (np.floating,)):
    return float(obj)
  if isinstance(obj, (np.bool_,)):
    return bool(obj)
  return obj


def main() -> None:
  _register_omegaconf_resolvers()
  p = argparse.ArgumentParser(description='MDLM 9x9 Sudoku stochastic orbit eval')
  p.add_argument('--checkpoint', type=str, required=True)
  p.add_argument('--model', type=str, default='sudoku_5m')
  p.add_argument(
    '--npy-root',
    type=str,
    default=str(MDLM_ROOT / 'dataset' / 'sudoku_9m_npy'),
    help='Directory with *_solution.npy / *_anchor.npy')
  p.add_argument(
    '--target-mask-npy',
    type=str,
    default='',
    help='Optional explicit target-mask npy. If omitted, tries split_target_mask.npy; '
         'if unavailable, targets default to non-anchor cells.')
  p.add_argument(
    '--split',
    type=str,
    choices=('train', 'valid', 'validation', 'test'),
    default='test')
  p.add_argument(
    '--task',
    type=str,
    choices=('infill', 'unconditional'),
    default='infill')
  p.add_argument('--max-puzzles', type=int, default=512)
  p.add_argument('--batch-size', type=int, default=64)
  p.add_argument('--num-steps', type=int, default=128)
  p.add_argument('--num-batches', type=int, default=8, help='For unconditional only')
  p.add_argument('--predictor', type=str, default='ddpm_cache')
  p.add_argument('--temperature', type=float, default=1.0)
  p.add_argument('--no-noise-removal', action='store_true')
  p.add_argument('--device', type=str, default='cuda')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--generator-seed', type=int, default=None)
  p.add_argument(
    '--explicit-generator',
    dest='explicit_generator',
    action='store_true',
    default=True)
  p.add_argument(
    '--no-explicit-generator',
    dest='explicit_generator',
    action='store_false')
  p.add_argument(
    '--reset-generator-each-orbit-transform',
    action='store_true',
    default=False,
    help='Reset generator to --generator-seed before each transform. This is not a true '
         'common-random-number coupling, but can be useful as a control.')
  p.add_argument(
    '--samples-per-transform',
    type=int,
    default=1,
    help='K stochastic samples for each D4 transform.')
  p.add_argument(
    '--control-canonical-k',
    type=int,
    default=0,
    help='If >0, run K independent samples on the canonical layout as sampling-variance control.')
  p.add_argument(
    '--success-metric',
    type=str,
    choices=SUCCESS_METRICS,
    default='exact',
    help='Which binary success flag to use for stochastic orbit metrics.')
  p.add_argument(
    '--compute-mod',
    dest='compute_mod',
    action='store_true',
    default=True,
    help='Compute marginal orbit divergence from canonicalized predictions.')
  p.add_argument(
    '--no-compute-mod',
    dest='compute_mod',
    action='store_false')
  p.add_argument(
    '--bootstrap-samples',
    type=int,
    default=1000,
    help='Bootstrap resamples over base puzzles for CI. Use 0 to disable.')
  p.add_argument(
    '--save-npz',
    action='store_true',
    default=False,
    help='Save flags and canonicalized predictions to output_json.with_suffix(.npz).')
  p.add_argument('--output-json', type=str, default='')
  args = p.parse_args()

  if args.samples_per_transform <= 0:
    raise ValueError('--samples-per-transform must be positive')
  if args.control_canonical_k < 0:
    raise ValueError('--control-canonical-k must be >= 0')

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('CUDA requested but not available.')
  device = torch.device(args.device)

  gen_seed = args.generator_seed if args.generator_seed is not None else args.seed
  rng_gen: Optional[torch.Generator] = None
  if args.explicit_generator:
    rng_gen = _make_torch_generator(device, gen_seed)

  cfg = _compose_config(
    checkpoint=args.checkpoint,
    model=args.model,
    num_steps=args.num_steps,
    batch_size=args.batch_size,
    predictor=args.predictor,
    noise_removal=not args.no_noise_removal,
  )
  OmegaConf.resolve(cfg)
  _configure_sampling_temperature(args.temperature)
  model = _load_model(cfg, device)

  npy_root = Path(args.npy_root)
  if not npy_root.is_absolute():
    npy_root = MDLM_ROOT / npy_root

  if args.task == 'unconditional':
    total, valid = _run_unconditional(
      model, args.num_batches, device, args.num_steps, generator=rng_gen)
    metrics: Dict[str, Any] = {
      'task': 'unconditional',
      'n_samples': total,
      'valid_sudoku_rate': valid / max(total, 1),
      'num_batches': args.num_batches,
      'num_steps': args.num_steps,
      'predictor': args.predictor,
      'checkpoint': args.checkpoint,
      'model': args.model,
      'temperature': args.temperature,
      'explicit_generator': args.explicit_generator,
      'generator_seed': gen_seed,
    }
  else:
    ds = SudokuNpyDataset(npy_root, split=args.split, use_anchors=True)
    n = min(len(ds), args.max_puzzles)
    solutions = _safe_numpy_slice(ds.solutions, n, np.int64)
    anchors = _safe_numpy_slice(ds.anchors, n, np.bool_)
    target_masks = _load_optional_target_masks(npy_root, args.split, n, args.target_mask_npy)
    target_mask_source = 'loaded'
    if target_masks is None:
      target_masks = ~anchors
      target_mask_source = 'non_anchor_default'

    transform_indices = _build_transform_indices()
    num_t = len(TRANSFORMS)
    k = int(args.samples_per_transform)

    # Success arrays use the selected success metric. Extra arrays help debugging.
    orbit_success = np.zeros((n, num_t, k), dtype=np.bool_)
    orbit_exact = np.zeros((n, num_t, k), dtype=np.bool_)
    orbit_target_exact = np.zeros((n, num_t, k), dtype=np.bool_)
    orbit_valid = np.zeros((n, num_t, k), dtype=np.bool_)
    orbit_clue_ok = np.zeros((n, num_t, k), dtype=np.bool_)

    canonical_preds = None
    if args.compute_mod or args.save_npz:
      canonical_preds = np.empty((n, num_t, k, 81), dtype=np.int16)

    transform_metrics: Dict[str, Any] = {}
    for t_i, name in enumerate(TRANSFORMS):
      idx = transform_indices[name]
      t_sol = solutions[:, idx]
      t_anchor = anchors[:, idx]
      t_target = target_masks[:, idx]

      if args.reset_generator_each_orbit_transform and rng_gen is not None:
        rng_gen.manual_seed(int(gen_seed))

      per_sample_metrics: List[Dict[str, float]] = []
      for sample_i in range(k):
        sample_metrics, pred_t = _sample_infill_arrays_with_scores(
          model=model,
          solutions=t_sol,
          anchors=t_anchor,
          target_masks=t_target,
          device=device,
          num_steps=args.num_steps,
          batch_size=args.batch_size,
          success_metric=args.success_metric,
          generator=rng_gen,
        )
        flags = sample_metrics['flags']
        orbit_success[:, t_i, sample_i] = flags['success']
        orbit_exact[:, t_i, sample_i] = flags['exact']
        orbit_target_exact[:, t_i, sample_i] = flags['target_exact']
        orbit_valid[:, t_i, sample_i] = flags['valid']
        orbit_clue_ok[:, t_i, sample_i] = flags['clue_ok']
        if canonical_preds is not None:
          canonical_preds[:, t_i, sample_i, :] = _canonicalize_from_transform(pred_t, idx)
        per_sample_metrics.append({
          'exact_match_rate': float(sample_metrics['exact_match_rate']),
          'target_exact_rate': float(sample_metrics['target_exact_rate']),
          'clue_consistency_rate': float(sample_metrics['clue_consistency_rate']),
          'valid_sudoku_rate': float(sample_metrics['valid_sudoku_rate']),
          'valid_and_clue_rate': float(sample_metrics['valid_and_clue_rate']),
          'selected_success_rate': float(sample_metrics['success_rate']),
          'target_cell_accuracy': float(sample_metrics['target_cell_accuracy']),
        })

      transform_metrics[name] = {
        'n_puzzles': int(n),
        'samples_per_transform': k,
        'single_sample_mean': {
          key: float(np.mean([m[key] for m in per_sample_metrics]))
          for key in per_sample_metrics[0].keys()
        },
        'pass_at_k_selected_success_rate': float(orbit_success[:, t_i, :].any(axis=1).mean()),
        'all_k_selected_success_rate': float(orbit_success[:, t_i, :].all(axis=1).mean()),
      }

    stochastic_summary = _summarize_stochastic_orbit(
      orbit_success,
      bootstrap_samples=int(args.bootstrap_samples),
      seed=int(args.seed) + 17,
    )

    # Backward-compatible one-sample view, using sample 0 only.
    one_sample_success = orbit_success[:, :, 0]
    one_sample_any = one_sample_success.any(axis=1)
    one_sample_all = one_sample_success.all(axis=1)

    metrics = {
      'task': 'infill',
      'split': args.split,
      'npy_root': str(npy_root),
      'target_mask_source': target_mask_source,
      'n_puzzles': int(n),
      'num_transforms': num_t,
      'samples_per_transform': k,
      'success_metric': args.success_metric,
      'transform_metrics': transform_metrics,
      'stochastic_orbit': stochastic_summary,
      'one_sample_orbit_control': {
        'selected_success_any_of_8_rate': float(one_sample_any.mean()),
        'selected_success_all_8_rate': float(one_sample_all.mean()),
        'selected_success_lsr_rate': float(one_sample_any.mean() - one_sample_all.mean()),
        'note': 'Uses only sample 0 per transform; for stochastic samplers this is a control, not the main LE metric.',
      },
      'num_steps': args.num_steps,
      'predictor': args.predictor,
      'checkpoint': args.checkpoint,
      'model': args.model,
      'temperature': args.temperature,
      'explicit_generator': args.explicit_generator,
      'generator_seed': gen_seed,
      'reset_generator_each_orbit_transform': args.reset_generator_each_orbit_transform,
    }

    if args.compute_mod:
      assert canonical_preds is not None
      vocab_size = max(int(getattr(model, 'mask_index', 9)) + 1, 9)
      mod_mean, mod_per_puzzle = _compute_mod(canonical_preds, target_masks, vocab_size=vocab_size)
      lo, hi = _bootstrap_ci(mod_per_puzzle, int(args.bootstrap_samples), int(args.seed) + 23)
      metrics['marginal_orbit_divergence'] = {
        'mean': float(mod_mean),
        'ci95_low': lo,
        'ci95_high': hi,
        'vocab_size': int(vocab_size),
        'computed_over': 'canonical target_mask positions',
      }

    if args.control_canonical_k > 0:
      metrics['canonical_control'] = _run_canonical_control_k(
        model=model,
        solutions=solutions,
        anchors=anchors,
        target_masks=target_masks,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        k=int(args.control_canonical_k),
        success_metric=args.success_metric,
        generator=rng_gen,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed) + 31,
      )

    if args.save_npz and args.output_json:
      out_path = Path(args.output_json)
      npz_path = out_path.with_suffix('.npz')
      npz_path.parent.mkdir(parents=True, exist_ok=True)
      np.savez_compressed(
        npz_path,
        orbit_success=orbit_success,
        orbit_exact=orbit_exact,
        orbit_target_exact=orbit_target_exact,
        orbit_valid=orbit_valid,
        orbit_clue_ok=orbit_clue_ok,
        canonical_preds=canonical_preds if canonical_preds is not None else np.asarray([], dtype=np.int16),
        target_masks=target_masks,
        anchors=anchors,
        solutions=solutions,
        transforms=np.asarray(TRANSFORMS),
      )
      metrics['npz_path'] = str(npz_path)

  safe_metrics = _json_sanitize(metrics)
  print(json.dumps(safe_metrics, indent=2))
  if args.output_json:
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(safe_metrics, indent=2), encoding='utf-8')
    print(f'Wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
  main()
