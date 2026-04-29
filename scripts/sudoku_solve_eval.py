#!/usr/bin/env python3
"""Evaluate an MDLM Sudoku checkpoint by sampling (infill or unconditional).

Uses the same DDPM cache sampler as training, optionally starting from
puzzles (clues fixed, blanks masked) like ``sudoku9-anchors`` memmaps.

Example (from repo root, GPU node):

  python scripts/sudoku_solve_eval.py \\
    --checkpoint outputs/.../checkpoints/best.ckpt \\
    --model sudoku_5m \\
    --npy-root dataset/sudoku_9m_npy \\
    --split test \\
    --task infill \\
    --max-puzzles 512 \\
    --batch-size 64 \\
    --num-steps 128

Greedy decoding (same intent as temperature 0 in softmax sampling):

  python scripts/sudoku_solve_eval.py ... --temperature 0

Explicit sampling RNG (DDPM categorical steps only; recommended):

  python scripts/sudoku_solve_eval.py ... --generator-seed 42

Same RNG stream start for each orbit transform (optional):

  python scripts/sudoku_solve_eval.py ... \\
    --reset-generator-each-orbit-transform
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Repo imports (run with cwd = mdlm root, or PYTHONPATH=.)
MDLM_ROOT = Path(__file__).resolve().parent.parent
if str(MDLM_ROOT) not in sys.path:
  sys.path.insert(0, str(MDLM_ROOT))

import diffusion as diffusion_mod
import dataloader as dataloader_mod
import temperature_sampling
from sudoku_dataloader import SudokuNpyDataset, SudokuTokenizer

# Snapshot before any patch so temperature=1 restores exact repo behavior.
_ORIGINAL_SAMPLE_CATEGORICAL = diffusion_mod._sample_categorical


def _make_torch_generator(device: torch.device, seed: int) -> torch.Generator:
  """Explicit RNG for sampling (isolates eval noise from global torch.manual_seed)."""
  if device.type == 'cuda':
    gen = torch.Generator(device=device)
  else:
    gen = torch.Generator()
  gen.manual_seed(int(seed))
  return gen


def _configure_sampling_temperature(
    temperature: float,
    *,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> None:
  """Patch categorical sampling (temperature + optional top-k / top-p)."""
  temperature_sampling.configure_eval_sampling(
    diffusion_mod,
    _ORIGINAL_SAMPLE_CATEGORICAL,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
  )


def _register_omegaconf_resolvers() -> None:
  # Match resolvers used by main.py/config.yaml.
  OmegaConf.register_new_resolver('cwd', os.getcwd, replace=True)
  OmegaConf.register_new_resolver(
    'mdlm_root',
    lambda: str(MDLM_ROOT),
    replace=True)
  OmegaConf.register_new_resolver(
    'device_count',
    torch.cuda.device_count,
    replace=True)
  OmegaConf.register_new_resolver('eval', eval, replace=True)
  OmegaConf.register_new_resolver(
    'div_up',
    lambda x, y: (x + y - 1) // y,
    replace=True)


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
      if frozenset(g[br : br + 3, bc : bc + 3].ravel()) != digits:
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
  with initialize_config_dir(
      version_base=None, config_dir=str(MDLM_ROOT / 'configs')):
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
    # Fallback: direct state-dict files.
    state_dict = payload

  # OSCAR checkpoints include a frozen reference copy under ref_backbone.*.
  # Strip it for evaluation with the base Diffusion class.
  if isinstance(state_dict, dict):
    state_dict = {
      k: v for k, v in state_dict.items()
      if not k.startswith('ref_backbone.')
    }

  model = diffusion_mod.Diffusion(cfg, tokenizer=tok)
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if unexpected:
    raise RuntimeError(f'Unexpected state_dict keys: {unexpected[:8]}')

  # Orbit-KTO .pt may not include EMA payload; disable EMA unless explicitly present.
  ema_state = None
  if isinstance(payload, dict):
    ema_state = payload.get('ema_state_dict')
  if ema_state is not None and getattr(model, 'ema', None) is not None:
    model.ema.load_state_dict(ema_state)
  else:
    model.ema = None

  if missing:
    # Missing keys are expected in some cross-mode checkpoints (e.g., EMA blocks).
    print(f'[eval] load_state_dict missing keys (showing up to 8): {missing[:8]}')
  model = model.to(device)
  model.eval()
  return model


@torch.no_grad()
def _run_infill(
    model: diffusion_mod.Diffusion,
    loader: DataLoader,
    device: torch.device,
    num_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int, int, int, int]:
  """Returns (n, exact, valid, clue_cells, correct_blank_cells)."""
  n_total = 0
  n_exact = 0
  n_valid = 0
  clue_cells = 0
  correct_blank_cells = 0

  mask_id = model.mask_index
  for batch in loader:
    sol = batch['input_ids'].to(device)
    anchor = batch['anchor_mask'].to(device)
    x_init = torch.where(anchor, sol, torch.full_like(sol, mask_id))
    pred = model.restore_model_and_sample(
      num_steps, x_init=x_init, generator=generator)
    pred_cpu = pred.cpu().numpy()
    sol_cpu = sol.cpu().numpy()
    anchor_cpu = anchor.cpu().numpy()

    for i in range(pred_cpu.shape[0]):
      n_total += 1
      p = pred_cpu[i]
      s = sol_cpu[i]
      a = anchor_cpu[i]
      if np.array_equal(p, s):
        n_exact += 1
      if _is_valid_completed_sudoku(p):
        n_valid += 1
      clue_cells += int(a.sum())
      blanks = ~a
      correct_blank_cells += int((p[blanks] == s[blanks]).sum())

  return n_total, n_exact, n_valid, clue_cells, correct_blank_cells


def _run_infill_arrays(
    model: diffusion_mod.Diffusion,
    solutions: np.ndarray,
    anchors: np.ndarray,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int, int, int, int]:
  """Same metrics as _run_infill, but from numpy arrays [N,81]."""
  n_total = 0
  n_exact = 0
  n_valid = 0
  clue_cells = 0
  correct_blank_cells = 0

  mask_id = model.mask_index
  n = solutions.shape[0]
  for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    sol = torch.from_numpy(solutions[start:end]).to(device=device, dtype=torch.long)
    anchor = torch.from_numpy(anchors[start:end]).to(device=device, dtype=torch.bool)
    x_init = torch.where(anchor, sol, torch.full_like(sol, mask_id))
    pred = model.restore_model_and_sample(
      num_steps, x_init=x_init, generator=generator)
    pred_cpu = pred.cpu().numpy()
    sol_cpu = sol.cpu().numpy()
    anchor_cpu = anchor.cpu().numpy()

    for i in range(pred_cpu.shape[0]):
      n_total += 1
      p = pred_cpu[i]
      s = sol_cpu[i]
      a = anchor_cpu[i]
      if np.array_equal(p, s):
        n_exact += 1
      if _is_valid_completed_sudoku(p):
        n_valid += 1
      clue_cells += int(a.sum())
      blanks = ~a
      correct_blank_cells += int((p[blanks] == s[blanks]).sum())

  return n_total, n_exact, n_valid, clue_cells, correct_blank_cells


def _run_infill_arrays_with_flags(
    model: diffusion_mod.Diffusion,
    solutions: np.ndarray,
    anchors: np.ndarray,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int, int, int, int, np.ndarray, np.ndarray]:
  """_run_infill_arrays + per-puzzle exact/valid success flags."""
  n_total = 0
  n_exact = 0
  n_valid = 0
  clue_cells = 0
  correct_blank_cells = 0
  exact_flags: List[bool] = []
  valid_flags: List[bool] = []

  mask_id = model.mask_index
  n = solutions.shape[0]
  for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    sol = torch.from_numpy(solutions[start:end]).to(device=device, dtype=torch.long)
    anchor = torch.from_numpy(anchors[start:end]).to(device=device, dtype=torch.bool)
    x_init = torch.where(anchor, sol, torch.full_like(sol, mask_id))
    pred = model.restore_model_and_sample(
      num_steps, x_init=x_init, generator=generator)
    pred_cpu = pred.cpu().numpy()
    sol_cpu = sol.cpu().numpy()
    anchor_cpu = anchor.cpu().numpy()

    for i in range(pred_cpu.shape[0]):
      n_total += 1
      p = pred_cpu[i]
      s = sol_cpu[i]
      a = anchor_cpu[i]
      is_exact = bool(np.array_equal(p, s))
      is_valid = bool(_is_valid_completed_sudoku(p))
      if is_exact:
        n_exact += 1
      if is_valid:
        n_valid += 1
      exact_flags.append(is_exact)
      valid_flags.append(is_valid)
      clue_cells += int(a.sum())
      blanks = ~a
      correct_blank_cells += int((p[blanks] == s[blanks]).sum())

  return (
    n_total, n_exact, n_valid, clue_cells, correct_blank_cells,
    np.asarray(exact_flags, dtype=np.bool_),
    np.asarray(valid_flags, dtype=np.bool_))


@torch.no_grad()
def _run_canonical_control_k(
    model: diffusion_mod.Diffusion,
    solutions: np.ndarray,
    anchors: np.ndarray,
    device: torch.device,
    num_steps: int,
    batch_size: int,
    k: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:
  """Canonical layout control: run K independent samples per puzzle."""
  if k <= 0:
    return {}

  exact_success = np.zeros((solutions.shape[0], k), dtype=np.bool_)
  valid_success = np.zeros((solutions.shape[0], k), dtype=np.bool_)
  exact_single_rates: List[float] = []
  valid_single_rates: List[float] = []

  for i in range(k):
    (n_total, n_exact, n_valid, _clue_cells, _correct_blank,
     exact_flags, valid_flags) = _run_infill_arrays_with_flags(
      model=model,
      solutions=solutions,
      anchors=anchors,
      device=device,
      num_steps=num_steps,
      batch_size=batch_size,
      generator=generator)
    exact_success[:, i] = exact_flags
    valid_success[:, i] = valid_flags
    exact_single_rates.append(float(n_exact / max(n_total, 1)))
    valid_single_rates.append(float(n_valid / max(n_total, 1)))

  return {
    'k': int(k),
    'exact_single_sample_mean_rate': float(np.mean(exact_single_rates)),
    'valid_single_sample_mean_rate': float(np.mean(valid_single_rates)),
    'exact_any_of_k_rate': float(exact_success.any(axis=1).mean()),
    'exact_all_k_rate': float(exact_success.all(axis=1).mean()),
    'valid_any_of_k_rate': float(valid_success.any(axis=1).mean()),
    'valid_all_k_rate': float(valid_success.all(axis=1).mean()),
  }


def _build_transform_indices() -> Dict[str, np.ndarray]:
  base = np.arange(81, dtype=np.int64).reshape(9, 9)
  out: Dict[str, np.ndarray] = {}
  for name in TRANSFORMS:
    out[name] = _apply_transform(base, name).reshape(-1)
  return out


@torch.no_grad()
def _run_unconditional(
    model: diffusion_mod.Diffusion,
    num_batches: int,
    device: torch.device,
    num_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, int]:
  """Returns (n_samples, n_valid_sudoku)."""
  bsz = int(model.config.loader.eval_batch_size)
  n_valid = 0
  n_total = 0
  for _ in range(num_batches):
    pred = model.restore_model_and_sample(
      num_steps, x_init=None, generator=generator)
    pred_cpu = pred.cpu().numpy()
    for i in range(pred_cpu.shape[0]):
      n_total += 1
      if _is_valid_completed_sudoku(pred_cpu[i]):
        n_valid += 1
  return n_total, n_valid


def main() -> None:
  _register_omegaconf_resolvers()
  p = argparse.ArgumentParser(description='MDLM 9x9 Sudoku sampling eval')
  p.add_argument('--checkpoint', type=str, required=True)
  p.add_argument('--model', type=str, default='sudoku_5m')
  p.add_argument(
    '--npy-root',
    type=str,
    default=str(MDLM_ROOT / 'dataset' / 'sudoku_9m_npy'),
    help='Directory with *_solution.npy / *_anchor.npy',
  )
  p.add_argument(
    '--split',
    type=str,
    choices=('train', 'valid', 'validation', 'test'),
    default='test',
    help='Which memmap split to read',
  )
  p.add_argument(
    '--task',
    type=str,
    choices=('infill', 'unconditional'),
    default='infill',
    help='infill: clues fixed (needs anchor npy). unconditional: all-mask prior.',
  )
  p.add_argument('--max-puzzles', type=int, default=512)
  p.add_argument('--batch-size', type=int, default=64)
  p.add_argument('--num-steps', type=int, default=128)
  p.add_argument('--num-batches', type=int, default=8, help='For unconditional only')
  p.add_argument('--predictor', type=str, default='ddpm_cache')
  p.add_argument(
    '--temperature',
    type=float,
    default=1.0,
    help=(
      'Categorical sampling temperature inside diffusion steps. '
      '1.0 = repo default sampler; 0 = greedy (argmax); '
      '<1 sharper, >1 flatter (after softmax(log(p)/T)).'),
  )
  p.add_argument(
    '--sample-top-k',
    type=int,
    default=0,
    help='If >0, apply top-k on probs after softmax(log(p)/T), before Gumbel sampling.',
  )
  p.add_argument(
    '--sample-top-p',
    type=float,
    default=1.0,
    help='If <1.0, apply nucleus (top-p) after top-k. 1.0 disables.',
  )
  p.add_argument('--no-noise-removal', action='store_true')
  p.add_argument('--device', type=str, default='cuda')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument(
    '--explicit-generator',
    dest='explicit_generator',
    action='store_true',
    default=True,
    help='Use torch.Generator(device).manual_seed(...) for DDPM categorical noise '
         '(recommended; isolates sampling RNG from global torch.manual_seed).',
  )
  p.add_argument(
    '--no-explicit-generator',
    dest='explicit_generator',
    action='store_false',
    help='Do not pass a Generator (legacy: rely on global RNG only).',
  )
  p.add_argument(
    '--generator-seed',
    type=int,
    default=None,
    help='Seed for explicit Generator; defaults to --seed when omitted.',
  )
  p.add_argument(
    '--reset-generator-each-orbit-transform',
    action='store_true',
    default=False,
    help=(
      'Before each D4 orbit transform, call generator.manual_seed(generator-seed). '
      'Useful so each layout sees the same RNG stream start (still differs inside '
      'sampling because masks/clues differ).'),
  )
  p.add_argument(
    '--control-canonical-k',
    type=int,
    default=0,
    help='If >0, run canonical layout K independent samples as control.')
  p.add_argument('--output-json', type=str, default='')
  args = p.parse_args()

  if args.sample_top_k < 0:
    raise ValueError('--sample-top-k must be >= 0')

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('CUDA requested but not available.')
  device = torch.device(args.device)

  gen_seed = (
    args.generator_seed if args.generator_seed is not None else args.seed)
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
  _configure_sampling_temperature(
    args.temperature,
    top_k=args.sample_top_k if args.sample_top_k > 0 else None,
    top_p=args.sample_top_p if args.sample_top_p < 1.0 - 1e-12 else None,
  )

  model = _load_model(cfg, device)

  npy_root = Path(args.npy_root)
  if not npy_root.is_absolute():
    npy_root = MDLM_ROOT / npy_root

  if args.task == 'infill':
    ds = SudokuNpyDataset(npy_root, split=args.split, use_anchors=True)
    n = min(len(ds), args.max_puzzles)
    solutions = np.asarray(ds.solutions[:n], dtype=np.int64)
    anchors = np.asarray(ds.anchors[:n], dtype=np.bool_)
    transform_indices = _build_transform_indices()
    transform_metrics: Dict[str, Any] = {}
    totals = dict(total=0, exact=0, valid=0, clue_cells=0, correct_blank=0)
    orbit_exact_success = np.zeros((n, len(TRANSFORMS)), dtype=np.bool_)
    orbit_valid_success = np.zeros((n, len(TRANSFORMS)), dtype=np.bool_)
    for t_i, name in enumerate(TRANSFORMS):
      if (
          args.reset_generator_each_orbit_transform
          and rng_gen is not None):
        rng_gen.manual_seed(int(gen_seed))
      idx = transform_indices[name]
      t_sol = solutions[:, idx]
      t_anchor = anchors[:, idx]
      (total, exact, valid, clue_cells, correct_blank,
       exact_flags, valid_flags) = _run_infill_arrays_with_flags(
        model=model,
        solutions=t_sol,
        anchors=t_anchor,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        generator=rng_gen)
      orbit_exact_success[:, t_i] = exact_flags
      orbit_valid_success[:, t_i] = valid_flags
      blank_cells = 81 * total - clue_cells
      transform_metrics[name] = {
        'n_puzzles': total,
        'exact_match_rate': exact / max(total, 1),
        'valid_sudoku_rate': valid / max(total, 1),
        'blank_cell_accuracy': correct_blank / max(blank_cells, 1),
      }
      totals['total'] += total
      totals['exact'] += exact
      totals['valid'] += valid
      totals['clue_cells'] += clue_cells
      totals['correct_blank'] += correct_blank

    blank_cells = 81 * totals['total'] - totals['clue_cells']
    orbit_exact_any = orbit_exact_success.any(axis=1)
    orbit_exact_all = orbit_exact_success.all(axis=1)
    orbit_valid_any = orbit_valid_success.any(axis=1)
    orbit_valid_all = orbit_valid_success.all(axis=1)
    metrics: Dict[str, Any] = {
      'task': 'infill',
      'split': args.split,
      'npy_root': str(npy_root),
      'n_puzzles': n,
      'num_transforms': len(TRANSFORMS),
      'transform_metrics': transform_metrics,
      'macro_average': {
        'exact_match_rate': float(np.mean(
          [transform_metrics[t]['exact_match_rate'] for t in TRANSFORMS])),
        'valid_sudoku_rate': float(np.mean(
          [transform_metrics[t]['valid_sudoku_rate'] for t in TRANSFORMS])),
        'blank_cell_accuracy': float(np.mean(
          [transform_metrics[t]['blank_cell_accuracy'] for t in TRANSFORMS])),
      },
      'micro_average': {
        'exact_match_rate': totals['exact'] / max(totals['total'], 1),
        'valid_sudoku_rate': totals['valid'] / max(totals['total'], 1),
        'blank_cell_accuracy': totals['correct_blank'] / max(blank_cells, 1),
      },
      'orbit_accuracy': {
        # "8개 중 하나라도 성공" / "8개 전부 성공" (per puzzle over D4 orbit)
        'exact_any_of_8_rate': float(orbit_exact_any.mean()),
        'exact_all_8_rate': float(orbit_exact_all.mean()),
        'valid_any_of_8_rate': float(orbit_valid_any.mean()),
        'valid_all_8_rate': float(orbit_valid_all.mean()),
      },
      'num_steps': args.num_steps,
      'predictor': args.predictor,
      'checkpoint': args.checkpoint,
      'model': args.model,
      'temperature': args.temperature,
      'sample_top_k': int(args.sample_top_k),
      'sample_top_p': float(args.sample_top_p),
      'explicit_generator': args.explicit_generator,
      'generator_seed': gen_seed,
      'reset_generator_each_orbit_transform': (
        args.reset_generator_each_orbit_transform),
    }
    if args.control_canonical_k > 0:
      metrics['canonical_control'] = _run_canonical_control_k(
        model=model,
        solutions=solutions,
        anchors=anchors,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        k=int(args.control_canonical_k),
        generator=rng_gen)
  else:
    total, valid = _run_unconditional(
      model, args.num_batches, device, args.num_steps, generator=rng_gen)
    metrics = {
      'task': 'unconditional',
      'n_samples': total,
      'valid_sudoku_rate': valid / max(total, 1),
      'num_batches': args.num_batches,
      'num_steps': args.num_steps,
      'predictor': args.predictor,
      'checkpoint': args.checkpoint,
      'model': args.model,
      'temperature': args.temperature,
      'sample_top_k': int(args.sample_top_k),
      'sample_top_p': float(args.sample_top_p),
      'explicit_generator': args.explicit_generator,
      'generator_seed': gen_seed,
      'reset_generator_each_orbit_transform': (
        args.reset_generator_each_orbit_transform),
    }

  print(json.dumps(metrics, indent=2))
  if args.output_json:
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
  main()
