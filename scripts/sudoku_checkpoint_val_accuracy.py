#!/usr/bin/env python3
"""Infill sampling accuracy over many Sudoku MDLM checkpoints (val .npy).

Same loading / sampling stack as ``scripts/sudoku_solve_eval.py`` (DDPM cache,
temperature). By default evaluates **canonical** puzzle layout
only (one pass over ``max_samples``). Pass ``--d8-orbit`` to match full eval:
D4×2 transforms with macro-averaged exact rate.

Example (repo root, GPU):

  python scripts/sudoku_checkpoint_val_accuracy.py \\
    --checkpoints-dir outputs/sudoku3m-only/sudoku-50m/2026.05.02/210723/checkpoints \\
    --model sudoku_50m \\
    --val-npy-root dataset/3m_only_val_npy \\
    --max-samples 500 \\
    --step-multiple 50000 \\
    --include-max-step \\
    --num-steps 128

  Last five checkpoints by global step (``N-M.ckpt`` with largest ``M``):

    python scripts/sudoku_checkpoint_val_accuracy.py \\
      --checkpoints-dir outputs/.../checkpoints \\
      --model sudoku_50m \\
      --last-n 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

MDLM_ROOT = Path(__file__).resolve().parent.parent
if str(MDLM_ROOT) not in sys.path:
  sys.path.insert(0, str(MDLM_ROOT))

from sudoku_dataloader import SudokuNpyDataset  # noqa: E402


def _import_solve_eval():
  path = Path(__file__).resolve().parent / 'sudoku_solve_eval.py'
  spec = importlib.util.spec_from_file_location('_sudoku_solve_eval', path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f'Cannot load {path}')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


_CKPT_RE = re.compile(r'^(\d+)-(\d+)\.ckpt$')


def _parse_epoch_step(name: str) -> Optional[Tuple[int, int]]:
  m = _CKPT_RE.match(name)
  if not m:
    return None
  return int(m.group(1)), int(m.group(2))


def _list_filtered_checkpoints(
    ckpt_dir: Path,
    *,
    step_multiple: int,
    include_max_step: bool,
) -> List[Path]:
  numbered: List[Tuple[int, int, Path]] = []
  for p in sorted(ckpt_dir.iterdir()):
    if not p.is_file() or p.suffix != '.ckpt':
      continue
    parsed = _parse_epoch_step(p.name)
    if parsed is None:
      continue
    ep, st = parsed
    numbered.append((st, ep, p))

  if not numbered:
    return []

  max_step = max(t[0] for t in numbered)
  out: List[Path] = []
  for st, _ep, p in sorted(numbered, key=lambda x: x[0]):
    pick = False
    if step_multiple <= 0:
      pick = True
    elif st > 0 and st % step_multiple == 0:
      pick = True
    elif include_max_step and st == max_step:
      pick = True
    if pick:
      out.append(p)

  seen = set()
  deduped: List[Path] = []

  def _step_key(path: Path) -> int:
    ps = _parse_epoch_step(path.name)
    return ps[1] if ps else 0

  for p in sorted(out, key=_step_key):
    key = str(p)
    if key not in seen:
      seen.add(key)
      deduped.append(p)
  return deduped


def _list_last_n_checkpoints(ckpt_dir: Path, n: int) -> List[Path]:
  """Highest global-step ``N-M.ckpt`` files, returned in increasing M order."""
  if n <= 0:
    return []
  numbered: List[Tuple[int, Path]] = []
  for p in sorted(ckpt_dir.iterdir()):
    if not p.is_file() or p.suffix != '.ckpt':
      continue
    parsed = _parse_epoch_step(p.name)
    if parsed is None:
      continue
    _ep, st = parsed
    numbered.append((st, p))
  if not numbered:
    return []
  numbered.sort(key=lambda x: x[0])
  tail = numbered[-n:]
  return [path for _st, path in tail]


def _eval_one_checkpoint(
    *,
    se: Any,
    ckpt_path: Path,
    model_name: str,
    batch_size: int,
    num_steps: int,
    predictor: str,
    noise_removal: bool,
    solutions: np.ndarray,
    anchors: np.ndarray,
    device: torch.device,
    temperature: float,
    explicit_generator: bool,
    generator_seed: int,
    d8_orbit: bool,
) -> Dict[str, Any]:
  se._configure_sampling_temperature(temperature)
  cfg = se._compose_config(
    checkpoint=str(ckpt_path),
    model=model_name,
    num_steps=num_steps,
    batch_size=batch_size,
    predictor=predictor,
    noise_removal=noise_removal,
  )
  OmegaConf.resolve(cfg)
  model = se._load_model(cfg, device)

  rng_gen: Optional[torch.Generator] = None
  if explicit_generator:
    rng_gen = se._make_torch_generator(device, int(generator_seed))

  if not d8_orbit:
    n_total, n_exact, n_valid, clue_cells, correct_blank = se._run_infill_arrays(
      model=model,
      solutions=solutions,
      anchors=anchors,
      device=device,
      num_steps=num_steps,
      batch_size=batch_size,
      generator=rng_gen,
    )
    blank_cells = 81 * n_total - clue_cells
    metrics = {
      'layout': 'canonical',
      'n_puzzles': n_total,
      'exact_match_rate': n_exact / max(n_total, 1),
      'valid_sudoku_rate': n_valid / max(n_total, 1),
      'blank_cell_accuracy': correct_blank / max(blank_cells, 1),
    }
  else:
    transform_indices = se._build_transform_indices()
    totals = dict(total=0, exact=0, valid=0, clue_cells=0, correct_blank=0)
    per_t: Dict[str, Dict[str, float]] = {}
    for name in se.TRANSFORMS:
      idx = transform_indices[name]
      t_sol = solutions[:, idx]
      t_anchor = anchors[:, idx]
      total, exact, valid, clue_cells, correct_blank = se._run_infill_arrays(
        model=model,
        solutions=t_sol,
        anchors=t_anchor,
        device=device,
        num_steps=num_steps,
        batch_size=batch_size,
        generator=rng_gen,
      )
      blank_cells = 81 * total - clue_cells
      per_t[name] = {
        'n_puzzles': float(total),
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
    metrics = {
      'layout': 'd8_orbit',
      'num_transforms': len(se.TRANSFORMS),
      'transform_metrics': per_t,
      'macro_average': {
        'exact_match_rate': float(np.mean(
          [per_t[t]['exact_match_rate'] for t in se.TRANSFORMS])),
        'valid_sudoku_rate': float(np.mean(
          [per_t[t]['valid_sudoku_rate'] for t in se.TRANSFORMS])),
        'blank_cell_accuracy': float(np.mean(
          [per_t[t]['blank_cell_accuracy'] for t in se.TRANSFORMS])),
      },
      'micro_average': {
        'exact_match_rate': totals['exact'] / max(totals['total'], 1),
        'valid_sudoku_rate': totals['valid'] / max(totals['total'], 1),
        'blank_cell_accuracy': totals['correct_blank'] / max(blank_cells, 1),
      },
    }

  parsed = _parse_epoch_step(ckpt_path.name)
  step = parsed[1] if parsed else -1
  epoch = parsed[0] if parsed else -1
  out: Dict[str, Any] = {
    'checkpoint': str(ckpt_path),
    'epoch': epoch,
    'global_step': step,
    'metrics': metrics,
  }
  return out


def main() -> None:
  p = argparse.ArgumentParser(
    description='Val infill accuracy (exact / valid Sudoku) over checkpoints')
  p.add_argument('--checkpoints-dir', type=str, required=True)
  p.add_argument('--model', type=str, default='sudoku_50m')
  p.add_argument(
    '--val-npy-root',
    type=str,
    default=str(MDLM_ROOT / 'dataset' / '3m_only_val_npy'),
  )
  p.add_argument(
    '--split',
    type=str,
    default='valid',
    choices=('train', 'valid', 'validation', 'test'),
  )
  p.add_argument('--max-samples', type=int, default=500)
  p.add_argument('--batch-size', type=int, default=64)
  p.add_argument('--step-multiple', type=int, default=50000)
  p.add_argument('--include-max-step', action='store_true')
  p.add_argument(
    '--last-n',
    type=int,
    default=0,
    help='If >0, take only the N highest-step numbered *.ckpt in the directory '
         '(increasing step order). Ignores --step-multiple and --include-max-step '
         'unless you pass explicit --checkpoints.',
  )
  p.add_argument('--device', type=str, default='cuda')
  p.add_argument('--num-steps', type=int, default=128)
  p.add_argument('--predictor', type=str, default='ddpm_cache')
  p.add_argument('--no-noise-removal', action='store_true')
  p.add_argument(
    '--temperature',
    type=float,
    default=1.0,
    help='1.0 = default sampler; 0 = greedy (argmax)',
  )
  p.add_argument('--seed', type=int, default=0)
  p.add_argument(
    '--generator-seed',
    type=int,
    default=None,
    help='Defaults to --seed',
  )
  p.add_argument(
    '--no-explicit-generator',
    dest='explicit_generator',
    action='store_false',
    default=True,
  )
  p.add_argument(
    '--d8-orbit',
    action='store_true',
    help='Evaluate all D4×2 transforms like sudoku_solve_eval.py (8× cost)',
  )
  p.add_argument('--checkpoints', type=str, default='')
  p.add_argument('--output-json', type=str, default='')
  args = p.parse_args()

  if str(args.device).startswith('cuda') and not torch.cuda.is_available():
    raise SystemExit('CUDA requested but not available.')

  ckpt_dir = Path(args.checkpoints_dir)
  if not ckpt_dir.is_absolute():
    ckpt_dir = MDLM_ROOT / ckpt_dir

  if args.checkpoints.strip():
    ckpts = [Path(x.strip()) for x in args.checkpoints.split(',') if x.strip()]
    for i, c in enumerate(ckpts):
      if not c.is_absolute():
        ckpts[i] = MDLM_ROOT / c
  elif int(args.last_n) > 0:
    ckpts = _list_last_n_checkpoints(ckpt_dir, int(args.last_n))
    if not ckpts:
      raise SystemExit(
        f'No numbered *.ckpt under {ckpt_dir} (--last-n={int(args.last_n)}).')
  else:
    mult = int(args.step_multiple)
    ckpts = _list_filtered_checkpoints(
      ckpt_dir,
      step_multiple=mult,
      include_max_step=bool(args.include_max_step),
    )
    if not ckpts:
      raise SystemExit(
        f'No numbered *.ckpt under {ckpt_dir} after filter '
        f'(step_multiple={mult}, include_max_step={args.include_max_step}).')

  val_root = Path(args.val_npy_root)
  if not val_root.is_absolute():
    val_root = MDLM_ROOT / val_root

  ds = SudokuNpyDataset(val_root, split=args.split, use_anchors=True)
  n = min(len(ds), int(args.max_samples))
  if n <= 0:
    raise SystemExit('No validation samples.')
  solutions = np.asarray(ds.solutions[:n], dtype=np.int64)
  anchors = np.asarray(ds.anchors[:n], dtype=np.bool_)

  torch.manual_seed(int(args.seed))
  np.random.seed(int(args.seed))
  gen_seed = (
    int(args.generator_seed) if args.generator_seed is not None else int(args.seed))

  device = torch.device(args.device)
  se = _import_solve_eval()
  se._register_omegaconf_resolvers()

  rows: List[Dict[str, Any]] = []
  for ck in ckpts:
    print(f'[val_accuracy] sampling {ck.name} ...', file=sys.stderr, flush=True)
    row = _eval_one_checkpoint(
      se=se,
      ckpt_path=ck,
      model_name=args.model,
      batch_size=int(args.batch_size),
      num_steps=int(args.num_steps),
      predictor=str(args.predictor),
      noise_removal=not args.no_noise_removal,
      solutions=solutions,
      anchors=anchors,
      device=device,
      temperature=float(args.temperature),
      explicit_generator=bool(args.explicit_generator),
      generator_seed=gen_seed,
      d8_orbit=bool(args.d8_orbit),
    )
    rows.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)

  summary: Dict[str, Any] = {
    'val_npy_root': str(val_root),
    'split': args.split,
    'max_samples': n,
    'num_steps': int(args.num_steps),
    'temperature': float(args.temperature),
    'd8_orbit': bool(args.d8_orbit),
    'step_multiple': int(args.step_multiple),
    'include_max_step': bool(args.include_max_step),
    'last_n': int(args.last_n),
    'model': args.model,
    'results': rows,
  }
  print(json.dumps(summary, indent=2))
  if args.output_json:
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
  main()


# python scripts/sudoku_checkpoint_val_accuracy.py \
#   --checkpoints-dir /home/hongsunw/projects/def-rahulgk/hongsunw/mdlm/outputs/sudoku3m-only/sudoku-50m/2026.05.02/164430/checkpoints \
#   --model sudoku_50m \
#   --val-npy-root dataset/3m_only_val_npy \
#   --max-samples 500 \
#   --step-multiple 100000 \
#   --include-max-step \
#   --num-steps 128


# python scripts/sudoku_checkpoint_val_accuracy.py \
#   --checkpoints-dir /home/hongsunw/projects/def-rahulgk/hongsunw/mdlm/outputs/sudoku3m-only/sudoku-50m/2026.05.02/164430/checkpoints \
#   --model sudoku_50m \
#   --val-npy-root dataset/3m_only_val_npy \
#   --max-samples 2000 \
#   --last-n 10 \
#   --num-steps 128