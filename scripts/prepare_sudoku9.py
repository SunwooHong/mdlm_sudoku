#!/usr/bin/env python3
"""Build Sudoku .npy memmaps for MDLM training.

Either:
  (A) Single Kaggle-style CSV + hash split by solution (default), or
  (B) --from-preprocessed-dir: train.csv / validation.csv / test.csv from
      `preprocess_sudoku_9m.py` (same 90/5/5 split, no re-hashing).

Output:
  <out_dir>/train_solution.npy, train_anchor.npy, ...
  meta.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np


PUZZLE_COL_CANDIDATES = [
    'quizzes', 'quiz', 'puzzle', 'puzzles', 'question', 'questions']
SOLUTION_COL_CANDIDATES = [
    'solutions', 'solution', 'answer', 'answers', 'text']


def normalize_grid_string(x: str) -> str:
  x = str(x).strip().replace(' ', '').replace('\n', '').replace('\t', '')
  return x


def detect_columns(columns: List[str]) -> Tuple[str, str]:
  lower_to_original = {c.lower(): c for c in columns}
  puzzle_col = None
  solution_col = None
  for c in PUZZLE_COL_CANDIDATES:
    if c in lower_to_original:
      puzzle_col = lower_to_original[c]
      break
  for c in SOLUTION_COL_CANDIDATES:
    if c in lower_to_original:
      solution_col = lower_to_original[c]
      break
  if puzzle_col is None or solution_col is None:
    raise ValueError(
        f'Could not detect puzzle/solution columns. Found={columns!r}')
  return puzzle_col, solution_col


def encode_solution(solution: str) -> np.ndarray:
  solution = normalize_grid_string(solution)
  if len(solution) != 81:
    raise ValueError(f'Solution length must be 81, got {len(solution)}')
  if any(ch not in '123456789' for ch in solution):
    raise ValueError('Solution must contain only digits 1-9')
  return np.fromiter(
      (ord(ch) - ord('1') for ch in solution),
      dtype=np.uint8,
      count=81)


def encode_anchor_mask(puzzle: str) -> np.ndarray:
  puzzle = normalize_grid_string(puzzle)
  if len(puzzle) != 81:
    raise ValueError(f'Puzzle length must be 81, got {len(puzzle)}')
  return np.fromiter(
      (0 if ch in '0.' else 1 for ch in puzzle),
      dtype=np.uint8,
      count=81)


def stable_split(key: str, valid_pct: float, test_pct: float) -> str:
  h = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
  r = int.from_bytes(h, byteorder='big', signed=False) / float(2**64)
  if r < test_pct:
    return 'test'
  if r < test_pct + valid_pct:
    return 'valid'
  return 'train'


def iter_pairs_csv(
    csv_path: Path,
    chunksize: int,
    limit: int | None = None) -> Iterator[Tuple[str, str]]:
  import pandas as pd

  seen = 0
  for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunksize):
    puzzle_col, solution_col = detect_columns(list(chunk.columns))
    for puzzle, solution in zip(chunk[puzzle_col], chunk[solution_col]):
      yield normalize_grid_string(str(puzzle)), normalize_grid_string(
          str(solution))
      seen += 1
      if limit is not None and seen >= limit:
        return


def count_rows_csv(path: Path) -> int:
  with path.open('r', newline='') as f:
    return max(0, sum(1 for _ in f) - 1)


def write_preprocessed_dir(src_dir: Path, out_dir: Path) -> Dict[str, int]:
  mapping = [
      ('train', 'train.csv'),
      ('valid', 'validation.csv'),
      ('test', 'test.csv'),
  ]
  counts: Dict[str, int] = {}
  for split, fname in mapping:
    p = src_dir / fname
    if not p.exists():
      raise FileNotFoundError(f'Missing {p}')
    counts[split] = count_rows_csv(p)

  out_dir.mkdir(parents=True, exist_ok=True)
  mmaps: Dict[str, Tuple[Any, Any]] = {}
  for split in counts:
    mmaps[split] = (
        np.lib.format.open_memmap(
            out_dir / f'{split}_solution.npy',
            mode='w+',
            dtype=np.uint8,
            shape=(counts[split], 81)),
        np.lib.format.open_memmap(
            out_dir / f'{split}_anchor.npy',
            mode='w+',
            dtype=np.uint8,
            shape=(counts[split], 81)),
    )

  bad = 0
  for split, fname in mapping:
    path = src_dir / fname
    sol_mm, anc_mm = mmaps[split]
    row = 0
    with path.open('r', newline='') as f:
      reader = csv.DictReader(f)
      puzzle_col, solution_col = detect_columns(reader.fieldnames or [])
      for rec in reader:
        try:
          puzzle = normalize_grid_string(str(rec[puzzle_col]))
          solution = normalize_grid_string(str(rec[solution_col]))
          sol_mm[row] = encode_solution(solution)
          anc_mm[row] = encode_anchor_mask(puzzle)
          row += 1
        except Exception:
          bad += 1
    sol_mm.flush()
    anc_mm.flush()

  meta = {'counts': counts, 'bad_rows': bad, 'source': 'preprocessed_csv_dir'}
  with (out_dir / 'meta.json').open('w') as fp:
    json.dump(meta, fp, indent=2)
  print(json.dumps(meta, indent=2))
  return counts


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', type=Path, help='Kaggle-style sudoku CSV.')
  parser.add_argument('--out_dir', type=Path, required=True)
  parser.add_argument('--train_pct', type=float, default=0.90)
  parser.add_argument('--valid_pct', type=float, default=0.05)
  parser.add_argument('--test_pct', type=float, default=0.05)
  parser.add_argument(
      '--split_by',
      choices=['solution', 'puzzle'],
      default='solution')
  parser.add_argument('--chunksize', type=int, default=200_000)
  parser.add_argument('--limit', type=int, default=None)
  parser.add_argument(
      '--from-preprocessed-dir',
      type=Path,
      default=None,
      help='Use train.csv/validation.csv/test.csv from preprocess_sudoku_9m.py')
  args = parser.parse_args()

  if args.from_preprocessed_dir is not None:
    write_preprocessed_dir(args.from_preprocessed_dir, args.out_dir)
    print('Done (from preprocessed dir).')
    return

  if args.csv is None:
    raise SystemExit('Provide --csv or --from-preprocessed-dir')

  s = args.train_pct + args.valid_pct + args.test_pct
  if abs(s - 1.0) > 1e-6:
    raise ValueError(f'Split proportions must sum to 1.0, got {s}')

  args.out_dir.mkdir(parents=True, exist_ok=True)

  counts: Dict[str, int] = {'train': 0, 'valid': 0, 'test': 0}
  bad = 0
  print('Pass 1/2: counting...')
  for puzzle, solution in iter_pairs_csv(args.csv, args.chunksize, args.limit):
    try:
      _ = encode_solution(solution)
      _ = encode_anchor_mask(puzzle)
    except Exception:
      bad += 1
      continue
    key = solution if args.split_by == 'solution' else puzzle
    counts[stable_split(key, args.valid_pct, args.test_pct)] += 1

  print(f'counts={counts}, bad_rows={bad}')

  mmaps: Dict[str, Tuple[Any, Any]] = {}
  for split, n in counts.items():
    mmaps[split] = (
        np.lib.format.open_memmap(
            args.out_dir / f'{split}_solution.npy',
            mode='w+',
            dtype=np.uint8,
            shape=(n, 81)),
        np.lib.format.open_memmap(
            args.out_dir / f'{split}_anchor.npy',
            mode='w+',
            dtype=np.uint8,
            shape=(n, 81)),
    )

  offsets = {'train': 0, 'valid': 0, 'test': 0}
  bad2 = 0
  print('Pass 2/2: writing...')
  for puzzle, solution in iter_pairs_csv(args.csv, args.chunksize, args.limit):
    try:
      sol_ids = encode_solution(solution)
      anchor = encode_anchor_mask(puzzle)
    except Exception:
      bad2 += 1
      continue
    key = solution if args.split_by == 'solution' else puzzle
    split = stable_split(key, args.valid_pct, args.test_pct)
    j = offsets[split]
    mmaps[split][0][j] = sol_ids
    mmaps[split][1][j] = anchor
    offsets[split] += 1

  for split in counts:
    mmaps[split][0].flush()
    mmaps[split][1].flush()

  meta = {
      'counts': counts,
      'bad_rows': bad + bad2,
      'train_pct': args.train_pct,
      'valid_pct': args.valid_pct,
      'test_pct': args.test_pct,
      'split_by': args.split_by,
      'vocab': {
          'digit_ids': '1..9 -> 0..8',
          'mask_id': 9,
          'vocab_size': 10,
      },
  }
  with (args.out_dir / 'meta.json').open('w') as fp:
    json.dump(meta, fp, indent=2)
  print('Done.')
  print(json.dumps(meta, indent=2))


if __name__ == '__main__':
  main()
