#!/usr/bin/env python3
"""Create a Sudoku test split from sudoku-3m.csv (difficulty-aware sampling).

Input CSV expected columns:
  - puzzle
  - solution
  - difficulty (float-like)

Two modes:

1) **Balanced integer buckets (default)** — same as before:
   - Filter difficulty to ``[min_difficulty, max_difficulty)`` (default ``[0, 4)``).
   - Stratify by ``int(difficulty)`` in ``{0,1,2,3}`` and sample ``--test-size`` as evenly
     as possible across buckets (respecting per-bucket capacity).

2) **Explicit strata (``--strata``)** — custom half-open intervals ``[lo, hi)`` each with a count:
   - Example: ``--strata 0:1:250 --strata 1:2:250`` → 500 rows, half easy ``[0,1)``, half ``[1,2)``.

Output directory contains:
  - test.csv           (text,puzzle,solution,difficulty,clues)
  - train.csv          (header only; for ``prepare_sudoku9.py --from-preprocessed-dir``)
  - validation.csv     (header only)
  - metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _normalize_grid(s: str) -> str:
  return str(s).strip().replace(" ", "").replace("\t", "").replace("\n", "")


def _valid_puzzle(s: str) -> bool:
  return len(s) == 81 and all(ch in "0123456789." for ch in s)


def _valid_solution(s: str) -> bool:
  return len(s) == 81 and all(ch in "123456789" for ch in s)


def _difficulty_bucket(difficulty: float) -> int:
  b = int(difficulty)
  if b < 0:
    return 0
  if b > 3:
    return 3
  return b


def _compute_targets(total: int, capacities: Dict[int, int]) -> Dict[int, int]:
  buckets = [0, 1, 2, 3]
  base = total // len(buckets)
  rem = total % len(buckets)
  target = {b: base + (1 if i < rem else 0) for i, b in enumerate(buckets)}

  for b in buckets:
    if target[b] > capacities[b]:
      target[b] = capacities[b]

  assigned = sum(target.values())
  need = total - assigned
  if need <= 0:
    return target

  for b in buckets:
    if need <= 0:
      break
    spare = capacities[b] - target[b]
    if spare <= 0:
      continue
    add = min(spare, need)
    target[b] += add
    need -= add
  return target


def _parse_stratum(spec: str) -> Tuple[float, float, int]:
  parts = spec.strip().split(":")
  if len(parts) != 3:
    raise ValueError(
        f'Invalid --strata {spec!r}; expected MIN:MAX:COUNT (half-open [MIN, MAX)).')
  lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
  if not (hi > lo):
    raise ValueError(f'Invalid stratum bounds in {spec!r}: need MAX > MIN.')
  if n < 0:
    raise ValueError(f'Invalid count in {spec!r}.')
  return lo, hi, n


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument(
      "--input-csv",
      type=Path,
      default=Path("dataset/sudoku-3m.csv"),
      help="Path to sudoku-3m.csv",
  )
  p.add_argument(
      "--output-dir",
      type=Path,
      default=Path("dataset/sudoku_3m_eval_d0_3"),
      help="Directory to write test.csv (+ empty train/validation.csv).",
  )
  p.add_argument(
      "--test-size",
      type=int,
      default=20000,
      help="Total rows (balanced-bucket mode only). Ignored when --strata is set.",
  )
  p.add_argument(
      "--strata",
      action="append",
      default=None,
      metavar="MIN:MAX:COUNT",
      help=(
          "Half-open difficulty interval [MIN, MAX) with COUNT samples. "
          "Repeatable, e.g. --strata 0:1:250 --strata 1:2:250. "
          "When set, overrides balanced 0/1/2/3 bucket mode."),
  )
  p.add_argument("--seed", type=int, default=42)
  p.add_argument(
      "--min-difficulty",
      type=float,
      default=0.0,
      help="Balanced-bucket mode only: global filter lower bound.",
  )
  p.add_argument(
      "--max-difficulty",
      type=float,
      default=4.0,
      help="Balanced-bucket mode only: global filter upper bound (exclusive).",
  )
  return p.parse_args()


def main() -> None:
  args = parse_args()
  rng = random.Random(args.seed)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  strata_specs: List[Tuple[float, float, int]] | None = None
  if args.strata:
    strata_specs = [_parse_stratum(s) for s in args.strata]

  skipped_invalid = 0
  skipped_range = 0
  skipped_no_stratum = 0
  total_rows = 0

  # Balanced mode: bucket 0..3 within global [min, max)
  idx_by_bucket: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
  # Strata mode: one index list per stratum
  idx_by_stratum: List[List[int]] = (
      [[] for _ in range(len(strata_specs))] if strata_specs else [])

  with args.input_csv.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    needed = {"puzzle", "solution", "difficulty"}
    if not needed.issubset(set(reader.fieldnames or [])):
      raise ValueError(
          f"Input CSV must include columns {sorted(needed)}; "
          f"got {reader.fieldnames}")

    for i, row in enumerate(reader):
      total_rows += 1
      puzzle = _normalize_grid(row["puzzle"])
      solution = _normalize_grid(row["solution"])
      try:
        difficulty = float(str(row["difficulty"]).strip())
      except Exception:
        skipped_invalid += 1
        continue

      if not _valid_puzzle(puzzle) or not _valid_solution(solution):
        skipped_invalid += 1
        continue

      if strata_specs:
        placed = False
        for si, (lo, hi, _) in enumerate(strata_specs):
          if lo <= difficulty < hi:
            idx_by_stratum[si].append(i)
            placed = True
            break
        if not placed:
          skipped_no_stratum += 1
        continue

      if not (args.min_difficulty <= difficulty < args.max_difficulty):
        skipped_range += 1
        continue
      idx_by_bucket[_difficulty_bucket(difficulty)].append(i)

  if strata_specs:
    capacities = [len(lst) for lst in idx_by_stratum]
    targets = []
    for si, (_, _, want) in enumerate(strata_specs):
      targets.append(min(want, capacities[si]))
    selected_row_ids: set[int] = set()
    for si, (_, _, _) in enumerate(strata_specs):
      k = targets[si]
      pool = idx_by_stratum[si]
      sampled = rng.sample(pool, k) if k > 0 else []
      selected_row_ids.update(sampled)
    targets_dict = {
        f"stratum_{si}": targets[si] for si in range(len(strata_specs))}
    capacities_dict = {
        f"stratum_{si}": capacities[si] for si in range(len(strata_specs))}
    stratum_defs = [
        {
            "index": si,
            "min": strata_specs[si][0],
            "max": strata_specs[si][1],
            "requested": strata_specs[si][2],
            "capacity": capacities[si],
            "selected": targets[si],
        }
        for si in range(len(strata_specs))
    ]
  else:
    capacities = {b: len(v) for b, v in idx_by_bucket.items()}
    available = sum(capacities.values())
    if available == 0:
      raise RuntimeError("No rows matched difficulty range and validity checks.")

    target_total = min(args.test_size, available)
    targets = _compute_targets(target_total, capacities)
    sampled_indices: Dict[int, set[int]] = {}
    for b in [0, 1, 2, 3]:
      k = targets[b]
      sampled = rng.sample(idx_by_bucket[b], k) if k > 0 else []
      sampled_indices[b] = set(sampled)
    selected_row_ids = set().union(*sampled_indices.values())
    targets_dict = {str(k): v for k, v in targets.items()}
    capacities_dict = {str(k): v for k, v in capacities.items()}
    stratum_defs = None

  test_path = args.output_dir / "test.csv"
  train_path = args.output_dir / "train.csv"
  valid_path = args.output_dir / "validation.csv"

  with test_path.open("w", newline="", encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["text", "puzzle", "solution", "difficulty", "clues"])

    with args.input_csv.open("r", newline="", encoding="utf-8") as f_in:
      reader = csv.DictReader(f_in)
      for i, row in enumerate(reader):
        if i not in selected_row_ids:
          continue
        puzzle = _normalize_grid(row["puzzle"])
        solution = _normalize_grid(row["solution"])
        difficulty = float(str(row["difficulty"]).strip())
        clues = int(str(row.get("clues", "0")).strip() or 0)
        writer.writerow([solution, puzzle, solution, difficulty, clues])

  header = ["text", "puzzle", "solution", "difficulty", "clues"]
  for p in [train_path, valid_path]:
    with p.open("w", newline="", encoding="utf-8") as fp:
      csv.writer(fp).writerow(header)

  metadata: Dict[str, object] = {
      "input_csv": str(args.input_csv),
      "output_dir": str(args.output_dir),
      "seed": args.seed,
      "mode": "strata" if strata_specs else "balanced_buckets_0_3",
      "actual_test_size": len(selected_row_ids),
      "total_rows_scanned": total_rows,
      "skipped_invalid_rows": skipped_invalid,
  }
  if strata_specs:
    metadata["strata"] = stratum_defs
    metadata["skipped_outside_strata_rows"] = skipped_no_stratum
  else:
    metadata["difficulty_range"] = [args.min_difficulty, args.max_difficulty]
    metadata["requested_test_size"] = args.test_size
    metadata["bucket_definitions"] = {
        "0": "[0.0, 1.0) within global range",
        "1": "[1.0, 2.0)",
        "2": "[2.0, 3.0)",
        "3": "[3.0, 4.0)",
    }
    metadata["bucket_capacities"] = capacities_dict
    metadata["bucket_selected"] = targets_dict
    metadata["skipped_out_of_range_rows"] = skipped_range

  (args.output_dir / "metadata.json").write_text(
      json.dumps(metadata, indent=2),
      encoding="utf-8")

  print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
  main()
