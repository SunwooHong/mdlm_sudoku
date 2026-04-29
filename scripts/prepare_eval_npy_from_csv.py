#!/usr/bin/env python3
"""Prepare eval npy files (solution/anchor) from split CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def _row_to_arrays(row: dict) -> tuple[np.ndarray, np.ndarray]:
  puzzle = str(row["puzzle"]).strip()
  solution = str(row["solution"]).strip()
  if len(puzzle) != 81 or len(solution) != 81:
    raise ValueError("Each puzzle/solution must have length 81.")
  sol = np.asarray([int(ch) - 1 for ch in solution], dtype=np.int64)
  if ((sol < 0) | (sol > 8)).any():
    raise ValueError("Solution contains non 1..9 digits.")
  anchor = np.asarray([ch in "123456789" for ch in puzzle], dtype=np.bool_)
  return sol, anchor


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--input-csv", required=True)
  p.add_argument("--output-dir", required=True)
  p.add_argument("--split", default="test")
  p.add_argument("--max-puzzles", type=int, default=1000)
  p.add_argument("--seed", type=int, default=42)
  args = p.parse_args()

  in_csv = Path(args.input_csv)
  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  rows = []
  with in_csv.open("r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows.append(row)
  if not rows:
    raise ValueError(f"Empty CSV: {in_csv}")

  rng = random.Random(args.seed)
  rng.shuffle(rows)
  picked = rows[: min(args.max_puzzles, len(rows))]

  solutions = []
  anchors = []
  source_counts = Counter()
  for row in picked:
    sol, anc = _row_to_arrays(row)
    solutions.append(sol)
    anchors.append(anc)
    source_counts[row.get("source", "<none>")] += 1

  sol_arr = np.stack(solutions, axis=0)
  anc_arr = np.stack(anchors, axis=0)
  np.save(out_dir / f"{args.split}_solution.npy", sol_arr)
  np.save(out_dir / f"{args.split}_anchor.npy", anc_arr)

  meta = {
    "input_csv": str(in_csv),
    "output_dir": str(out_dir),
    "split": args.split,
    "seed": args.seed,
    "max_puzzles": args.max_puzzles,
    "n_selected": int(sol_arr.shape[0]),
    "source_counts": dict(source_counts),
  }
  (out_dir / f"{args.split}_meta.json").write_text(
    json.dumps(meta, indent=2), encoding="utf-8")
  print(json.dumps(meta, indent=2))


if __name__ == "__main__":
  main()
