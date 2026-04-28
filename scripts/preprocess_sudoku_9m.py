#!/usr/bin/env python3
"""Preprocess 9M Sudoku CSV into strict train/validation/test splits.

Input format (Kaggle):
  puzzle,solution
  0700...,6795...

Output format:
  - train.csv
  - validation.csv
  - test.csv

Each output row has:
  text,puzzle,solution

Split is deterministic by hashing `solution` (+ seed), which keeps any rows
with the same solved board in exactly one split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass


@dataclass
class Counts:
    total: int = 0
    kept: int = 0
    invalid: int = 0
    train: int = 0
    validation: int = 0
    test: int = 0


def _is_valid_grid(s: str, allow_zero: bool) -> bool:
    if len(s) != 81:
        return False
    if allow_zero:
        return all(ch in "0123456789" for ch in s)
    return all(ch in "123456789" for ch in s)


def _split_of_solution(solution: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    key = f"{seed}:{solution}".encode("utf-8")
    h = hashlib.sha1(key).digest()
    bucket = int.from_bytes(h[:8], byteorder="big", signed=False) / float(2**64)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "validation"
    return "test"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="dataset/sudoku_9m.csv",
        help="Path to Kaggle sudoku csv (default: dataset/sudoku_9m.csv)",
    )
    p.add_argument(
        "--output-dir",
        default="dataset/sudoku_9m_processed",
        help="Directory for processed split csv files",
    )
    p.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    p.add_argument("--train-ratio", type=float, default=0.90, help="Train split ratio")
    p.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Ratios must be positive and sum to < 1.0")

    os.makedirs(args.output_dir, exist_ok=True)
    out_paths = {
        "train": os.path.join(args.output_dir, "train.csv"),
        "validation": os.path.join(args.output_dir, "validation.csv"),
        "test": os.path.join(args.output_dir, "test.csv"),
    }

    counts = Counts()
    out_files = {k: open(v, "w", newline="") for k, v in out_paths.items()}
    out_writers = {k: csv.writer(f) for k, f in out_files.items()}
    for writer in out_writers.values():
        writer.writerow(["text", "puzzle", "solution"])

    try:
        with open(args.input, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "puzzle" not in reader.fieldnames or "solution" not in reader.fieldnames:
                raise ValueError("Input csv must contain 'puzzle' and 'solution' columns")

            for row in reader:
                counts.total += 1
                puzzle = row["puzzle"].strip()
                solution = row["solution"].strip()

                if not _is_valid_grid(puzzle, allow_zero=True):
                    counts.invalid += 1
                    continue
                if not _is_valid_grid(solution, allow_zero=False):
                    counts.invalid += 1
                    continue

                split = _split_of_solution(
                    solution=solution,
                    seed=args.seed,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                )
                # text는 MDLM tokenizer 파이프라인에서 바로 사용 가능한 문자열 컬럼.
                out_writers[split].writerow([solution, puzzle, solution])

                counts.kept += 1
                if split == "train":
                    counts.train += 1
                elif split == "validation":
                    counts.validation += 1
                else:
                    counts.test += 1
    finally:
        for f in out_files.values():
            f.close()

    meta_path = os.path.join(args.output_dir, "metadata.txt")
    with open(meta_path, "w") as mf:
        mf.write(f"input={args.input}\n")
        mf.write(f"seed={args.seed}\n")
        mf.write(f"ratios=train:{args.train_ratio},val:{args.val_ratio},test:{test_ratio}\n")
        mf.write(f"total_rows={counts.total}\n")
        mf.write(f"kept_rows={counts.kept}\n")
        mf.write(f"invalid_rows={counts.invalid}\n")
        mf.write(f"train_rows={counts.train}\n")
        mf.write(f"validation_rows={counts.validation}\n")
        mf.write(f"test_rows={counts.test}\n")

    print("Preprocessing done.")
    print(f"Output dir: {args.output_dir}")
    print(
        f"Rows - total: {counts.total}, kept: {counts.kept}, invalid: {counts.invalid}, "
        f"train: {counts.train}, validation: {counts.validation}, test: {counts.test}"
    )


if __name__ == "__main__":
    main()
