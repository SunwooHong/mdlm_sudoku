#!/usr/bin/env python3
"""Build 3M-only and 3M+9M Sudoku training splits.

Usage:
  python scripts/build_sudoku_splits.py \
    --data-dir dataset \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple


BIN_NAMES = ("d0", "d1_2", "d2_3", "d3p")


def diff_bin_3m(difficulty: float) -> str:
    if difficulty == 0:
        return "d0"
    if 1 <= difficulty < 2:
        return "d1_2"
    if 2 <= difficulty < 3:
        return "d2_3"
    if difficulty >= 3:
        return "d3p"
    raise ValueError(f"Unexpected 3M difficulty value: {difficulty}")


def count_givens(puzzle: str) -> int:
    return sum(ch in "123456789" for ch in puzzle)


def pick_quartile_edges(hist: Dict[int, int]) -> Tuple[int, int, int]:
    total = sum(hist.values())
    if total == 0:
        raise ValueError("Empty histogram")

    thresholds = [total * 0.25, total * 0.5, total * 0.75]
    edges: List[int] = []
    running = 0
    for g in range(82):
        running += hist.get(g, 0)
        while len(edges) < 3 and running >= thresholds[len(edges)]:
            edges.append(g)
    while len(edges) < 3:
        edges.append(81)
    return edges[0], edges[1], edges[2]


def make_9m_bin_fn(edges: Tuple[int, int, int]) -> Callable[[int], str]:
    e1, e2, e3 = edges

    def _bin(givens: int) -> str:
        if givens <= e1:
            return "d0"
        if givens <= e2:
            return "d1_2"
        if givens <= e3:
            return "d2_3"
        return "d3p"

    return _bin


def write_rows(
    source_csv: Path,
    source_name: str,
    idx_to_split: Dict[int, str],
    out_writers: Dict[str, csv.DictWriter],
    bin_fn: Callable[[Dict[str, str]], str],
    diff_fn: Callable[[Dict[str, str]], str],
) -> None:
    with source_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            split = idx_to_split.get(idx)
            if not split:
                continue
            puzzle = row["puzzle"]
            solution = row["solution"]
            givens = count_givens(puzzle)
            out_writers[split].writerow(
                {
                    "source": source_name,
                    "puzzle": puzzle,
                    "solution": solution,
                    "difficulty": diff_fn(row),
                    "difficulty_bin": bin_fn(row),
                    "proxy_givens": givens,
                }
            )


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir).resolve()
    csv_3m = data_dir / "sudoku-3m.csv"
    csv_9m = data_dir / "sudoku_9m.csv"
    if not csv_3m.exists() or not csv_9m.exists():
        raise FileNotFoundError("Expected dataset/sudoku-3m.csv and dataset/sudoku_9m.csv")

    # ---------- pass 1: index rows ----------
    idx_by_bin_3m: Dict[str, List[int]] = {k: [] for k in BIN_NAMES}
    hist_9m = Counter()
    val_test_candidates_9m: List[int] = []
    seen_9m = 0

    with csv_3m.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            d = float(row["difficulty"])
            idx_by_bin_3m[diff_bin_3m(d)].append(idx)

    with csv_9m.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            givens = count_givens(row["puzzle"])
            hist_9m[givens] += 1
            seen_9m += 1
            if len(val_test_candidates_9m) < 2000:
                val_test_candidates_9m.append(idx)
            else:
                j = rng.randint(0, seen_9m - 1)
                if j < 2000:
                    val_test_candidates_9m[j] = idx

    # ---------- sample 3M splits ----------
    for b in BIN_NAMES:
        rng.shuffle(idx_by_bin_3m[b])

    need_3m_only_post = 20000
    need_3m_mix_post = 10000
    need_val = 500
    need_test = 500
    needed_each_bin = need_val + need_test + need_3m_only_post + need_3m_mix_post

    selected_3m = {}
    for b in BIN_NAMES:
        if len(idx_by_bin_3m[b]) < needed_each_bin:
            raise ValueError(f"3M bin {b} has {len(idx_by_bin_3m[b])}, need {needed_each_bin}")
        arr = idx_by_bin_3m[b]
        selected_3m[b] = {
            "val": arr[0:need_val],
            "test": arr[need_val : need_val + need_test],
            "post_3m_only": arr[need_val + need_test : need_val + need_test + need_3m_only_post],
            "post_3m_mix": arr[
                need_val + need_test + need_3m_only_post : need_val + need_test + need_3m_only_post + need_3m_mix_post
            ],
        }

    idx_to_split_3m_only: Dict[int, str] = {}
    idx_to_split_3m_mix: Dict[int, str] = {}
    used_3m_only = set()
    used_3m_mix = set()

    for b in BIN_NAMES:
        for idx in selected_3m[b]["val"]:
            idx_to_split_3m_only[idx] = "val"
            idx_to_split_3m_mix[idx] = "val"
            used_3m_only.add(idx)
            used_3m_mix.add(idx)
        for idx in selected_3m[b]["test"]:
            idx_to_split_3m_only[idx] = "test"
            idx_to_split_3m_mix[idx] = "test"
            used_3m_only.add(idx)
            used_3m_mix.add(idx)
        for idx in selected_3m[b]["post_3m_only"]:
            idx_to_split_3m_only[idx] = "post_training"
            used_3m_only.add(idx)
        for idx in selected_3m[b]["post_3m_mix"]:
            idx_to_split_3m_mix[idx] = "post_training"
            used_3m_mix.add(idx)

    total_3m_rows = sum(len(v) for v in idx_by_bin_3m.values())
    for idx in range(total_3m_rows):
        if idx not in used_3m_only:
            idx_to_split_3m_only[idx] = "pretraining"
        if idx not in used_3m_mix:
            idx_to_split_3m_mix[idx] = "pretraining"

    # ---------- sample 9M splits ----------
    rng.shuffle(val_test_candidates_9m)
    val_9m = set(val_test_candidates_9m[:1000])
    test_9m = set(val_test_candidates_9m[1000:2000])
    holdout_9m = val_9m | test_9m

    edges = pick_quartile_edges(hist_9m)
    bin_9m = make_9m_bin_fn(edges)
    # Memory-safe stratified sampling for 9M post-training.
    # Keep only reservoir buffers, not all candidate indices.
    seen_per_bin = {k: 0 for k in BIN_NAMES}
    reservoir_per_bin: Dict[str, List[int]] = {k: [] for k in BIN_NAMES}
    with csv_9m.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx in holdout_9m:
                continue
            b = bin_9m(count_givens(row["puzzle"]))
            seen_per_bin[b] += 1
            seen = seen_per_bin[b]
            res = reservoir_per_bin[b]
            if len(res) < 10000:
                res.append(idx)
            else:
                j = rng.randint(0, seen - 1)
                if j < 10000:
                    res[j] = idx

    post_9m = set()
    for b in BIN_NAMES:
        if len(reservoir_per_bin[b]) < 10000:
            raise ValueError(f"9M proxy bin {b} has only {len(reservoir_per_bin[b])} candidates")
        post_9m.update(reservoir_per_bin[b])

    idx_to_split_9m_mix: Dict[int, str] = {}
    for idx in val_9m:
        idx_to_split_9m_mix[idx] = "val"
    for idx in test_9m:
        idx_to_split_9m_mix[idx] = "test"
    for idx in post_9m:
        idx_to_split_9m_mix[idx] = "post_training"
    for idx in range(seen_9m):
        if idx not in idx_to_split_9m_mix:
            idx_to_split_9m_mix[idx] = "pretraining"

    # ---------- write outputs ----------
    out_3m_only = data_dir / "3m_only"
    out_3m_mix = data_dir / "3m_9m"
    out_3m_only.mkdir(parents=True, exist_ok=True)
    out_3m_mix.mkdir(parents=True, exist_ok=True)

    fieldnames = ["source", "puzzle", "solution", "difficulty", "difficulty_bin", "proxy_givens"]

    def open_writers(root: Path) -> Tuple[Dict[str, csv.DictWriter], Dict[str, object]]:
        files = {}
        writers = {}
        for split in ("pretraining", "post_training", "val", "test"):
            fp = root / f"{split}.csv"
            fh = fp.open("w", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            files[split] = fh
            writers[split] = w
        return writers, files

    writers_3m_only, files_3m_only = open_writers(out_3m_only)
    writers_3m_mix, files_3m_mix = open_writers(out_3m_mix)

    write_rows(
        source_csv=csv_3m,
        source_name="kaggle_3m",
        idx_to_split=idx_to_split_3m_only,
        out_writers=writers_3m_only,
        bin_fn=lambda r: diff_bin_3m(float(r["difficulty"])),
        diff_fn=lambda r: r["difficulty"],
    )
    write_rows(
        source_csv=csv_3m,
        source_name="kaggle_3m",
        idx_to_split=idx_to_split_3m_mix,
        out_writers=writers_3m_mix,
        bin_fn=lambda r: diff_bin_3m(float(r["difficulty"])),
        diff_fn=lambda r: r["difficulty"],
    )
    write_rows(
        source_csv=csv_9m,
        source_name="kaggle_9m",
        idx_to_split=idx_to_split_9m_mix,
        out_writers=writers_3m_mix,
        bin_fn=lambda r: bin_9m(count_givens(r["puzzle"])),
        diff_fn=lambda _: "",
    )

    for fh in files_3m_only.values():
        fh.close()
    for fh in files_3m_mix.values():
        fh.close()

    meta = {
        "seed": args.seed,
        "3m_bins": {
            "d0": "difficulty == 0",
            "d1_2": "1 <= difficulty < 2",
            "d2_3": "2 <= difficulty < 3",
            "d3p": "difficulty >= 3",
        },
        "9m_proxy": {
            "name": "number_of_givens",
            "quartile_edges": {
                "edge1": edges[0],
                "edge2": edges[1],
                "edge3": edges[2],
            },
        },
        "targets": {
            "3m_only": {
                "post_training_each_bin": 20000,
                "val_each_bin": 500,
                "test_each_bin": 500,
            },
            "3m_9m": {
                "post_training_3m_each_bin": 10000,
                "post_training_9m_each_bin": 10000,
                "val_3m_each_bin": 500,
                "val_9m_random": 1000,
                "test_3m_each_bin": 500,
                "test_9m_random": 1000,
            },
        },
    }

    (out_3m_only / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_3m_mix / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote splits to: {out_3m_only} and {out_3m_mix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="dataset", help="Directory containing sudoku-3m.csv and sudoku_9m.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
