#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
bash scripts/grokking/train_sudoku_grokking.sh 3m-only 5m
