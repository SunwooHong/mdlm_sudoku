#!/usr/bin/env python3
"""Wrapper entrypoint for Sudoku post-training SFT.

Keeps the implementation under `posttrain/` while exposing the expected
`scripts/sudoku_posttrain_sft.py` CLI path.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
  repo_root = Path(__file__).resolve().parent.parent
  impl_path = repo_root / 'posttrain' / 'sudoku_posttrain_sft.py'
  if not impl_path.exists():
    raise FileNotFoundError(f'Missing implementation script: {impl_path}')
  runpy.run_path(str(impl_path), run_name='__main__')


if __name__ == '__main__':
  main()
