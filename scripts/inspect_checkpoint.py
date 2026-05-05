#!/usr/bin/env python3
"""Print training step / epoch stored inside a checkpoint file.

Supports:
  - Lightning ``*.ckpt`` (keys ``global_step``, ``epoch``, …)
  - Post-train ``*.pt`` from ``posttrain/sudoku_posttrain_sft.py`` (``step`` in payload)

Examples::

  python scripts/inspect_checkpoint.py outputs/.../checkpoints/best.ckpt
  python scripts/inspect_checkpoint.py outputs/posttrain/.../best.pt
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

# Lightning pickles may reference repo modules during unpickling (e.g. sudoku_dataloader).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import torch


def _torch_load(path: Path) -> dict:
  kw = {'map_location': 'cpu'}
  sig = inspect.signature(torch.load)
  if 'weights_only' in sig.parameters:
    kw['weights_only'] = False
  obj = torch.load(str(path), **kw)
  if not isinstance(obj, dict):
    raise TypeError(f'Expected dict checkpoint, got {type(obj)}')
  return obj


def main() -> None:
  p = argparse.ArgumentParser(description='Inspect Lightning or post-train checkpoint metadata.')
  p.add_argument('checkpoint', type=str, help='Path to .ckpt or .pt')
  args = p.parse_args()
  path = Path(args.checkpoint)
  if not path.is_file():
    raise SystemExit(f'Not a file: {path}')

  payload = _torch_load(path)
  out: dict = {'path': str(path.resolve())}

  if 'global_step' in payload:
    out['kind'] = 'lightning'
    out['global_step'] = int(payload['global_step'])
    out['epoch'] = int(payload.get('epoch', -1))
    out['pytorch_lightning_version'] = payload.get('pytorch_lightning_version')
  elif 'step' in payload and 'model_state_dict' in payload:
    out['kind'] = 'posttrain_pt'
    out['step'] = int(payload['step'])
    if isinstance(payload.get('metrics'), dict):
      out['metrics'] = payload['metrics']
  else:
    out['kind'] = 'unknown'
    out['keys'] = sorted(payload.keys())[:40]

  print(json.dumps(out, indent=2))


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
