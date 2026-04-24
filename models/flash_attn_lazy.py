"""Lazy import for flash-attn.

Importing `diffusion` / `models` should not require flash-attn to be installed.
DIT / AR forward still needs it; a clear ImportError is raised on first use.
"""

from __future__ import annotations

import typing

_flash_attn = None
_flash_attn_rotary = None


def get_flash_attn() -> typing.Tuple[typing.Any, typing.Any]:
  """Return (flash_attn, flash_attn.layers.rotary) modules."""
  global _flash_attn, _flash_attn_rotary
  if _flash_attn is not None:
    return _flash_attn, _flash_attn_rotary
  try:
    import flash_attn as fa
    import flash_attn.layers.rotary as far
  except ImportError as e:
    raise ImportError(
        'The `flash_attn` package is required for backbone=dit and backbone=ar. '
        'On a GPU node with CUDA toolkit (nvcc), from the repo root run:\n'
        '  ./scripts/install_cuda_extensions_uv.sh\n'
        'Or install manually, e.g.:\n'
        '  uv pip install --no-build-isolation flash-attn==2.5.6'
    ) from e
  _flash_attn = fa
  _flash_attn_rotary = far
  return _flash_attn, _flash_attn_rotary
