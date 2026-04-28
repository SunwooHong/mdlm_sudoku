"""Lazy import for flash-attn + PyTorch fallbacks for uniform-length batches."""

from __future__ import annotations

import typing

import torch
import torch.nn.functional as F

_flash_attn: typing.Optional[typing.Any] = None
_flash_attn_rotary: typing.Optional[typing.Any] = None
_flash_attempted = False


def try_get_flash_attn() -> typing.Optional[typing.Tuple[typing.Any, typing.Any]]:
  """Return (flash_attn, rotary) or None if flash-attn is not installed."""
  global _flash_attn, _flash_attn_rotary, _flash_attempted
  if _flash_attempted:
    if _flash_attn is None:
      return None
    return _flash_attn, _flash_attn_rotary
  _flash_attempted = True
  try:
    import flash_attn as fa
    import flash_attn.layers.rotary as far
  except ImportError:
    _flash_attn = None
    _flash_attn_rotary = None
    return None
  _flash_attn = fa
  _flash_attn_rotary = far
  return _flash_attn, _flash_attn_rotary


def get_flash_attn() -> typing.Tuple[typing.Any, typing.Any]:
  """Strict: require flash-attn (e.g. variable-length packed batches)."""
  pair = try_get_flash_attn()
  if pair is None:
    raise ImportError(
        'The `flash_attn` package is required for variable-length packed '
        'attention and for best throughput. On a GPU node with CUDA toolkit '
        '(nvcc), from the repo root run:\n'
        '  ./scripts/install_cuda_extensions_uv.sh\n'
        'Or install manually, e.g.:\n'
        '  uv pip install --no-build-isolation flash-attn==2.5.6\n'
        'Uniform (batch x seq) workloads can still run without flash-attn via '
        'PyTorch SDPA fallback.'
    ) from None
  return pair


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_qkv(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
  """Apply RoPE to packed QKV (same layout as flash_attn rotary helper)."""
  cos_h = cos[0, :, 0, 0, : cos.shape[-1] // 2]
  sin_h = sin[0, :, 0, 0, : sin.shape[-1] // 2]
  pair = try_get_flash_attn()
  if pair is not None:
    _, far = pair
    return far.apply_rotary_emb_qkv_(qkv, cos_h, sin_h)
  cos_f = torch.cat([cos_h, cos_h], dim=-1).to(device=qkv.device, dtype=qkv.dtype)
  sin_f = torch.cat([sin_h, sin_h], dim=-1).to(device=qkv.device, dtype=qkv.dtype)
  cos_f = cos_f.unsqueeze(0).unsqueeze(2)
  sin_f = sin_f.unsqueeze(0).unsqueeze(2)
  q, k, v = qkv.unbind(2)
  q = q * cos_f + _rotate_half(q) * sin_f
  k = k * cos_f + _rotate_half(k) * sin_f
  return torch.stack((q, k, v), dim=2)


def _cu_seqlens_is_uniform_packed(
    cu_seqlens: torch.Tensor, batch_size: int, seq_len: int
) -> bool:
  if cu_seqlens.numel() != batch_size + 1:
    return False
  expected = torch.arange(
      0,
      (batch_size + 1) * seq_len,
      step=seq_len,
      dtype=cu_seqlens.dtype,
      device=cu_seqlens.device,
  )
  return bool(torch.equal(cu_seqlens, expected))


def qkvpacked_attention(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    batch_size: int,
    n_heads: int,
    causal: bool,
    dropout_p: float = 0.0,
) -> torch.Tensor:
  """Match flash_attn_varlen_qkvpacked_func for uniform packed batches without flash."""
  pair = try_get_flash_attn()
  if pair is not None:
    fa, _ = pair
    return fa.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen, dropout_p, causal=causal
    )
  if not _cu_seqlens_is_uniform_packed(cu_seqlens, batch_size, max_seqlen):
    raise ImportError(
        'Non-uniform sequence packing requires flash-attn. '
        'Run ./scripts/install_cuda_extensions_uv.sh on a GPU node with nvcc.'
    ) from None
  seq_len = max_seqlen
  total, three, nh, hd = qkv.shape
  if three != 3 or nh != n_heads or total != batch_size * seq_len:
    raise ValueError(
        f'Expected qkv shape (batch*seq={batch_size * seq_len}, 3, {n_heads}, d), '
        f'got {qkv.shape}'
    )
  qkv_bs = qkv.view(batch_size, seq_len, 3, n_heads, hd)
  q = qkv_bs[:, :, 0].transpose(1, 2)
  k = qkv_bs[:, :, 1].transpose(1, 2)
  v = qkv_bs[:, :, 2].transpose(1, 2)
  out = F.scaled_dot_product_attention(
      q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=causal
  )
  return out.transpose(1, 2).reshape(batch_size * seq_len, n_heads, hd)
