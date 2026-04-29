"""Temperature / Top-k / Top-p truncation for eval patches of diffusion._sample_categorical."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F


def truncate_probs_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
  """Keep top-k mass along last dim; renormalize."""
  if k <= 0:
    return probs
  vocab = probs.shape[-1]
  if k >= vocab:
    return probs
  topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
  out = torch.zeros_like(probs)
  out.scatter_(-1, topk_idx, topk_vals)
  return out / out.sum(dim=-1, keepdim=True).clamp(min=1e-30)


def truncate_probs_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
  """Nucleus (top-p) truncation on normalized probs; renormalize."""
  if p >= 1.0 - 1e-12:
    return probs
  sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
  cumsum = sorted_probs.cumsum(dim=-1)
  sorted_remove = cumsum > p
  sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
  sorted_remove[..., 0] = False
  filtered = sorted_probs.masked_fill(sorted_remove, 0.0)
  out = torch.zeros_like(probs)
  out.scatter_(-1, sorted_idx, filtered)
  return out / out.sum(dim=-1, keepdim=True).clamp(min=1e-30)


def configure_eval_sampling(
    diffusion_mod: Any,
    original_sample_categorical: Callable[..., torch.Tensor],
    *,
    temperature: float,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> None:
  """Patch diffusion_mod._sample_categorical for eval (temperature + optional truncation).

  Baseline: ``temperature == 1`` and no top-k / top-p restores ``original_sample_categorical``.
  """
  orig = original_sample_categorical
  t = float(temperature)
  k = int(top_k) if top_k is not None else 0
  if top_p is None or float(top_p) >= 1.0 - 1e-12:
    p_thr = 1.0
  else:
    p_thr = float(top_p)
    if p_thr <= 0.0:
      raise ValueError(f'sample-top-p must be > 0 when below 1.0, got {p_thr}')

  if t < 0:
    raise ValueError(f'temperature must be >= 0, got {t}')
  if k < 0:
    raise ValueError(f'sample-top-k must be >= 0, got {k}')

  use_trunc = (k > 0) or (p_thr < 1.0 - 1e-12)

  def _apply_trunc(scaled: torch.Tensor) -> torch.Tensor:
    out = scaled
    if k > 0:
      out = truncate_probs_top_k(out, k)
    if p_thr < 1.0 - 1e-12:
      out = truncate_probs_top_p(out, p_thr)
    return out

  if abs(t - 1.0) < 1e-7 and not use_trunc:
    diffusion_mod._sample_categorical = orig
    return

  temp_eps = 1e-12
  if t <= temp_eps:

    def _greedy_sample(
        categorical_probs: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
      del generator
      if use_trunc:
        logits = torch.log(categorical_probs.clamp(min=1e-30))
        scaled = F.softmax(logits, dim=-1)
        scaled = _apply_trunc(scaled)
        return scaled.argmax(dim=-1)
      return categorical_probs.argmax(dim=-1)

    diffusion_mod._sample_categorical = _greedy_sample
    return

  def _temp_sample(
      categorical_probs: torch.Tensor,
      generator: Optional[torch.Generator] = None,
  ) -> torch.Tensor:
    logits = torch.log(categorical_probs.clamp(min=1e-30))
    scaled = F.softmax(logits / t, dim=-1)
    scaled = _apply_trunc(scaled)
    if generator is None:
      rand_src = torch.rand_like(scaled)
    else:
      rand_src = torch.rand(
        scaled.shape,
        device=scaled.device,
        dtype=scaled.dtype,
        generator=generator,
      )
    gumbel_norm = 1e-10 - (rand_src + 1e-10).log()
    return (scaled / gumbel_norm).argmax(dim=-1)

  diffusion_mod._sample_categorical = _temp_sample
