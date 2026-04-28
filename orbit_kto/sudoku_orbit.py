
"""Utilities for Sudoku D4 orbit transforms and orbit-aware weighting.

Designed to plug into SunwooHong/mdlm_sudoku without modifying the
core model classes beyond an optional main.py mode hook.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


ALL_TRANSFORMS: Tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_diagonal",
)


def _reshape_board(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 81:
        raise ValueError(f"Expected trailing dimension 81, got {tuple(x.shape)}")
    return x.view(*x.shape[:-1], 9, 9)


def _flatten_board(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(*x.shape[:-2], 81)


def apply_transform_81(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply a D4 transform to a Sudoku board tensor with trailing dim 81.

    Supports int/float/bool tensors of shape [..., 81].
    """
    b = _reshape_board(x)
    if name == "identity":
        y = b
    elif name == "rot90":
        y = torch.rot90(b, 1, dims=(-2, -1))
    elif name == "rot180":
        y = torch.rot90(b, 2, dims=(-2, -1))
    elif name == "rot270":
        y = torch.rot90(b, 3, dims=(-2, -1))
    elif name == "flip_lr":
        y = torch.flip(b, dims=(-1,))
    elif name == "flip_ud":
        y = torch.flip(b, dims=(-2,))
    elif name == "transpose":
        y = b.transpose(-2, -1)
    elif name == "anti_diagonal":
        # Reflection over the anti-diagonal.
        y = torch.rot90(b.transpose(-2, -1), 2, dims=(-2, -1))
    else:
        raise KeyError(f"Unknown transform: {name}")
    return _flatten_board(y.contiguous())


def invert_transform_name(name: str) -> str:
    if name in {"identity", "rot180", "flip_lr", "flip_ud", "transpose", "anti_diagonal"}:
        return name
    if name == "rot90":
        return "rot270"
    if name == "rot270":
        return "rot90"
    raise KeyError(f"Unknown transform: {name}")


def choose_transforms(num_transforms: int, shuffle: bool = False) -> List[str]:
    if num_transforms <= 0:
        raise ValueError("num_transforms must be positive")
    if num_transforms >= len(ALL_TRANSFORMS):
        names = list(ALL_TRANSFORMS)
    else:
        names = list(ALL_TRANSFORMS[:num_transforms])
    if shuffle:
        random.shuffle(names)
    return names


def make_orbit_tensors(
    solution: torch.Tensor,
    anchor_mask: torch.Tensor,
    mask_index: int,
    transform_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    """Expand a base batch [B,81] into an orbit batch [B*M,81]."""
    if solution.ndim != 2 or solution.shape[-1] != 81:
        raise ValueError(f"Expected solution [B,81], got {tuple(solution.shape)}")
    if anchor_mask.shape != solution.shape:
        raise ValueError(
            f"anchor_mask shape {tuple(anchor_mask.shape)} does not match "
            f"solution shape {tuple(solution.shape)}"
        )
    B = solution.shape[0]
    sols = []
    anchors = []
    starts = []
    base_index = []
    transform_index = []
    for tid, name in enumerate(transform_names):
        sol_t = apply_transform_81(solution, name)
        anc_t = apply_transform_81(anchor_mask, name).bool()
        start_t = torch.where(
            anc_t,
            sol_t,
            torch.full_like(sol_t, fill_value=mask_index),
        )
        sols.append(sol_t)
        anchors.append(anc_t)
        starts.append(start_t)
        base_index.append(torch.arange(B, device=solution.device, dtype=torch.long))
        transform_index.append(torch.full((B,), tid, device=solution.device, dtype=torch.long))
    return {
        "solution": torch.cat(sols, dim=0),
        "anchor_mask": torch.cat(anchors, dim=0),
        "start_x": torch.cat(starts, dim=0),
        "attention_mask": (~torch.cat(anchors, dim=0)).long(),
        "base_index": torch.cat(base_index, dim=0),
        "transform_index": torch.cat(transform_index, dim=0),
    }


def exact_match(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return exact board match per row."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return pred.eq(target).all(dim=-1).long()


def orbit_counterexample_weights(
    labels: torch.Tensor,
    base_index: torch.Tensor,
    num_transforms: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Orbit-KTO positive and negative weights.

    labels: [N] with 0/1 exact-correct indicators.
    base_index: [N] mapping each transformed example to its base puzzle.
    Returns:
        w_pos [N], w_neg [N]
    where
        w_neg = (1-c_j) * (#successes in orbit) / (M-1)
        w_pos = c_j * (#failures in orbit) / (M-1)
    which is the normalized counterexample count used in the paper.
    """
    if labels.ndim != 1 or base_index.ndim != 1 or labels.shape != base_index.shape:
        raise ValueError("labels and base_index must be 1D tensors of the same shape")
    device = labels.device
    N = labels.shape[0]
    w_pos = torch.zeros(N, dtype=torch.float32, device=device)
    w_neg = torch.zeros(N, dtype=torch.float32, device=device)
    if num_transforms <= 1:
        return w_pos, w_neg

    num_bases = int(base_index.max().item()) + 1 if N > 0 else 0
    succ = torch.zeros(num_bases, dtype=torch.float32, device=device)
    succ.scatter_add_(0, base_index, labels.float())
    fail = float(num_transforms) - succ
    norm = float(num_transforms - 1)

    w_neg = (1.0 - labels.float()) * succ[base_index] / norm
    w_pos = labels.float() * fail[base_index] / norm
    return w_pos, w_neg


def orbit_statistics(
    labels: torch.Tensor,
    base_index: torch.Tensor,
    num_transforms: int,
) -> Dict[str, torch.Tensor]:
    """Compute Any/All/LSR-style batch statistics for transformed examples."""
    device = labels.device
    num_bases = int(base_index.max().item()) + 1 if labels.numel() > 0 else 0
    succ = torch.zeros(num_bases, dtype=torch.float32, device=device)
    succ.scatter_add_(0, base_index, labels.float())
    any_ok = (succ > 0).float()
    all_ok = (succ == float(num_transforms)).float()
    return {
        "any": any_ok.mean() if num_bases > 0 else torch.tensor(0.0, device=device),
        "all": all_ok.mean() if num_bases > 0 else torch.tensor(0.0, device=device),
        "lsr": (any_ok - all_ok).mean() if num_bases > 0 else torch.tensor(0.0, device=device),
    }
