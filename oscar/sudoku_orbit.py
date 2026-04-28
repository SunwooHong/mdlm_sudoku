from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import torch

TRANSFORM_NAMES: List[str] = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_diagonal",
]


def _index_grid() -> torch.Tensor:
    return torch.arange(81, dtype=torch.long).reshape(9, 9)


@lru_cache(maxsize=None)
def get_perm(name: str) -> torch.Tensor:
    idx = _index_grid()
    if name == "identity":
        out = idx
    elif name == "rot90":
        out = torch.rot90(idx, k=1, dims=(0, 1))
    elif name == "rot180":
        out = torch.rot90(idx, k=2, dims=(0, 1))
    elif name == "rot270":
        out = torch.rot90(idx, k=3, dims=(0, 1))
    elif name == "flip_lr":
        out = torch.flip(idx, dims=(1,))
    elif name == "flip_ud":
        out = torch.flip(idx, dims=(0,))
    elif name == "transpose":
        out = idx.t()
    elif name == "anti_diagonal":
        out = torch.rot90(idx.t(), k=2, dims=(0, 1))
    else:
        raise KeyError(f"Unknown transform: {name}")
    return out.reshape(-1).clone()


@lru_cache(maxsize=None)
def get_inv_perm(name: str) -> torch.Tensor:
    return torch.argsort(get_perm(name))



def _normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else ndim + dim



def apply_perm(x: torch.Tensor, perm: torch.Tensor, dim: int = -1) -> torch.Tensor:
    dim = _normalize_dim(dim, x.ndim)
    perm = perm.to(device=x.device)
    return x.index_select(dim, perm)



def apply_transform(x: torch.Tensor, name: str, dim: int = -1) -> torch.Tensor:
    return apply_perm(x, get_perm(name), dim=dim)



def inverse_transform(x: torch.Tensor, name: str, dim: int = -1) -> torch.Tensor:
    return apply_perm(x, get_inv_perm(name), dim=dim)



def stack_orbit_views(x: torch.Tensor, names: Iterable[str], dim: int = -1, stack_dim: int = 1) -> torch.Tensor:
    views = [apply_transform(x, name, dim=dim) for name in names]
    return torch.stack(views, dim=stack_dim)
