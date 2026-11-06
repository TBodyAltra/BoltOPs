"""Dense operation semantics, independent of kernel tile configuration."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScaleSpec:
    dtype: torch.dtype
    block: tuple[int, int]
    encoding: str = "numeric"


@dataclass(frozen=True)
class DenseProblem:
    m: int
    n: int
    k: int
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    out_dtype: torch.dtype
    a_scale: ScaleSpec
    b_scale: ScaleSpec
    arch: str
    a_aligned_16: bool = True
    b_aligned_16: bool = True
