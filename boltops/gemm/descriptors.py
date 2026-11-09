# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""GEMM semantics shared across dense, batched and grouped implementations."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json

import torch


@dataclass(frozen=True, init=False)
class KernelConfig:
    # Canonical JSON keeps nested options immutable, hashable and serializable.
    _params_json: str

    def __init__(self, block_m, block_n, block_k, **options):
        if any(type(v) is not int or v <= 0 for v in (block_m, block_n, block_k)):
            raise ValueError("kernel block sizes must be positive integers")
        params = dict(options, BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k)
        object.__setattr__(self, "_params_json", json.dumps(params, sort_keys=True, allow_nan=False))

    @classmethod
    def from_params(cls, params):
        if not isinstance(params, dict):
            raise ValueError("kernel params must be an object")
        options = dict(params)
        try:
            blocks = [options.pop("BLOCK_SIZE_" + d) for d in ("M", "N", "K")]
        except KeyError as error:
            raise ValueError("kernel params require BLOCK_SIZE_M/N/K") from error
        return cls(*blocks, **options)

    @property
    def params(self):
        return json.loads(self._params_json)

    @property
    def block_m(self):
        return self.params["BLOCK_SIZE_M"]

    @property
    def block_n(self):
        return self.params["BLOCK_SIZE_N"]

    @property
    def block_k(self):
        return self.params["BLOCK_SIZE_K"]


@dataclass(frozen=True)
class ScaleSpec:
    dtype: torch.dtype
    block: tuple[int, int]
    encoding: str = "numeric"


@dataclass(frozen=True)
class OperandSpec:
    storage_dtype: torch.dtype
    format: str = "native"
    packing: Optional[str] = None
    pack_axis: Optional[int] = None
    values_per_storage: int = 1
    zero_point_dtype: Optional[torch.dtype] = None
    strides: Optional[tuple[int, ...]] = None

    def __post_init__(self):
        if self.format == "native":
            if (self.packing is not None or self.pack_axis is not None or
                    self.values_per_storage != 1 or self.zero_point_dtype is not None):
                raise ValueError("native operands cannot carry packed-format metadata")
        elif (self.format not in ("int4", "fp4") or not self.packing or
              self.pack_axis is None or self.values_per_storage <= 1):
            raise ValueError("packed operands require a format, packing name, axis and pack factor")


@dataclass(frozen=True)
class GroupSpec:
    # M denotes total flattened rows; equal groups each contain M/count rows.
    mode: str
    count: int

    def __post_init__(self):
        if (self.mode not in ("equal", "offsets", "masked", "indexed") or
                type(self.count) is not int or self.count <= 0):
            raise ValueError("invalid grouped GEMM addressing contract")


@dataclass(frozen=True)
class GemmProblem:
    kind: str
    m: int
    n: int
    k: int
    a: OperandSpec
    b: OperandSpec
    out_dtype: torch.dtype
    arch: str
    a_scale: Optional[ScaleSpec] = None
    b_scale: Optional[ScaleSpec] = None
    batch: int = 1
    groups: Optional[GroupSpec] = None
    a_aligned_16: bool = True
    b_aligned_16: bool = True
    num_cus: Optional[int] = None
    activation_quantization: Optional[str] = None
    has_bias: bool = False
    input_dtype: Optional[torch.dtype] = field(init=False, default=None)
    a_layout: Optional[str] = field(init=False, default=None)
    b_layout: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        if self.kind == "dense":
            if self.unscaled and self.a_dtype == self.b_dtype:
                object.__setattr__(self, "input_dtype", self.a_dtype)
            object.__setattr__(self, "a_layout",
                               "k_contiguous" if self.a.strides == (self.k, 1) else "strided")
            object.__setattr__(self, "b_layout",
                               "k_contiguous" if self.b.strides == (self.k, 1) else
                               "n_contiguous" if self.b.strides == (1, self.n) else "strided")
        if self.num_cus is not None and (type(self.num_cus) is not int or self.num_cus <= 0):
            raise ValueError("num_cus must be a positive integer when specified")
        if self.kind not in ("dense", "batched", "grouped") or min(self.m, self.n, self.k) < 0:
            raise ValueError("invalid GEMM kind or shape")
        if self.batch <= 0 or (self.kind != "batched" and self.batch != 1):
            raise ValueError("batch count belongs only to batched GEMM")
        if (self.kind == "grouped") != (self.groups is not None):
            raise ValueError("group metadata belongs only to grouped GEMM")
        if self.groups and self.groups.mode == "equal" and self.m % self.groups.count:
            raise ValueError("equal groups must divide the total M rows")

    @property
    def interface(self):
        if self.kind == "batched" and self.activation_quantization == "mxfp4":
            return "batched_a16wfp4_online_quant"
        if self.kind in ("batched", "grouped"):
            return self.kind
        return "dense" if self.unscaled else "scaled"

    @property
    def a_dtype(self):
        return self.a.storage_dtype

    @property
    def b_dtype(self):
        return self.b.storage_dtype

    @property
    def native_operands(self):
        return self.a.format == self.b.format == "native"

    @property
    def unscaled(self):
        return self.a_scale is None and self.b_scale is None
