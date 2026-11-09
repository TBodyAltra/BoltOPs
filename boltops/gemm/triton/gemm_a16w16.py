"""Vendored aiter Triton ``gemm_a16w16`` (FP16/BF16).

Upstream: ROCm/aiter ``aiter.ops.triton.gemm.basic.gemm_a16w16``.
vLLM call site: ``from aiter.ops.triton.gemm_a16w16 import gemm_a16w16`` then
``gemm_a16w16(x, weight, bias)`` with ``weight`` shape ``[N, K]`` (computes
``Y = X @ W^T``).

Gluon backend and ``torch_compile_guard`` are not vendored; this wrapper always
uses the Triton kernel path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton

from boltops.gemm.triton.activation import (
    _get_activation_from_str,
)
from boltops.gemm.triton.gemm_a16w16_op import (
    _gemm_a16_w16_kernel,
)
from boltops.gemm.triton.gemm_a16w16_op import (
    _get_config as _get_triton_config,
)
from boltops.gemm.triton.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from ..descriptors import KernelConfig


def gemm_a16w16(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    skip_reduce: bool = False,
    kernel_type: str = "bandwidth_bound",
    backend: Optional[str] = None,
    problem=None,
) -> torch.Tensor:
    """16-bit matmul ``Y = X @ W^T`` (aiter / vLLM ROCm Triton path).

    Args:
        a: ``[M, K]`` FP16/BF16
        b: ``[N, K]`` FP16/BF16 (internally transposed)
        bias: optional ``[N]``
        dtype: output dtype; defaults to ``a.dtype``
        out: optional preallocated ``[M, N]``
        config: optional launch config dict (else JSON table lookup)
        activation: optional epilogue activation name
        skip_reduce: if True and split-K, return partials ``[NUM_KSPLIT, M, N]``
        kernel_type / backend: accepted for API compatibility; Triton-only here
    """
    if out is not None:
        dtype = out.dtype
    del kernel_type  # gluon-only upstream
    if backend is not None and backend.lower() not in ("triton",):
        raise ValueError(
            f"boltops vendored gemm_a16w16 only supports backend='triton'; "
            f"got {backend!r}"
        )

    if dtype is None or not isinstance(dtype, torch.dtype):
        dtype = a.dtype

    assert a.dtype in (torch.float16, torch.bfloat16), (
        f"Activations (x) must be fp16 or bf16, got {a.dtype}"
    )
    assert b.dtype in (torch.float16, torch.bfloat16), (
        f"Weights (w) must be fp16 or bf16, got {b.dtype}"
    )
    assert a.shape[1] == b.shape[1], "Incompatible matrix shapes."

    M, K = a.shape
    N, _ = b.shape
    w_t = b.T

    if config is None:
        config, _ = _get_triton_config(M, N, K)
    else:
        config = dict(config)

    if out is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        out = torch.empty((M, N), dtype=dtype, device=a.device)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=out.device if out is not None else a.device,
        )
    else:
        y_pp = None

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16_w16_kernel[grid](
        a,
        w_t,
        bias,
        out if config["NUM_KSPLIT"] == 1 else y_pp,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        out.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        out.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=(bias is not None),
        SKIP_REDUCE=skip_reduce,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            out,
            bias,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            out.stride(0),
            out.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=(bias is not None),
            activation=_get_activation_from_str(activation) if activation else "",
            use_activation=activation is not None,
            KERNEL_NAME="_gemm_a16w16_reduce_kernel",
        )

    return out


__all__ = ["gemm_a16w16"]


def _native_dense(p):
    return p.kind == "dense" and p.native_operands and p.unscaled and p.a_dtype == p.b_dtype


def _general_configurations(p):
    if not _native_dense(p) or p.a_dtype not in (torch.float16, torch.bfloat16):
        return ()
    if p.out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return ()
    from .gemm_a16w16_op import _get_config
    from .gemm_config_utils import compute_splitk_params
    cfg = dict(_get_config(p.m, p.n, p.k)[0])
    cfg["NUM_KSPLIT"] = 1
    cfg = compute_splitk_params(cfg, p.k)
    return (KernelConfig.from_params(cfg),)
