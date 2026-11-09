# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Vendored aiter Triton batched BF16/FP16 GEMM.

Upstream: ROCm/aiter ``aiter.ops.triton.batched_gemm_bf16``.

Computes ``Y[i] = X[i] @ W[i]^T`` with ``W`` shaped ``[B, N, K]``.

HCU tuned JSON (optional), searched in this order::

    configs/batched_gemm_bf16/BATCHED_BF16-B={B}-N={N}-K={K}_{dtype}_{device}.json
    configs/batched_gemm_bf16/BATCHED_BF16-N={N}-K={K}_{dtype}_{device}.json
    configs/batched_gemm_bf16/BATCHED_BF16-DEFAULT_{dtype}_{device}.json

For BF16, each lookup level tries a native ``bf16`` file first and then the
corresponding ``fp16`` file. Both use the same 16-bit kernel configuration, so
existing FP16 tuning remains usable until a BF16-specific table is provided.

Flat ``{M: launch_config, ...}`` or AWQ-style
``{ "{N}_{K}": {M: launch_config, ...} }``; nearest ``M`` wins. This keeps
the existing N/K files compatible while allowing batch-specialized tuning.
Missing files fall back to built-in large/small tiles (AMD-only keys filtered
at launch).
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any, Optional

import torch
import triton
import triton.language as tl
from ..descriptors import KernelConfig

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "batched_gemm_bf16",
)

_LAUNCH_KEYS = frozenset(
    {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "use_block_pingpong",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "optimize_epilogue",
        "amdgpu_enable_max_ilp_scheduling_strategy",
        "BATCH_LAST_GRID",
    }
)

# aiter MI large/small without waves_per_eu / matrix_instr_nonkdim
_FALLBACK_LARGE = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "num_warps": 8,
    "num_stages": 2,
}
_FALLBACK_SMALL = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 2,
}


def _device_name_for_config() -> str:
    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name().replace(" ", "_")
    low = name.lower()
    if low.startswith("bw") or "bw200" in low:
        return "BW200"
    if "k100" in low:
        return "K100_AI"
    return name


def _dtype_tag(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype).replace("torch.", "")


def _filter_launch_config(config: dict) -> dict:
    return {k: v for k, v in config.items() if k in _LAUNCH_KEYS}


@functools.lru_cache
def get_batched_gemm_a16w16_configs(
    B: int, N: int, K: int, dtype_tag: str, layout: str = "tn"
) -> Optional[dict[int, Any]]:
    """Load the most specialized HCU table available for ``(B, N, K)``.

    Lookup order mirrors upstream AITER: B/N/K-specialized, N/K-specialized,
    then device default. At each level BF16 falls back to FP16 because both
    dtypes share this 16-bit kernel. Existing N/K JSON files remain valid.
    """
    device = _device_name_for_config()
    dtype_tags = (dtype_tag, "fp16") if dtype_tag == "bf16" else (dtype_tag,)
    layout = "tn" if layout == "nt" else layout
    assert layout in ("tn", "tt"), f"Unsupported {layout=}"
    prefix = "BATCHED_BF16" if layout == "tn" else "BATCHED_BF16-TT"
    filenames = tuple(
        filename
        for pattern in (
            "{prefix}-B={B}-N={N}-K={K}_{dtype}_{device}.json",
            "{prefix}-N={N}-K={K}_{dtype}_{device}.json",
            "{prefix}-DEFAULT_{dtype}_{device}.json",
        )
        for candidate_dtype in dtype_tags
        for filename in (
            pattern.format(
                B=B,
                N=N,
                K=K,
                dtype=candidate_dtype,
                device=device,
                prefix=prefix,
            ),
        )
    )
    for filename in filenames:
        path = os.path.join(_CONFIG_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        # Accept flat M tables plus the wrappers used by existing/tuning tools.
        for shape_key in (f"{B}_{N}_{K}", f"{N}_{K}", "any"):
            if shape_key in data and isinstance(data[shape_key], dict):
                data = data[shape_key]
                break
        return {int(k): v for k, v in data.items()}
    return None


def _get_config(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    B: int = 1,
    layout: str = "tn",
) -> dict:
    configs = get_batched_gemm_a16w16_configs(B, N, K, _dtype_tag(dtype), layout)
    if configs:
        return _filter_launch_config(
            configs[min(configs.keys(), key=lambda x: abs(x - M))]
        )
    if M + N >= 4096:
        return dict(_FALLBACK_LARGE)
    return dict(_FALLBACK_SMALL)


def _infer_weight_layout(x: torch.Tensor, w: torch.Tensor) -> str:
    """Infer TN versus TT storage for logical ``w[B,N,K]`` from its strides."""
    assert w.shape[2] == x.shape[2], (
        f"Incompatible K dimensions: x.shape={tuple(x.shape)}, "
        f"w.shape={tuple(w.shape)}"
    )
    k_contiguous = w.stride(2) == 1
    n_contiguous = w.stride(1) == 1
    if k_contiguous:
        return "tn"
    if n_contiguous:
        return "tt"
    raise AssertionError(
        "w must have either contiguous K (TN) or contiguous N (TT): "
        f"w.shape={tuple(w.shape)}, w.stride={w.stride()}"
    )


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _batched_gemm_a16w16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_biasb,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    BATCH_LAST_GRID: tl.constexpr,
):
    """Batched matmul ``C[i] = A[i] @ B[i]`` (+ optional bias). Not for direct use."""

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_biasb > 0)

    # Tensile launches this workload with the tile coordinate in grid.x and
    # batch in grid.z.  Keep the legacy ordering available for existing config
    # files, while allowing tuned shapes to use the same hardware dispatch
    # order as rocBLAS.
    if BATCH_LAST_GRID:
        pid = tl.program_id(axis=0)
        batch_id = tl.program_id(axis=2)
    else:
        batch_id = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )

        accumulator += tl.dot(a, b, input_precision="ieee")

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


def batched_gemm_a16w16(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    problem=None,
    a_scale=None,
    b_scale=None,
    group_offsets=None,
) -> torch.Tensor:
    """FP16/BF16 batched GEMM ``Y[i] = X[i] @ W[i].T`` (+ optional bias).

    Args:
        a: ``[B, M, K]`` BF16/FP16
        b: logical ``[B, N, K]`` tensor. Both operand strides are preserved.
            Explicit configurations support A K/M-contiguous and B K/N-contiguous storage.
        bias: optional ``[B, 1, N]``
        dtype: output dtype (``bfloat16`` or ``float16``)
        splitK: must be ``None`` (unsupported on this Triton path)
        out: optional preallocated ``[B, M, N]``
        config: optional launch dict; else HCU JSON or built-in fallback
    """
    if out is not None:
        dtype = out.dtype
    assert a.ndim == 3 and b.ndim == 3, "x and w must be rank-3 tensors"
    assert a.shape[0] == b.shape[0], "Incompatible Batch dimensions!!!"
    assert a.shape[2] == b.shape[2], "Incompatible K dimensions"

    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a16w16"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    weight_layout = _infer_weight_layout(a, b) if config is None else None
    b = b.transpose(1, 2)

    B = a.shape[0]
    M = a.shape[1]
    K = a.shape[2]
    N = b.shape[2]

    has_bias = bias is not None
    if out is None:
        out = torch.empty((B, M, N), dtype=dtype, device=a.device)

    if config is None:
        config = _get_config(M, N, K, dtype, B=B, layout=weight_layout)
    else:
        config = dict(config)

    # Triton needs a real pointer; unused when HAS_BIAS is False.
    bias_arg = bias if has_bias else out
    bias_stride = bias.stride(0) if has_bias else 1

    def grid(META):
        grid_mn = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]
        )
        if META.get("BATCH_LAST_GRID", False):
            return (grid_mn, 1, B)
        return (B, grid_mn)

    config.setdefault("BATCH_LAST_GRID", False)

    _batched_gemm_a16w16_kernel[grid](
        a,
        b,
        out,
        bias_arg,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        bias_stride,
        has_bias,
        **config,
    )

    return out


def _native(p):
    return (p.kind == 'grouped' and p.groups.mode == 'equal' and p.native_operands and p.unscaled and
            p.a_dtype == p.b_dtype and p.a_dtype in (torch.float16, torch.bfloat16) and
            p.out_dtype in (torch.float16, torch.bfloat16))


def _native_config(p):
    if not _native(p):
        return ()
    if p.b.strides is None:
        return ()
    layout = 'tn' if p.b.strides[-1] == 1 else 'tt' if p.b.strides[-2] == 1 else None
    if layout is None:
        return ()
    cfg = _get_config(p.m // p.groups.count, p.n, p.k, p.out_dtype, B=p.groups.count, layout=layout)
    return (KernelConfig.from_params(cfg),)
