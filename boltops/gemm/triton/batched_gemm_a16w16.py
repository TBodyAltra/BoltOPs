# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Vendored aiter Triton batched BF16/FP16 GEMM.

Upstream: ROCm/aiter ``aiter.ops.triton.batched_gemm_bf16``.

Computes ``Y[i] = X[i] @ W[i]^T`` with ``W`` shaped ``[B, N, K]``.


Launch parameters are supplied by the public interface configuration map.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

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
        config: required launch parameter dictionary
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

    b = b.transpose(1, 2)

    B = a.shape[0]
    M = a.shape[1]
    K = a.shape[2]
    N = b.shape[2]

    has_bias = bias is not None
    if out is None:
        out = torch.empty((B, M, N), dtype=dtype, device=a.device)

    if config is None:
        raise ValueError("batched_gemm_a16w16 requires an explicit config; use gemm_batched or gemm_grouped")
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


def _supports_grouped(p, config):
    return (p.kind == 'grouped' and p.groups.mode == 'equal' and p.native_operands and p.unscaled and
            p.a_dtype == p.b_dtype and p.a_dtype in (torch.float16, torch.bfloat16) and
            p.out_dtype in (torch.float16, torch.bfloat16))
