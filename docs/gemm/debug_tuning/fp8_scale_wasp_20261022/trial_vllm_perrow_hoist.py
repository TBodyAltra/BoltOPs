"""Vendored vLLM Triton W8A8 block-scaled FP8 GEMM.

Source: ``vllm/model_executor/layers/quantization/utils/fp8_utils.py``
(``_w8a8_triton_block_scaled_mm`` / ``w8a8_triton_block_scaled_mm``).
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any, Optional

import torch
import triton
import triton.language as tl

@triton.jit
def decode_fp8_scale(scale, FP8_INPUT: tl.constexpr):
    if FP8_INPUT and scale.dtype == tl.uint8:
        pair = tl.inline_asm_elementwise(
            "v_cvt_pk_f32_e8m0 $0, $1;", constraints="=v,v",
            args=[scale.to(tl.uint32)], dtype=tl.uint64, is_pure=True, pack=1,
        )
        return pair.to(tl.uint32).to(tl.float32, bitcast=True)
    else:
        return scale





_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "w8a8_triton_block_scaled_mm",
)


def _device_name_for_config() -> str:
    if not torch.cuda.is_available():
        return "unknown"
    name = torch.cuda.get_device_name().replace(" ", "_")
    return name


@functools.lru_cache
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> Optional[dict[int, Any]]:
    """Return optimized configs keyed by batch size, or None for defaults."""
    device_name = _device_name_for_config()
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )
    config_file_path = os.path.join(_CONFIG_DIR, json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            return {int(key): val for key, val in json.load(f).items()}
    return None


@triton.jit
def _w8a8_triton_block_scaled_mm(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_SCALE_PER_ROW: tl.constexpr = False,
    WARP_SPECIALIZE: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if A_SCALE_PER_ROW:
        a_row_scale = decode_fp8_scale(tl.load(As_ptrs), A.dtype.element_ty.is_fp8())
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPECIALIZE):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        partial = tl.dot(a, b)
        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_scale_k = 0 if A_SCALE_PER_ROW else offs_ks
        if A_SCALE_PER_ROW:
            a_s = a_row_scale
        else:
            a_s = decode_fp8_scale(tl.load(As_ptrs + a_scale_k * stride_As_k), A.dtype.element_ty.is_fp8())
        b_s = decode_fp8_scale(tl.load(Bs_ptrs + offs_ks * stride_Bs_k), B.dtype.element_ty.is_fp8())

        accumulator += partial * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_triton_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Block-wise quantized matmul (vLLM W8A8 / FP8 block-scale Triton path).

    Args:
        A: activation, e.g. ``[..., K]`` FP8
        B: weight ``[N, K]`` FP8
        As: ``[..., ceil(K/block_k)]`` per-token-group scale, or ``[..., 1]``
            for one scale per token (floating values or uint8 E8M0)
        Bs: per-block scale for ``B`` ``[ceil(N/block_n), ceil(K/block_k)]``;
            uint8 storage is decoded as E8M0 for FP8 operands
        block_size: ``[block_n, block_k]``, e.g. ``[128, 128]``
        output_dtype: FP16 / BF16 / FP32
    """
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert As.shape[-1] in (1, triton.cdiv(A.shape[-1], block_k))
    a_scale_per_row = As.shape[-1] == 1
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    configs = get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_size[0],
            "BLOCK_SIZE_K": block_size[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        A_SCALE_PER_ROW=a_scale_per_row,
        **config,
    )
    return C
