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

from boltops.utility.fp8_scale import decode_fp8_scale
from ..descriptors import KernelConfig

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
    K,
    group_n,
    group_k,
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
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_scale_k = 0 if A_SCALE_PER_ROW else offs_ks
        a_s = decode_fp8_scale(tl.load(As_ptrs + a_scale_k * stride_As_k), A.dtype.element_ty.is_fp8())
        b_s = decode_fp8_scale(tl.load(Bs_ptrs + offs_ks * stride_Bs_k), B.dtype.element_ty.is_fp8())

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
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
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: list[int] = None,
    output_dtype: torch.dtype = torch.float16,
    out=None,
    config=None,
    problem=None,
) -> torch.Tensor:
    """Block-wise quantized matmul (vLLM W8A8 / FP8 block-scale Triton path).

    Args:
        a: activation, e.g. ``[..., K]`` FP8
        b: weight ``[N, K]`` FP8
        a_scale: ``[..., ceil(K/block_k)]`` per-token-group scale, or ``[..., 1]``
            for one scale per token (floating values or uint8 E8M0)
        b_scale: per-block scale for ``B`` ``[ceil(N/block_n), ceil(K/block_k)]``;
            uint8 storage is decoded as E8M0 for FP8 operands
        block_size: ``[block_n, block_k]``, e.g. ``[128, 128]``
        output_dtype: FP16 / BF16 / FP32
    """
    if problem is not None:
        block_size = problem.b_scale.block
    if out is not None:
        output_dtype = out.dtype
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert a.shape[-1] == b.shape[-1]
    assert a.shape[:-1] == a_scale.shape[:-1] and a.is_contiguous()
    assert a_scale.shape[-1] in (1, triton.cdiv(a.shape[-1], block_k))
    a_scale_per_row = a_scale.shape[-1] == 1
    M = a.numel() // a.shape[-1]

    assert b.ndim == 2 and b_scale.ndim == 2
    N, K = b.shape
    assert triton.cdiv(N, block_n) == b_scale.shape[0]
    assert triton.cdiv(K, block_k) == b_scale.shape[1]

    C_shape = a.shape[:-1] + (N,)
    C = out if out is not None else a.new_empty(C_shape, dtype=output_dtype)

    if config is None:
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
        a,
        b,
        C,
        a_scale,
        b_scale,
        M,
        N,
        K,
        block_n,
        block_k,
        a.stride(-2),
        a.stride(-1),
        b.stride(1),
        b.stride(0),
        C.stride(-2),
        C.stride(-1),
        a_scale.stride(-2),
        a_scale.stride(-1),
        b_scale.stride(1),
        b_scale.stride(0),
        A_SCALE_PER_ROW=a_scale_per_row,
        **config,
    )
    return C


_FLOAT_SCALES = (torch.float16, torch.bfloat16, torch.float32)
_INPUTS = (torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16, torch.int8)
_OUTPUTS = (torch.float16, torch.bfloat16, torch.float32)


def _common(p):
    return (p.a_dtype != torch.int8 and p.kind == 'dense' and p.native_operands and p.a_dtype == p.b_dtype and
            p.a_dtype in _INPUTS and p.out_dtype in _OUTPUTS and
            p.a.strides == (p.k, 1) and p.b.strides == (p.k, 1) and
            all(s is not None and s.encoding == 'numeric' and s.dtype in _FLOAT_SCALES
                for s in (p.a_scale, p.b_scale)))


def _block_config(p):
    if not _common(p) or p.has_bias:
        return None
    am, ak = p.a_scale.block
    bn, bk = p.b_scale.block
    if am != 1 or ak != bk or min(bn, bk) < 16 or bn & (bn - 1) or bk & (bk - 1):
        return None
    get_configs = get_w8a8_block_fp8_configs
    _filter_launch_config = dict
    configs = get_configs(p.n, p.k, bn, bk)
    cfg = (_filter_launch_config(configs[min(configs, key=lambda m: abs(m - p.m))]) if configs else
           {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': bn, 'BLOCK_SIZE_K': bk})
    # An iteration must not cross a scale-group boundary.
    if cfg['BLOCK_SIZE_K'] > bk or bk % cfg['BLOCK_SIZE_K']:
        return None
    return cfg


def _block_configs(p):
    config = _block_config(p)
    if config is None:
        return ()
    config = dict(config)
    config.setdefault("GROUP_SIZE_M", 32)
    config.setdefault("num_warps", 4)
    config.setdefault("num_stages", 2)
    return (KernelConfig.from_params(config),)
