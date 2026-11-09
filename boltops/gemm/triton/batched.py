# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Launch native batched GEMM with the selected config and original strides."""

_PARAMETER_NAMES = {
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages", "use_block_pingpong", "waves_per_eu",
    "matrix_instr_nonkdim", "optimize_epilogue",
    "amdgpu_enable_max_ilp_scheduling_strategy", "BATCH_LAST_GRID",
}


def validate_config(params):
    required = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                "num_warps", "num_stages", "BATCH_LAST_GRID"}
    if not required <= params.keys() or params.keys() - _PARAMETER_NAMES:
        return False
    positive = required - {"BATCH_LAST_GRID"}
    if any(type(params[k]) is not int or params[k] <= 0 for k in positive):
        return False
    if any(params[k] & (params[k] - 1) for k in
           ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps")):
        return False
    return type(params["BATCH_LAST_GRID"]) is bool


def launch_native(config, p, a, b, bias, out):
    from .batched_gemm_bf16 import _batched_gemm_bf16_kernel
    params = config.params

    def grid(meta):
        tiles = ((p.m + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"] *
                 ((p.n + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"]))
        return (tiles, 1, p.batch) if meta["BATCH_LAST_GRID"] else (p.batch, tiles)

    _batched_gemm_bf16_kernel[grid](
        a, b, out, bias if bias is not None else out, p.m, p.n, p.k,
        *a.stride(), b.stride(0), b.stride(2), b.stride(1), *out.stride(),
        bias.stride(0) if bias is not None else 1, bias is not None, **params)
