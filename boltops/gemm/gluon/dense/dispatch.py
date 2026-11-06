"""Registered block-scaled implementations under the shared GEMM framework."""
import math

import torch
import triton

from ..registry import Implementation, KernelConfig


def _row_major(strides, rows, cols):
    return (strides is not None and len(strides) == 2 and
            (rows == 1 or strides[0] == cols) and (cols == 1 or strides[1] == 1))


def _scaled_dense_layout(p):
    return (p.native_operands and _row_major(p.a.strides, p.m, p.k) and
            _row_major(p.b.strides, p.n, p.k))


def _fp8_fp32_k128(p):
    return (_scaled_dense_layout(p) and p.a_scale is not None and p.b_scale is not None and
            p.a_dtype == p.b_dtype == torch.float8_e4m3fn and
            p.out_dtype == torch.bfloat16 and
            p.a_scale.dtype == p.b_scale.dtype == torch.float32 and
            p.a_scale.encoding == p.b_scale.encoding == "numeric" and
            p.a_scale.block == (1, 128) and p.b_scale.block == (128, 128))


_MLS_CONFIG = KernelConfig(128, 256, 128)
_FP8_CONFIG = KernelConfig(64, 128, 128)


def _supports_mls(p, config):
    # This resource/geometry contract is specific to the measured implementation.
    return (config == _MLS_CONFIG and p.arch == "gfx946" and _fp8_fp32_k128(p) and
            p.m % config.block_m == 0 and p.n % config.block_n == 0 and
            256 <= p.k <= 6144 and p.k % 256 == 0 and
            p.a_aligned_16 and p.b_aligned_16)


def _supports_masked_fp8(p, config):
    return config == _FP8_CONFIG and _fp8_fp32_k128(p)


def _general_configs(p):
    if not _scaled_dense_layout(p) or p.a_scale is None or p.b_scale is None:
        return ()
    q = math.gcd(p.a_scale.block[1], p.b_scale.block[1])
    return (KernelConfig(32, 64, max(32, min(128, triton.next_power_of_2(q)))),)


def _supports_general(p, config):
    # Native input/accumulator legality remains in the target MMAC database.
    return (config in _general_configs(p) and
            all(scale.encoding in ("numeric", "e8m0") and
                (scale.encoding != "e8m0" or scale.dtype == torch.uint8)
                for scale in (p.a_scale, p.b_scale)))


IMPLEMENTATIONS = (
    Implementation("mls_wdra", "dense", "boltops.gemm.gluon.dense.kernels.mls_wdra", "launch",
                   lambda p: (_MLS_CONFIG,), _supports_mls),
    Implementation("masked_fp8", "dense", "boltops.gemm.gluon.dense.kernels.masked", "launch_fp8",
                   lambda p: (_FP8_CONFIG,), _supports_masked_fp8),
    Implementation("block_scaled", "dense", "boltops.gemm.gluon.dense.kernels.masked", "launch_general",
                   _general_configs, _supports_general),
)
