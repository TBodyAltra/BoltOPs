# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Lazy adapter to the shared FP16/BF16 batched GEMM launcher."""


def launch_native(config, p, a, b, bias, out):
    from .batched_gemm_a16w16 import batched_gemm_a16w16
    batched_gemm_a16w16(a, b, bias, dtype=out.dtype, y=out, config=config.params)
