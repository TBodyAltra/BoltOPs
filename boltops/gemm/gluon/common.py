# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Shared Gluon helpers for GEMM pipelines."""

from triton.experimental import gluon as g


@g.jit
def _idle_partition():
    pass
