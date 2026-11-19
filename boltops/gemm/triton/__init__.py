# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Triton GEMM backends, loaded by each operation on demand."""

import os

_BF16_MLS_REQUIRED_ENV = {
    "AMDGCN_USE_BUFFER_OPS": "1",
    "TRITON_BUFFER_CACHE_SWIZZLE": "1",
    "TRITON_MLS_B_INTERLEAVE": "2",
}
# Interleave is not a launch option (or a cache-key component on the archived
# compiler). Require process-start configuration, never switch it during calls.
_BF16_MLS_IMPORT_ENV = {key: os.environ.get(key) for key in _BF16_MLS_REQUIRED_ENV}
