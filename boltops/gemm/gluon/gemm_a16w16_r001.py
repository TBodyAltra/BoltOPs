# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Fixed R001 BF16 TN GEMM with explicit two-bank LDS scheduling.

A[64,2048], physical W[64,2048], B=W.T, C[64,64]. One wave per TG,
16 output TGs, 4 BK128 chunks per bank. No assembly rewriting or Split-K.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacLayout, matrix_load, sched_barrier, wait_lgkmcnt,
)


@g.jit
def _produce(A, B, sa, sb, om, on, START: gl.constexpr):
    for j in gl.static_range(4):
        matrix_load(sa[j], A, (64, 2048), (2048, 1),
                    (om, (START + j) * 128), warps=(1, 1))
        matrix_load(sb[j], B, (2048, 64), (1, 2048),
                    ((START + j) * 128, on), warps=(1, 1))


@g.jit
def _read(sa, sb, F: gl.constexpr):
    a = async_copy.load_shared_relaxed(sa, gl.DotOperandLayout(0, F, 4))
    b = async_copy.load_shared_relaxed(sb, gl.DotOperandLayout(1, F, 4))
    return a, b


@g.jit
def r001_bf16_tn(
    A, B, C,
    BLOCK_SIZE_M: gl.constexpr = 16,
    BLOCK_SIZE_N: gl.constexpr = 16,
    BLOCK_SIZE_K: gl.constexpr = 128,
    LOOP_UNROLL_FACTOR: gl.constexpr = 4,
):
    gl.static_assert(BLOCK_SIZE_M == 16 and BLOCK_SIZE_N == 16)
    gl.static_assert(BLOCK_SIZE_K == 128 and LOOP_UNROLL_FACTOR == 4)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16)
    gl.static_assert(B.dtype.element_ty == gl.bfloat16)
    gl.static_assert(C.dtype.element_ty == gl.bfloat16)
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], mmac_layout=MmacLayout.LIT,
        warps_per_cta=[1, 1])
    SA: gl.constexpr = MlsSharedLayout(0, (16, 32), bits=16, order=(1, 0))
    SB: gl.constexpr = MlsSharedLayout(1, (32, 16), bits=16, order=(0, 1))
    a0 = (gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA),
          gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA),
          gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA),
          gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA))
    b0 = (gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB),
          gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB),
          gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB),
          gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB))
    a1 = (gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA),
          gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA),
          gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA),
          gl.allocate_shared_memory(gl.bfloat16, (16, 128), SA))
    b1 = (gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB),
          gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB),
          gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB),
          gl.allocate_shared_memory(gl.bfloat16, (128, 16), SB))
    pid = gl.program_id(0)
    om = (pid // 4) * 16
    on = (pid % 4) * 16
    acc = gl.full((16, 16), 0, gl.float32, F)
    _produce(A, B, a0, b0, om, on, 0)
    for batch in gl.static_range(4):
        if batch % 2 == 0:
            ca, cb = a0, b0
            na, nb = a1, b1
        else:
            ca, cb = a1, b1
            na, nb = a0, b0
        async_copy.wait_group(0)
        x0, y0 = _read(ca[0], cb[0], F)
        x1, y1 = _read(ca[1], cb[1], F)
        x2, y2 = _read(ca[2], cb[2], F)
        x3, y3 = _read(ca[3], cb[3], F)
        sched_barrier(0)
        if batch < 3:
            _produce(A, B, na, nb, om, on, (batch + 1) * 4)
        sched_barrier(0)
        wait_lgkmcnt(0)
        acc = gl.amd.cdna4.mfma(x0, y0, acc)
        acc = gl.amd.cdna4.mfma(x1, y1, acc)
        acc = gl.amd.cdna4.mfma(x2, y2, acc)
        acc = gl.amd.cdna4.mfma(x3, y3, acc)
        # Keep next-batch VMEM waits/DS reads after the current MMAC group.
        sched_barrier(0)
    m = om + gl.arange(0, 16, layout=gl.SliceLayout(1, F))
    n = on + gl.arange(0, 16, layout=gl.SliceLayout(0, F))
    gl.amd.cdna3.buffer_store(acc.to(gl.bfloat16), C, m[:, None] * 64 + n[None, :])
