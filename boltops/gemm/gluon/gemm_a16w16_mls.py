# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""BF16 complete-tile MLS rings for TN and TT weights."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacLayout, matrix_load, sched_barrier, wait_lgkmcnt,
    abarrier_arrive, abarrier_seq, abarrier_init, abarrier_try_wait, ebarrier_sync,
)

from .common import _idle_partition
from .gemm_a16w16_tt import gemm_a16w16_tt_kernel as _gemm_a16w16_tt_kernel


@g.jit
def _load_mls_batch(A, B, sa, sb, om, on, START: gl.constexpr):
    for j in gl.static_range(4):
        matrix_load(sa[j], A, (64, 2048), (2048, 1),
                    (om, (START + j) * 128), warps=(1, 1))
        matrix_load(sb[j], B, (2048, 64), (1, 2048),
                    ((START + j) * 128, on), warps=(1, 1))


@g.jit
def _read_mmac_operands(sa, sb, F: gl.constexpr):
    a = async_copy.load_shared_relaxed(sa, gl.DotOperandLayout(0, F, 4))
    b = async_copy.load_shared_relaxed(sb, gl.DotOperandLayout(1, F, 4))
    return a, b


@g.jit
def _mls_producer(A, B, a0, b0, a1, b1):
    # Per-slot EMPTY=2*slot, READY=2*slot+1. One wave contributes to each.
    # Prime EMPTY phase 0; later phases are released by the Consumer only.
    for bid in gl.static_range(4):
        abarrier_init(bid, 1)
    abarrier_arrive(0)
    abarrier_arrive(2)
    ebarrier_sync(0, 2)
    pid = gl.program_id(0)
    om, on = (pid // 4) * 16, (pid % 4) * 16
    for batch in gl.static_range(4):
        if batch % 2 == 0:
            sa, sb = a0, b0
        elif batch % 2 == 1:
            sa, sb = a1, b1
        abarrier_try_wait((batch % 2) * 2, batch // 2)
        # READY tracks completion of both A and B MLS transfers.
        abarrier_seq((batch % 2) * 2 + 1)
        _load_mls_batch(A, B, sa, sb, om, on, batch * 4)
        abarrier_arrive((batch % 2) * 2 + 1)
    ebarrier_sync(15, 2)


@g.jit
def _mmac_consumer(C, a0, b0, a1, b1, F: gl.constexpr):
    ebarrier_sync(0, 2)
    pid = gl.program_id(0)
    om, on = (pid // 4) * 16, (pid % 4) * 16
    acc = gl.full((16, 16), 0, gl.float32, F)
    for batch in gl.static_range(4):
        if batch % 2 == 0:
            ca, cb = a0, b0
        elif batch % 2 == 1:
            ca, cb = a1, b1
        abarrier_try_wait((batch % 2) * 2 + 1, batch // 2)
        x0, y0 = _read_mmac_operands(ca[0], cb[0], F)
        x1, y1 = _read_mmac_operands(ca[1], cb[1], F)
        x2, y2 = _read_mmac_operands(ca[2], cb[2], F)
        x3, y3 = _read_mmac_operands(ca[3], cb[3], F)
        sched_barrier(0)
        wait_lgkmcnt(15)
        sched_barrier(0)
        acc = gl.amd.cdna4.mfma(x0, y0, acc)
        sched_barrier(0)
        wait_lgkmcnt(15)
        acc = gl.amd.cdna4.mfma(x1, y1, acc)
        sched_barrier(0)
        wait_lgkmcnt(8)
        acc = gl.amd.cdna4.mfma(x2, y2, acc)
        sched_barrier(0)
        wait_lgkmcnt(0)
        # All reads of this slot have returned; Producer can refill it while
        # Consumer executes the last eight K16 MMAC instructions.
        abarrier_arrive((batch % 2) * 2)
        acc = gl.amd.cdna4.mfma(x3, y3, acc)
        sched_barrier(0)
    m = om + gl.arange(0, 16, layout=gl.SliceLayout(1, F))
    n = on + gl.arange(0, 16, layout=gl.SliceLayout(0, F))
    gl.amd.cdna3.buffer_store(acc.to(gl.bfloat16), C, m[:, None] * 64 + n[None, :])
    ebarrier_sync(15, 2)


@g.jit
def _gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel(
    A, B, C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 16,
    BLOCK_SIZE_N: gl.constexpr = 16,
    BLOCK_SIZE_K: gl.constexpr = 128,
    LOOP_UNROLL_FACTOR: gl.constexpr = 4,
    B_K_CONTIGUOUS: gl.constexpr = True,
):
    gl.static_assert(B_K_CONTIGUOUS, "this pipeline requires TN weights")
    gl.static_assert(M == 64 and N == 64 and K == 2048,
                     "this pipeline requires MNK=(64,64,2048)")
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
    # HCU removes the empty default: exactly one Producer + one Consumer.
    # Register requests are hints; WDRA is disabled by the recorded config.
    gl.warp_specialize([
        (_idle_partition, ()),
        (_mls_producer, (A, B, a0, b0, a1, b1)),
        (_mmac_consumer, (C, a0, b0, a1, b1, F)),
    ], [1, 1], [32, 160])


# Fixed BF16 TT pipeline: 16x32 per TG, three 24-KiB LDS slots.
# Public weights remain W[N,K]; N-contiguous W is logical B[K,N] contiguous.
@g.jit
def _load_tt_mls_batch(A, B, sa, sb, om, on, START: gl.constexpr,
                    BK: gl.constexpr, CHUNKS: gl.constexpr):
    for i in gl.static_range(CHUNKS):
        matrix_load(sa[i], A, (64, 2048), (2048, 1),
                    (om, (START + i) * BK), warps=(1, 1))
        matrix_load(sb[i], B, (2048, 64), (64, 1),
                    ((START + i) * BK, on), warps=(1, 1))

@g.jit
def _tt_mls_producer(A, B, a0, b0, a1, b1, a2, b2, BK: gl.constexpr, CHUNKS: gl.constexpr):
    # Three EMPTY/READY pairs. Each is owned by one Producer/Consumer wave.
    for bid in gl.static_range(6):
        abarrier_init(bid, 1)
    abarrier_arrive(0)
    abarrier_arrive(2)
    abarrier_arrive(4)
    ebarrier_sync(0, 2)
    pid = gl.program_id(0)
    om, on = (pid // 2) * 16, (pid % 2) * 32
    for batch in gl.static_range(2048 // (BK * CHUNKS)):
        if batch % 3 == 0:
            sa, sb = a0, b0
        elif batch % 3 == 1:
            sa, sb = a1, b1
        elif batch % 3 == 2:
            sa, sb = a2, b2
        abarrier_try_wait((batch % 3) * 2, (batch // 3) % 2)
        abarrier_seq((batch % 3) * 2 + 1)
        _load_tt_mls_batch(A, B, sa, sb, om, on, batch * CHUNKS, BK, CHUNKS)
        abarrier_arrive((batch % 3) * 2 + 1)
    ebarrier_sync(15, 2)

@g.jit
def _consume_tt_batch(ca, cb, acc, F: gl.constexpr, CHUNKS: gl.constexpr,
                   BK: gl.constexpr, RELEASE: gl.constexpr):
    xs = ()
    ys = ()
    for i in gl.static_range(CHUNKS):
        x, y = _read_mmac_operands(ca[i], cb[i], F)
        xs += (x,)
        ys += (y,)
    sched_barrier(0)
    # Each BK32 pair emits one A and two B DS instructions.
    for i in gl.static_range(CHUNKS):
        if i == CHUNKS - 1:
            # The first chunk uses backend operand-dependency waits. Before
            # releasing the slot, all remaining DS reads must have completed.
            wait_lgkmcnt(0)
            abarrier_arrive(RELEASE)
        sched_barrier(0)
        acc = gl.amd.cdna4.mfma(xs[i], ys[i], acc)
        sched_barrier(0)
    return acc

@g.jit
def _tt_mmac_consumer(C, a0, b0, a1, b1, a2, b2, F: gl.constexpr,
                   BK: gl.constexpr, CHUNKS: gl.constexpr):
    ebarrier_sync(0, 2)
    pid = gl.program_id(0)
    om, on = (pid // 2) * 16, (pid % 2) * 32
    acc = gl.full((16, 32), 0, gl.float32, F)
    for batch in gl.static_range(2048 // (BK * CHUNKS)):
        if batch % 3 == 0:
            ca, cb = a0, b0
        elif batch % 3 == 1:
            ca, cb = a1, b1
        elif batch % 3 == 2:
            ca, cb = a2, b2
        abarrier_try_wait((batch % 3) * 2 + 1, (batch // 3) % 2)
        acc = _consume_tt_batch(ca, cb, acc, F, CHUNKS, BK, (batch % 3) * 2)
    m = om + gl.arange(0, 16, layout=gl.SliceLayout(1, F))
    n = on + gl.arange(0, 32, layout=gl.SliceLayout(0, F))
    gl.amd.cdna3.buffer_store(acc.to(gl.bfloat16), C, m[:, None] * 64 + n[None, :])
    ebarrier_sync(15, 2)

@g.jit
def _gemm_a16w16_bf16_tt_bm16_bn32_bk128_kernel(
    A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr=16, BLOCK_SIZE_N: gl.constexpr=32,
    BLOCK_SIZE_K: gl.constexpr=128,
    B_K_CONTIGUOUS: gl.constexpr=False,
):
    gl.static_assert(not B_K_CONTIGUOUS, "this pipeline requires TT weights")
    gl.static_assert(M == 64 and N == 64 and K == 2048)
    gl.static_assert(BLOCK_SIZE_M == 16 and BLOCK_SIZE_N == 32)
    gl.static_assert(BLOCK_SIZE_K == 128)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16)
    gl.static_assert(B.dtype.element_ty == gl.bfloat16)
    gl.static_assert(C.dtype.element_ty == gl.bfloat16)
    BK: gl.constexpr = BLOCK_SIZE_K
    CHUNKS: gl.constexpr = 2
    gl.static_assert(2048 % (BK * CHUNKS) == 0)
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 1], tiles_per_warp=[1, 2])
    SA: gl.constexpr = MlsSharedLayout(0, (16, 32), bits=16, order=(1, 0))
    SB: gl.constexpr = MlsSharedLayout(1, (16, 32), bits=16, order=(1, 0))
    a0, b0, a1, b1 = (), (), (), ()
    for i in gl.static_range(CHUNKS):
        a0 += (gl.allocate_shared_memory(gl.bfloat16, (16, BK), SA),)
        b0 += (gl.allocate_shared_memory(gl.bfloat16, (BK, 32), SB),)
        a1 += (gl.allocate_shared_memory(gl.bfloat16, (16, BK), SA),)
        b1 += (gl.allocate_shared_memory(gl.bfloat16, (BK, 32), SB),)
    a2, b2 = (), ()
    for i in gl.static_range(CHUNKS):
        a2 += (gl.allocate_shared_memory(gl.bfloat16, (16, BK), SA),)
        b2 += (gl.allocate_shared_memory(gl.bfloat16, (BK, 32), SB),)
    gl.warp_specialize([
        (_idle_partition, ()),
        (_tt_mls_producer, (A, B, a0, b0, a1, b1, a2, b2, BK, CHUNKS)),
        (_tt_mmac_consumer, (C, a0, b0, a1, b1, a2, b2, F, BK, CHUNKS)),
    ], [1, 1], [32, 240])
