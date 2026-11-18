# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Complete-tile BF16 TN GEMM with a producer/consumer MLS ring.

A[M,K] and public W[N,K] are K-contiguous; the MLS B view
is [K,N] with element strides (1,K). No operand transpose/copy is performed. Each TG owns one output tile, accumulates the complete K
in FP32, and stores BF16. There is no Split-K or cross-TG reduction.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacLayout, matrix_load, wait_lgkmcnt, sched_barrier,
    abarrier_init, abarrier_arrive, abarrier_seq, abarrier_try_wait,
    ebarrier_sync, get_wave_id,
)


@g.jit
def _idle_partition():
    pass


@g.jit
def _load_mls_batch(A, B, sa, sb, om, on, start,
                    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                    BK: gl.constexpr, CHUNKS: gl.constexpr, PW: gl.constexpr,
                    BM: gl.constexpr, BN: gl.constexpr):
    AM: gl.constexpr = min(PW, BM // 16)
    BNW: gl.constexpr = min(PW, BN // 16)
    for i in gl.static_range(CHUNKS):
        matrix_load(sa[i], A, (M, K), (K, 1),
                    (om, (start + i) * BK), warps=(AM, PW // AM))
        matrix_load(sb[i], B, (K, N), (1, K),
                    ((start + i) * BK, on), warps=(PW // BNW, BNW))



@g.jit
def _mls_producer(A, B, aa, bb,
                  M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                  CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                  PW: gl.constexpr, CW: gl.constexpr):
    # One physical wave initializes all EMPTY/READY pairs. Every consumer
    # releases one EMPTY arrival; every producer contributes one READY arrival.
    if get_wave_id() == 0:
        for s in gl.static_range(SLOTS):
            abarrier_init(s * 2, CW)
            abarrier_init(s * 2 + 1, PW)
            abarrier_arrive(s * 2, CW)
    ebarrier_sync(0, PW + CW)
    pid = gl.program_id(0)
    om, on = (pid // (N // BN)) * BM, (pid % (N // BN)) * BN
    for batch in gl.static_range(K // (BK * CHUNKS)):
        # Keep tuple subscripts as compile-time expressions: assigning this
        # remainder to an ordinary local would turn it into a tensor index.
        abarrier_try_wait((batch % SLOTS) * 2, (batch // SLOTS) % 2)
        abarrier_seq((batch % SLOTS) * 2 + 1)
        _load_mls_batch(A, B, aa[batch % SLOTS], bb[batch % SLOTS], om, on,
                        batch * CHUNKS, M, N, K, BK, CHUNKS, PW, BM, BN)
        abarrier_arrive((batch % SLOTS) * 2 + 1)
    ebarrier_sync(15, PW + CW)


@g.jit
def _consume_mls_batch(sa, sb, acc, F: gl.constexpr,
                       CHUNKS: gl.constexpr, release: gl.constexpr,
                       EARLY_RELEASE: gl.constexpr, DEFER_RELEASE: gl.constexpr):
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(sa[i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(sb[i], gl.DotOperandLayout(1, F, 4)),)
    sched_barrier(0)
    if EARLY_RELEASE:
        # Finish the whole operand batch once, then let the producer refill
        # its LDS slot while both BK chunks execute from VGPRs.
        wait_lgkmcnt(0)
        abarrier_arrive(release)
    for i in gl.static_range(CHUNKS):
        if not EARLY_RELEASE and not DEFER_RELEASE and i == CHUNKS - 1:
            # All DS reads must finish before the producer reuses this slot.
            # The last MMAC can overlap that refill because operands are VGPRs.
            wait_lgkmcnt(0)
            abarrier_arrive(release)
        sched_barrier(0)
        acc = gl.amd.cdna4.mfma(xx[i], yy[i], acc)
        sched_barrier(0)
    if DEFER_RELEASE:
        # Each batch's reads are waited on only as its MMAC operands are
        # consumed. Release the slot after both BK chunks have issued.
        abarrier_arrive(release)
    return acc


@g.jit
def _prefetch_mls_batches(aa, bb, acc, F: gl.constexpr,
                          K: gl.constexpr, BK: gl.constexpr,
                          CHUNKS: gl.constexpr, SLOTS: gl.constexpr):
    gl.static_assert(SLOTS >= 2)
    abarrier_try_wait(1, 0)
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(aa[0][i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(bb[0][i], gl.DotOperandLayout(1, F, 4)),)
    wait_lgkmcnt(0)
    abarrier_arrive(0)
    for batch in gl.static_range(K // (BK * CHUNKS)):
        nx, ny = (), ()
        for i in gl.static_range(CHUNKS):
            if batch + 1 < K // (BK * CHUNKS):
                if i == 0:
                    abarrier_try_wait(((batch + 1) % SLOTS) * 2 + 1,
                                     ((batch + 1) // SLOTS) % 2)
                nx += (async_copy.load_shared_relaxed(
                    aa[(batch + 1) % SLOTS][i], gl.DotOperandLayout(0, F, 4)),)
                ny += (async_copy.load_shared_relaxed(
                    bb[(batch + 1) % SLOTS][i], gl.DotOperandLayout(1, F, 4)),)
                sched_barrier(0)
            acc = gl.amd.cdna4.mfma(xx[i], yy[i], acc)
            sched_barrier(0)
        if batch + 1 < K // (BK * CHUNKS):
            # The last prefetched chunk has overlapped the current final
            # MMAC. All DS reads must finish before its LDS slot is released.
            wait_lgkmcnt(0)
            abarrier_arrive(((batch + 1) % SLOTS) * 2)
            xx, yy = nx, ny
    return acc


@g.jit
def _consume_two_mls_batches(aa, bb, acc, F: gl.constexpr,
                             K: gl.constexpr, BK: gl.constexpr,
                             CHUNKS: gl.constexpr, SLOTS: gl.constexpr):
    gl.static_assert(SLOTS >= 2 and (K // (BK * CHUNKS)) % 2 == 0)
    for group in gl.static_range(K // (BK * CHUNKS * 2)):
        xx, yy = (), ()
        for j in gl.static_range(2):
            abarrier_try_wait(((group * 2 + j) % SLOTS) * 2 + 1,
                             ((group * 2 + j) // SLOTS) % 2)
            for i in gl.static_range(CHUNKS):
                xx += (async_copy.load_shared_relaxed(
                    aa[(group * 2 + j) % SLOTS][i], gl.DotOperandLayout(0, F, 4)),)
                yy += (async_copy.load_shared_relaxed(
                    bb[(group * 2 + j) % SLOTS][i], gl.DotOperandLayout(1, F, 4)),)
        sched_barrier(0)
        for j in gl.static_range(2):
            for i in gl.static_range(CHUNKS):
                acc = gl.amd.cdna4.mfma(xx[j * CHUNKS + i], yy[j * CHUNKS + i], acc)
                sched_barrier(0)
        # Keep both slots until all required operands have been consumed.
        # An arrive between the two MMAC batches would cause the compiler
        # to drain unrelated outstanding LDS reads with lgkmcnt(0).
        for j in gl.static_range(2):
            abarrier_arrive(((group * 2 + j) % SLOTS) * 2)
            sched_barrier(0)
    return acc


@g.jit
def _mmac_consumer(C, aa, bb, F: gl.constexpr,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                   CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                   PW: gl.constexpr, CW: gl.constexpr, EARLY_RELEASE: gl.constexpr,
                   READ_AHEAD: gl.constexpr, GROUP_BATCHES: gl.constexpr,
                   DEFER_RELEASE: gl.constexpr):
    ebarrier_sync(0, PW + CW)
    pid = gl.program_id(0)
    om, on = (pid // (N // BN)) * BM, (pid % (N // BN)) * BN
    acc = gl.full((BM, BN), 0, gl.float32, F)
    if GROUP_BATCHES:
        acc = _consume_two_mls_batches(aa, bb, acc, F, K, BK, CHUNKS, SLOTS)
    elif READ_AHEAD:
        acc = _prefetch_mls_batches(aa, bb, acc, F, K, BK, CHUNKS, SLOTS)
    else:
        for batch in gl.static_range(K // (BK * CHUNKS)):
            abarrier_try_wait((batch % SLOTS) * 2 + 1, (batch // SLOTS) % 2)
            acc = _consume_mls_batch(aa[batch % SLOTS], bb[batch % SLOTS], acc,
                                     F, CHUNKS, (batch % SLOTS) * 2, EARLY_RELEASE, DEFER_RELEASE)
    rows = om + gl.arange(0, BM, layout=gl.SliceLayout(1, F))
    cols = on + gl.arange(0, BN, layout=gl.SliceLayout(0, F))
    gl.amd.cdna3.buffer_store(acc.to(gl.bfloat16), C,
                             rows[:, None] * N + cols[None, :])
    ebarrier_sync(15, PW + CW)


@g.jit
def gemm_a16w16_tn_kernel(
    A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr, CHUNKS: gl.constexpr = 1,
    SLOTS: gl.constexpr = 2, PW: gl.constexpr = 1, CW: gl.constexpr = 4,
    LTS: gl.constexpr = False, B_K_CONTIGUOUS: gl.constexpr = True,
    EARLY_RELEASE: gl.constexpr = False, READ_AHEAD: gl.constexpr = False,
    GROUP_BATCHES: gl.constexpr = False, DEFER_RELEASE: gl.constexpr = False,
):
    gl.static_assert(B_K_CONTIGUOUS, "this pipeline requires TN weights")
    gl.static_assert(not (EARLY_RELEASE and DEFER_RELEASE))
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    WN: gl.constexpr = BN // 16
    WM: gl.constexpr = BM // 16
    gl.static_assert(CW == WM * WN and PW <= CW and PW + CW <= 16)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and
                     B.dtype.element_ty == gl.bfloat16 and
                     C.dtype.element_ty == gl.bfloat16)
    gl.static_assert(PW == 1 or PW == 2 or PW == 4)
    gl.static_assert(SLOTS >= 1 and SLOTS <= 4 and (CHUNKS == 1 or CHUNKS == 2 or CHUNKS == 4))
    gl.static_assert(M % BM == 0 and N % BN == 0 and K % (BK * CHUNKS) == 0)
    gl.static_assert(CW >= WN and CW % WN == 0 and BM % (16 * WM) == 0)
    gl.static_assert(BN % (16 * WN) == 0)
    gl.static_assert(BM == 16 or BM == 32 or BM == 64 or BM == 128)
    gl.static_assert(BN == 16 or BN == 32 or BN == 64)
    # The current reader requires each wave's non-K span to fit one MLS tile.
    gl.static_assert(BM // WM == 16 and BN // WN == 16)
    gl.static_assert(BK % 64 == 0)
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        mmac_layout=MmacLayout.LTS if LTS else MmacLayout.LIT,
        warps_per_cta=[WM, WN], tiles_per_warp=[BM // WM // 16, BN // WN // 16],
    )
    SA: gl.constexpr = MlsSharedLayout(
        0, (16, 32), bits=16, order=(1, 0))
    SB: gl.constexpr = MlsSharedLayout(
        1, (32, 16), bits=16, order=(0, 1))
    aa, bb = (), ()
    for slot in gl.static_range(SLOTS):
        sa, sb = (), ()
        for i in gl.static_range(CHUNKS):
            sa += (gl.allocate_shared_memory(gl.bfloat16, (BM, BK), SA),)
            sb += (gl.allocate_shared_memory(gl.bfloat16, (BK, BN), SB),)
        aa += (sa,)
        bb += (sb,)
    gl.warp_specialize([
        (_idle_partition, ()),
        (_mls_producer, (A, B, aa, bb, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW)),
        (_mmac_consumer, (C, aa, bb, F, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW, EARLY_RELEASE, READ_AHEAD, GROUP_BATCHES, DEFER_RELEASE)),
    ], [PW, CW], [48, 240])
