# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Complete-tile BF16 TT GEMM with a producer/consumer MLS ring.

A[M,K] and physical B[K,N] are row-contiguous; public w[N,K] is a view
of that B storage. Each TG owns one output tile, accumulates the complete K
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
                    BK: gl.constexpr, CHUNKS: gl.constexpr, PW: gl.constexpr, TN: gl.constexpr):
    for i in gl.static_range(CHUNKS):
        matrix_load(sa[i], A, (M, K), (K, 1),
                    (om, (start + i) * BK), warps=(PW, 1))
        matrix_load(sb[i], B, (K, N), (1, K) if TN else (N, 1),
                    ((start + i) * BK, on), warps=(1, PW))


@g.jit
def _mls_producer(A, B, aa, bb,
                  M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                  CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                  PW: gl.constexpr, CW: gl.constexpr, TN: gl.constexpr):
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
                        batch * CHUNKS, M, N, K, BK, CHUNKS, PW, TN)
        abarrier_arrive((batch % SLOTS) * 2 + 1)
    ebarrier_sync(15, PW + CW)


@g.jit
def _consume_mls_batch(sa, sb, acc, F: gl.constexpr,
                       CHUNKS: gl.constexpr, release: gl.constexpr, WAIT: gl.constexpr):
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(sa[i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(sb[i], gl.DotOperandLayout(1, F, 4)),)
    sched_barrier(0)
    for i in gl.static_range(CHUNKS):
        if WAIT == 1:
            wait_lgkmcnt(15 if i < CHUNKS - 2 else (8 if i == CHUNKS - 2 else 0))
        if i == CHUNKS - 1:
            # All DS reads must finish before the producer reuses this slot.
            # The last MMAC can overlap that refill because operands are VGPRs.
            wait_lgkmcnt(0)
            abarrier_arrive(release)
        sched_barrier(0)
        acc = gl.amd.cdna4.mfma(xx[i], yy[i], acc)
        sched_barrier(0)
    return acc


@g.jit
def _load_operand_batch(sa, sb, F: gl.constexpr, CHUNKS: gl.constexpr):
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(sa[i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(sb[i], gl.DotOperandLayout(1, F, 4)),)
    return xx, yy


@g.jit
def _mmac_consumer(C, aa, bb, F: gl.constexpr,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                   CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                   PW: gl.constexpr, CW: gl.constexpr, PREFETCH: gl.constexpr, WAIT: gl.constexpr):
    ebarrier_sync(0, PW + CW)
    pid = gl.program_id(0)
    om, on = (pid // (N // BN)) * BM, (pid % (N // BN)) * BN
    acc = gl.full((BM, BN), 0, gl.float32, F)
    if PREFETCH:
        gl.static_assert(SLOTS >= 2)
        abarrier_try_wait(1, 0)
        xx, yy = _load_operand_batch(aa[0], bb[0], F, CHUNKS)
        wait_lgkmcnt(0)
        abarrier_arrive(0)
        for batch in gl.static_range(K // (BK * CHUNKS)):
            if batch + 1 < K // (BK * CHUNKS):
                abarrier_try_wait(((batch + 1) % SLOTS) * 2 + 1,
                                 ((batch + 1) // SLOTS) % 2)
                nx, ny = _load_operand_batch(aa[(batch + 1) % SLOTS],
                                            bb[(batch + 1) % SLOTS], F, CHUNKS)
                sched_barrier(0)
            for i in gl.static_range(CHUNKS):
                acc = gl.amd.cdna4.mfma(xx[i], yy[i], acc)
            sched_barrier(0)
            if batch + 1 < K // (BK * CHUNKS):
                # The next reads overlap this batch's MMAC. Release their LDS
                # slot only after completion; retain their operands in VGPRs.
                wait_lgkmcnt(0)
                abarrier_arrive(((batch + 1) % SLOTS) * 2)
                xx, yy = nx, ny
    else:
        for batch in gl.static_range(K // (BK * CHUNKS)):
            abarrier_try_wait((batch % SLOTS) * 2 + 1, (batch // SLOTS) % 2)
            acc = _consume_mls_batch(aa[batch % SLOTS], bb[batch % SLOTS], acc,
                                     F, CHUNKS, (batch % SLOTS) * 2, WAIT)
    rows = om + gl.arange(0, BM, layout=gl.SliceLayout(1, F))
    cols = on + gl.arange(0, BN, layout=gl.SliceLayout(0, F))
    gl.amd.cdna3.buffer_store(acc.to(gl.bfloat16), C,
                             rows[:, None] * N + cols[None, :])
    ebarrier_sync(15, PW + CW)


@g.jit
def candidate_kernel(
    A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr, CHUNKS: gl.constexpr = 1,
    SLOTS: gl.constexpr = 2, PW: gl.constexpr = 1, CW: gl.constexpr = 4,
    LTS: gl.constexpr = True, B_K_CONTIGUOUS: gl.constexpr = False,
    PREFETCH: gl.constexpr = False, WN: gl.constexpr = 1,
    WAIT: gl.constexpr = 0,
):
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    WM: gl.constexpr = CW // WN
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and
                     B.dtype.element_ty == gl.bfloat16 and
                     C.dtype.element_ty == gl.bfloat16)
    gl.static_assert(PW == 1 or PW == 2 or PW == 4)
    gl.static_assert(SLOTS >= 1 and SLOTS <= 4 and (CHUNKS == 1 or CHUNKS == 2 or CHUNKS == 4 or CHUNKS == 8))
    gl.static_assert(M % BM == 0 and N % BN == 0 and K % (BK * CHUNKS) == 0)
    gl.static_assert(CW >= WN and CW % WN == 0 and BM % (16 * WM) == 0)
    gl.static_assert(BM == 16 or BM == 32 or BM == 64 or BM == 128)
    gl.static_assert(BN == 16 or BN == 32 or BN == 64)
    gl.static_assert(BM != 32 or WM == 2)
    # The current reader requires each wave's non-K span to fit one MLS tile.
    gl.static_assert(BM // WM <= (16 if BM <= 32 else 64))
    gl.static_assert(BK % 64 == 0)
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        mmac_layout=MmacLayout.LTS if LTS else MmacLayout.LIT,
        warps_per_cta=[WM, WN], tiles_per_warp=[BM // WM // 16, BN // WN // 16],
    )
    SA: gl.constexpr = MlsSharedLayout(
        0, (16, 32) if BM <= 32 else (64, 64), bits=16, order=(1, 0))
    SB: gl.constexpr = MlsSharedLayout(
        1, ((32, 16) if BN <= 32 else (64, 64)) if B_K_CONTIGUOUS else ((16, 32) if BN == 32 else (64, 64)),
        bits=16, order=(0, 1) if B_K_CONTIGUOUS else (1, 0))
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
        (_mls_producer, (A, B, aa, bb, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW, B_K_CONTIGUOUS)),
        (_mmac_consumer, (C, aa, bb, F, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW, PREFETCH, WAIT)),
    ], [PW, CW], [48, 240])
