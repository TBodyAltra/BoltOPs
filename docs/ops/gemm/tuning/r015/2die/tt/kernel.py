# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""BF16 TT/TN GEMM kernels for arbitrary nonnegative M, N and K.

A and C are contiguous row-major. TT B is contiguous KxN; set
B_K_CONTIGUOUS for contiguous NxK. Launch ceil(M/BM)*ceil(N/BN) groups.
A single M tile with complete N/K tiles uses the MLS ring; all edges and
multiple M tiles use masked Gluon loads. No post-compilation assembly edits.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacLayout, matrix_load, sched_barrier, wait_lgkmcnt,
    abarrier_arrive, abarrier_seq, abarrier_init, abarrier_try_wait, ebarrier_sync, get_wave_id,
)

@g.jit
def _idle_partition():
    pass

# BF16 configurable TN/TT MLS ring

@g.jit
def _load_tile_mls_batch(A, B, sa, sb, om, on, start,
                    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                    BK: gl.constexpr, CHUNKS: gl.constexpr, PW: gl.constexpr,
                    BM: gl.constexpr, BN: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr):
    AM: gl.constexpr = min(PW, BM // 16) if B_K_CONTIGUOUS else PW
    BNW: gl.constexpr = min(PW, BN // 16) if B_K_CONTIGUOUS else PW
    B_STRIDES: gl.constexpr = (1, K) if B_K_CONTIGUOUS else (N, 1)
    B_BOUNDS: gl.constexpr = (0, 1) if K % BK != 0 or N % BN != 0 else ()
    for i in gl.static_range(CHUNKS):
        matrix_load(sa[i], A, (M, K), (K, 1),
                    (om, (start + i) * BK), warps=(AM, PW // AM), boundary_check=(0, 1))
        matrix_load(sb[i], B, (K, N), B_STRIDES,
                    ((start + i) * BK, on), warps=(PW // BNW, BNW), boundary_check=B_BOUNDS)


@g.jit
def _tile_mls_producer(A, B, aa, bb,
                  M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                  CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                  PW: gl.constexpr, CW: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr):
    # One physical wave initializes all EMPTY/READY pairs. Every consumer
    # releases one EMPTY arrival; every producer contributes one READY arrival.
    if get_wave_id() == 0:
        for s in gl.static_range(SLOTS):
            abarrier_init(s * 2, CW)
            abarrier_init(s * 2 + 1, PW)
            abarrier_arrive(s * 2, CW)
    ebarrier_sync(0, PW + CW)
    pid = gl.program_id(0)
    # A single M tile has a constant row offset for every eligible shape.
    om: gl.constexpr = 0
    on = pid * BN
    for batch in gl.static_range(K // (BK * CHUNKS)):
        # Keep tuple subscripts as compile-time expressions: assigning this
        # remainder to an ordinary local would turn it into a tensor index.
        abarrier_try_wait((batch % SLOTS) * 2, (batch // SLOTS) % 2)
        abarrier_seq((batch % SLOTS) * 2 + 1)
        _load_tile_mls_batch(A, B, aa[batch % SLOTS], bb[batch % SLOTS], om, on,
                        batch * CHUNKS, M, N, K, BK, CHUNKS, PW, BM, BN, B_K_CONTIGUOUS)
        abarrier_arrive((batch % SLOTS) * 2 + 1)
    ebarrier_sync(15, PW + CW)


@g.jit
def _consume_mls_batch(sa, sb, acc, F: gl.constexpr,
                       CHUNKS: gl.constexpr, release: gl.constexpr, DEFER_RELEASE: gl.constexpr):
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(sa[i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(sb[i], gl.DotOperandLayout(1, F, 4)),)
    sched_barrier(0)
    for i in gl.static_range(CHUNKS):
        if not DEFER_RELEASE and i == CHUNKS - 1:
            # All DS reads must finish before the producer reuses this slot.
            # The last MMAC can overlap that refill because operands are VGPRs.
            wait_lgkmcnt(0)
            abarrier_arrive(release)
        sched_barrier(0)
        acc = gl.amd.cdna4.mfma(xx[i], yy[i], acc)
        sched_barrier(0)
    if DEFER_RELEASE:
        # Operand dependencies complete the reads before this slot is reused.
        abarrier_arrive(release)
    return acc


@g.jit
def _load_operand_batch(sa, sb, F: gl.constexpr, CHUNKS: gl.constexpr):
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(sa[i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(sb[i], gl.DotOperandLayout(1, F, 4)),)
    return xx, yy


@g.jit
def _tile_mmac_consumer(C, aa, bb, F: gl.constexpr,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                   CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                   PW: gl.constexpr, CW: gl.constexpr, PREFETCH: gl.constexpr,
                   DEFER_RELEASE: gl.constexpr):
    ebarrier_sync(0, PW + CW)
    pid = gl.program_id(0)
    # A single M tile has a constant row offset for every eligible shape.
    om: gl.constexpr = 0
    on = pid * BN
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
                                     F, CHUNKS, (batch % SLOTS) * 2, DEFER_RELEASE)
    rows = om + gl.arange(0, BM, layout=gl.SliceLayout(1, F))
    cols = on + gl.arange(0, BN, layout=gl.SliceLayout(0, F))
    gl.store(C + rows[:, None] * N + cols[None, :], acc.to(gl.bfloat16), (rows[:, None] < M) & (cols[None, :] < N))
    ebarrier_sync(15, PW + CW)


@g.jit
def _masked_gemm(A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                 BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                 B_K_CONTIGUOUS: gl.constexpr):
    # This partition uses the default warp group. All three problem axes are
    # masked independently, including the final K iteration and zero-length K.
    F: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
        instr_shape=[16, 16, 16], warps_per_cta=[1, gl.num_warps()],
        mmac_layout=MmacLayout.LTS, tiles_per_warp=[1, 1])
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, gl.num_warps()], [1, 0])
    pid = gl.program_id(0)
    nt = gl.cdiv(N, BN)
    rm = (pid // nt) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    rn = (pid % nt) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    rk_a = gl.arange(0, BK, layout=gl.SliceLayout(0, L))
    rk_b = gl.arange(0, BK, layout=gl.SliceLayout(1, L))
    acc = gl.full((BM, BN), 0, gl.float32, F)
    for block in range(gl.cdiv(K, BK)):
        ka = block * BK + rk_a
        kb = block * BK + rk_b
        av = gl.load(A + rm[:, None] * K + ka[None, :],
                     (rm[:, None] < M) & (ka[None, :] < K), other=0)
        if B_K_CONTIGUOUS:
            bp = B + rn[None, :] * K + kb[:, None]
        else:
            bp = B + kb[:, None] * N + rn[None, :]
        bv = gl.load(bp, (kb[:, None] < K) & (rn[None, :] < N), other=0)
        av = gl.convert_layout(av, gl.DotOperandLayout(0, F, 4))
        bv = gl.convert_layout(bv, gl.DotOperandLayout(1, F, 4))
        acc = gl.amd.cdna4.mfma(av, bv, acc)
    rr = (pid // nt) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, F))
    cc = (pid % nt) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, F))
    gl.store(C + rr[:, None] * N + cc[None, :], acc.to(gl.bfloat16),
             (rr[:, None] < M) & (cc[None, :] < N))


@g.jit
def _gemm_a16w16_kernel(
    A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr, CHUNKS: gl.constexpr = 1,
    SLOTS: gl.constexpr = 2, PW: gl.constexpr = 1, CW: gl.constexpr = 4,
    LTS: gl.constexpr = True, B_K_CONTIGUOUS: gl.constexpr = False,
    PREFETCH: gl.constexpr = False, DEFER_RELEASE: gl.constexpr = False,
):
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    gl.static_assert(M >= 0 and N >= 0 and K >= 0)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and
                     B.dtype.element_ty == gl.bfloat16 and
                     C.dtype.element_ty == gl.bfloat16)
    gl.static_assert(PW >= 1 and CW >= 1 and PW + CW <= 16)
    gl.static_assert(SLOTS >= 1 and SLOTS <= 4 and (CHUNKS == 1 or CHUNKS == 2 or CHUNKS == 4))
    gl.static_assert(BM == 16 or BM == 32 or BM == 64 or BM == 128)
    gl.static_assert(BK % 64 == 0)
    gl.static_assert(not (PREFETCH and DEFER_RELEASE))
    if B_K_CONTIGUOUS:
        WN: gl.constexpr = BN // 16
        WM: gl.constexpr = BM // 16
        gl.static_assert(BN == 16 or BN == 32 or BN == 64)
        gl.static_assert(CW == WM * WN and PW <= CW)
        gl.static_assert(PW == 1 or PW == 2 or PW == 4)
        gl.static_assert(PW % min(PW, WM) == 0 and PW % min(PW, WN) == 0)
    else:
        WN: gl.constexpr = max(1, BN // 64)
        WM: gl.constexpr = CW // WN
        gl.static_assert(PW == 1 or PW == 2)
        gl.static_assert(BN == 32 or BN == 64 or BN == 128 or BN == 256 or BN == 512)
        gl.static_assert(CW >= WN and CW % WN == 0 and BM % (16 * WM) == 0)
        gl.static_assert(BM != 32 or WM == 2)
        # Each wave's non-K span must fit one MLS tile.
        gl.static_assert(BM // WM <= (16 if BM <= 32 else 64))
    if M <= BM and N % BN == 0 and K > 0 and K % (BK * CHUNKS) == 0:
        F: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 16],
            mmac_layout=MmacLayout.LTS if LTS else MmacLayout.LIT,
            warps_per_cta=[WM, WN], tiles_per_warp=[BM // WM // 16, BN // WN // 16],
        )
        SA: gl.constexpr = MlsSharedLayout(
            0, (16, 32) if B_K_CONTIGUOUS or BM <= 32 else (64, 64), bits=16, order=(1, 0))
        SB: gl.constexpr = MlsSharedLayout(
            1, (32, 16) if B_K_CONTIGUOUS else ((16, 32) if BN == 32 else (64, 64)),
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
            (_tile_mls_producer, (A, B, aa, bb, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW, B_K_CONTIGUOUS)),
            (_tile_mmac_consumer, (C, aa, bb, F, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW, PREFETCH, DEFER_RELEASE)),
        ], [PW, CW], [48, 240])
    else:
        _masked_gemm(A, B, C, M, N, K, BM, BN, BK, B_K_CONTIGUOUS)
