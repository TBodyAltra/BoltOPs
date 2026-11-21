# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""BF16 complete-tile MLS rings for TN and TT weights."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacInterleave2Layout, MmacLayout, matrix_load, matrix_store, sched_barrier, wait_lgkmcnt,
    abarrier_arrive, abarrier_seq, abarrier_init, abarrier_try_wait, ebarrier_sync, get_wave_id,
)

from .common import _idle_partition
from .gemm_a16w16_persistent import _cast_for_matrix_store


# BF16 small-tile TN MLS ring

@g.jit
def _tn_idle_partition():
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
def _consume_tn_batch(sa, sb, acc, F: gl.constexpr,
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
def _mmac_consumer(C, aa, bb, F: gl.constexpr,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                   CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                   PW: gl.constexpr, CW: gl.constexpr, WAIT: gl.constexpr):
    ebarrier_sync(0, PW + CW)
    pid = gl.program_id(0)
    om, on = (pid // (N // BN)) * BM, (pid % (N // BN)) * BN
    acc = gl.full((BM, BN), 0, gl.float32, F)
    for batch in gl.static_range(K // (BK * CHUNKS)):
        abarrier_try_wait((batch % SLOTS) * 2 + 1, (batch // SLOTS) % 2)
        acc = _consume_tn_batch(aa[batch % SLOTS], bb[batch % SLOTS], acc,
                                 F, CHUNKS, (batch % SLOTS) * 2, WAIT)
    rows = om + gl.arange(0, BM, layout=gl.SliceLayout(1, F))
    cols = on + gl.arange(0, BN, layout=gl.SliceLayout(0, F))
    gl.amd.cdna3.buffer_store(acc.to(gl.bfloat16), C,
                             rows[:, None] * N + cols[None, :])
    ebarrier_sync(15, PW + CW)


@g.jit
def gemm_a16w16_tn_kernel(
    A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 16, BLOCK_SIZE_N: gl.constexpr = 16,
    BLOCK_SIZE_K: gl.constexpr = 128, B_K_CONTIGUOUS: gl.constexpr = True,
    LOOP_UNROLL_FACTOR: gl.constexpr = 4,
    CHUNKS: gl.constexpr = 4, SLOTS: gl.constexpr = 2,
):
    gl.static_assert(B_K_CONTIGUOUS, "this pipeline requires TN weights")
    gl.static_assert(M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0)
    gl.static_assert((CHUNKS == 1 or CHUNKS == 2 or CHUNKS == 4) and SLOTS >= 1 and SLOTS <= 4)
    gl.static_assert(K % (BLOCK_SIZE_K * CHUNKS) == 0)
    gl.static_assert(LOOP_UNROLL_FACTOR == 4)
    gl.static_assert((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) == (16, 16, 128))
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and
                     B.dtype.element_ty == gl.bfloat16 and
                     C.dtype.element_ty == gl.bfloat16)
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], mmac_layout=MmacLayout.LIT,
        warps_per_cta=[1, 1], tiles_per_warp=[1, 1],
    )
    SA: gl.constexpr = MlsSharedLayout(0, (16, 32), bits=16, order=(1, 0))
    SB: gl.constexpr = MlsSharedLayout(1, (32, 16), bits=16, order=(0, 1))
    aa, bb = (), ()
    for slot in gl.static_range(SLOTS):
        sa, sb = (), ()
        for i in gl.static_range(CHUNKS):
            sa += (gl.allocate_shared_memory(gl.bfloat16, (BM, BK), SA),)
            sb += (gl.allocate_shared_memory(gl.bfloat16, (BK, BN), SB),)
        aa += (sa,)
        bb += (sb,)
    gl.warp_specialize([
        (_tn_idle_partition, ()),
        (_mls_producer, (A, B, aa, bb, M, N, K, BM, BN, BK, CHUNKS, SLOTS, 1, 1, True)),
        (_mmac_consumer, (C, aa, bb, F, M, N, K, BM, BN, BK, CHUNKS, SLOTS, 1, 1, 1)),
    ], [1, 1], [48, 240])


_gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel = gemm_a16w16_tn_kernel


# BF16 fixed-tile TT MLS ring

@g.jit
def _read_mmac_operands(sa, sb, F: gl.constexpr):
    a = async_copy.load_shared_relaxed(sa, gl.DotOperandLayout(0, F, 4))
    b = async_copy.load_shared_relaxed(sb, gl.DotOperandLayout(1, F, 4))
    return a, b


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


# BF16 configurable TN/TT MLS ring

@g.jit
def _load_tile_mls_batch(A, B, sa, sb, om, on, start,
                    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                    BK: gl.constexpr, CHUNKS: gl.constexpr, PW: gl.constexpr,
                    BM: gl.constexpr, BN: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr, B_FIRST: gl.constexpr):
    AM: gl.constexpr = min(PW, BM // 16) if B_K_CONTIGUOUS else PW
    BNW: gl.constexpr = min(PW, BN // 16) if B_K_CONTIGUOUS else PW
    B_STRIDES: gl.constexpr = (1, K) if B_K_CONTIGUOUS else (N, 1)
    for i in gl.static_range(CHUNKS):
        if B_FIRST:
            matrix_load(sb[i], B, (K, N), B_STRIDES,
                        ((start + i) * BK, on), warps=(PW // BNW, BNW))
            matrix_load(sa[i], A, (M, K), (K, 1),
                        (om, (start + i) * BK), warps=(AM, PW // AM))
        else:
            matrix_load(sa[i], A, (M, K), (K, 1),
                        (om, (start + i) * BK), warps=(AM, PW // AM))
            matrix_load(sb[i], B, (K, N), B_STRIDES,
                        ((start + i) * BK, on), warps=(PW // BNW, BNW))


@g.jit
def _produce_tile_mls_batches(A, B, aa, bb, om, on,
                              M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                              BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                              CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                              PW: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr,
                              B_FIRST: gl.constexpr, SKIP_INITIAL_SLOT_WAIT: gl.constexpr):
    for batch in gl.static_range(K // (BK * CHUNKS)):
        # Keep tuple subscripts as compile-time expressions: assigning this
        # remainder to an ordinary local would turn it into a tensor index.
        # The first pass consumes slots seeded at initialization.
        if not SKIP_INITIAL_SLOT_WAIT or batch >= SLOTS:
            abarrier_try_wait((batch % SLOTS) * 2, (batch // SLOTS) % 2)
        abarrier_seq((batch % SLOTS) * 2 + 1)
        _load_tile_mls_batch(A, B, aa[batch % SLOTS], bb[batch % SLOTS], om, on,
                        batch * CHUNKS, M, N, K, BK, CHUNKS, PW, BM, BN, B_K_CONTIGUOUS, B_FIRST)
        abarrier_arrive((batch % SLOTS) * 2 + 1)


@g.jit
def _tile_mls_producer(A, B, aa, bb,
                  M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                  CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                  PW: gl.constexpr, CW: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr,
                  LOAD_ORDER_GROUP_SIZE: gl.constexpr, SKIP_INITIAL_SLOT_WAIT: gl.constexpr):
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
    if LOAD_ORDER_GROUP_SIZE and (pid % (2 * LOAD_ORDER_GROUP_SIZE)) >= LOAD_ORDER_GROUP_SIZE:
        _produce_tile_mls_batches(A, B, aa, bb, om, on,
                                  M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, B_K_CONTIGUOUS, True,
                                  SKIP_INITIAL_SLOT_WAIT)
    else:
        _produce_tile_mls_batches(A, B, aa, bb, om, on,
                                  M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, B_K_CONTIGUOUS, False,
                                  SKIP_INITIAL_SLOT_WAIT)
    ebarrier_sync(15, PW + CW)


@g.jit
def _consume_mls_batch(sa, sb, acc, F: gl.constexpr, B_LAYOUT: gl.constexpr,
                       CHUNKS: gl.constexpr, release: gl.constexpr, DEFER_RELEASE: gl.constexpr):
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(sa[i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(sb[i], B_LAYOUT),)
    sched_barrier(0)
    for i in gl.static_range(CHUNKS):
        if release >= 0 and not DEFER_RELEASE and i == CHUNKS - 1:
            # All DS reads must finish before the producer reuses this slot.
            # The last MMAC can overlap that refill because operands are VGPRs.
            wait_lgkmcnt(0)
            abarrier_arrive(release)
        sched_barrier(0)
        acc = gl.amd.cdna4.mfma(xx[i], yy[i], acc)
        sched_barrier(0)
    if release >= 0 and DEFER_RELEASE:
        # Operand dependencies complete the reads before this slot is reused.
        abarrier_arrive(release)
    return acc


@g.jit
def _load_operand_batch(sa, sb, F: gl.constexpr, B_LAYOUT: gl.constexpr, CHUNKS: gl.constexpr):
    xx, yy = (), ()
    for i in gl.static_range(CHUNKS):
        xx += (async_copy.load_shared_relaxed(sa[i], gl.DotOperandLayout(0, F, 4)),)
        yy += (async_copy.load_shared_relaxed(sb[i], B_LAYOUT),)
    return xx, yy


@g.jit
def _tile_mmac_consumer(C, aa, bb, F: gl.constexpr,
                   M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                   CHUNKS: gl.constexpr, SLOTS: gl.constexpr,
                   PW: gl.constexpr, CW: gl.constexpr, PREFETCH: gl.constexpr,
                   DEFER_RELEASE: gl.constexpr, USE_MATRIX_STORE: gl.constexpr, SKIP_FINAL_SLOT_RELEASE: gl.constexpr,
                   CONSUMER_N_SPLITS: gl.constexpr, SUBTILE: gl.constexpr):
    ebarrier_sync(0, PW + CW)
    pid = gl.program_id(0)
    om, on = (pid // (N // BN)) * BM, (pid % (N // BN)) * BN
    TILE_N: gl.constexpr = BN // CONSUMER_N_SPLITS
    if CONSUMER_N_SPLITS == 1:
        tile_b = bb
    else:
        on += SUBTILE * TILE_N
        tile_b = ()
        for slot in gl.static_range(SLOTS):
            chunks = ()
            for i in gl.static_range(CHUNKS):
                chunks += (bb[slot][i].slice(SUBTILE * TILE_N, TILE_N, dim=1),)
            tile_b += (chunks,)
    if USE_MATRIX_STORE:
        # Matrix stores require the matching B and accumulator interleave.
        ACC_LAYOUT: gl.constexpr = MmacInterleave2Layout(F)
        B_LAYOUT: gl.constexpr = MmacInterleave2Layout(gl.DotOperandLayout(1, F, 4))
    else:
        ACC_LAYOUT: gl.constexpr = F
        B_LAYOUT: gl.constexpr = gl.DotOperandLayout(1, F, 4)
    acc = gl.full((BM, TILE_N), 0, gl.float32, ACC_LAYOUT)
    if PREFETCH:
        gl.static_assert(SLOTS >= 2)
        abarrier_try_wait(1, 0)
        xx, yy = _load_operand_batch(aa[0], tile_b[0], F, B_LAYOUT, CHUNKS)
        wait_lgkmcnt(0)
        abarrier_arrive(0)
        for batch in gl.static_range(K // (BK * CHUNKS)):
            if batch + 1 < K // (BK * CHUNKS):
                abarrier_try_wait(((batch + 1) % SLOTS) * 2 + 1,
                                 ((batch + 1) // SLOTS) % 2)
                nx, ny = _load_operand_batch(aa[(batch + 1) % SLOTS],
                                            tile_b[(batch + 1) % SLOTS], F, B_LAYOUT, CHUNKS)
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
            # A slot needs an EMPTY arrival only if the producer will use it again.
            if not SKIP_FINAL_SLOT_RELEASE or batch + SLOTS < K // (BK * CHUNKS):
                acc = _consume_mls_batch(aa[batch % SLOTS], tile_b[batch % SLOTS], acc,
                                         F, B_LAYOUT, CHUNKS, (batch % SLOTS) * 2, DEFER_RELEASE)
            else:
                acc = _consume_mls_batch(aa[batch % SLOTS], tile_b[batch % SLOTS], acc,
                                         F, B_LAYOUT, CHUNKS, -1, DEFER_RELEASE)
    if USE_MATRIX_STORE:
        ebarrier_sync(15, PW + CW)
        matrix_store(C, _cast_for_matrix_store(acc, gl.bfloat16),
                     (M, N), (N, 1), (om, on), tile=(32, 32))
    else:
        rows = om + gl.arange(0, BM, layout=gl.SliceLayout(1, F))
        cols = on + gl.arange(0, BN, layout=gl.SliceLayout(0, F))
        gl.amd.cdna3.buffer_store(acc.to(gl.bfloat16), C,
                                 rows[:, None] * N + cols[None, :])
        ebarrier_sync(15, PW + CW)


@g.jit
def _gemm_a16w16_kernel(
    A, B, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr, CHUNKS: gl.constexpr = 1,
    SLOTS: gl.constexpr = 2, PW: gl.constexpr = 1, CW: gl.constexpr = 4,
    LTS: gl.constexpr = True, B_K_CONTIGUOUS: gl.constexpr = False,
    PREFETCH: gl.constexpr = False, DEFER_RELEASE: gl.constexpr = False,
    LOAD_ORDER_GROUP_SIZE: gl.constexpr = 0,
    CONSUMER_WARPS_N: gl.constexpr = None,
    SKIP_INITIAL_SLOT_WAIT: gl.constexpr = False,
    USE_MATRIX_STORE: gl.constexpr = False,
    SKIP_FINAL_SLOT_RELEASE: gl.constexpr = False,
    CONSUMER_N_SPLITS: gl.constexpr = 1,
):
    BM: gl.constexpr = BLOCK_SIZE_M
    BN: gl.constexpr = BLOCK_SIZE_N
    BK: gl.constexpr = BLOCK_SIZE_K
    gl.static_assert(M > 0 and N > 0 and K > 0)
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and
                     B.dtype.element_ty == gl.bfloat16 and
                     C.dtype.element_ty == gl.bfloat16)
    gl.static_assert(PW >= 1 and CW >= 1 and PW + CW <= 16)
    gl.static_assert(SLOTS >= 1 and SLOTS <= 4 and (CHUNKS == 1 or CHUNKS == 2 or CHUNKS == 4))
    gl.static_assert(M % BM == 0 and N % BN == 0 and K % (BK * CHUNKS) == 0)
    gl.static_assert(BM == 16 or BM == 32 or BM == 64 or BM == 128)
    gl.static_assert(BK % 64 == 0)
    gl.static_assert(not (PREFETCH and (DEFER_RELEASE or SKIP_FINAL_SLOT_RELEASE)))
    if B_K_CONTIGUOUS:
        WN: gl.constexpr = BN // 16
        WM: gl.constexpr = BM // 16
        gl.static_assert(BN == 16 or BN == 32 or BN == 64)
        gl.static_assert(CW == WM * WN and PW <= CW)
        gl.static_assert(PW == 1 or PW == 2 or PW == 4)
        gl.static_assert(PW % min(PW, WM) == 0 and PW % min(PW, WN) == 0)
    else:
        WN: gl.constexpr = max(1, BN // 64) if CONSUMER_WARPS_N is None else CONSUMER_WARPS_N
        WM: gl.constexpr = CW // WN
        gl.static_assert(PW == 1 or PW == 2)
        gl.static_assert(BN == 32 or BN == 64 or BN == 128 or BN == 256 or BN == 512)
        gl.static_assert(CW >= WN and CW % WN == 0 and BM % (16 * WM) == 0)
        gl.static_assert(BM != 32 or WM == 2)
        # Each wave's non-K span must fit one MLS tile.
        gl.static_assert(BM // WM <= (16 if BM <= 32 else 64))
    if USE_MATRIX_STORE:
        gl.static_assert(LTS and BM % 32 == 0 and (BN // WN) % 32 == 0)
    gl.static_assert(CONSUMER_N_SPLITS == 1 or CONSUMER_N_SPLITS == 2)
    if CONSUMER_N_SPLITS == 2:
        gl.static_assert(not B_K_CONTIGUOUS and WN == 2 and CW == 4 and USE_MATRIX_STORE)
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16],
        mmac_layout=MmacLayout.LTS if LTS else MmacLayout.LIT,
        warps_per_cta=[WM, WN // CONSUMER_N_SPLITS],
        tiles_per_warp=[BM // WM // 16, BN // WN // 16],
    )
    SA: gl.constexpr = MlsSharedLayout(
        0, (16, 32) if B_K_CONTIGUOUS or BM <= 32 else (64, 64), bits=16, order=(1, 0))
    SB: gl.constexpr = MlsSharedLayout(
        1, (32, 16) if B_K_CONTIGUOUS else ((16, 32) if BN == 32 else (64, 64)),
        bits=16, alt=1 if USE_MATRIX_STORE else 0,
        order=(0, 1) if B_K_CONTIGUOUS else (1, 0))
    aa, bb = (), ()
    for slot in gl.static_range(SLOTS):
        sa, sb = (), ()
        for i in gl.static_range(CHUNKS):
            sa += (gl.allocate_shared_memory(gl.bfloat16, (BM, BK), SA),)
            sb += (gl.allocate_shared_memory(gl.bfloat16, (BK, BN), SB),)
        aa += (sa,)
        bb += (sb,)
    if CONSUMER_N_SPLITS == 1:
        gl.warp_specialize([
            (_idle_partition, ()),
            (_tile_mls_producer,
             (A, B, aa, bb, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW,
              B_K_CONTIGUOUS, LOAD_ORDER_GROUP_SIZE, SKIP_INITIAL_SLOT_WAIT)),
            (_tile_mmac_consumer,
             (C, aa, bb, F, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW,
              PREFETCH, DEFER_RELEASE, USE_MATRIX_STORE, SKIP_FINAL_SLOT_RELEASE,
              CONSUMER_N_SPLITS, 0)),
        ], [PW, CW], [48, 240])
    else:
        # Consumer partitions share the ring and own disjoint output columns.
        gl.warp_specialize([
            (_idle_partition, ()),
            (_tile_mls_producer,
             (A, B, aa, bb, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW,
              B_K_CONTIGUOUS, LOAD_ORDER_GROUP_SIZE, SKIP_INITIAL_SLOT_WAIT)),
            (_tile_mmac_consumer,
             (C, aa, bb, F, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW,
              PREFETCH, DEFER_RELEASE, USE_MATRIX_STORE, SKIP_FINAL_SLOT_RELEASE,
              CONSUMER_N_SPLITS, 0)),
            (_tile_mmac_consumer,
             (C, aa, bb, F, M, N, K, BM, BN, BK, CHUNKS, SLOTS, PW, CW,
              PREFETCH, DEFER_RELEASE, USE_MATRIX_STORE, SKIP_FINAL_SLOT_RELEASE,
              CONSUMER_N_SPLITS, 1)),
        ], [PW, CW // 2, CW // 2], [48, 240, 240])
