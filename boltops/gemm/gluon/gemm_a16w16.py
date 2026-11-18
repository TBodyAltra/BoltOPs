# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Gluon dense BF16/FP16 GEMM kernels, selected by tile name.

All entries share one public contract: x[M,K] and w[N,K], with the logical
transpose of w handled inside the kernel and no operand copy. Fixed BF16 TN
and TT entries cover R001; the persistent entry also supports both layouts.

R001 is a fixed 64x64x2048 WASP pipeline: one Producer and one Consumer
wave per TG, two 32-KiB LDS slots, 16 output TGs and 4 BK128 chunks per slot.

The TT entry uses 16x32 output tiles, eight TGs, and three 24-KiB slots.

The persistent 256x256x64 pipeline bounds active thread groups by the device
CU count and output tile count. Its two 128-column panels and packed epilogue
use native HCU MLS operations, with bounded loads and masked tail stores.

The BF16 TT 256x128x64 Stream-K pipeline uses two Producer partitions and two
Consumer partitions, a 96-KiB LDS ring, and one persistent TG per device CU.
It integrates the R020/V5 gluon_shape_reuse pipeline.
The host launcher supplies the task table, split-tile scratch and padded edges.
"""
from .gemm_a16w16_tt import gemm_a16w16_tt_kernel as _gemm_a16w16_tt_kernel

import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacInterleave2Layout, MmacLayout, matrix_load, matrix_store,
    sched_barrier, wait_lgkmcnt,
    abarrier_arrive, abarrier_seq, abarrier_init, abarrier_try_wait, ebarrier_sync,
)
try:
    from triton.experimental.gluon.language.amd.hcu import abarrier_wait
except ImportError:
    from triton.experimental.gluon.language.amd.hcu import abarrier_try_wait as abarrier_wait


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
def _idle_partition():
    pass


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


# Each LDS buffer has an empty/full barrier pair; both consumers release it.
_EMPTY_BARRIER_BASE = gl.constexpr(1)
_FULL_BARRIER_BASE = gl.constexpr(2)
_EMPTY_BARRIER_ARRIVALS = gl.constexpr(8)
_FULL_BARRIER_ARRIVALS = gl.constexpr(4)
_NUM_BUFFERS = gl.constexpr(2)
_BARRIER_INIT_WAVE = gl.constexpr(4)
_GET_WAVE_ID_ASM = gl.constexpr("s_getreg_b32 $0, hwreg(HW_REG_HW_ID1, 0, 5)")
_SYNC_THREAD_GROUP_ASM = gl.constexpr("s_ebarrier_sync 15, 0\n\ts_mov_b32 $0, 0")
_SYNC_PARTITION_ASM = gl.constexpr("s_ebarrier_sync 4, 4\n\ts_mov_b32 $0, 0")
_INVALIDATE_BARRIERS_ASM = gl.constexpr(
    "s_abarrier_inv 1\n\ts_abarrier_inv 2\n\t"
    "s_abarrier_inv 3\n\ts_abarrier_inv 4\n\ts_mov_b32 $0, 0"
)
_WAIT_LDS_READS_ASM = gl.constexpr("s_waitcnt lgkmcnt(0)\n\ts_mov_b32 $0, 0")
_BEGIN_MMAC_TURN_ASM = gl.constexpr(
    "s_ebarrier_sync 1, 8\n\ts_setprio 1\n\ts_mov_b32 $0, 0"
)
_END_MMAC_TURN_ASM = gl.constexpr(
    "s_setprio 0\n\ts_ebarrier_sync 1, 8\n\ts_mov_b32 $0, 0"
)
_SYNC_CONSUMERS_ASM = gl.constexpr("s_ebarrier_sync 1, 8\n\ts_mov_b32 $0, 0")
_RESET_PRIORITY_ASM = gl.constexpr("s_setprio 0\n\ts_mov_b32 $0, 0")


@triton.constexpr_function
def _barrier_init_asm(barrier_id, arrival_count):
    return f"s_abarrier_init {barrier_id}, {arrival_count}\n\ts_mov_b32 $0, 0"


@triton.constexpr_function
def _barrier_arrive_asm(barrier_id, arrival_count):
    return f"s_abarrier_arrive $0, {barrier_id}, {arrival_count}"


@g.jit
def _emit_scalar_asm(ASM: gl.constexpr):
    gl.inline_asm_elementwise(
        ASM, constraints="=s,~{memory}", args=[],
        dtype=gl.int32, is_pure=False, pack=1,
    )


@g.jit
def _get_wave_id():
    return gl.inline_asm_elementwise(
        _GET_WAVE_ID_ASM, constraints="=s", args=[],
        dtype=gl.int32, is_pure=True, pack=1,
    )


@g.jit
def _initialize_barrier(
    BARRIER_ID: gl.constexpr,
    ARRIVAL_COUNT: gl.constexpr,
    INITIAL_CREDITS: gl.constexpr,
    wave_id,
):
    INIT_ASM: gl.constexpr = _barrier_init_asm(BARRIER_ID, ARRIVAL_COUNT)
    if wave_id == _BARRIER_INIT_WAVE:
        _emit_scalar_asm(INIT_ASM)
    _emit_scalar_asm(_SYNC_PARTITION_ASM)
    if INITIAL_CREDITS > 0:
        CREDIT_ASM: gl.constexpr = _barrier_arrive_asm(BARRIER_ID, INITIAL_CREDITS)
        gl.inline_asm_elementwise(
            CREDIT_ASM, constraints="=s,~{memory}", args=[],
            dtype=gl.int32, is_pure=False, pack=1,
        )


@g.jit
def _initialize_pipeline_barriers(INITIALIZE: gl.constexpr):
    if INITIALIZE == 1:
        wave_id = _get_wave_id()
        _initialize_barrier(
            _EMPTY_BARRIER_BASE, _EMPTY_BARRIER_ARRIVALS, _NUM_BUFFERS, wave_id,
        )
        _initialize_barrier(_FULL_BARRIER_BASE, _FULL_BARRIER_ARRIVALS, 0, wave_id)
        _initialize_barrier(
            _EMPTY_BARRIER_BASE + 2, _EMPTY_BARRIER_ARRIVALS, _NUM_BUFFERS, wave_id,
        )
        _initialize_barrier(_FULL_BARRIER_BASE + 2, _FULL_BARRIER_ARRIVALS, 0, wave_id)
    _emit_scalar_asm(_SYNC_THREAD_GROUP_ASM)


@g.jit
def _finalize_pipeline_barriers(INVALIDATE: gl.constexpr):
    _emit_scalar_asm(_SYNC_THREAD_GROUP_ASM)
    if INVALIDATE == 1:
        _emit_scalar_asm(_INVALIDATE_BARRIERS_ASM)


@g.jit
def _advance_buffer_phase(buffer_index, barrier_phase):
    buffer_index = (buffer_index + 1) % _NUM_BUFFERS
    return buffer_index, barrier_phase ^ (1 - buffer_index)


@g.jit
def _persistent_producer(
    A,
    B,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    a_buffer0,
    a_buffer1,
    b_buffer0,
    b_buffer1,
    NUM_THREAD_GROUPS: gl.constexpr,
    B_K_CONTIGUOUS: gl.constexpr,
):
    _initialize_pipeline_barriers(0)
    B_STRIDES: gl.constexpr = (1, K) if B_K_CONTIGUOUS else (N, 1)
    # Two-axis checks also handle fully out-of-bounds native load tiles.
    A_BOUNDS: gl.constexpr = (0, 1) if M % 256 != 0 or K % 64 != 0 else ()
    B_BOUNDS: gl.constexpr = (0, 1) if K % 64 != 0 or N % 256 != 0 else ()
    NUM_N_TILES: gl.constexpr = (N + 255) // 256
    NUM_OUTPUT_TILES: gl.constexpr = ((M + 255) // 256) * NUM_N_TILES
    NUM_K_TILES: gl.constexpr = (K + 63) // 64
    buffer_index = 0
    barrier_phase = 0
    for tile_id in range(gl.program_id(0), NUM_OUTPUT_TILES, NUM_THREAD_GROUPS):
        row_offset = (tile_id // NUM_N_TILES) * 256
        col_offset = (tile_id % NUM_N_TILES) * 256
        for k_tile in range(NUM_K_TILES):
            k_offset = k_tile * 64
            abarrier_wait(_EMPTY_BARRIER_BASE + buffer_index * 2, barrier_phase)
            abarrier_seq(_FULL_BARRIER_BASE + buffer_index * 2)
            if buffer_index == 0:
                matrix_load(
                    a_buffer0, A, (M, K), (K, 1), (row_offset, k_offset),
                    warps=(4, 1), boundary_check=A_BOUNDS,
                )
                matrix_load(
                    b_buffer0, B, (K, N), B_STRIDES, (k_offset, col_offset),
                    warps=(1, 4), boundary_check=B_BOUNDS,
                )
            else:
                matrix_load(
                    a_buffer1, A, (M, K), (K, 1), (row_offset, k_offset),
                    warps=(4, 1), boundary_check=A_BOUNDS,
                )
                matrix_load(
                    b_buffer1, B, (K, N), B_STRIDES, (k_offset, col_offset),
                    warps=(1, 4), boundary_check=B_BOUNDS,
                )
            abarrier_arrive(_FULL_BARRIER_BASE + buffer_index * 2)
            buffer_index, barrier_phase = _advance_buffer_phase(
                buffer_index, barrier_phase,
            )
    _finalize_pipeline_barriers(0)


@triton.constexpr_function
def _matrix_store_pair_layout(layout):
    # Make register bit 2 the low bit to cast MLS pairs (0,4), (1,5), (2,6), (3,7).
    register_bases = list(layout.reg_bases)
    register_bases = [
        register_bases[2], register_bases[0], register_bases[1],
    ] + register_bases[3:]
    return gl.DistributedLinearLayout(
        register_bases, layout.lane_bases, layout.warp_bases,
        layout.block_bases, layout.shape,
    )


@g.jit
def _cast_for_matrix_store(accumulator, OUTPUT_DTYPE: gl.constexpr):
    # Layout conversions only rename registers; each packed pair shares one VGPR.
    ACCUMULATOR_LAYOUT: gl.constexpr = accumulator.type.layout
    LINEAR_LAYOUT: gl.constexpr = gl.to_linear_layout(
        ACCUMULATOR_LAYOUT, accumulator.shape,
    )
    PACKED_LAYOUT: gl.constexpr = _matrix_store_pair_layout(LINEAR_LAYOUT)
    value = gl.convert_layout(accumulator, PACKED_LAYOUT).to(OUTPUT_DTYPE)
    return gl.convert_layout(value, ACCUMULATOR_LAYOUT)


@g.jit
def _persistent_consumer(
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    a_buffer0,
    a_buffer1,
    b_buffer0,
    b_buffer1,
    MFMA_LAYOUT: gl.constexpr,
    OUTPUT_DTYPE: gl.constexpr,
    CONSUMER_ID: gl.constexpr,
    NUM_THREAD_GROUPS: gl.constexpr,
    FORCE_MASKED_STORE: gl.constexpr,
):
    _initialize_pipeline_barriers(1 if CONSUMER_ID == 0 else 0)
    ROWS_PER_CONSUMER: gl.constexpr = 128
    ACCUMULATOR_LAYOUT: gl.constexpr = MmacInterleave2Layout(MFMA_LAYOUT)
    A_OPERAND_LAYOUT: gl.constexpr = gl.DotOperandLayout(0, MFMA_LAYOUT, 4)
    B_OPERAND_LAYOUT: gl.constexpr = MmacInterleave2Layout(
        gl.DotOperandLayout(1, MFMA_LAYOUT, 4),
    )
    a_slice0 = a_buffer0.slice(
        CONSUMER_ID * ROWS_PER_CONSUMER, ROWS_PER_CONSUMER, dim=0,
    )
    a_slice1 = a_buffer1.slice(
        CONSUMER_ID * ROWS_PER_CONSUMER, ROWS_PER_CONSUMER, dim=0,
    )
    # Each 128-column panel gives every N wave one native 64-column MLS tile.
    b_buffer0_panel0 = b_buffer0.slice(0, 128, dim=1)
    b_buffer0_panel1 = b_buffer0.slice(128, 128, dim=1)
    b_buffer1_panel0 = b_buffer1.slice(0, 128, dim=1)
    b_buffer1_panel1 = b_buffer1.slice(128, 128, dim=1)
    NUM_N_TILES: gl.constexpr = (N + 255) // 256
    NUM_OUTPUT_TILES: gl.constexpr = ((M + 255) // 256) * NUM_N_TILES
    NUM_K_TILES: gl.constexpr = (K + 63) // 64
    buffer_index = 0
    barrier_phase = 0
    for tile_id in range(gl.program_id(0), NUM_OUTPUT_TILES, NUM_THREAD_GROUPS):
        accumulator0 = gl.full(
            (ROWS_PER_CONSUMER, 128), 0, gl.float32, ACCUMULATOR_LAYOUT,
        )
        accumulator1 = gl.full(
            (ROWS_PER_CONSUMER, 128), 0, gl.float32, ACCUMULATOR_LAYOUT,
        )
        # Stagger consumer 1 so consumer 0's exit releases its next MMAC turn.
        if CONSUMER_ID == 1:
            _emit_scalar_asm(_SYNC_CONSUMERS_ASM)
        for k_tile in range(NUM_K_TILES):
            abarrier_wait(_FULL_BARRIER_BASE + buffer_index * 2, barrier_phase)
            if buffer_index == 0:
                a_values = async_copy.load_shared_relaxed(a_slice0, A_OPERAND_LAYOUT)
                b_values0 = async_copy.load_shared_relaxed(
                    b_buffer0_panel0, B_OPERAND_LAYOUT,
                )
                b_values1 = async_copy.load_shared_relaxed(
                    b_buffer0_panel1, B_OPERAND_LAYOUT,
                )
            else:
                a_values = async_copy.load_shared_relaxed(a_slice1, A_OPERAND_LAYOUT)
                b_values0 = async_copy.load_shared_relaxed(
                    b_buffer1_panel0, B_OPERAND_LAYOUT,
                )
                b_values1 = async_copy.load_shared_relaxed(
                    b_buffer1_panel1, B_OPERAND_LAYOUT,
                )
            # Both consumers must finish reading before the producer reuses LDS.
            _emit_scalar_asm(_WAIT_LDS_READS_ASM)
            abarrier_arrive(_EMPTY_BARRIER_BASE + buffer_index * 2)
            sched_barrier(0)
            _emit_scalar_asm(_BEGIN_MMAC_TURN_ASM)
            accumulator0 = gl.amd.cdna4.mfma(a_values, b_values0, accumulator0)
            accumulator1 = gl.amd.cdna4.mfma(a_values, b_values1, accumulator1)
            if CONSUMER_ID == 1:
                # The final turn has no peer left for another rendezvous.
                if k_tile + 1 < NUM_K_TILES:
                    _emit_scalar_asm(_END_MMAC_TURN_ASM)
                else:
                    _emit_scalar_asm(_RESET_PRIORITY_ASM)
            else:
                _emit_scalar_asm(_END_MMAC_TURN_ASM)
            buffer_index, barrier_phase = _advance_buffer_phase(
                buffer_index, barrier_phase,
            )
        row_offset = (tile_id // NUM_N_TILES) * 256 + CONSUMER_ID * ROWS_PER_CONSUMER
        col_offset = (tile_id % NUM_N_TILES) * 256
        for panel_id in gl.static_range(2):
            accumulator = accumulator0 if panel_id == 0 else accumulator1
            panel_col_offset = col_offset + panel_id * 128
            if M % 256 == 0 and N % 256 == 0 and not FORCE_MASKED_STORE:
                matrix_store(
                    C, _cast_for_matrix_store(accumulator, OUTPUT_DTYPE),
                    (M, N), (N, 1), (row_offset, panel_col_offset), tile=(32, 32),
                )
            else:
                rows = row_offset + gl.arange(
                    0, ROWS_PER_CONSUMER, layout=gl.SliceLayout(1, ACCUMULATOR_LAYOUT),
                )
                cols = panel_col_offset + gl.arange(
                    0, 128, layout=gl.SliceLayout(0, ACCUMULATOR_LAYOUT),
                )
                gl.store(
                    C + rows[:, None] * N + cols[None, :], accumulator.to(OUTPUT_DTYPE),
                    (rows[:, None] < M) & (cols[None, :] < N),
                )
    _finalize_pipeline_barriers(1)


@g.jit
def _idle_default_partition():
    pass


@g.jit
def _gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel(
    A,
    B,
    C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    NUM_SMS: gl.constexpr,
    B_K_CONTIGUOUS: gl.constexpr = True,
    FORCE_MASKED_STORE: gl.constexpr = False,
):
    """Run a double-buffered persistent GEMM with two consumer partitions."""
    gl.static_assert(
        (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) == (256, 256, 64),
        "this pipeline requires tile (256,256,64)",
    )
    gl.static_assert(NUM_SMS > 0, "the device CU count must be positive")
    gl.static_assert(M > 0 and N > 0 and K > 0)
    NUM_THREAD_GROUPS: gl.constexpr = min(
        NUM_SMS, ((M + 255) // 256) * ((N + 255) // 256),
    )
    OUTPUT_DTYPE: gl.constexpr = A.dtype.element_ty
    gl.static_assert(OUTPUT_DTYPE == gl.float16 or OUTPUT_DTYPE == gl.bfloat16)
    gl.static_assert(B.dtype.element_ty == OUTPUT_DTYPE)
    gl.static_assert(C.dtype.element_ty == OUTPUT_DTYPE)
    A_SHARED_LAYOUT: gl.constexpr = MlsSharedLayout(0, (64, 64), bits=16)
    B_SHARED_LAYOUT: gl.constexpr = MlsSharedLayout(
        1, (64, 64), bits=16, alt=1,
        order=(0, 1) if B_K_CONTIGUOUS else (1, 0),
    )
    MFMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[2, 2], tiles_per_warp=[4, 4],
    )
    a_buffer0 = gl.allocate_shared_memory(OUTPUT_DTYPE, (256, 64), A_SHARED_LAYOUT)
    a_buffer1 = gl.allocate_shared_memory(OUTPUT_DTYPE, (256, 64), A_SHARED_LAYOUT)
    b_buffer0 = gl.allocate_shared_memory(OUTPUT_DTYPE, (64, 256), B_SHARED_LAYOUT)
    b_buffer1 = gl.allocate_shared_memory(OUTPUT_DTYPE, (64, 256), B_SHARED_LAYOUT)
    gl.warp_specialize(
        [
            (_idle_default_partition, ()),
            (
                _persistent_producer,
                (A, B, M, N, K, a_buffer0, a_buffer1, b_buffer0, b_buffer1,
                 NUM_THREAD_GROUPS, B_K_CONTIGUOUS),
            ),
            (
                _persistent_consumer,
                (C, M, N, K, a_buffer0, a_buffer1, b_buffer0, b_buffer1,
                 MFMA_LAYOUT, OUTPUT_DTYPE, 0, NUM_THREAD_GROUPS, FORCE_MASKED_STORE),
            ),
            (
                _persistent_consumer,
                (C, M, N, K, a_buffer0, a_buffer1, b_buffer0, b_buffer1,
                 MFMA_LAYOUT, OUTPUT_DTYPE, 1, NUM_THREAD_GROUPS, FORCE_MASKED_STORE),
            ),
        ],
        [4, 4, 4], [16, 244, 244],
    )


# Two-producer/two-consumer Stream-K pipeline. The host launcher reserves
# 96 KiB dynamic LDS for the explicitly addressed, double-buffered MLS ring.
@triton.constexpr_function
def _streamk_accumulator_registers():
    # One 4-register fragment per (M16, N16), in the Gluon accumulator layout.
    return tuple(61 - 4 * i + j for i in range(16) for j in range(4))


@triton.constexpr_function
def _streamk_k64_asm():
    # pack=64: outputs 0..63, tied accumulator inputs 64..127, then seven
    # scalar arguments with 64 repeated operands each. Only the first scalar
    # operand of each group is referenced by the assembly.
    lines = ['s_abarrier_try_wait s86, $320, $512']
    for slot in (0, 4, 1, 5, 2, 6, 3, 7):
        base = 68 + 4 * slot
        offset = (slot % 4) * 1024 + (slot // 4) * 4096
        lines.append(f'ds_read_matrix_trans_format v[{base}:{base+3}], '
                     f'$128 offset:{offset} element:2 row:2 col:1')
    # Issue B while the A LDS reads are outstanding.
    lines.append('s_abarrier_try_wait s86, $448, $512')
    for slot, offset in enumerate((0, 1024, 4096, 5120, 2048, 3072, 6144, 7168)):
        base = 100 + 4 * slot
        lines.append(f'ds_read_matrix_format v[{base}:{base+3}], '
                     f'$192 offset:{offset} element:2 row:2 col:1')
    # A reads precede B reads. Release each ring slot only after its reads
    # complete. Keep the measured wait sequence, including lgkmcnt(4).
    lines += ['s_waitcnt lgkmcnt(8)', 's_abarrier_arrive s86, $256, 1',
              's_waitcnt lgkmcnt(4)', 's_waitcnt lgkmcnt(0)',
              's_abarrier_arrive s86, $384, 1']
    count = 0
    for half in range(2):
        for m in range(4):
            for k in range(2):
                for n in (3, 2, 1, 0):
                    c = 61 - 4 * (4 * m + n)
                    a = 68 + 16 * half + 4 * m + 2 * k
                    b = 100 + 16 * half + (0, 2, 8, 10)[n] + 4 * k
                    lines.append(f'v_mmac_f32_16x16x16_bf16 v[{c}:{c+3}], '
                                 f'v[{a}:{a+1}], v[{b}:{b+1}], v[{c}:{c+3}] lit')
                    count += 1
                    if count == 24:
                        lines.append('s_setprio 3')
    lines.append('s_setprio 0')
    return '\n'.join(lines)


@triton.constexpr_function
def _streamk_k64_constraints():
    return ','.join([f'={{v{i}}}' for i in _streamk_accumulator_registers()]
                    + [str(i) for i in range(64)] + ['s'] * (7 * 64)
                    + [f'~{{v{i}}}' for i in range(68, 132)]
                    + ['~{s86}', '~{memory}'])


@g.jit
def _streamk_consume_k64(acc, a_address, b_address, a_empty, a_ready, b_empty, b_ready, phase):
    return gl.inline_asm_elementwise(
        _streamk_k64_asm(), _streamk_k64_constraints(),
        [acc, a_address, b_address, a_empty, a_ready, b_empty, b_ready, phase],
        dtype=gl.float32, is_pure=False, pack=64)


@g.jit
def _streamk_wave_id():
    return gl.inline_asm_elementwise('s_getreg_b32 $0, hwreg(HW_REG_HW_ID1, 0, 5)',
                                    '=s', [], dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _streamk_rendezvous():
    gl.inline_asm_elementwise(
        's_waitcnt lgkmcnt(0)\ns_ebarrier_sync 14, 12\ns_mov_b32 $0, 0',
        '=s,~{memory}', [], dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _streamk_initialize(GROUP: gl.constexpr):
    if GROUP == 0:
        if _streamk_wave_id() == 0:
            gl.inline_asm_elementwise(
                's_abarrier_init 0, 4\ns_abarrier_init 1, 2\n'
                's_abarrier_init 2, 4\ns_abarrier_init 3, 2\n'
                's_abarrier_init 4, 4\ns_abarrier_init 5, 2\n'
                's_abarrier_init 6, 4\ns_abarrier_init 7, 2\n'
                's_abarrier_init 8, 8\ns_abarrier_init 9, 2\n'
                's_abarrier_init 10, 8\ns_abarrier_init 11, 2\ns_mov_b32 $0, 0',
                '=s,~{memory}', [], dtype=gl.int32, is_pure=False, pack=1)
    _streamk_rendezvous()
    if GROUP == 0:
        gl.inline_asm_elementwise(
            's_abarrier_arrive $0, 0, 2\ns_abarrier_arrive $0, 2, 2\n'
            's_abarrier_arrive $0, 4, 2\ns_abarrier_arrive $0, 6, 2\n'
            's_abarrier_arrive $0, 8, 4\ns_abarrier_arrive $0, 10, 4',
            '=s,~{memory}', [], dtype=gl.int32, is_pure=False, pack=1)
    _streamk_rendezvous()


@g.jit
def _streamk_wait_barrier(barrier, phase):
    gl.inline_asm_elementwise('s_abarrier_try_wait $0, $1, $2',
                             '=s,s,s,~{memory}', [barrier, phase],
                             dtype=gl.int32, is_pure=False, pack=1)


@triton.constexpr_function
def _streamk_matrix_asm(priority):
    lines = ['s_mov_b64 s[80:81], $1', 's_and_b32 s81, s81, 0xffff',
             's_mov_b32 s82, $4', 's_mov_b32 s83, 0x10000',
             's_add_i32 s84, $2, 1024', 's_add_i32 s85, $2, 2048',
             's_add_i32 s86, $2, 3072']
    if priority:
        lines.append('s_setprio 1')
    lines += ['s_abarrier_seq $3',
              'matrix_load_64x16_b16 s[80:83] $2 t bps lds',
              'matrix_load_64x16_b16 s[80:83] s84 moffset:16 t bps lds',
              'matrix_load_64x16_b16 s[80:83] s85 moffset:32 t bps lds',
              'matrix_load_64x16_b16 s[80:83] s86 moffset:48 t bps lds',
              's_abarrier_arrive $0, $3, 1']
    if priority:
        lines.append('s_setprio 0')
    return '\n'.join(lines)


@g.jit
def _streamk_matrix_to_lds(pointer, address, ready, STRIDE: gl.constexpr, IS_B: gl.constexpr):
    gl.inline_asm_elementwise(
        _streamk_matrix_asm(IS_B),
        '=s,s,s,s,s,~{s80},~{s81},~{s82},~{s83},~{s84},~{s85},~{s86},~{scc},~{memory}',
        [pointer, address, ready, STRIDE], dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _streamk_load_task(Schedule, index, RECORDS: gl.constexpr):
    ptr = Schedule + gl.program_id(0) * RECORDS * 4 + index * 4
    tile, start, length, tag = gl.inline_asm_elementwise(
        's_setprio 0\ns_load_dwordx4 s[80:83], $4, 0\ns_waitcnt lgkmcnt(0)',
        '={s80},={s81},={s82},={s83},s,~{memory}', [ptr],
        dtype=(gl.int32, gl.int32, gl.int32, gl.int32), is_pure=False, pack=1)
    return tile, start, length, tag


@g.jit
def _streamk_quad_addresses(offsets):
    even, odd = gl.split(gl.reshape(offsets, (128, 64, 2)))
    base, ignored = gl.split(gl.reshape(even, (128, 32, 2)))
    doubled = gl.reshape(gl.join(base, base), (128, 64))
    expanded = gl.reshape(gl.join(doubled, doubled), (128, 128))
    return gl.convert_layout(expanded, offsets.type.layout)


@triton.constexpr_function
def _streamk_epilogue_asm(output_bytes):
    # Tie the accumulator across the whole epilogue so the allocator does not
    # need a second live 64-register copy at the split-K control-flow merge.
    # Scalar arguments: output, peer workspace, flag, epoch, tag. Vector
    # arguments: repeated quad offsets, already scaled to byte addresses.
    lines = ['s_cmp_gt_i32 $384, 0',
             's_cbranch_scc1 .Lall_bps_publish_${:uid}',
             's_cmp_eq_i32 $384, 0',
             's_cbranch_scc1 .Lall_bps_store_${:uid}',
             's_mov_b64 s[80:81], $256', 's_and_b32 s81, s81, 0xffff',
             's_mov_b32 s82, -1', 's_mov_b32 s83, 0x27000',
             's_mov_b64 s[84:85], exec', 's_mov_b64 exec, 1',
             '.Lall_bps_acquire_${:uid}:',
             'buffer_load_dword v65, off, s[80:83], 0 glc',
             's_waitcnt vmcnt(0)', 'v_readfirstlane_b32 s86, v65',
             's_cmp_eq_i32 s86, $320',
             's_cbranch_scc0 .Lall_bps_acquire_${:uid}',
             's_mov_b64 exec, s[84:85]',
             's_mov_b64 s[80:81], $192', 's_and_b32 s81, s81, 0xffff']
    for i in range(16):
        base = 65 + 4 * i
        lines.append(f'buffer_load_dwordx4 v[{base}:{base+3}], '
                     f'${448+4*i}, s[80:83], 0 offen glc')
    lines.append('s_waitcnt vmcnt(0)')
    for i in range(16):
        c, peer = 61 - 4 * i, 65 + 4 * i
        for j in (0, 2):
            lines.append(f'v_pk_add_f32 v[{c+j}:{c+j+1}], '
                         f'v[{c+j}:{c+j+1}], v[{peer+j}:{peer+j+1}]')
    lines += ['.Lall_bps_store_${:uid}:',
              's_mov_b64 s[80:81], $128', 's_and_b32 s81, s81, 0xffff',
              f's_mov_b32 s82, {output_bytes}', 's_mov_b32 s83, 0x27000']
    # Reverse packing order, with packed values above every FP32 accumulator.
    for i in reversed(range(32)):
        c = 61 - 4 * (i // 2) + 2 * (i % 2)
        lines.append(f'v_cvt_pk_bf16_f32 v{65+i}, v{c}, v{c+1}')
    for i in range(16):
        base = 65 + 2 * i
        lines.append(f'buffer_store_dwordx2 v[{base}:{base+1}], '
                     f'${512+4*i}, s[80:83], 0 offen')
    lines += ['s_branch .Lall_bps_done_${:uid}',
              '.Lall_bps_publish_${:uid}:',
              's_mov_b64 s[80:81], $192', 's_and_b32 s81, s81, 0xffff',
              's_mov_b32 s82, -1', 's_mov_b32 s83, 0x27000']
    for i in range(16):
        c = 61 - 4 * i
        lines.append(f'buffer_store_dwordx4 v[{c}:{c+3}], '
                     f'${448+4*i}, s[80:83], 0 offen glc')
    lines += ['s_waitcnt vmcnt(0)',
              's_mov_b64 s[80:81], $256', 's_and_b32 s81, s81, 0xffff',
              'v_mov_b32 v65, $320',
              's_mov_b64 s[84:85], exec', 's_mov_b64 exec, 1',
              'buffer_store_dword v65, off, s[80:83], 0 glc',
              's_waitcnt vmcnt(0)', 's_mov_b64 exec, s[84:85]',
              '.Lall_bps_done_${:uid}:']
    return '\n'.join(lines)


@triton.constexpr_function
def _streamk_epilogue_constraints():
    return ','.join([f'={{v{i}}}' for i in _streamk_accumulator_registers()]
                    + [str(i) for i in range(64)] + ['s'] * (5 * 64) + ['v'] * (2 * 64)
                    + [f'~{{v{i}}}' for i in range(65, 129)]
                    + ['~{s80}', '~{s81}', '~{s82}', '~{s83}',
                       '~{s84}', '~{s85}', '~{s86}', '~{scc}', '~{memory}'])


@g.jit
def _streamk_finish_task(acc, output, peer, flag, epoch, tag, local_offsets, output_offsets,
                OUTPUT_BYTES: gl.constexpr):
    gl.inline_asm_elementwise(
        _streamk_epilogue_asm(OUTPUT_BYTES), _streamk_epilogue_constraints(),
        [acc, output.to(gl.uint64), peer.to(gl.uint64), flag.to(gl.uint64), epoch, tag,
         _streamk_quad_addresses(local_offsets) * 4, _streamk_quad_addresses(output_offsets) * 2],
        dtype=gl.float32, is_pure=False, pack=64)


@triton.constexpr_function
def _streamk_edge_epilogue_asm():
    # Add one scalar bound operand, moving both repeated vector groups by 64.
    source = _streamk_epilogue_asm(0)
    for operand in reversed(range(448, 576)):
        source = source.replace(f'${operand}', f'${operand + 64}')
    return source.replace('s_mov_b32 s82, 0\ns_mov_b32 s83, 0x27000',
                          's_mov_b32 s82, $448\ns_mov_b32 s83, 0x27000')


@triton.constexpr_function
def _streamk_edge_epilogue_constraints():
    return _streamk_epilogue_constraints().replace(',s' * 320 + ',', ',s' * 384 + ',', 1)


@g.jit
def _streamk_finish_edge_task(acc, output, peer, flag, epoch, tag, limit, local_offsets, output_offsets):
    gl.inline_asm_elementwise(
        _streamk_edge_epilogue_asm(), _streamk_edge_epilogue_constraints(),
        [acc, output.to(gl.uint64), peer.to(gl.uint64), flag.to(gl.uint64), epoch, tag, limit,
         _streamk_quad_addresses(local_offsets) * 4, _streamk_quad_addresses(output_offsets) * 2],
        dtype=gl.float32, is_pure=False, pack=64)


@g.jit
def _streamk_producer(A, B, EdgeM, EdgeKA, EdgeKB, Schedule, M: gl.constexpr, N: gl.constexpr,
             K: gl.constexpr, RECORDS: gl.constexpr, GROUP: gl.constexpr):
    _streamk_initialize(GROUP)
    wid = _streamk_wave_id() - GROUP * 2
    task = 0
    counter = 0
    tile, start, length, tag = _streamk_load_task(Schedule, task, RECORDS)
    while length > 0:
        row = tile // (N // 128) * 256 + GROUP * 128 + wid * 64
        col = tile % (N // 128) * 128
        ap = A + row * K + start
        if M % 256:
            if tile // (N // 128) == M // 256:
                ap = EdgeM + (GROUP * 128 + wid * 64) * K + start
        bp = B + start * N + col + wid * 64
        for step in range(length // 64):
            stage = counter & 1
            phase = (counter >> 1) & 1
            # Shared B is submitted first by _streamk_producer partition 0. The other
            # _streamk_producer partition supplies A only, as in the measured kernel.
            if GROUP == 0:
                _streamk_wait_barrier(8 + stage * 2, phase)
                if K % 64:
                    if start + step * 64 >= K // 64 * 64:
                        _streamk_matrix_to_lds(EdgeKB + col + wid * 64,
                                      65536 + stage * 16384 + wid * 8192,
                                      9 + stage * 2, N, True)
                    else:
                        _streamk_matrix_to_lds(bp + step * 64 * N,
                                      65536 + stage * 16384 + wid * 8192,
                                      9 + stage * 2, N, True)
                else:
                    _streamk_matrix_to_lds(bp + step * 64 * N,
                                  65536 + stage * 16384 + wid * 8192,
                                  9 + stage * 2, N, True)
            _streamk_empty = GROUP * 4 + stage * 2
            _streamk_wait_barrier(_streamk_empty, phase)
            if K % 64:
                if start + step * 64 >= K // 64 * 64:
                    _streamk_matrix_to_lds(EdgeKA + row * 64,
                                  GROUP * 32768 + stage * 16384 + wid * 8192,
                                  _streamk_empty + 1, 64, False)
                else:
                    _streamk_matrix_to_lds(ap + step * 64,
                                  GROUP * 32768 + stage * 16384 + wid * 8192,
                                  _streamk_empty + 1, K, False)
            else:
                _streamk_matrix_to_lds(ap + step * 64,
                              GROUP * 32768 + stage * 16384 + wid * 8192,
                              _streamk_empty + 1, K, False)
            counter += 1
        task += 1
        tile, start, length, tag = _streamk_load_task(Schedule, task, RECORDS)
    _streamk_rendezvous()


@g.jit
def _streamk_consumer(C, Schedule, Workspace, Flags, Epoch, M: gl.constexpr, N: gl.constexpr,
             RECORDS: gl.constexpr, GROUP: gl.constexpr):
    _streamk_initialize(2 + GROUP)
    wid = _streamk_wave_id() - 4 - GROUP * 4
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], mmac_layout=gl.amd.MmacLayout.LIT,
        warps_per_cta=[2, 2], tiles_per_warp=[4, 4])
    rm = gl.arange(0, 128, gl.SliceLayout(1, mma))
    rn = gl.arange(0, 128, gl.SliceLayout(0, mma))
    task = 0
    counter = 0
    tile, start, length, tag = _streamk_load_task(Schedule, task, RECORDS)
    while length > 0:
        acc = gl.full((128, 128), 0, gl.float32, mma)
        for step in range(length // 64):
            stage = counter & 1
            phase = (counter >> 1) & 1
            _streamk_empty = GROUP * 4 + stage * 2
            acc = _streamk_consume_k64(
                acc, GROUP * 32768 + stage * 16384 + (wid >> 1) * 8192,
                65536 + stage * 16384 + (wid & 1) * 8192,
                _streamk_empty, _streamk_empty + 1, 8 + stage * 2, 9 + stage * 2, phase)
            counter += 1
        local = rm[:, None] * 128 + rn[None, :]
        offsets = rm[:, None] * N + rn[None, :]
        base = (tile // (N // 128) * 256 + GROUP * 128) * N + tile % (N // 128) * 128
        output = C + base
        slot = gl.abs(tag) - 1
        if M % 256:
            _streamk_finish_edge_task(acc, output, Workspace + slot * 32768 + GROUP * 16384,
                             Flags + slot * 8 + GROUP * 4 + wid, Epoch, tag,
                             gl.maximum((M * N - base) * 2, 0), local, offsets)
        else:
            _streamk_finish_task(acc, output, Workspace + slot * 32768 + GROUP * 16384,
                        Flags + slot * 8 + GROUP * 4 + wid, Epoch, tag, local, offsets,
                        0xffffffff)
        task += 1
        tile, start, length, tag = _streamk_load_task(Schedule, task, RECORDS)
    _streamk_rendezvous()


@g.jit
def _streamk_empty():
    pass


@g.jit
def _gemm_a16w16_bf16_tt_bm256_bn128_bk64_persistent_kernel(A, B, C, Schedule, Workspace, Flags, EdgeM, EdgeKA, EdgeKB, Epoch,
                    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, RECORDS: gl.constexpr):
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and C.dtype.element_ty == gl.bfloat16)
    gl.static_assert(M > 0 and N > 0 and K > 0 and M % 4 == 0 and N % 128 == 0 and K % 16 == 0)
    gl.warp_specialize([
        (_streamk_empty, ()),
        (_streamk_producer, (A, B, EdgeM, EdgeKA, EdgeKB, Schedule, M, N, K, RECORDS, 0)),
        (_streamk_producer, (A, B, EdgeM, EdgeKA, EdgeKB, Schedule, M, N, K, RECORDS, 1)),
        (_streamk_consumer, (C, Schedule, Workspace, Flags, Epoch, M, N, RECORDS, 0)),
        (_streamk_consumer, (C, Schedule, Workspace, Flags, Epoch, M, N, RECORDS, 1)),
    ], [2, 2, 4, 4], [48, 48, 144, 144])


@g.jit
def _gemm_a16w16_bf16_tt_prepare_edges_kernel(A, B, EdgeM, EdgeKA, EdgeKB, M: gl.constexpr, N: gl.constexpr,
                  K: gl.constexpr, BLOCK: gl.constexpr):
    """Pad only the final M tile and K64 slice; all other input stays in place."""
    pid = gl.program_id(0)
    x = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [64], [4], [0]))
    EDGE_M: gl.constexpr = 256 * K if M % 256 else 0
    EDGE_KA: gl.constexpr = ((M + 255) // 256) * 256 * 64 if K % 64 else 0
    M_BLOCKS: gl.constexpr = (EDGE_M + BLOCK - 1) // BLOCK
    A_BLOCKS: gl.constexpr = (EDGE_KA + BLOCK - 1) // BLOCK
    if pid < M_BLOCKS:
        offset = pid * BLOCK + x
        row, col = offset // K, offset % K
        value = gl.load(A + (M // 256 * 256 + row) * K + col,
                        (offset < EDGE_M) & (row < M % 256), other=0)
        gl.store(EdgeM + offset, value, offset < EDGE_M)
    elif pid < M_BLOCKS + A_BLOCKS:
        offset = (pid - M_BLOCKS) * BLOCK + x
        row, col = offset // 64, offset % 64
        value = gl.load(A + row * K + K // 64 * 64 + col,
                        (offset < EDGE_KA) & (row < M) & (col < K % 64), other=0)
        gl.store(EdgeKA + offset, value, offset < EDGE_KA)
    else:
        offset = (pid - M_BLOCKS - A_BLOCKS) * BLOCK + x
        row, col = offset // N, offset % N
        value = gl.load(B + (K // 64 * 64 + row) * N + col,
                        (offset < 64 * N) & (row < K % 64), other=0)
        gl.store(EdgeKB + offset, value, offset < 64 * N)
