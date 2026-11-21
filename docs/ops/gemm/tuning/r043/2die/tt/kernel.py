# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# SPDX-License-Identifier: MIT

import triton


from triton.experimental import gluon as g


from triton.experimental.gluon import language as gl


from triton.experimental.gluon.language.amd.cdna4 import async_copy


from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacInterleave2Layout, MmacLayout, matrix_load, matrix_store,
    sched_barrier, wait_lgkmcnt,
    abarrier_arrive, abarrier_seq, abarrier_init, abarrier_try_wait, ebarrier_sync, get_wave_id,
)


try:
    from triton.experimental.gluon.language.amd.hcu import abarrier_wait
except ImportError:
    from triton.experimental.gluon.language.amd.hcu import abarrier_try_wait as abarrier_wait


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
    A_BOUNDS: gl.constexpr = (0, 1)
    B_BOUNDS: gl.constexpr = (0, 1)
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
            if row_offset + ROWS_PER_CONSUMER <= M and panel_col_offset + 128 <= N and not FORCE_MASKED_STORE:
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
