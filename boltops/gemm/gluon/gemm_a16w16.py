# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Gluon dense BF16/FP16 TN GEMM kernels, selected by tile name.

Two entries share one public contract: x[M,K] and w[N,K], with the logical
transpose of w handled inside the kernel and no operand copy. R001 requires
K-contiguous x/w (TN). The persistent entry also accepts N-contiguous w (TT).

R001 is a fixed 64x64x2048 pipeline with two LDS banks per operand tile and one
wave per TG. It is a small-shape kernel: 16 output TGs, 4 BK128 chunks per bank.

R020 is a 256x256x64 persistent pipeline for complete 2048x2048x4096 tiles. One
warp-specialized TG owns three 4-wave partitions (an idle default partition, a
matrix-load producer and two 128-row MMAC consumers) and walks output tiles in
grid strides, so 48 TGs cover the 64-tile grid in two passes.
"""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MlsSharedLayout, MmacInterleave2Layout, MmacLayout, matrix_load, matrix_store,
    sched_barrier, wait_lgkmcnt,
    abarrier_arrive, abarrier_seq, abarrier_wait,
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
def _gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel(
    A, B, C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr = 16,
    BLOCK_SIZE_N: gl.constexpr = 16,
    BLOCK_SIZE_K: gl.constexpr = 128,
    LOOP_UNROLL_FACTOR: gl.constexpr = 4,
):
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


# ---------------------------------------------------------------------------
# R020 persistent 256x256x64 pipeline.
#
# LDS holds two A/B slot pairs. The producer waits for a slot to be reported
# empty, fills it, then publishes it full; both consumers read that slot, each
# taking its own 128 rows of A. Slot phases, the abarrier identities and the
# 4/4/4 partition split are tied to NUM_SMS == 48, so all three are compile-time
# legality checks rather than tuning knobs.
# ---------------------------------------------------------------------------

_R020_EMPTY_ID = gl.constexpr(1)
_R020_FULL_ID = gl.constexpr(2)
_R020_EMPTY_COUNT = gl.constexpr(8)
_R020_FULL_COUNT = gl.constexpr(4)
_R020_SLOTS = gl.constexpr(2)
_R020_INIT_WAVE = gl.constexpr(4)
_R020_WAVE_ID = gl.constexpr("s_getreg_b32 $0, hwreg(HW_REG_HW_ID1, 0, 5)")
_R020_TG_SYNC = gl.constexpr("s_ebarrier_sync 15, 0\n\ts_mov_b32 $0, 0")
_R020_PART_SYNC = gl.constexpr("s_ebarrier_sync 4, 4\n\ts_mov_b32 $0, 0")
_R020_INVALIDATE = gl.constexpr(
    "s_abarrier_inv 1\n\ts_abarrier_inv 2\n\t"
    "s_abarrier_inv 3\n\ts_abarrier_inv 4\n\ts_mov_b32 $0, 0"
)
_R020_DRAIN = gl.constexpr("s_waitcnt lgkmcnt(0)\n\ts_mov_b32 $0, 0")
_R020_TURN_ENTER = gl.constexpr(
    "s_ebarrier_sync 1, 8\n\ts_setprio 1\n\ts_mov_b32 $0, 0"
)
_R020_TURN_EXIT = gl.constexpr("s_setprio 0\n\ts_ebarrier_sync 1, 8\n\ts_mov_b32 $0, 0")
_R020_TURN_ONLY = gl.constexpr("s_ebarrier_sync 1, 8\n\ts_mov_b32 $0, 0")
_R020_PRIO_ZERO = gl.constexpr("s_setprio 0\n\ts_mov_b32 $0, 0")


@triton.constexpr_function
def _r020_init_asm(barrier_id, count):
    return f"s_abarrier_init {barrier_id}, {count}\n\ts_mov_b32 $0, 0"


@triton.constexpr_function
def _r020_credit_asm(barrier_id, count):
    return f"s_abarrier_arrive $0, {barrier_id}, {count}"


@g.jit
def _r020_emit(code: gl.constexpr):
    gl.inline_asm_elementwise(code, constraints="=s,~{memory}", args=[],
                              dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _r020_wave_id():
    return gl.inline_asm_elementwise(_R020_WAVE_ID, constraints="=s", args=[],
                                     dtype=gl.int32, is_pure=True, pack=1)


@g.jit
def _r020_init_barrier(barrier_id: gl.constexpr, count: gl.constexpr,
                       credit: gl.constexpr, wave):
    INIT: gl.constexpr = _r020_init_asm(barrier_id, count)
    if wave == _R020_INIT_WAVE:
        _r020_emit(INIT)
    _r020_emit(_R020_PART_SYNC)
    if credit > 0:
        CREDIT: gl.constexpr = _r020_credit_asm(barrier_id, credit)
        gl.inline_asm_elementwise(CREDIT, constraints="=s,~{memory}", args=[],
                                  dtype=gl.int32, is_pure=False, pack=1)


@g.jit
def _r020_initialize(init: gl.constexpr):
    if init == 1:
        wave = _r020_wave_id()
        _r020_init_barrier(_R020_EMPTY_ID, _R020_EMPTY_COUNT, _R020_SLOTS, wave)
        _r020_init_barrier(_R020_FULL_ID, _R020_FULL_COUNT, 0, wave)
        _r020_init_barrier(_R020_EMPTY_ID + 2, _R020_EMPTY_COUNT, _R020_SLOTS, wave)
        _r020_init_barrier(_R020_FULL_ID + 2, _R020_FULL_COUNT, 0, wave)
    _r020_emit(_R020_TG_SYNC)


@g.jit
def _r020_finalize(invalidate: gl.constexpr):
    _r020_emit(_R020_TG_SYNC)
    if invalidate == 1:
        _r020_emit(_R020_INVALIDATE)


@g.jit
def _r020_advance(slot, phase):
    slot = (slot + 1) % _R020_SLOTS
    return slot, phase ^ (1 - slot)


@g.jit
def _r020_producer(A, B, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   sm_a0, sm_a1, sm_b0, sm_b1, NUM_SMS: gl.constexpr,
                   B_K_CONTIGUOUS: gl.constexpr):
    _r020_initialize(0)
    B_STRIDES: gl.constexpr = (1, K) if B_K_CONTIGUOUS else (N, 1)
    num_pid_n: gl.constexpr = N // 256
    num_tiles: gl.constexpr = (M // 256) * num_pid_n
    k_steps: gl.constexpr = K // 64
    slot = 0
    phase = 0
    for tile_id in range(gl.program_id(0), num_tiles, NUM_SMS):
        off_m = (tile_id // num_pid_n) * 256
        off_n = (tile_id % num_pid_n) * 256
        for k in range(k_steps):
            off_k = k * 64
            abarrier_wait(_R020_EMPTY_ID + slot * 2, phase)
            abarrier_seq(_R020_FULL_ID + slot * 2)
            if slot == 0:
                matrix_load(sm_a0, A, (M, K), (K, 1), (off_m, off_k), warps=(4, 1))
                matrix_load(sm_b0, B, (K, N), B_STRIDES, (off_k, off_n), warps=(1, 4))
            else:
                matrix_load(sm_a1, A, (M, K), (K, 1), (off_m, off_k), warps=(4, 1))
                matrix_load(sm_b1, B, (K, N), B_STRIDES, (off_k, off_n), warps=(1, 4))
            abarrier_arrive(_R020_FULL_ID + slot * 2)
            slot, phase = _r020_advance(slot, phase)
    _r020_finalize(0)


@g.jit
def _r020_consumer(C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                   sm_a0, sm_a1, sm_b0, sm_b1, F: gl.constexpr,
                   DTYPE: gl.constexpr, GROUP: gl.constexpr,
                   NUM_SMS: gl.constexpr):
    _r020_initialize(1 if GROUP == 0 else 0)
    GROUP_M: gl.constexpr = 128
    RF: gl.constexpr = MmacInterleave2Layout(F)
    AL: gl.constexpr = gl.DotOperandLayout(0, F, 4)
    BL: gl.constexpr = MmacInterleave2Layout(gl.DotOperandLayout(1, F, 4))
    a0 = sm_a0.slice(GROUP * GROUP_M, GROUP_M, dim=0)
    a1 = sm_a1.slice(GROUP * GROUP_M, GROUP_M, dim=0)
    num_pid_n: gl.constexpr = N // 256
    num_tiles: gl.constexpr = (M // 256) * num_pid_n
    k_steps: gl.constexpr = K // 64
    slot = 0
    phase = 0
    for tile_id in range(gl.program_id(0), num_tiles, NUM_SMS):
        acc = gl.full((GROUP_M, 256), 0, gl.float32, RF)
        # Offset group 1 by one rendezvous, so group 0's exit releases
        # group 1's entry rather than releasing both MMAC groups together.
        if GROUP == 1 and k_steps > 0:
            _r020_emit(_R020_TURN_ONLY)
        for k in range(k_steps):
            abarrier_wait(_R020_FULL_ID + slot * 2, phase)
            if slot == 0:
                a = async_copy.load_shared_relaxed(a0, AL)
                b = async_copy.load_shared_relaxed(sm_b0, BL)
            else:
                a = async_copy.load_shared_relaxed(a1, AL)
                b = async_copy.load_shared_relaxed(sm_b1, BL)
            # Both consumers must finish reading the slot before it is released.
            _r020_emit(_R020_DRAIN)
            abarrier_arrive(_R020_EMPTY_ID + slot * 2)
            sched_barrier(0)
            # One consumer holds the MMAC pipe at a time; the other waits here.
            _r020_emit(_R020_TURN_ENTER)
            acc = gl.amd.cdna4.mfma(a, b, acc)
            if GROUP == 1:
                # The final group-1 entry already pairs with group 0's final
                # exit. A further rendezvous would have no peer and deadlock.
                if k + 1 < k_steps:
                    _r020_emit(_R020_TURN_EXIT)
                else:
                    _r020_emit(_R020_PRIO_ZERO)
            else:
                _r020_emit(_R020_TURN_EXIT)
            slot, phase = _r020_advance(slot, phase)
        off_m = (tile_id // num_pid_n) * 256 + GROUP * GROUP_M
        off_n = (tile_id % num_pid_n) * 256
        matrix_store(C, acc.to(DTYPE), (M, N), (N, 1), (off_m, off_n), tile=(32, 32))
    _r020_finalize(1)


@g.jit
def _r020_idle():
    pass


@g.jit
def _gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel(
    A, B, C,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    NUM_SMS: gl.constexpr,
    B_K_CONTIGUOUS: gl.constexpr = True,
):
    """Persistent TN/TT GEMM; the historical TN symbol remains compatible."""
    gl.static_assert(
        (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) == (256, 256, 64),
        "this pipeline requires tile (256,256,64)")
    gl.static_assert(NUM_SMS == 48, "this pipeline requires 48 resident TGs")
    gl.static_assert(M % 256 == 0 and N % 256 == 0 and K % 64 == 0,
                     "this pipeline requires complete 256x256x64 output tiles")
    DTYPE: gl.constexpr = A.dtype.element_ty
    gl.static_assert(DTYPE == gl.float16 or DTYPE == gl.bfloat16)
    gl.static_assert(B.dtype.element_ty == DTYPE)
    gl.static_assert(C.dtype.element_ty == DTYPE)
    SA: gl.constexpr = MlsSharedLayout(0, (64, 64), bits=16)
    SB: gl.constexpr = MlsSharedLayout(
        1, (64, 64), bits=16, alt=1,
        order=(0, 1) if B_K_CONTIGUOUS else (1, 0))
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[2, 2], tiles_per_warp=[4, 8])
    sm_a0 = gl.allocate_shared_memory(DTYPE, (256, 64), SA)
    sm_a1 = gl.allocate_shared_memory(DTYPE, (256, 64), SA)
    sm_b0 = gl.allocate_shared_memory(DTYPE, (64, 256), SB)
    sm_b1 = gl.allocate_shared_memory(DTYPE, (64, 256), SB)
    gl.warp_specialize(
        [(_r020_idle, ()),
         (_r020_producer,
          (A, B, M, N, K, sm_a0, sm_a1, sm_b0, sm_b1, NUM_SMS, B_K_CONTIGUOUS)),
         (_r020_consumer,
          (C, M, N, K, sm_a0, sm_a1, sm_b0, sm_b1, F, DTYPE, 0, NUM_SMS)),
         (_r020_consumer,
          (C, M, N, K, sm_a0, sm_a1, sm_b0, sm_b1, F, DTYPE, 1, NUM_SMS))],
        [4, 4, 4], [16, 244, 244])
