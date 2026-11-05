"""Dense FP8 E4M3FN GEMM with FP32 M1/K128 and N128/K128 scales.

A[M,K], B[N,K], scales[M,ceil(K/128)] and [ceil(N/128),ceil(K/128)].
Complete tiles use the MLS pipeline; masked native MMAC handles other shapes.
"""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MmacLayout,
    MlsSharedLayout,
    matrix_load,
    matrix_store,
    MmacInterleave2Layout,
    get_lds_slice_address,
    get_wave_id,
    abarrier_init,
    abarrier_wait as aw,
    abarrier_seq as seq,
    abarrier_arrive as arrive,
    ebarrier_arrive,
    ebarrier_sync,
    sched_barrier,
    packed_scale_fma,
)


@g.jit
def initialize(wave):
    if wave == 0:
        abarrier_init(0, 4)
        abarrier_init(1, 4)
        abarrier_init(2, 4)
        abarrier_init(3, 4)
        abarrier_init(6, 4)
        abarrier_init(7, 4)
        abarrier_init(8, 8)
        abarrier_init(9, 4)
        abarrier_init(10, 4)
        abarrier_init(11, 4)
        abarrier_init(12, 8)
        abarrier_init(13, 4)
        abarrier_init(14, 8)
    if wave < 4:
        ebarrier_arrive(8, 8)
    ebarrier_sync(14, 16)


@g.jit
def drain():
    gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\n\ts_mov_b32 $0, 0",
        constraints="=s,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def turn(GROUP: gl.constexpr, WAIT: gl.constexpr):
    if WAIT:
        ebarrier_sync(8 + GROUP, 8)
    else:
        ebarrier_arrive(9 - GROUP, 8)
        gl.inline_asm_elementwise(
            "s_setprio 0\n\ts_mov_b32 $0, 0",
            constraints="=s,~{memory}", args=[], dtype=gl.int32,
            is_pure=False, pack=1,
        )


@g.jit
def empty():
    pass


@g.jit
def producer(
    A,
    B,
    AS,
    BS,
    smAS0,
    smAS1,
    smBS,
    smA0,
    smA1,
    smB0,
    smB1,
    K: gl.constexpr,
    GROUP: gl.constexpr,
):
    wave = get_wave_id()
    initialize(wave)
    SL: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    for i in range(K // 128):
        if i % 2 == 0:
            smA = smA0
            smB = smB0
        else:
            smA = smA1
            smB = smB1
        slot = i % 2
        if GROUP == 0:
            if i >= 2:
                aw(slot * 2, ((i // 2) - 1) % 2)
            seq(slot * 2 + 1)
            matrix_load(smA, A, (64, K), (K, 1), (0, i * 128), pred=wave % 4 == 0)
            arrive(slot * 2 + 1)
            if i >= 2:
                aw(8 + slot * 4, ((i // 2) - 1) % 2)
            seq(9 + slot * 4)
            matrix_load(smB, B, (K, 256), (1, K), (i * 128, 0))
            arrive(9 + slot * 4)
        else:
            if i >= 2:
                aw(6 + slot * 4, ((i // 2) - 1) % 2)
            seq(7 + slot * 4)
            matrix_load(
                smA, A + 64 * K, (64, K), (K, 1), (0, i * 128), pred=wave % 4 == 0
            )
            ar = gl.arange(0, 128, layout=SL)
            async_copy.buffer_load_to_shared(smAS0.index(i), AS, ar * (K // 128) + i)
            bi = gl.arange(0, 2, layout=SL)
            async_copy.buffer_load_to_shared(smBS.index(i), BS, bi * (K // 128) + i)
            arrive(7 + slot * 4)
            aw(14, i % 2)


@g.jit
def wait_ab():
    gl.inline_asm_elementwise(
        "s_setprio 3\n\ts_waitcnt lgkmcnt(9)\n\ts_mov_b32 $0, 0",
        constraints="=s,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def ready_a(i, GROUP: gl.constexpr):
    if GROUP == 0:
        aw((i % 2) * 2 + 1, (i // 2) % 2)
    else:
        aw(7 + (i % 2) * 4, (i // 2) % 2)


@g.jit
def ready_b(i):
    aw(9 + (i % 2) * 4, (i // 2) % 2)


@g.jit
def ready_scale(i, GROUP: gl.constexpr):
    if GROUP == 0:
        aw(7 + (i % 2) * 4, (i // 2) % 2)
    # ACK means readiness observed, not completion of pending LDS reads.
    # Scale slices are immutable; producer writes the following distinct slice.
    gl.inline_asm_elementwise(
        "s_abarrier_arrive $0, 14, 1",
        constraints="=s",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def release(i, GROUP: gl.constexpr):
    wait_ab()
    if GROUP == 0:
        arrive((i % 2) * 2)
    else:
        arrive(6 + (i % 2) * 4)
    arrive(8 + (i % 2) * 4)


@g.jit
def scales(
    smAS0,
    smAS1,
    smBS,
    i,
    m,
    lane_m_byte,
    bs_group,
    GROUP: gl.constexpr,
    F: gl.constexpr,
):
    base = get_lds_slice_address(smAS0.index(i)) + GROUP * 256
    base = base + lane_m_byte
    sa = gl.load(base.to(gl.uint64).to(gl.pointer_type(gl.float32, 3)) + (m // 4) * 4)
    bp = get_lds_slice_address(smBS.index(i))
    sb = gl.inline_asm_elementwise(
        "ds_read_b64 $0, $1",
        constraints="=v,v,~{memory}",
        args=[bp],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )
    return sa, sb


@g.jit
def select_bs(sb, bs_group):
    lo = sb.to(gl.uint32)
    hi = (sb >> 32).to(gl.uint32)
    selected = gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\n\tv_cmp_ne_u32_e64 vcc, 0, $3\n\tv_cndmask_b32_e32 $0, $1, $2, vcc",
        constraints="=v,v,v,v,~{vcc},~{memory}",
        args=[lo, hi, bs_group],
        dtype=gl.uint32,
        is_pure=False,
        pack=1,
    )
    return selected.to(gl.float32, bitcast=True)


@g.jit
def consumer(
    AS,
    BS,
    C,
    smAS0,
    smAS1,
    smBS,
    smA0,
    smA1,
    smB0,
    smB1,
    K: gl.constexpr,
    N: gl.constexpr,
    GROUP: gl.constexpr,
    F: gl.constexpr,
):
    wave = get_wave_id()
    initialize(wave)
    RF: gl.constexpr = MmacInterleave2Layout(F)
    acc = gl.full((64, 256), 0, gl.float32, RF)
    zero_mmac = gl.inline_asm_elementwise(
        "v_mov_b32 $0, 0\n\tv_mov_b32 $1, 0\n\tv_mov_b32 $2, 0\n\tv_mov_b32 $3, 0",
        constraints="=v,=v,=v,=v,v,v,v,v",
        args=[gl.full((64, 256), 0, gl.int32, RF)],
        dtype=gl.float32,
        is_pure=True,
        pack=4,
    )
    m = gl.arange(0, 64, layout=gl.SliceLayout(1, RF))
    bs_group = (wave % 4) // 2
    lane_m_byte = gl.inline_asm_elementwise(
        "v_mbcnt_lo_u32_b32 $0, -1, 0\n\tv_mbcnt_hi_u32_b32 $0, -1, $0\n\tv_bfe_u32 $0, $0, 4, 2\n\tv_lshlrev_b32 $0, 2, $0",
        constraints="=v",
        args=[],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )
    ready_a(0, GROUP)
    a = async_copy.load_shared_relaxed(smA0, gl.DotOperandLayout(0, F, 8))
    ready_b(0)
    b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
    sched_barrier(0)
    ready_scale(0, GROUP)
    sa, sb = scales(smAS0, smAS1, smBS, 0, m, lane_m_byte, bs_group, GROUP, RF)
    release(0, GROUP)
    for pair in range(K // 256):
        i = pair * 2
        sched_barrier(0)
        turn(GROUP, True)
        dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
        turn(GROUP, False)
        selected_bs = select_bs(sb, bs_group)
        j = i + 1
        ready_a(j, GROUP)
        if j % 2 == 0:
            a = async_copy.load_shared_relaxed(smA0, gl.DotOperandLayout(0, F, 8))
        else:
            a = async_copy.load_shared_relaxed(smA1, gl.DotOperandLayout(0, F, 8))
        ready_b(j)
        if j % 2 == 0:
            b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
        else:
            b = async_copy.load_shared_relaxed(smB1, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
        sched_barrier(0)
        ready_scale(j, GROUP)
        next_sa, next_sb = scales(
            smAS0, smAS1, smBS, j, m, lane_m_byte, bs_group, GROUP, RF
        )
        acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
        release(j, GROUP)
        sa = next_sa
        sb = next_sb
        sched_barrier(0)
        if pair + 1 < K // 256:
            i = pair * 2 + 1
            sched_barrier(0)
            turn(GROUP, True)
            dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
            turn(GROUP, False)
            selected_bs = select_bs(sb, bs_group)
            j = i + 1
            ready_a(j, GROUP)
            if j % 2 == 0:
                a = async_copy.load_shared_relaxed(smA0, gl.DotOperandLayout(0, F, 8))
            else:
                a = async_copy.load_shared_relaxed(smA1, gl.DotOperandLayout(0, F, 8))
            ready_b(j)
            if j % 2 == 0:
                b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
            else:
                b = async_copy.load_shared_relaxed(smB1, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
            sched_barrier(0)
            ready_scale(j, GROUP)
            next_sa, next_sb = scales(
                smAS0, smAS1, smBS, j, m, lane_m_byte, bs_group, GROUP, RF
            )
            acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
            release(j, GROUP)
            sa = next_sa
            sb = next_sb
            sched_barrier(0)
    turn(GROUP, True)
    dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
    turn(GROUP, False)
    selected_bs = select_bs(sb, bs_group)
    acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
    packed = acc.to(gl.bfloat16).to(gl.float16, bitcast=True)
    matrix_store(
        C.to(gl.pointer_type(gl.float16)),
        packed,
        (128, N),
        (N, 1),
        (GROUP * 64, 0),
        tile=(32, 32),
        interleave=1,
        version=3,
        order=(1, 0),
    )


@g.jit
def kernel(
    A,
    B,
    AS,
    BS,
    C,
    N: gl.constexpr,
    K: gl.constexpr,
    SA: gl.constexpr,
    SB: gl.constexpr,
    F: gl.constexpr,
):
    tile_m = gl.program_id(0) * 128
    tile_n = gl.program_id(1) * 256
    A += tile_m * K
    B += tile_n * K
    AS += tile_m * (K // 128)
    BS += (tile_n // 128) * (K // 128)
    C += tile_m * N + tile_n
    scaleL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    sas0 = gl.allocate_shared_memory(gl.float32, (K // 128, 128), scaleL)
    sas1 = sas0
    sbs = gl.allocate_shared_memory(gl.float32, (K // 128, 2), scaleL)
    a0 = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a1 = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a0_next = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a1_next = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    b_next = gl.allocate_shared_memory(gl.float8e4nv, (128, 256), SB)
    b = gl.allocate_shared_memory(gl.float8e4nv, (128, 256), SB)
    gl.warp_specialize(
        [
            (empty, ()),
            (producer, (A, B, AS, BS, sas0, sas1, sbs, a0, a0_next, b, b_next, K, 0)),
            (producer, (A, B, AS, BS, sas0, sas1, sbs, a1, a1_next, b, b_next, K, 1)),
            (
                consumer,
                (AS, BS, C, sas0, sas1, sbs, a0, a0_next, b, b_next, K, N, 0, F),
            ),
            (
                consumer,
                (AS, BS, C, sas0, sas1, sbs, a1, a1_next, b, b_next, K, N, 1, F),
            ),
        ],
        [4, 4, 4, 4],
        [8, 8, 240, 240],
    )


def launch_config():
    """Fixed MLS pipeline configuration for complete tiles."""
    # Native interleave2 B/C mappings avoid register shuffles at both transfers.
    sb = MlsSharedLayout(1, (128, 64), alt=1, order=(0, 1))
    return dict(
        SA=MlsSharedLayout(0, (64, 128)),
        SB=sb,
        F=gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            mmac_layout=MmacLayout.LTS,
            warps_per_cta=[1, 4],
            tiles_per_warp=[4, 4],
        ),
        num_warps=4,
        num_stages=2,
        wasp_partition_warps=(4, 4, 4, 4),
        wasp_partition_regs=(8, 8, 240, 240),
        wasp_wdra=True,
    )


@g.jit
def _masked_kernel(A, B, AS, BS, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        mmac_layout=MmacLayout.LTS,
        warps_per_cta=[1, 4],
        tiles_per_warp=[4, 2],
    )
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    m = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, L))
    kk_a = gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    kk_b = gl.arange(0, 128, layout=gl.SliceLayout(1, L))
    n = gl.program_id(1) * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    rm = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, F))
    rn = gl.program_id(1) * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, F))
    groups: gl.constexpr = gl.cdiv(K, 128)
    acc = gl.full((64, 128), 0, gl.float32, F)
    for i in range(groups):
        ka = i * 128 + kk_a
        kb = i * 128 + kk_b
        a = gl.load(
            A.to(gl.pointer_type(gl.uint8)) + m[:, None].to(gl.int64) * K + ka[None, :],
            (m[:, None] < M) & (ka[None, :] < K),
            other=0,
        )
        b = gl.load(
            B.to(gl.pointer_type(gl.uint8)) + n[None, :].to(gl.int64) * K + kb[:, None],
            (n[None, :] < N) & (kb[:, None] < K),
            other=0,
        )
        a = gl.convert_layout(a.to(gl.float8e4nv, bitcast=True), gl.DotOperandLayout(0, F, 8))
        b = gl.convert_layout(b.to(gl.float8e4nv, bitcast=True), gl.DotOperandLayout(1, F, 8))
        dot = gl.amd.cdna4.mfma(a, b, gl.full((64, 128), 0, gl.float32, F))
        sa = gl.load(AS + rm * groups + i, rm < M, other=0)
        sb = gl.load(BS + gl.program_id(1) * groups + i)
        acc = packed_scale_fma(dot, sa[:, None], sb, acc)
    gl.store(
        C + rm[:, None] * N + rn[None, :],
        acc.to(gl.bfloat16),
        (rm[:, None] < M) & (rn[None, :] < N),
    )


def _uses_mls_pipeline(m, n, k, a, b):
    # A/B double buffering occupies 96 KiB; immutable scale slices grow with K.
    # K6144 uses 120.375 KiB. The two-iteration pipeline requires full pairs.
    return (
        m % 128 == 0
        and n % 256 == 0
        and 256 <= k <= 6144
        and k % 256 == 0
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
    )


def gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16(
    a, b, a_scale, b_scale, *, scale_recipe=((1, 128), (128, 128)), out_dtype=None
):
    """Compatibility name for gemm_block_scaled; defaults preserve the old recipe.

    Other dtypes, scale blocks and output dtypes use the general native path.
    """
    from .gemm_block_scaled import gemm_block_scaled
    return gemm_block_scaled(a, b, a_scale, b_scale, scale_recipe=scale_recipe, out_dtype=out_dtype)
