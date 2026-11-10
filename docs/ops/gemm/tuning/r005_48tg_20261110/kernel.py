"""Fixed 64x256x128 MLS/WDRA48 pipeline for E4M3FN and FP32 K128 scales.

Tile geometry, LDS buffering and partition resources form one implementation.
Select a different pipeline to change them; get_launch_config is not a tile tuner.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy
from triton.experimental.gluon.language.amd.hcu import (
    MmacLayout,
    MlsSharedLayout,
    matrix_load,
    matrix_store,
    MmacInterleave2Layout,
    get_wave_id,
    set_priority,
    wait_lgkmcnt,
    abarrier_init,
    abarrier_try_wait,
    abarrier_seq,
    abarrier_arrive,
    ebarrier_arrive,
    ebarrier_sync,
    sched_barrier,
    packed_scale_fma,
)



@g.jit
def _initialize_pipeline_barriers(wave):
    if wave == 0:
        abarrier_init(0, 2)
        abarrier_init(1, 2)
        abarrier_init(2, 2)
        abarrier_init(3, 2)
        abarrier_init(4, 2)
        abarrier_init(5, 2)
        abarrier_init(6, 2)
        abarrier_init(7, 2)
        abarrier_init(8, 4)
        abarrier_init(9, 2)
        abarrier_init(10, 2)
        abarrier_init(11, 2)
        abarrier_init(12, 4)
        abarrier_init(13, 2)
        abarrier_init(14, 4)
    if wave < 2:
        ebarrier_arrive(8, 4)
    ebarrier_sync(14, 8)


@g.jit
def _wait_all_lds_reads():
    wait_lgkmcnt(0)


@g.jit
def _synchronize_mmac_turn(GROUP: gl.constexpr, WAIT: gl.constexpr):
    if WAIT:
        ebarrier_sync(8 + GROUP, 4)
    else:
        ebarrier_arrive(9 - GROUP, 4)
        set_priority(0)


@g.jit
def _idle_default_partition():
    pass


@g.jit
def _publish_scale_batch(AS, BS, smAS, smBS, start, ROWS, K: gl.constexpr, DEPTH: gl.constexpr, SLOT: gl.constexpr):
    SL: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
    ar = gl.arange(0, 64, layout=SL)
    bi = gl.arange(0, 2, layout=SL)
    abarrier_seq(SLOT)
    for item in gl.static_range(DEPTH):
        index = start + item
        if index < K // 128:
            async_copy.buffer_load_to_shared(smAS.index(index), AS, gl.where(ar < ROWS, ar, 0) * (K // 128) + index)
            async_copy.buffer_load_to_shared(smBS.index(index), BS, (bi * 0) * (K // 128) + index)
    abarrier_arrive(SLOT)


@g.jit
def _produce_ab_and_scales(
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
    ROWS,
):
    wave = get_wave_id()
    _initialize_pipeline_barriers(wave)
    if GROUP == 1:
        _publish_scale_batch(AS, BS, smAS0, smBS, 0, ROWS, K, 2, 4)
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
                abarrier_try_wait(slot * 2, ((i // 2) - 1) % 2)
            abarrier_seq(slot * 2 + 1)
            matrix_load(smA, A, (gl.minimum(16, ROWS), K), (K, 1), (0, i * 128), pred=wave % 2 == 0, boundary_check=(0,))
            abarrier_arrive(slot * 2 + 1)
            if i >= 2:
                abarrier_try_wait(8 + slot * 4, ((i // 2) - 1) % 2)
            abarrier_seq(9 + slot * 4)
            matrix_load(smB, B, (K, 128), (1, K), (i * 128, 0), warps=(1, 2), pred=wave % 2 < 2)
            abarrier_arrive(9 + slot * 4)
        else:
            if i >= 2:
                abarrier_try_wait(6 + slot * 4, ((i // 2) - 1) % 2)
            abarrier_seq(7 + slot * 4)
            matrix_load(
                smA, A + 16 * K, (gl.maximum(0, ROWS - 16), K), (K, 1), (0, i * 128), pred=wave % 2 == 0, boundary_check=(0,)
            )
            abarrier_arrive(7 + slot * 4)
            # Publish the next scale batch after this A tile has been issued.
            # ACK14 on every iteration prevents a scale ready barrier from
            # being reused before both consumers observed its prior phase.
            if (i + 1) % 2 == 0 and i + 1 < K // 128:
                batch = (i + 1) // 2
                if batch % 2 == 0:
                    _publish_scale_batch(AS, BS, smAS0, smBS, i + 1, ROWS, K, 2, 4)
                else:
                    _publish_scale_batch(AS, BS, smAS0, smBS, i + 1, ROWS, K, 2, 5)
            abarrier_try_wait(14, i % 2)


@g.jit
def _wait_ab_reads_and_raise_priority():
    set_priority(3)
    wait_lgkmcnt(9)


@g.jit
def _wait_a_tile_ready(i, GROUP: gl.constexpr):
    if GROUP == 0:
        abarrier_try_wait((i % 2) * 2 + 1, (i // 2) % 2)
    else:
        abarrier_try_wait(7 + (i % 2) * 4, (i // 2) % 2)


@g.jit
def _wait_b_tile_ready(i):
    abarrier_try_wait(9 + (i % 2) * 4, (i // 2) % 2)


@g.jit
def _ack_scale_tile_ready(i, GROUP: gl.constexpr):
    if i % 2 == 0:
        batch = i // 2
        if batch % 2 == 0:
            abarrier_try_wait(4, (batch // 2) % 2)
        else:
            abarrier_try_wait(5, (batch // 2) % 2)
    # Per-round ACK remains unchanged. The slices are immutable; the ready
    # wait happens once per batch before any consumer reads that batch.
    gl.inline_asm_elementwise(
        "s_abarrier_arrive $0, 14, 1",
        constraints="=s",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _release_ab_lds_slots(i, GROUP: gl.constexpr):
    _wait_ab_reads_and_raise_priority()
    if GROUP == 0:
        abarrier_arrive((i % 2) * 2)
    else:
        abarrier_arrive(6 + (i % 2) * 4)
    abarrier_arrive(8 + (i % 2) * 4)


@g.jit
def _prefetch_scale_tile(
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
    # release/3.6.x exposes descriptor loads; the reference's raw LDS
    # slice pointer API is not present in this release branch.
    sa = smAS0.index(i).slice(GROUP * 16, 16).load(gl.SliceLayout(1, F))
    bits = smBS.index(i)._reinterpret(gl.uint64, (1,), gl.SwizzledSharedLayout(1, 1, 1, [0]))
    sb = bits.load(gl.BlockedLayout([1], [64], [2], [0])).reshape(())
    return sa, sb


@g.jit
def _select_b_scale_for_wave(sb, bs_group):
    lo = sb.to(gl.uint32)
    hi = (sb >> 32).to(gl.uint32)
    wait_lgkmcnt(0)
    selected = gl.where(bs_group != 0, hi, lo)
    return selected.to(gl.float32, bitcast=True)


@g.jit
def _consume_ab_and_accumulate(
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
    ROWS,
):
    wave = get_wave_id()
    _initialize_pipeline_barriers(wave)
    RF: gl.constexpr = MmacInterleave2Layout(F)
    acc = gl.full((16, 128), 0, gl.float32, RF)
    # TODO: replace with gl.full once LLVM keeps the zero VGPR tuple outside
    # the loop instead of rematerializing it and repacking registers per MMAC.
    zero_mmac = gl.inline_asm_elementwise(
        "v_mov_b32 $0, 0\n\tv_mov_b32 $1, 0\n\tv_mov_b32 $2, 0\n\tv_mov_b32 $3, 0",
        constraints="=v,=v,=v,=v,v,v,v,v",
        args=[gl.full((16, 128), 0, gl.int32, RF)],
        dtype=gl.float32,
        is_pure=True,
        pack=4,
    )
    m = gl.arange(0, 16, layout=gl.SliceLayout(1, RF))
    bs_group = (wave % 2) // 2
    # Preserve a common lane base so A-scale reads combine into ds_read2.
    lane_m_byte = (m % 4) * 4
    _wait_a_tile_ready(0, GROUP)
    a = async_copy.load_shared_relaxed(smA0.slice(0, 16, 0), gl.DotOperandLayout(0, F, 8))
    _wait_b_tile_ready(0)
    b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
    sched_barrier(0)
    _ack_scale_tile_ready(0, GROUP)
    sa, sb = _prefetch_scale_tile(smAS0, smAS1, smBS, 0, m, lane_m_byte, bs_group, GROUP, RF)
    _release_ab_lds_slots(0, GROUP)
    for pair in range(K // 256):
        i = pair * 2
        sched_barrier(0)
        _synchronize_mmac_turn(GROUP, True)
        dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
        _synchronize_mmac_turn(GROUP, False)
        selected_bs = _select_b_scale_for_wave(sb, bs_group)
        j = i + 1
        _wait_a_tile_ready(j, GROUP)
        if j % 2 == 0:
            a = async_copy.load_shared_relaxed(smA0.slice(0, 16, 0), gl.DotOperandLayout(0, F, 8))
        else:
            a = async_copy.load_shared_relaxed(smA1.slice(0, 16, 0), gl.DotOperandLayout(0, F, 8))
        _wait_b_tile_ready(j)
        if j % 2 == 0:
            b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
        else:
            b = async_copy.load_shared_relaxed(smB1, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
        sched_barrier(0)
        acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
        sched_barrier(0)
        _release_ab_lds_slots(j, GROUP)
        sched_barrier(0)
        _ack_scale_tile_ready(j, GROUP)
        next_sa, next_sb = _prefetch_scale_tile(
            smAS0, smAS1, smBS, j, m, lane_m_byte, bs_group, GROUP, RF
        )
        sched_barrier(0)
        sa = next_sa
        sb = next_sb
        sched_barrier(0)
        if pair + 1 < K // 256:
            i = pair * 2 + 1
            sched_barrier(0)
            _synchronize_mmac_turn(GROUP, True)
            dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
            _synchronize_mmac_turn(GROUP, False)
            selected_bs = _select_b_scale_for_wave(sb, bs_group)
            j = i + 1
            _wait_a_tile_ready(j, GROUP)
            if j % 2 == 0:
                a = async_copy.load_shared_relaxed(smA0.slice(0, 16, 0), gl.DotOperandLayout(0, F, 8))
            else:
                a = async_copy.load_shared_relaxed(smA1.slice(0, 16, 0), gl.DotOperandLayout(0, F, 8))
            _wait_b_tile_ready(j)
            if j % 2 == 0:
                b = async_copy.load_shared_relaxed(smB0, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
            else:
                b = async_copy.load_shared_relaxed(smB1, MmacInterleave2Layout(gl.DotOperandLayout(1, F, 8)))
            sched_barrier(0)
            acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
            sched_barrier(0)
            _release_ab_lds_slots(j, GROUP)
            sched_barrier(0)
            _ack_scale_tile_ready(j, GROUP)
            next_sa, next_sb = _prefetch_scale_tile(
                smAS0, smAS1, smBS, j, m, lane_m_byte, bs_group, GROUP, RF
            )
            sched_barrier(0)
            sa = next_sa
            sb = next_sb
            sched_barrier(0)
    _synchronize_mmac_turn(GROUP, True)
    dot = gl.amd.cdna4.mfma(a, b, zero_mmac)
    _synchronize_mmac_turn(GROUP, False)
    selected_bs = _select_b_scale_for_wave(sb, bs_group)
    # Keep scale reads ahead of the final scale arithmetic.
    sched_barrier(0)
    acc = packed_scale_fma(dot, sa[:, None], selected_bs, acc)
    packed = acc.to(gl.bfloat16).to(gl.float16, bitcast=True)
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, RF))
    rows = GROUP * 16 + m
    gl.store(C + rows[:, None] * N + cols[None, :], acc.to(gl.bfloat16), mask=rows[:, None] < ROWS)


@g.jit
def _gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16_bm64_bn256_bk128_kernel(
    A,
    B,
    AS,
    BS,
    C,
    N: gl.constexpr,
    K: gl.constexpr,
    M: gl.constexpr,
    SA: gl.constexpr,
    SB: gl.constexpr,
    F: gl.constexpr,
):
    tile_m = gl.program_id(0) * 24
    rows = gl.minimum(24, M - tile_m)
    tile_n = gl.program_id(1) * 128
    A += tile_m * K
    B += tile_n * K
    AS += tile_m * (K // 128)
    BS += (tile_n // 128) * (K // 128)
    C += tile_m * N + tile_n
    scaleL: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    sas0 = gl.allocate_shared_memory(gl.float32, (K // 128, 64), scaleL)
    sas1 = sas0
    sbs = gl.allocate_shared_memory(gl.float32, (K // 128, 2), scaleL)
    a0 = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a1 = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a0_next = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    a1_next = gl.allocate_shared_memory(gl.float8e4nv, (64, 128), SA)
    b_next = gl.allocate_shared_memory(gl.float8e4nv, (128, 128), SB)
    b = gl.allocate_shared_memory(gl.float8e4nv, (128, 128), SB)
    gl.warp_specialize(
        [
            (_idle_default_partition, ()),
            (_produce_ab_and_scales, (A, B, AS, BS, sas0, sas1, sbs, a0, a0_next, b, b_next, K, 0, rows)),
            (_produce_ab_and_scales, (A, B, AS, BS, sas0, sas1, sbs, a1, a1_next, b, b_next, K, 1, rows)),
            (
                _consume_ab_and_accumulate,
                (AS, BS, C, sas0, sas1, sbs, a0, a0_next, b, b_next, K, N, 0, F, rows),
            ),
            (
                _consume_ab_and_accumulate,
                (AS, BS, C, sas0, sas1, sbs, a1, a1_next, b, b_next, K, N, 1, F, rows),
            ),
        ],
        [2, 2, 2, 2],
        [8, 8, 240, 240],
    )


def get_launch_config():
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
            warps_per_cta=[1, 2],
            tiles_per_warp=[1, 4],
        ),
        num_warps=2,
        num_stages=2,
        wasp_partition_warps=(2, 2, 2, 2),
        wasp_wdra=False,
    )


kernel = _gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16_bm64_bn256_bk128_kernel
