# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""BF16 TN/TT Stream-K kernels, task scheduling, and dynamic LDS launch."""

from collections import defaultdict
from functools import lru_cache

import torch
import triton
from triton import knobs
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.runtime.driver import driver


# BF16 TN/TT Stream-K kernels

# Two-producer/two-consumer Stream-K pipeline. The host launcher reserves
# 96 KiB dynamic LDS for the explicitly addressed, double-buffered MLS ring.
@triton.constexpr_function
def _streamk_accumulator_registers():
    # One 4-register fragment per (M16, N16), in the Gluon accumulator layout.
    return tuple(61 - 4 * i + j for i in range(16) for j in range(4))


@triton.constexpr_function
def _streamk_k64_asm(B_K_CONTIGUOUS):
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
    if B_K_CONTIGUOUS:
        # K-contiguous weights use the transposed LDS read to feed the same
        # LIT registers. The K halves are 4096 bytes apart and N panels 2048.
        b_offsets = (0, 32, 2048, 2080, 4096, 4128, 6144, 6176)
        b_read = 'ds_read_matrix_trans_format'
        b_format = 'element:2 row:1 col:2'
    else:
        b_offsets = (0, 1024, 4096, 5120, 2048, 3072, 6144, 7168)
        b_read = 'ds_read_matrix_format'
        b_format = 'element:2 row:2 col:1'
    for slot, offset in enumerate(b_offsets):
        base = 100 + 4 * slot
        lines.append(f'{b_read} v[{base}:{base+3}], $192 offset:{offset} {b_format}')
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
def _streamk_consume_k64(acc, a_address, b_address, a_empty, a_ready, b_empty, b_ready, phase, B_K_CONTIGUOUS: gl.constexpr):
    return gl.inline_asm_elementwise(
        _streamk_k64_asm(B_K_CONTIGUOUS), _streamk_k64_constraints(),
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
def _streamk_matrix_asm(priority, b_k_contiguous):
    lines = ['s_mov_b64 s[80:81], $1', 's_and_b32 s81, s81, 0xffff',
             's_mov_b32 s82, $4', 's_mov_b32 s83, 0x10000',
             's_add_i32 s84, $2, 1024', 's_add_i32 s85, $2, 2048',
             's_add_i32 s86, $2, 3072']
    if priority:
        lines.append('s_setprio 1')
    layout = 'r' if priority and b_k_contiguous else 't'
    lines += ['s_abarrier_seq $3',
              f'matrix_load_64x16_b16 s[80:83] $2 {layout} bps lds',
              f'matrix_load_64x16_b16 s[80:83] s84 moffset:16 {layout} bps lds',
              f'matrix_load_64x16_b16 s[80:83] s85 moffset:32 {layout} bps lds',
              f'matrix_load_64x16_b16 s[80:83] s86 moffset:48 {layout} bps lds',
              's_abarrier_arrive $0, $3, 1']
    if priority:
        lines.append('s_setprio 0')
    return '\n'.join(lines)


@g.jit
def _streamk_matrix_to_lds(pointer, address, ready, STRIDE: gl.constexpr, IS_B: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr = False):
    gl.inline_asm_elementwise(
        _streamk_matrix_asm(IS_B, B_K_CONTIGUOUS),
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
             K: gl.constexpr, RECORDS: gl.constexpr, GROUP: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr):
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
        if B_K_CONTIGUOUS:
            bp = B + (col + wid * 64) * K + start
        else:
            bp = B + start * N + col + wid * 64
        for step in range(length // 64):
            stage = counter & 1
            phase = (counter >> 1) & 1
            # Shared B is submitted first by _streamk_producer partition 0. The other
            # _streamk_producer partition supplies A only, as in the measured kernel.
            if GROUP == 0:
                _streamk_wait_barrier(8 + stage * 2, phase)
                if B_K_CONTIGUOUS:
                    _streamk_matrix_to_lds(bp + step * 64,
                                  65536 + stage * 16384 + wid * 8192,
                                  9 + stage * 2, K, True, True)
                elif K % 64:
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
             RECORDS: gl.constexpr, GROUP: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr):
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
                _streamk_empty, _streamk_empty + 1, 8 + stage * 2, 9 + stage * 2, phase, B_K_CONTIGUOUS)
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
                    M: gl.constexpr, N: gl.constexpr, K: gl.constexpr, RECORDS: gl.constexpr, B_K_CONTIGUOUS: gl.constexpr = False):
    gl.static_assert(A.dtype.element_ty == gl.bfloat16 and C.dtype.element_ty == gl.bfloat16)
    gl.static_assert(M > 0 and N > 0 and K > 0 and M % 4 == 0 and N % 128 == 0 and K % 16 == 0)
    gl.static_assert(not B_K_CONTIGUOUS or K % 64 == 0)
    gl.warp_specialize([
        (_streamk_empty, ()),
        (_streamk_producer, (A, B, EdgeM, EdgeKA, EdgeKB, Schedule, M, N, K, RECORDS, 0, B_K_CONTIGUOUS)),
        (_streamk_producer, (A, B, EdgeM, EdgeKA, EdgeKB, Schedule, M, N, K, RECORDS, 1, B_K_CONTIGUOUS)),
        (_streamk_consumer, (C, Schedule, Workspace, Flags, Epoch, M, N, RECORDS, 0, B_K_CONTIGUOUS)),
        (_streamk_consumer, (C, Schedule, Workspace, Flags, Epoch, M, N, RECORDS, 1, B_K_CONTIGUOUS)),
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


# Stream-K host scheduling and launch

_STREAMK_SHARED_BYTES = 96 * 1024
_STREAMK_I32_MAX = (1 << 31) - 1


def _streamk_supports_shape(m, n, k, layout):
    """Whether the native BF16 kernel can encode the shape (independent of CU count)."""
    if (layout not in ("TT", "TN") or min(m, n, k) <= 0 or m % 4 or n % 128
            or k % 16 or (layout == "TN" and k % 64)):
        return False
    padded_m, padded_k = triton.cdiv(m, 256) * 256, triton.cdiv(k, 64) * 64
    # Producer and edge-prep pointer arithmetic uses i32 element offsets,
    # including padded extents. The epilogue uses i32 byte offsets and, for
    # partial M tiles, clamps a signed i32 byte bound for the buffer descriptor.
    return max(padded_m * k, padded_k * n, 2 * padded_m * n) <= _STREAMK_I32_MAX


def _active_tg_count(tiles, steps, cu_count):
    # No empty TGs, and no more than two owners per tile. Two interior
    # boundaries in the same tile would need a third owner; tile-aligned
    # boundaries don't split anything. Keep the original CU count when safe.
    work = tiles * steps
    for count in range(min(cu_count, work, 2 * tiles), tiles, -1):
        last_split_tile = -1
        for pid in range(1, count):
            tile, offset = divmod(pid * work // count, steps)
            if offset:
                if tile == last_split_tile:
                    break
                last_split_tile = tile
        else:
            return count
    # At most one TG per tile always admits the balanced contiguous partition.
    return min(cu_count, tiles)


@lru_cache(maxsize=128)
def _make_schedule(m, n, k, cu_count, *, schedule_group_m=4, compact_schedule=False):
    if not _streamk_supports_shape(m, n, k, "TT") or cu_count <= 0:
        raise ValueError("unsupported Stream-K Gluon shape or CU count")
    if type(schedule_group_m) is not int or schedule_group_m <= 0:
        raise ValueError("Stream-K Gluon requires a positive integer schedule_group_m")
    if type(compact_schedule) is not bool:
        raise ValueError("Stream-K Gluon requires a boolean compact_schedule")
    rows, cols, steps = triton.cdiv(m, 256), n // 128, triton.cdiv(k, 64)
    tiles = rows * cols
    active_tgs = _active_tg_count(tiles, steps, cu_count)
    # Traverse groups of M tiles to reuse the B panels; group size 1 is row-major.
    physical = [row * cols + col for first in range(0, rows, schedule_group_m)
                for col in range(cols) for row in range(first, min(first + schedule_group_m, rows))]
    tasks = [[] for _ in range(active_tgs)]
    by_tile = defaultdict(list)
    work = tiles * steps
    for pid in range(active_tgs):
        cursor, end = pid * work // active_tgs, (pid + 1) * work // active_tgs
        while cursor < end:
            tile, start = divmod(cursor, steps)
            stop = min(end, (tile + 1) * steps)
            record = [physical[tile], start * 64, (stop - cursor) * 64, 0]
            tasks[pid].append(record)
            by_tile[record[0]].append(record)
            cursor = stop
    slots = 0
    for _, fragments in sorted(by_tile.items()):
        if len(fragments) > 2:
            raise RuntimeError("Stream-K Gluon cannot encode more than two fragments per tile")
        if len(fragments) == 2:
            # The higher-K owner publishes; the lower-K owner merges and stores.
            fragments[0][3], fragments[1][3] = -(slots + 1), slots + 1
            slots += 1
    # Include a zero-length sentinel for every TG, as in the source schedule.
    records = max(map(len, tasks)) + 1 if compact_schedule else tiles + 1
    # Table and scratch pointers are indexed with signed i32 element offsets.
    if max(active_tgs * records * 4, slots * 32768) > _STREAMK_I32_MAX:
        raise ValueError("Stream-K Gluon schedule exceeds its i32 offset bounds")
    table = tuple(tuple(record) for tg in tasks
                  for record in tg + [[0, 0, 0, 0]] * (records - len(tg)))
    return table, records, slots


@lru_cache(maxsize=128)
def _device_schedule(m, n, k, cu_count, device, stream, *, schedule_group_m=4, compact_schedule=False):
    # Key by stream so the asynchronous schedule upload never crosses streams.
    table, records, slots = _make_schedule(
        m, n, k, cu_count, schedule_group_m=schedule_group_m, compact_schedule=compact_schedule,
    )
    return torch.tensor(table, dtype=torch.int32, device=device), records, slots


def _compile_kernel(kernel, m, n, k, records, options, target, *, b_k_contiguous=False):
    return _compile_kernel_for_device(
        kernel, m, n, k, records, options, target, torch.cuda.current_device(), b_k_contiguous,
    )


@lru_cache(maxsize=128)
def _compile_kernel_for_device(kernel, m, n, k, records, options, target, device_index, b_k_contiguous):
    # CompiledKernel retains HIP function handles from its first launch device.
    # Include the device index even when both GPUs have the same target ISA.
    constexprs = dict(M=m, N=n, K=k, RECORDS=records, B_K_CONTIGUOUS=b_k_contiguous)
    source = GluonASTSource(
        kernel,
        signature=dict(A="*bf16", B="*bf16", C="*bf16", Schedule="*i32",
                       Workspace="*fp32", Flags="*i32", EdgeM="*bf16",
                       EdgeKA="*bf16", EdgeKB="*bf16", Epoch="i32"),
        constexprs=constexprs,
        attrs={(i,): [("tt.divisibility", 16)] for i in range(9)},
    )
    return triton.compile(source, target=target, options=dict(options))


def _launch_gemm_a16w16_streamk(kernel, a, b, out, *, cu_count, config, layout):
    """Launch at most one TG per CU, reserving LDS for the inline MLS ops.

    The immutable task table is cached per stream. Scratch and zeroed flags
    belong to each invocation, so concurrent streams and CUDA graph replays
    cannot observe completion flags from a previous GEMM.
    """
    m, k = a.shape
    n = b.shape[0]
    if not _streamk_supports_shape(m, n, k, layout):
        raise ValueError("unsupported Stream-K Gluon shape or layout")
    if any(t.dtype != torch.bfloat16 for t in (a, b, out)):
        raise ValueError("Stream-K Gluon requires BF16 input and output")
    b_k_contiguous = layout == "TN"
    if (b.shape != (n, k) or out.shape != (m, n) or not out.is_contiguous()
            or a.stride() != (k, 1) or b.stride() != ((k, 1) if b_k_contiguous else (1, n))):
        raise ValueError("Stream-K Gluon requires dense K-contiguous x, TN or TT w, and contiguous output")
    for tensor in (a, b, out):
        address = tensor.data_ptr()
        if address % 16:
            raise ValueError("Stream-K Gluon requires 16-byte-aligned tensor pointers")
        # Inline MLS and buffer descriptors store only the low 48 address bits.
        if address + tensor.numel() * 2 > 1 << 48:
            raise ValueError("Stream-K Gluon tensor exceeds its 48-bit descriptor address range")
    tile = tuple(config.get(key) for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
    if tile != (256, 128, 64):
        raise ValueError("Stream-K Gluon requires a 256x128x64 tile")
    options = {key: tuple(value) if isinstance(value, list) else value
               for key, value in config.items()
               if key not in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "NUM_SMS",
                              "schedule_group_m", "compact_schedule")}
    if (options.get("num_warps") != 2
            or options.get("wasp_partition_warps") != (2, 2, 4, 4)
            or options.get("wasp_wdra") is not False):
        raise ValueError("Stream-K Gluon requires num_warps=2 and partitions (2,2,4,4) without WDRA")
    stream = torch.cuda.current_stream(a.device)
    schedule, records, slots = _device_schedule(
        m, n, k, cu_count, a.device, stream.cuda_stream,
        schedule_group_m=config.get("schedule_group_m", 4),
        compact_schedule=config.get("compact_schedule", False),
    )
    schedule.record_stream(stream)
    workspace = torch.empty(max(slots * 256 * 128, 1), dtype=torch.float32, device=a.device)
    # Reset on the GPU each invocation: using a host epoch alone would reuse a
    # captured epoch during graph replay and accept stale partial results.
    flags = torch.zeros(max(slots * 8, 1), dtype=torch.int32, device=a.device)
    edge_sizes = (256 * k if m % 256 else 0,
                  triton.cdiv(m, 256) * 256 * 64 if k % 64 else 0,
                  64 * n if k % 64 else 0)
    edges = [torch.empty(max(size, 1), dtype=a.dtype, device=a.device) for size in edge_sizes]
    layout_options = {"b_k_contiguous": True} if b_k_contiguous else {}
    compiled = _compile_kernel(kernel, m, n, k, records, tuple(sorted(options.items())),
                               driver.active.get_current_target(), **layout_options)
    # Initialize Triton's launcher before retrieving the module function.
    run = compiled.run
    if any(edge_sizes):
        prepare_edges = _gemm_a16w16_bf16_tt_prepare_edges_kernel
        prepare_edges[(sum(triton.cdiv(size, 1024) for size in edge_sizes),)](
            a, b, *edges, M=m, N=n, K=k, BLOCK=1024, num_warps=4, num_stages=1,
        )
    args = (a, b, out, schedule, workspace, flags, *edges, 1)
    grid = (schedule.shape[0] // records, 1, 1)
    # These LDS addresses are expressed in inline assembly and are invisible to
    # Triton's allocator. Override only the launch's dynamic-LDS reservation;
    # keep the compiled kernel and its cached metadata unchanged.
    packed = (compiled.metadata.num_warps, compiled.metadata.num_ctas, _STREAMK_SHARED_BYTES)
    run(*grid, stream.cuda_stream, compiled.function, packed,
        compiled.launch_metadata(grid, stream.cuda_stream, *args),
        knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
