"""Masked native block-scaled GEMM implementations, without the MLS pipeline."""
import math
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from ...scales import decode_block_scale
from triton.experimental.gluon.language.amd.hcu import (
    MmacLayout, get_mmac_instruction, packed_scale_fma,
)


@g.jit
def _gemm_block_scaled_fp8_fp32_masked_kernel(A, B, AS, BS, C, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
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



@g.jit
def _gemm_block_scaled_masked_kernel(A, B, AS, BS, C, M: gl.constexpr, N: gl.constexpr,
                         K: gl.constexpr, AM: gl.constexpr, AK: gl.constexpr,
                         BN: gl.constexpr, BK_SCALE: gl.constexpr,
                         Q: gl.constexpr, DOT_BK: gl.constexpr,
                         A_E8M0: gl.constexpr = False, B_E8M0: gl.constexpr = False):
    DOT_ACC: gl.constexpr = (gl.float64 if A.dtype.element_ty == gl.float64 else
                            gl.int32 if A.dtype.element_ty.is_int() else gl.float32)
    MATH_ACC: gl.constexpr = (gl.float64 if DOT_ACC == gl.float64 or
                             AS.dtype.element_ty == gl.float64 or BS.dtype.element_ty == gl.float64
                             else gl.float32)
    INSTRUCTION: gl.constexpr = get_mmac_instruction(
        A.dtype.element_ty, B.dtype.element_ty, DOT_ACC, DOT_BK)
    F: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=INSTRUCTION[0], mmac_layout=MmacLayout.LTS,
        warps_per_cta=[2, 2], tiles_per_warp=[1, 2],
        element_bitwidth=64 if DOT_ACC == gl.float64 else 32,
    )
    L: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    m = gl.program_id(0) * 32 + gl.arange(0, 32, layout=gl.SliceLayout(1, L))
    n = gl.program_id(1) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, L))
    ka = gl.arange(0, DOT_BK, layout=gl.SliceLayout(0, L))
    kb = gl.arange(0, DOT_BK, layout=gl.SliceLayout(1, L))
    rm = gl.program_id(0) * 32 + gl.arange(0, 32, layout=gl.SliceLayout(1, F))
    rn = gl.program_id(1) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, F))
    acc = gl.full((32, 64), 0, MATH_ACC, F)
    for group in range(gl.cdiv(K, Q)):
        lo = group * Q
        hi = gl.minimum(lo + Q, K)
        partial = gl.full((32, 64), 0, DOT_ACC, F)
        for chunk in range(gl.cdiv(hi - lo, DOT_BK)):
            ax = lo + chunk * DOT_BK + ka
            bx = lo + chunk * DOT_BK + kb
            if A.dtype.element_ty.primitive_bitwidth == 8:
                av = gl.load(A.to(gl.pointer_type(gl.uint8)) + m[:, None].to(gl.int64) * K + ax[None, :],
                             (m[:, None] < M) & (ax[None, :] < hi), other=0)
                a = av.to(A.dtype.element_ty, bitcast=True)
            else:
                a = gl.load(A + m[:, None].to(gl.int64) * K + ax[None, :],
                            (m[:, None] < M) & (ax[None, :] < hi), other=0)
            if B.dtype.element_ty.primitive_bitwidth == 8:
                bv = gl.load(B.to(gl.pointer_type(gl.uint8)) + n[None, :].to(gl.int64) * K + bx[:, None],
                             (n[None, :] < N) & (bx[:, None] < hi), other=0)
                b = bv.to(B.dtype.element_ty, bitcast=True)
            else:
                b = gl.load(B + n[None, :].to(gl.int64) * K + bx[:, None],
                            (n[None, :] < N) & (bx[:, None] < hi), other=0)
            a = gl.convert_layout(a, gl.DotOperandLayout(0, F, INSTRUCTION[1]))
            b = gl.convert_layout(b, gl.DotOperandLayout(1, F, INSTRUCTION[1]))
            partial = gl.amd.cdna4.mfma(a, b, partial)
        ia = (rm // AM).to(gl.int64) * gl.cdiv(K, AK) + lo // AK
        ib = (rn // BN).to(gl.int64) * gl.cdiv(K, BK_SCALE) + lo // BK_SCALE
        if AS.dtype.element_ty.primitive_bitwidth == 8:
            sa_bits = gl.load(AS.to(gl.pointer_type(gl.uint8)) + ia, rm < M, other=0)
            sa = decode_block_scale(sa_bits.to(AS.dtype.element_ty, bitcast=True), A_E8M0).to(MATH_ACC)
        else:
            sa = gl.load(AS + ia, rm < M, other=0).to(MATH_ACC)
        if BS.dtype.element_ty.primitive_bitwidth == 8:
            sb_bits = gl.load(BS.to(gl.pointer_type(gl.uint8)) + ib, rn < N, other=0)
            sb = decode_block_scale(sb_bits.to(BS.dtype.element_ty, bitcast=True), B_E8M0).to(MATH_ACC)
        else:
            sb = gl.load(BS + ib, rn < N, other=0).to(MATH_ACC)
        acc = gl.fma(partial.to(MATH_ACC) * sa[:, None], sb[None, :], acc)
    gl.store(C + rm[:, None].to(gl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty),
             (rm[:, None] < M) & (rn[None, :] < N))



def launch_gemm_block_scaled_fp8_fp32_masked(config, problem, a, b, a_scale, b_scale, out):
    _gemm_block_scaled_fp8_fp32_masked_kernel[(triton.cdiv(problem.m, config.block_m), triton.cdiv(problem.n, config.block_n))](
        a, b, a_scale, b_scale, out, M=problem.m, N=problem.n, K=problem.k,
        num_warps=4, num_stages=1)


def launch_gemm_block_scaled_masked(config, problem, a, b, a_scale, b_scale, out):
    am, ak = problem.a_scale.block
    bn, bk = problem.b_scale.block
    _gemm_block_scaled_masked_kernel[(triton.cdiv(problem.m, config.block_m), triton.cdiv(problem.n, config.block_n))](
        a, b, a_scale, b_scale, out, M=problem.m, N=problem.n, K=problem.k,
        AM=am, AK=ak, BN=bn, BK_SCALE=bk, Q=math.gcd(ak, bk), DOT_BK=config.block_k,
        A_E8M0=problem.a_scale.encoding == "e8m0", B_E8M0=problem.b_scale.encoding == "e8m0",
        num_warps=4, num_stages=1)
