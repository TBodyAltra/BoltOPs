"""Native block-scaled dense GEMM with independent A/B scale blocks."""
import math
import operator

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.hcu import HcuMmacLayout, MmacLayout, get_mmac_instruction

from .gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16 import (
    _masked_kernel, _uses_mls_pipeline, kernel, launch_config,
)


@g.jit
def _block_scaled_kernel(A, B, AS, BS, C, M: gl.constexpr, N: gl.constexpr,
                         K: gl.constexpr, AM: gl.constexpr, AK: gl.constexpr,
                         BN: gl.constexpr, BK_SCALE: gl.constexpr,
                         Q: gl.constexpr, DOT_BK: gl.constexpr):
    DOT_ACC: gl.constexpr = (gl.float64 if A.dtype.element_ty == gl.float64 else
                            gl.int32 if A.dtype.element_ty.is_int() else gl.float32)
    MATH_ACC: gl.constexpr = (gl.float64 if DOT_ACC == gl.float64 or
                             AS.dtype.element_ty == gl.float64 or BS.dtype.element_ty == gl.float64
                             else gl.float32)
    INSTRUCTION: gl.constexpr = get_mmac_instruction(
        A.dtype.element_ty, B.dtype.element_ty, DOT_ACC, DOT_BK)
    F: gl.constexpr = HcuMmacLayout(
        version=4, instr_shape=INSTRUCTION[0], mode=MmacLayout.LTS,
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
            sa = sa_bits.to(AS.dtype.element_ty, bitcast=True).to(MATH_ACC)
        else:
            sa = gl.load(AS + ia, rm < M, other=0).to(MATH_ACC)
        if BS.dtype.element_ty.primitive_bitwidth == 8:
            sb_bits = gl.load(BS.to(gl.pointer_type(gl.uint8)) + ib, rn < N, other=0)
            sb = sb_bits.to(BS.dtype.element_ty, bitcast=True).to(MATH_ACC)
        else:
            sb = gl.load(BS + ib, rn < N, other=0).to(MATH_ACC)
        acc = gl.fma(partial.to(MATH_ACC) * sa[:, None], sb[None, :], acc)
    gl.store(C + rm[:, None].to(gl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty),
             (rm[:, None] < M) & (rn[None, :] < N))


def _parse_recipe(scale_recipe):
    try:
        recipe = tuple(tuple(operator.index(x) for x in block) for block in scale_recipe)
        if len(recipe) != 2 or any(len(block) != 2 or any(x <= 0 for x in block) for block in recipe):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("scale_recipe must be two positive integer pairs: ((A_M,A_K),(B_N,B_K))") from None
    return recipe


def gemm_block_scaled(a, b, a_scale, b_scale, *,
                      scale_recipe=((1, 128), (128, 128)), out_dtype=None):
    """Compute block-scaled A[M,K] @ B[N,K].T using native MMAC.

    Scale shapes are [ceil(M/A_M),ceil(K/A_K)] and
    [ceil(N/B_N),ceil(K/B_K)]. Blocks may differ; K-group is contiguous.
    Scales contain numeric values (uint8 is numeric, not implicitly E8M0).
    Input/accumulator legality is queried from the target MMAC database;
    unsupported combinations raise compiler errors, without input upcasting.
    Output dtype defaults to BF16 and may be selected explicitly. Scale
    arithmetic uses FP32, promoted to FP64 for FP64 inputs or scales.
    The existing tuned recipe uses its MLS/masked kernels; other combinations
    use a general masked native-MMAC kernel with no performance claim.
    """
    recipe = _parse_recipe(scale_recipe)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected A[M,K] and B[N,K] with matching K")
    m, k = a.shape
    n = b.shape[0]
    (am, ak), (bn, bk) = recipe
    expected = (("a", a, (m, k)), ("b", b, (n, k)),
                ("a_scale", a_scale, (triton.cdiv(m, am), triton.cdiv(k, ak))),
                ("b_scale", b_scale, (triton.cdiv(n, bn), triton.cdiv(k, bk))))
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape} for scale_recipe={recipe}, got {tuple(tensor.shape)}")
    for _, tensor, _ in expected:
        if tensor.device != a.device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous on the same GPU")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    with torch.cuda.device(a.device):
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        if m == 0 or n == 0:
            return out
        tuned = (recipe == ((1, 128), (128, 128)) and
                 a.dtype == b.dtype == torch.float8_e4m3fn and
                 a_scale.dtype == b_scale.dtype == torch.float32 and
                 out_dtype == torch.bfloat16)
        if tuned:
            if _uses_mls_pipeline(m, n, k, a, b):
                kernel[(m // 128, n // 256)](a, b, a_scale, b_scale, out, N=n, K=k, **launch_config())
            else:
                _masked_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 128))](
                    a, b, a_scale, b_scale, out, M=m, N=n, K=k, num_warps=4, num_stages=1)
        elif k == 0:
            out.zero_()
        else:
            q = math.gcd(ak, bk)
            dot_bk = max(32, min(128, triton.next_power_of_2(q)))
            _block_scaled_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
                a, b, a_scale, b_scale, out, M=m, N=n, K=k,
                AM=am, AK=ak, BN=bn, BK_SCALE=bk, Q=q, DOT_BK=dot_bk,
                num_warps=4, num_stages=1)
    return out
