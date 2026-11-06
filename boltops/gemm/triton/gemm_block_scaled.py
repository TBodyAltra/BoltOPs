"""General native-dot block-scaled GEMM; independent A/B scale recipes."""
import math

import triton
import triton.language as tl


@triton.jit
def _decode_block_scale(value, E8M0: tl.constexpr):
    if E8M0:
        exponent = value.to(tl.uint32)
        bits = exponent << 23
        bits = tl.where(exponent == 0, 0x00400000, bits)
        bits = tl.where(exponent == 255, 0x7FC00000, bits)
        return bits.to(tl.float32, bitcast=True)
    else:
        return value


@triton.jit
def _gemm_block_scaled_kernel(
    A, B, AS, BS, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    AM: tl.constexpr, AK: tl.constexpr, BN: tl.constexpr, BK_SCALE: tl.constexpr,
    Q: tl.constexpr, DOT_BK: tl.constexpr,
    A_E8M0: tl.constexpr, B_E8M0: tl.constexpr,
    BM: tl.constexpr, BN_TILE: tl.constexpr,
):
    DOT_ACC: tl.constexpr = (tl.float64 if A.dtype.element_ty == tl.float64 else
                            tl.int32 if A.dtype.element_ty.is_int() else tl.float32)
    MATH_ACC: tl.constexpr = (tl.float64 if DOT_ACC == tl.float64 or
                             AS.dtype.element_ty == tl.float64 or BS.dtype.element_ty == tl.float64
                             else tl.float32)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN_TILE + tl.arange(0, BN_TILE)
    kk = tl.arange(0, DOT_BK)
    acc = tl.full((BM, BN_TILE), 0, MATH_ACC)
    for group in range(tl.cdiv(K, Q)):
        lo = group * Q
        hi = tl.minimum(lo + Q, K)
        partial = tl.full((BM, BN_TILE), 0, DOT_ACC)
        for chunk in range(tl.cdiv(hi - lo, DOT_BK)):
            k = lo + chunk * DOT_BK + kk
            # Load 8-bit storage as bytes so no numeric input conversion occurs.
            if A.dtype.element_ty.primitive_bitwidth == 8:
                av = tl.load(A.to(tl.pointer_type(tl.uint8)) + m[:, None].to(tl.int64) * K + k[None, :],
                             (m[:, None] < M) & (k[None, :] < hi), other=0)
                a = av.to(A.dtype.element_ty, bitcast=True)
            else:
                a = tl.load(A + m[:, None].to(tl.int64) * K + k[None, :],
                            (m[:, None] < M) & (k[None, :] < hi), other=0)
            if B.dtype.element_ty.primitive_bitwidth == 8:
                bv = tl.load(B.to(tl.pointer_type(tl.uint8)) + n[None, :].to(tl.int64) * K + k[:, None],
                             (n[None, :] < N) & (k[:, None] < hi), other=0)
                b = bv.to(B.dtype.element_ty, bitcast=True)
            else:
                b = tl.load(B + n[None, :].to(tl.int64) * K + k[:, None],
                            (n[None, :] < N) & (k[:, None] < hi), other=0)
            partial = tl.dot(a, b, partial, input_precision="ieee", out_dtype=DOT_ACC)
        ia = (m // AM).to(tl.int64) * tl.cdiv(K, AK) + lo // AK
        ib = (n // BN).to(tl.int64) * tl.cdiv(K, BK_SCALE) + lo // BK_SCALE
        if AS.dtype.element_ty.primitive_bitwidth == 8:
            sa_bits = tl.load(AS.to(tl.pointer_type(tl.uint8)) + ia, m < M, other=0)
            sa = _decode_block_scale(sa_bits.to(AS.dtype.element_ty, bitcast=True), A_E8M0).to(MATH_ACC)
        else:
            sa = tl.load(AS + ia, m < M, other=0).to(MATH_ACC)
        if BS.dtype.element_ty.primitive_bitwidth == 8:
            sb_bits = tl.load(BS.to(tl.pointer_type(tl.uint8)) + ib, n < N, other=0)
            sb = _decode_block_scale(sb_bits.to(BS.dtype.element_ty, bitcast=True), B_E8M0).to(MATH_ACC)
        else:
            sb = tl.load(BS + ib, n < N, other=0).to(MATH_ACC)
        acc = tl.fma(partial.to(MATH_ACC) * sa[:, None], sb[None, :], acc)
    tl.store(C + m[:, None].to(tl.int64) * N + n[None, :], acc.to(C.dtype.element_ty),
             (m[:, None] < M) & (n[None, :] < N))


def launch_gemm_block_scaled(config, problem, a, b, a_scale, b_scale, out):
    # TL's legacy frontend otherwise silently upcasts FNUZ to FP16 on HCU.
    # Keep this API's native-input contract instead of accepting that fallback.
    hcu_archs = {"gfx926", "gfx928", "gfx936", "gfx938", "gfx92a", "gfx946", "gfx948"}
    if problem.arch in hcu_archs and any(
        str(t.dtype) in ("torch.float8_e4m3fnuz", "torch.float8_e5m2fnuz") for t in (a, b)
    ):
        raise ValueError("native MMAC does not support FNUZ FP8 on HCU; input upcasting is disabled")
    am, ak = problem.a_scale.block
    bn, bk = problem.b_scale.block
    _gemm_block_scaled_kernel[(triton.cdiv(problem.m, config.block_m),
                              triton.cdiv(problem.n, config.block_n))](
        a, b, a_scale, b_scale, out, M=problem.m, N=problem.n, K=problem.k,
        AM=am, AK=ak, BN=bn, BK_SCALE=bk, Q=math.gcd(ak, bk), DOT_BK=config.block_k,
        A_E8M0=problem.a_scale.encoding == "e8m0", B_E8M0=problem.b_scale.encoding == "e8m0",
        BM=config.block_m, BN_TILE=config.block_n, num_warps=4, num_stages=1,
    )
