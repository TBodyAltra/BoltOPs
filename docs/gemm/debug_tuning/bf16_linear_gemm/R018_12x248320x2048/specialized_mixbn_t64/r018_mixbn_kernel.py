"""R018 mixed-BN GEMM: main panels of BLOCK_SIZE_N columns, remainder in narrow TAIL_BN panels.

Motivation (R018: M=12, N=248320, K=2048, TT, BF16). The shape is pure HBM-read bound (B is read exactly
once and the measured cycles match a B-only read probe within 0.4%), so the only structural loss left is
load imbalance: N/BLOCK_SIZE_N = 970 panels do not divide over 48 CUs, and the ~10 panels of the last
round read ~10 MB while 38 CUs idle, i.e. they run at ~1/5 of the chip's read bandwidth.

This kernel keeps the baseline tile for the first CU_MULT-aligned panels and covers the remaining columns
with TAIL_BN-wide panels, so the final round is spread over many more programs with proportionally fewer
bytes each. Output ownership is disjoint and every element is written exactly once: no partials, no
atomics, no reduction kernel. Grid = n_main + ceil((N - n_main*BLOCK_SIZE_N) / TAIL_BN), supplied by the
harness (bench.py grid_programs).

_gemm_bf16_generic_kernel keeps the baseline signature; TAIL_BN and CU_MULT are extra constexprs.
"""
import triton
import triton.language as tl


@triton.jit
def _panel(a_ptr, b_ptr, c_ptr, M, N, K,
           stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, om, on,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           M_ALIGNED: tl.constexpr, N_ALIGNED: tl.constexpr, K_ALIGNED: tl.constexpr,
           USE_MLS: tl.constexpr, UNROLL: tl.constexpr):
    """One [BM, BN] output panel, full K reduction. Same load/store structure as the baseline kernel."""
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    num_k = tl.cdiv(K, BK)
    if USE_MLS:
        for k in tl.range(0, num_k, loop_unroll_factor=UNROLL):
            ok = k * BK
            a = tl.matrix_load(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                               block_shape=[BM, BK], offsets=[om, ok],
                               boundary_check=((0, 1) if not K_ALIGNED else ((0,) if not M_ALIGNED else ())))
            b = tl.matrix_load(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                               block_shape=[BK, BN], offsets=[ok, on],
                               boundary_check=((0, 1) if not (K_ALIGNED and N_ALIGNED) else ()))
            acc = tl.dot(a, b, acc)
    else:
        offs_m = (om + tl.arange(0, BM)) if M_ALIGNED else ((om + tl.arange(0, BM)) % M)
        offs_n = (on + tl.arange(0, BN)) if N_ALIGNED else ((on + tl.arange(0, BN)) % N)
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_m.to(tl.int64)[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n.to(tl.int64)[None, :] * stride_bn
        for k in tl.range(0, num_k, loop_unroll_factor=UNROLL):
            if K_ALIGNED:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                rem = K - k * BK
                a = tl.load(a_ptrs, mask=offs_k[None, :] < rem, other=0.)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < rem, other=0.)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = (om + tl.arange(0, BM)).to(tl.int64)
    offs_cn = (on + tl.arange(0, BN)).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    if M_ALIGNED and N_ALIGNED:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


@triton.heuristics({
    "_M_ALIGNED": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "_N_ALIGNED": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "_K_ALIGNED": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    # tail columns are an exact multiple of TAIL_BN -> the tail panels need no N masking either
    "_TAIL_EXACT": lambda args: (args["N"] - (args["N"] // args["BLOCK_SIZE_N"]) // args["CU_MULT"]
                                 * args["CU_MULT"] * args["BLOCK_SIZE_N"]) % args["TAIL_BN"] == 0,
})
@triton.jit(do_not_specialize=["M", "N"])
def _gemm_bf16_generic_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_ck, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    USE_MLS: tl.constexpr,
    USE_MATRIX_STORE: tl.constexpr,
    TAIL_BN: tl.constexpr,
    CU_MULT: tl.constexpr,
    _M_ALIGNED: tl.constexpr,
    _N_ALIGNED: tl.constexpr,
    _K_ALIGNED: tl.constexpr,
    _TAIL_EXACT: tl.constexpr,
    ADD_BIAS: tl.constexpr = False,
    LOOP_UNROLL_FACTOR: tl.constexpr = 1,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    pid = tl.program_id(0)
    n_main = (N // BLOCK_SIZE_N) // CU_MULT * CU_MULT        # whole panels, multiple of CU_MULT
    tl.assume(n_main >= 0)
    if pid < n_main:
        _panel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
               stride_cm, stride_cn, 0, pid * BLOCK_SIZE_N,
               BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
               _M_ALIGNED, True, _K_ALIGNED, USE_MLS, LOOP_UNROLL_FACTOR)
    else:
        _panel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
               stride_cm, stride_cn, 0, n_main * BLOCK_SIZE_N + (pid - n_main) * TAIL_BN,
               BLOCK_SIZE_M, TAIL_BN, BLOCK_SIZE_K,
               _M_ALIGNED, _TAIL_EXACT, _K_ALIGNED, USE_MLS, LOOP_UNROLL_FACTOR)


def _warmup_generic_kernel(*args, grid, **kwargs):
    return _gemm_bf16_generic_kernel.run(*args, grid=grid, warmup=True, **kwargs)


_gemm_bf16_generic_kernel.warmup = _warmup_generic_kernel


def grid_programs(M, N, K, BLOCK_SIZE_N, TAIL_BN, CU_MULT, **_):
    """Host-side grid: aligned main panels + narrow tail panels (used by bench.py)."""
    n_main = (N // BLOCK_SIZE_N) // CU_MULT * CU_MULT
    rest = N - n_main * BLOCK_SIZE_N
    return n_main + -(-rest // TAIL_BN)
