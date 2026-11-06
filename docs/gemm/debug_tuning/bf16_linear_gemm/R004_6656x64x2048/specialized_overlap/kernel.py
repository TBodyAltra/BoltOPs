"""R004 专用交付候选：完整重叠 MLS 读取，仅对输出行所有权加掩码。

验证配置：M6656/N64/K2048，BM64/BN64/BK64，144 个程序，48 个 CU。
没有分 K、原子操作、临时归约或矩阵加载边界检查。
SK_SPLIT 在本内核中保留测试接口含义：所有权范围的对齐行数（16）。
"""
import triton
import triton.language as tl

@triton.jit
def _gemm_bf16_hsk_kernel(
    a_ptr, b_ptr, c_ptr, ws_ptr, cnt_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    NUM_DP,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SK_SPLIT: tl.constexpr, SK_FIRST: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, USE_MLS: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr = 1,
    ATOMIC_SEM: tl.constexpr = "relaxed", PARTIAL_FP32: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pn = pid % (N // BLOCK_SIZE_N)
    pr = pid // (N // BLOCK_SIZE_N)
    P = NUM_DP // (N // BLOCK_SIZE_N)
    ALIGN: tl.constexpr = SK_SPLIT
    r0 = (pr * (M // ALIGN) // P) * ALIGN
    r1 = ((pr + 1) * (M // ALIGN) // P) * ALIGN
    om = tl.minimum(r0, M - BLOCK_SIZE_M)
    on = pn * BLOCK_SIZE_N
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    for k in tl.range(0, K // BLOCK_SIZE_K, warp_specialize=WARP_SPECIALIZE,
                      loop_unroll_factor=LOOP_UNROLL_FACTOR):
        ok = k * BLOCK_SIZE_K
        a = tl.matrix_load(a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K], offsets=[om, ok], boundary_check=())
        b = tl.matrix_load(b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], offsets=[ok, on], boundary_check=())
        acc = tl.dot(a, b, acc)
    rm = om + tl.arange(0, BLOCK_SIZE_M)
    rn = on + tl.arange(0, BLOCK_SIZE_N)
    cp = c_ptr + rm.to(tl.int64)[:, None] * stride_cm + rn.to(tl.int64)[None, :] * stride_cn
    tl.store(cp, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] >= r0) & (rm[:, None] < r1))
