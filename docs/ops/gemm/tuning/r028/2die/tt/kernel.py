"""R028 BF16 TT kernel definitions, selected from suggested-splitk3-256x256-bk32.
Original kernel/helper bodies are preserved; host invocation is supplied by the reproducer.
"""


import triton
import triton.language as tl

@triton.jit
def _splitk_mls_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_MATRIX_STORE: tl.constexpr,
    SPLIT_K: tl.constexpr,
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
    split = tl.program_id(axis=1)
    c_ptr += split * M * N
    tl.static_assert(K % (SPLIT_K * BLOCK_SIZE_K) == 0)
    K_STEPS: tl.constexpr = K // (SPLIT_K * BLOCK_SIZE_K)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    mls_offs_am = pid_m * BLOCK_SIZE_M
    mls_offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K_STEPS, warp_specialize=WARP_SPECIALIZE):
        mls_offs_k = (split * K_STEPS + k) * BLOCK_SIZE_K
        a = tl.matrix_load(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            offsets=[mls_offs_am, mls_offs_k],
        )
        b = tl.matrix_load(
            b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            offsets=[mls_offs_k, mls_offs_bn],
        )
        accumulator = tl.dot(a, b, accumulator)

    if HAS_BIAS:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
        accumulator += bias
    c = accumulator.to(c_ptr.dtype.element_ty)

    if USE_MATRIX_STORE:
        # VGPR→GMEM matrix_store (prefer mmac_layout_force=3).
        tl.matrix_store(
            c_ptr,
            c,
            shape=[M, N],
            strides=[stride_cm, stride_cn],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            offsets=[mls_offs_am, mls_offs_bn],
        )
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def _reduce_split3(Workspace, C, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    value = tl.load(Workspace + offsets, mask, other=0)
    value += tl.load(Workspace + ELEMENTS + offsets, mask, other=0)
    value += tl.load(Workspace + 2 * ELEMENTS + offsets, mask, other=0)
    tl.store(C + offsets, value.to(C.dtype.element_ty), mask)
