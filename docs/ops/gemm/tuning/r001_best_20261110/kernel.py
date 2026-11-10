"""Direct BF16 GEMM, no workspace/reduction. Requires NUM_KSPLIT=1."""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _broadcast_lane0_i32(x):
    """Broadcast lane0's VGPR to an SGPR visible to all lanes in the wave."""
    return tl.inline_asm_elementwise(
        "v_readfirstlane_b32 $0, $1;",
        "=s,v",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.heuristics({
    "_NUM_PID_M": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"]),
    "_NUM_PID_N": lambda args: triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    "_EXACT_K": lambda args: args["K"] == args["NUM_KSPLIT"] * args["SPLITK_BLOCK_SIZE"]
        and args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0,
    "_FAST_K": lambda args: ((
        args["LOOP_UNROLL_FACTOR"] == 2
        and args.get("num_stages", 4) <= 2
        and args["SPLITK_BLOCK_SIZE"] >= 2 * args["BLOCK_SIZE_K"]
        and args["K"] >= args["NUM_KSPLIT"] * args["SPLITK_BLOCK_SIZE"]
        and args["K"] % (2 * args["BLOCK_SIZE_K"]) == 0
        and args["SPLITK_BLOCK_SIZE"] % (2 * args["BLOCK_SIZE_K"]) == 0
    ) or (
        args["K"] == args["NUM_KSPLIT"] * args["SPLITK_BLOCK_SIZE"]
        and args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0
        and args.get("num_stages", 2) >= 2
        and args["SPLITK_BLOCK_SIZE"] // args["BLOCK_SIZE_K"]
            >= args["LOOP_UNROLL_FACTOR"] * (args.get("num_stages", 2) - 1)
    )),
    "_M_ALIGNED": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "_N_ALIGNED": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
})
@triton.jit
def _gemm_bf16_fused_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_pk: tl.constexpr,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    USE_MLS: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr,
    _NUM_PID_M: tl.constexpr,
    _NUM_PID_N: tl.constexpr,
    _EXACT_K: tl.constexpr,
    _FAST_K: tl.constexpr,
    _M_ALIGNED: tl.constexpr,
    _N_ALIGNED: tl.constexpr,
):
    tl.static_assert(NUM_KSPLIT == 1)
    tl.static_assert(SPLITK_BLOCK_SIZE > 0)
    tl.static_assert(GROUP_SIZE_M > 0)
    tl.static_assert(not WARP_SPECIALIZE, "This fused reduction requires a single-wave TG")
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_pk > 0)
    tl.assume(stride_pm > 0)
    tl.assume(stride_pn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K >= 0)

    pid_unified = tl.program_id(axis=0)
    num_pid_m = _NUM_PID_M
    num_pid_n = _NUM_PID_N
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    tile_id = pid_m * num_pid_n + pid_n

    k_start = pid_k * SPLITK_BLOCK_SIZE
    if _EXACT_K:
        k_span = SPLITK_BLOCK_SIZE
        num_k = SPLITK_BLOCK_SIZE // BLOCK_SIZE_K
    else:
        k_end = tl.minimum(k_start + SPLITK_BLOCK_SIZE, K)
        if pid_k == NUM_KSPLIT - 1:
            k_end = K
        k_span = tl.maximum(k_end - k_start, 0)
        num_k = tl.cdiv(k_span, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    om = pid_m * BLOCK_SIZE_M
    on = pid_n * BLOCK_SIZE_N

    if USE_MLS:
        if _FAST_K:
            # Rebase to this slice: global shape=[M,K] would not protect the
            # boundary between adjacent K slices when their length is not BK-aligned.
            # Skip empty slices entirely, including the MLS pipeline prologue.
            if k_span > 0:
                a_slice = a_ptr + k_start * stride_ak
                b_slice = b_ptr + k_start * stride_bk
                for k in tl.range(
                    0,
                    k_span // BLOCK_SIZE_K,
                    warp_specialize=WARP_SPECIALIZE,
                    loop_unroll_factor=LOOP_UNROLL_FACTOR,
                ):
                    ok = k * BLOCK_SIZE_K
                    a = tl.matrix_load(
                        a_slice,
                        shape=[M, k_span],
                        strides=[stride_am, stride_ak],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                        offsets=[om, ok],
                        boundary_check=((0,) if not _M_ALIGNED else ()),
                    )
                    b = tl.matrix_load(
                        b_slice,
                        shape=[k_span, N],
                        strides=[stride_bk, stride_bn],
                        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                        offsets=[ok, on],
                        boundary_check=((1,) if not _N_ALIGNED else ()),
                    )
                    acc = tl.dot(a, b, acc)
        else:
            # Rebase to this slice: global shape=[M,K] would not protect the
            # boundary between adjacent K slices when their length is not BK-aligned.
            # Skip empty slices entirely, including the MLS pipeline prologue.
            if k_span > 0:
                a_slice = a_ptr + k_start * stride_ak
                b_slice = b_ptr + k_start * stride_bk
                for k in tl.range(
                    0,
                    num_k,
                    warp_specialize=WARP_SPECIALIZE,
                    loop_unroll_factor=LOOP_UNROLL_FACTOR,
                ):
                    ok = k * BLOCK_SIZE_K
                    a = tl.matrix_load(
                        a_slice,
                        shape=[M, k_span],
                        strides=[stride_am, stride_ak],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                        offsets=[om, ok],
                        boundary_check=(0, 1),
                    )
                    b = tl.matrix_load(
                        b_slice,
                        shape=[k_span, N],
                        strides=[stride_bk, stride_bn],
                        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                        offsets=[ok, on],
                        boundary_check=(0, 1),
                    )
                    acc = tl.dot(a, b, acc)
    else:
        offs_m = om + tl.arange(0, BLOCK_SIZE_M)
        offs_n = on + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        for k in tl.range(
            0,
            num_k,
            warp_specialize=WARP_SPECIALIZE,
            loop_unroll_factor=LOOP_UNROLL_FACTOR,
        ):
            rem = k_span - k * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < rem),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < rem) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = om + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = on + tl.arange(0, BLOCK_SIZE_N)
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def _warmup_fused_splitk(*args, grid, **kwargs):
    """Preserve the JIT kernel warmup interface through the heuristics wrapper."""
    return _gemm_bf16_fused_splitk_kernel.run(
        *args, grid=grid, warmup=True, **kwargs
    )


_gemm_bf16_fused_splitk_kernel.warmup = _warmup_fused_splitk
