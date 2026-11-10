"""Fused BF16 GEMM with in-kernel split-K reduce (single dispatch).

Each TG owns one (pid_m, pid_n, pid_k) tile-slice:
  1. GEMM over its K slice into an FP32 accumulator
  2. Store the FP32 partial into workspace[pid_k, om:om+BM, on:on+BN]
  3. tid==0 does tl.atomic_add on a per-tile int32 counter (acq_rel, gpu)
  4. is_last is broadcast within the wave via v_readfirstlane_b32, then the
     whole wave reduces all KSPLIT partials (volatile loads), stores BF16 C,
     and tid==0 resets the counter to 0

Dynamic M/N/K support
---------------------
The caller no longer passes EVEN_MNK. Instead `@triton.heuristics` derives
the compile-time alignment flags from the runtime arguments:

  _M_ALIGNED : M % BLOCK_SIZE_M == 0        -> else boundary_check on A rows
  _N_ALIGNED : N % BLOCK_SIZE_N == 0        -> else boundary_check on B cols
  _EVEN_K    : K % BLOCK_SIZE_K == 0 and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
                                            -> else boundary_check on the K
                                               extent of the *slice*
  _SINGLE_WAVE : num_warps == 1             -> required by the readfirstlane
                                               broadcast of is_last

K slicing: split `pid_k` covers [pid_k*SPLITK_BLOCK_SIZE, k_end) where k_end is
min(k_start + SPLITK_BLOCK_SIZE, K) and the last split always extends to K, so
no K tail is ever dropped. A split may be partial or even empty (K small);
an empty split writes a zero partial and still takes part in the counter
protocol, so the result stays correct.

The MLS `matrix_load` is given the *slice* as its logical tensor
(`shape=[M, k_span]` / `[k_span, N]`), so `boundary_check` zero-fills both the
M/N overhang and the K overhang of the slice (a K slice never reads into its
neighbour's K range).

Cross-TG global flags are intentionally NOT used: TGs that share a tile would
race on a shared flag word (observed as counters stuck at 8/16/24 and NaNs).

num_warps>1 caveat: readfirstlane is wave-scoped; only wave0 receives the
is_last bit. This is enforced with a static_assert (num_warps must be 1).
"""
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
    "_SINGLE_WAVE": lambda args: args.get("num_warps", 4) == 1,
    "_M_ALIGNED": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "_N_ALIGNED": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "_EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0
        and args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0,
})
@triton.jit
def _gemm_bf16_fused_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    partial_ptr,
    counter_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_pk,
    stride_pm,
    stride_pn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    USE_MLS: tl.constexpr,
    LOOP_UNROLL_FACTOR: tl.constexpr,
    _SINGLE_WAVE: tl.constexpr,
    _M_ALIGNED: tl.constexpr,
    _N_ALIGNED: tl.constexpr,
    _EVEN_K: tl.constexpr,
):
    tl.static_assert(NUM_KSPLIT > 0)
    tl.static_assert(SPLITK_BLOCK_SIZE > 0)
    tl.static_assert(GROUP_SIZE_M > 0)
    tl.static_assert(_SINGLE_WAVE, "Wave-only is_last broadcast requires num_warps == 1")
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
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tile_id = pid_m * num_pid_n + pid_n

    # K slice [k_start, k_end). The last split absorbs any K tail; a split
    # that starts past K is empty (k_span == 0, num_k == 0).
    k_start = pid_k * SPLITK_BLOCK_SIZE
    k_end = tl.minimum(k_start + SPLITK_BLOCK_SIZE, K)
    if pid_k == NUM_KSPLIT - 1:
        k_end = K
    k_span = tl.maximum(k_end - k_start, 0)
    num_k = tl.cdiv(k_span, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    om = pid_m * BLOCK_SIZE_M
    on = pid_n * BLOCK_SIZE_N

    a_slice = a_ptr + k_start * stride_ak
    b_slice = b_ptr + k_start * stride_bk
    if USE_MLS:
        for k in tl.range(
            0,
            num_k,
            warp_specialize=WARP_SPECIALIZE,
            loop_unroll_factor=LOOP_UNROLL_FACTOR,
        ):
            ok = k * BLOCK_SIZE_K
            # None entries are dropped by matrix_load, so aligned dims are
            # not checked at all (no extra instructions).
            a = tl.matrix_load(
                a_slice,
                shape=[M, k_span],
                strides=[stride_am, stride_ak],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                offsets=[om, ok],
                boundary_check=(
                    0 if not _M_ALIGNED else None,
                    1 if not _EVEN_K else None,
                ),
            )
            # NOTE: for B, a K-only check (boundary_check=(0,)) was observed to
            # produce NaN/garbage on gfx946 PMD (when N is aligned but K is
            # not). Checking dim 1 together with dim 0 is correct, so whenever
            # K needs a check, N is checked too (a no-op for an aligned N).
            b = tl.matrix_load(
                b_slice,
                shape=[k_span, N],
                strides=[stride_bk, stride_bn],
                block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                offsets=[ok, on],
                boundary_check=(
                    0 if not _EVEN_K else None,
                    1 if (not _N_ALIGNED or not _EVEN_K) else None,
                ),
            )
            acc = tl.dot(a, b, acc)
    else:
        offs_m = om + tl.arange(0, BLOCK_SIZE_M)
        offs_n = on + tl.arange(0, BLOCK_SIZE_N)
        if not _M_ALIGNED:
            offs_m = offs_m % M
        if not _N_ALIGNED:
            offs_n = offs_n % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_slice + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_slice + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        for k in tl.range(
            0,
            num_k,
            warp_specialize=WARP_SPECIALIZE,
            loop_unroll_factor=LOOP_UNROLL_FACTOR,
        ):
            if _EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                rem = k_span - k * BLOCK_SIZE_K
                a = tl.load(a_ptrs, mask=offs_k[None, :] < rem, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < rem, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = om + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = on + tl.arange(0, BLOCK_SIZE_N)
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    p_ptrs = (
        partial_ptr
        + pid_k * stride_pk
        + stride_pm * offs_cm[:, None]
        + stride_pn * offs_cn[None, :]
    )
    if _M_ALIGNED and _N_ALIGNED:
        tl.store(p_ptrs, acc)
    else:
        tl.store(p_ptrs, acc, mask=out_mask)

    # Arrival counter (one increment per TG). Release publishes partial stores.
    tid = libdevice.thread_idx(0)
    is_last = tl.full((), 0, tl.int32)
    if tid == 0:
        count = tl.atomic_add(counter_ptr + tile_id, 1, sem="acq_rel", scope="gpu")
        if count == NUM_KSPLIT - 1:
            is_last = 1
    # Wave-wide broadcast of lane0's decision (correct for num_warps==1).
    is_last_b = _broadcast_lane0_i32(is_last)

    if is_last_b != 0:
        # volatile: atomics order through L2; bypass possibly-stale L1 copies
        # of peers' partials written by other TGs.
        acc_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for s in tl.static_range(0, NUM_KSPLIT):
            sp = (
                partial_ptr
                + s * stride_pk
                + stride_pm * offs_cm[:, None]
                + stride_pn * offs_cn[None, :]
            )
            if _M_ALIGNED and _N_ALIGNED:
                part = tl.load(sp, volatile=True)
            else:
                part = tl.load(sp, mask=out_mask, other=0.0, volatile=True)
            acc_sum += part
        out = acc_sum.to(tl.bfloat16)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        if _M_ALIGNED and _N_ALIGNED:
            tl.store(c_ptrs, out)
        else:
            tl.store(c_ptrs, out, mask=out_mask)
        if tid == 0:
            tl.store(counter_ptr + tile_id, 0)


def _warmup_fused_splitk(*args, grid, **kwargs):
    """Preserve the JIT kernel warmup interface through the heuristics wrapper."""
    return _gemm_bf16_fused_splitk_kernel.run(
        *args, grid=grid, warmup=True, **kwargs
    )


_gemm_bf16_fused_splitk_kernel.warmup = _warmup_fused_splitk
