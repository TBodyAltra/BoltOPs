import triton
import triton.language as tl

from boltops.gemm.triton.kernel_repr import make_kernel_repr
from boltops.gemm.triton.pid_preprocessing import pid_grid, remap_xcd
from boltops.gemm.triton.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)

_gemm_a16w16_repr = make_kernel_repr(
    "_gemm_a16_w16_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "EVEN_MN",
        "cache_modifier",
        "activation",
        "use_activation",
        "ADD_BIAS",
        "SKIP_REDUCE",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["SPLITK_BLOCK_SIZE"]) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_gemm_a16w16_repr,
    do_not_specialize=["M", "N"],
)
def _gemm_a16_w16_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_unified = remap_xcd(pid_unified, num_pid_m * num_pid_n * NUM_KSPLIT, NUM_XCDS=8)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        if EVEN_MN:
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        if ADD_BIAS:
            if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                accumulator = tl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
                accumulator = tl.broadcast_to(
                    accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                )
            else:
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        else:
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        for k in range(num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )
            accumulator = tl.dot(a, b, acc=accumulator)
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if use_activation and NUM_KSPLIT == 1:
            accumulator = activation(accumulator)

        # Write back the block of the output matrix C with masks.
        c = accumulator.to(c_ptr.type.element_ty)
        offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        if EVEN_MN:
            tl.store(c_ptrs, c)
        else:
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)


def _get_config(
    M: int,
    N: int,
    K: int,
):
    config, is_tunned = get_gemm_config("GEMM-A16W16", M, N, K)
    return compute_splitk_params(config, K), is_tunned


# ---------------------------------------------------------------------------
# Autotune kernel (added in this copy; the kernel above is the unchanged
# reference "plain" kernel). Operator-level switches for the new features:
#   USE_MLS / USE_MATRIX_STORE : tl.matrix_load / tl.matrix_store
#   WARP_SPECIALIZE            : tl.range(..., warp_specialize=...) (WASP/WDRA
#                                partitioning itself is a launch option)
#   LOOP_UNROLL_FACTOR         : tl.range loop_unroll_factor
#   NUM_KSPLIT                 : >1 writes FP32 partials to c_ptr[pid_k] (3-D),
#                                reduced by a separate kernel (phase 2)
#   ADD_BIAS / bias_ptr        : optional [N] bias (NUM_KSPLIT == 1 only; with split-K
#                                the reduce kernel adds it)
# Launch-only features (no kernel change): wasp_wdra, wasp_partition_warps,
# use_block_pingpong, block_pingpong_async_*, use_async_copy, utc_warmup,
# enable_v_mmac_cluster, mmac_layout_force, ... ds_read_m is on by default.
#
# DYNAMIC M / N / K: there is no EVEN_MNK constexpr any more. M, N, K, strides and the
# split-K tail are runtime values and every tail is handled in the kernel, so one compiled
# kernel serves all shapes of a (tile, schedule) config.
#   * MLS path: matrix_load boundary_check is chosen by the heuristics below ((), (0,) or (0, 1); see the
#     kernel). The K bound handed
#     to the loads is the end of this program's K slice (k_end = min(k_start +
#     SPLITK_BLOCK_SIZE, K)): K % BLOCK_SIZE_K != 0 is zero-padded, the last split is
#     clipped at K and a split never reads into the next split's data.
#   * A split with k_start >= K runs zero iterations and stores an all-zero FP32 partial
#     (the reduce kernel never sees uninitialised memory).
#   * Non-MLS path: rows/cols wrap with % M / % N, K tail is masked, C store is masked.
#   * do_not_specialize=["M", "N"]; 64-bit address math for row/col/partial offsets.
#   * SPLITK_BLOCK_SIZE must be a multiple of BLOCK_SIZE_K; pass 0 when NUM_KSPLIT == 1.
#   * ADD_BIAS: [N], masked load, added in FP32 before the down-cast. Not added when
#     NUM_KSPLIT > 1 (pass the bias to the reduce kernel). With ADD_BIAS the epilogue always
#     uses the pointer store, even if USE_MATRIX_STORE=True (see below).
#
# HEURISTICS (host side, evaluated on every launch from the runtime M / N / K, in the style of
# wasp_splitk_two_stage.py). They are extra constexprs, so each distinct combination is its own
# compiled variant; the caller never passes them:
#   _M_ALIGNED   M % BLOCK_SIZE_M == 0   -> no M boundary_check / no % M wrap / no M store mask
#   _N_ALIGNED   N % BLOCK_SIZE_N == 0   -> no N store mask / no % N wrap
#   _K_ALIGNED   K % BLOCK_SIZE_K == 0 and (NUM_KSPLIT == 1 or SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0)
#                                        -> every K block of every split lies inside the split: no K
#                                           boundary_check / no K mask (an empty split runs 0 iterations)
#   MLS loads:   A unchecked if M and K aligned, (0,) if only M is ragged, else (0, 1);
#                B unchecked if K and N aligned, else (0, 1).
#   _USE_MSTORE  USE_MLS and USE_MATRIX_STORE and NUM_KSPLIT == 1 and not ADD_BIAS and N aligned
#                                        -> tl.matrix_store epilogue. It used to be a host-side check in
#                                           tune_lib.py (ValueError); now an unsupported combination
#                                           (N tail, bias, split-K) quietly takes the pointer store.
# Only the checks that are really needed stay in the kernel, e.g. a tile-aligned problem compiles to a
# kernel with no boundary_check at all (the old EVEN_MNK kernel, but chosen automatically).
#
# tl.matrix_store limitations (gfx946 PMD simulator, fp16, mmac_layout_force=3; verified with
# verify_dynamic_mnk.py):
#   * M tail works for BM=32/BN=256 (M=1, 13; TT and TN) but the simulator panics for BM=32/BN=128
#     with M=13 ("MLS: wong, non major filter is out of size").
#   * N tail is NOT usable: a partial last N tile panics the simulator ("major filter is out of size";
#     BN=128: N=33/60/64/65/129 panic, N=96..128 pass; BN=256: N=100/130/200/300 panic).
#     => matrix_store is only used when N % BLOCK_SIZE_N == 0 (heuristic _USE_MSTORE; otherwise the
#        kernel falls back to the pointer store). M % BLOCK_SIZE_M != 0 is not excluded: it is verified
#        for BM=32/BN=256 only, BM=32/BN=128 with an M tail still panics the simulator.
#   * With matrix_store in the program the accumulator's logical column order is permuted (load/store
#     permutations cancel in a plain GEMM): a bias add gives a column permutation of the bias and a
#     pointer-store fallback for N-tail tiles in the same kernel gives wrong values. Hence ADD_BIAS
#     disables matrix_store inside the kernel.
# ---------------------------------------------------------------------------
@triton.heuristics({
    "_M_ALIGNED": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "_N_ALIGNED": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "_K_ALIGNED": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0
        and (args["NUM_KSPLIT"] == 1 or args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
    "_USE_MSTORE": lambda args: bool(args["USE_MLS"] and args["USE_MATRIX_STORE"]
                                     and args["NUM_KSPLIT"] == 1 and not args.get("ADD_BIAS", False)
                                     and args["N"] % args["BLOCK_SIZE_N"] == 0),
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
    _M_ALIGNED: tl.constexpr,       # heuristics (see above): never passed by the caller
    _N_ALIGNED: tl.constexpr,
    _K_ALIGNED: tl.constexpr,
    _USE_MSTORE: tl.constexpr,
    ADD_BIAS: tl.constexpr = False,
    LOOP_UNROLL_FACTOR: tl.constexpr = 1,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    if USE_MLS:
        tl.assume(M > 0)
        tl.assume(N > 0)
        tl.assume(K > 0)

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

    # ---- K slice of this program: [k_start, k_end), k_span = k_end - k_start --
    if NUM_KSPLIT == 1:
        # SPLITK_BLOCK_SIZE is ignored (callers pass 0); K stays a runtime value.
        k_start = 0
        k_end = K
    else:
        k_start = pid_k * SPLITK_BLOCK_SIZE
        k_end = tl.minimum(k_start + SPLITK_BLOCK_SIZE, K)
    # k_span can be <= 0 for an empty split -> zero iterations, zero partial.
    k_span = tl.maximum(k_end - k_start, 0)
    num_k = tl.cdiv(k_span, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    om = pid_m * BLOCK_SIZE_M
    on = pid_n * BLOCK_SIZE_N

    if USE_MLS:
        # The bound along K is k_end (this split's end), so out-of-range K is masked at the global
        # tail and at split edges. The checks are chosen by the heuristics _M/_N/_K_ALIGNED; () = no check.
        # Measured on PMD (verify_generality.py / verify_dynamic_mnk.py): a SINGLE-dim check is only
        # correct for A's M dim, boundary_check=(0,) (M tail, K aligned). A=(1,), B=(0,), B=(1,) give wrong
        # values / NaN, so any K or N tail checks both dims (0, 1) of that operand.
        for k in tl.range(0, num_k, warp_specialize=WARP_SPECIALIZE,
                          loop_unroll_factor=LOOP_UNROLL_FACTOR):
            ok = k_start + k * BLOCK_SIZE_K
            a = tl.matrix_load(a_ptr, shape=[M, k_end], strides=[stride_am, stride_ak],
                               block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K], offsets=[om, ok],
                               boundary_check=((0, 1) if not _K_ALIGNED
                                               else ((0,) if not _M_ALIGNED else ())))
            b = tl.matrix_load(b_ptr, shape=[k_end, N], strides=[stride_bk, stride_bn],
                               block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N], offsets=[ok, on],
                               boundary_check=((0, 1) if not (_K_ALIGNED and _N_ALIGNED) else ()))
            acc = tl.dot(a, b, acc)
    else:
        if _M_ALIGNED:
            offs_m = om + tl.arange(0, BLOCK_SIZE_M)
        else:
            offs_m = (om + tl.arange(0, BLOCK_SIZE_M)) % M
        if _N_ALIGNED:
            offs_n = on + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_n = (on + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = (a_ptr + offs_m.to(tl.int64)[:, None] * stride_am
                  + (k_start + offs_k)[None, :] * stride_ak)
        b_ptrs = (b_ptr + (k_start + offs_k)[:, None] * stride_bk
                  + offs_n.to(tl.int64)[None, :] * stride_bn)
        for k in tl.range(0, num_k, warp_specialize=WARP_SPECIALIZE,
                          loop_unroll_factor=LOOP_UNROLL_FACTOR):
            if _K_ALIGNED:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                rem = k_span - k * BLOCK_SIZE_K
                a = tl.load(a_ptrs, mask=offs_k[None, :] < rem, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < rem, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # Bias [N] is added in FP32 before the down-cast. With NUM_KSPLIT > 1 the
    # partials are summed by the separate reduce kernel, which adds the bias
    # (ADD_BIAS=True there); adding it here too would count it NUM_KSPLIT times.
    if ADD_BIAS and NUM_KSPLIT == 1:
        offs_bias = on + tl.arange(0, BLOCK_SIZE_N)
        if _N_ALIGNED:
            bias = tl.load(bias_ptr + offs_bias).to(tl.float32)
        else:
            bias = tl.load(bias_ptr + offs_bias, mask=offs_bias < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    c = acc.to(c_ptr.dtype.element_ty)
    if _USE_MSTORE:
        # Heuristic _USE_MSTORE guarantees N % BLOCK_SIZE_N == 0 (no bias, no split-K); rows >= M of an
        # edge tile are dropped by boundary_check.
        tl.matrix_store(c_ptr, c, shape=[M, N], strides=[stride_cm, stride_cn],
                        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N], offsets=[om, on],
                        boundary_check=((0,) if not _M_ALIGNED else ()))
    else:
        offs_cm = om.to(tl.int64) + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = on.to(tl.int64) + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (c_ptr + pid_k.to(tl.int64) * stride_ck + stride_cm * offs_cm[:, None]
                  + stride_cn * offs_cn[None, :])
        if _M_ALIGNED and _N_ALIGNED:
            tl.store(c_ptrs, c)
        else:
            tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def _warmup_generic_kernel(*args, grid, **kwargs):
    """Keep the JIT `kernel.warmup(...)` interface through the heuristics wrapper."""
    return _gemm_bf16_generic_kernel.run(*args, grid=grid, warmup=True, **kwargs)


_gemm_bf16_generic_kernel.warmup = _warmup_generic_kernel
