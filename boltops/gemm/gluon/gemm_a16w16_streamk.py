# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Task scheduling and dynamic-LDS launch for the BF16 Stream-K kernel."""

from collections import defaultdict
from functools import lru_cache

import torch
import triton
from triton import knobs
from triton.experimental.gluon._runtime import GluonASTSource
from triton.runtime.driver import driver


_SHARED_BYTES = 96 * 1024


@lru_cache(maxsize=128)
def _make_schedule(m, n, k, cu_count):
    if min(m, n, k, cu_count) <= 0 or m % 4 or n % 128 or k % 16:
        raise ValueError("Stream-K Gluon requires M % 4 = N % 128 = K % 16 = 0")
    rows, cols, steps = triton.cdiv(m, 256), n // 128, triton.cdiv(k, 64)
    tiles = rows * cols
    if tiles < cu_count:
        raise ValueError("Stream-K Gluon requires at least one output tile per CU")
    # Traverse groups of four M tiles to reuse the B panels.
    physical = [row * cols + col for first in range(0, rows, 4)
                for col in range(cols) for row in range(first, min(first + 4, rows))]
    tasks = [[] for _ in range(cu_count)]
    by_tile = defaultdict(list)
    work = tiles * steps
    for pid in range(cu_count):
        cursor, end = pid * work // cu_count, (pid + 1) * work // cu_count
        while cursor < end:
            tile, start = divmod(cursor, steps)
            stop = min(end, (tile + 1) * steps)
            record = [physical[tile], start * 64, (stop - cursor) * 64, 0]
            tasks[pid].append(record)
            by_tile[record[0]].append(record)
            cursor = stop
    slots = 0
    for _, fragments in sorted(by_tile.items()):
        if len(fragments) == 2:
            # The higher-K owner publishes; the lower-K owner merges and stores.
            fragments[0][3], fragments[1][3] = -(slots + 1), slots + 1
            slots += 1
    # Include a zero-length sentinel for every TG, as in the source schedule.
    records = tiles + 1
    table = tuple(tuple(record) for tg in tasks
                  for record in tg + [[0, 0, 0, 0]] * (records - len(tg)))
    return table, records, slots


@lru_cache(maxsize=128)
def _device_schedule(m, n, k, cu_count, device, stream):
    # Key by stream so the asynchronous schedule upload never crosses streams.
    table, records, slots = _make_schedule(m, n, k, cu_count)
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


def _launch_gemm_a16w16_streamk(kernel, prepare_edges, a, b, out, *, cu_count, config):
    """Launch one TG per CU, reserving LDS used by the kernel's inline MLS ops.

    The immutable task table is cached per stream. Scratch and zeroed flags
    belong to each invocation, so concurrent streams and CUDA graph replays
    cannot observe completion flags from a previous GEMM.
    """
    m, k = a.shape
    n = b.shape[0]
    if a.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("Stream-K Gluon requires BF16 input and output")
    b_k_contiguous = b.stride() == (k, 1)
    if a.stride() != (k, 1) or b.stride() not in ((k, 1), (1, n)):
        raise ValueError("Stream-K Gluon requires contiguous x and TN or TT w")
    if b_k_contiguous and k % 64:
        raise ValueError("Stream-K Gluon requires K divisible by 64 for TN")
    if any(t.data_ptr() % 16 for t in (a, b, out)):
        raise ValueError("Stream-K Gluon requires 16-byte-aligned tensor pointers")
    tile = tuple(config.get(key) for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
    if tile != (256, 128, 64):
        raise ValueError("Stream-K Gluon requires a 256x128x64 tile")
    options = {key: tuple(value) if isinstance(value, list) else value
               for key, value in config.items()
               if key not in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "NUM_SMS")}
    if (options.get("num_warps") != 2
            or options.get("wasp_partition_warps") != (2, 2, 4, 4)
            or options.get("wasp_wdra") is not False):
        raise ValueError("Stream-K Gluon requires num_warps=2 and partitions (2,2,4,4) without WDRA")
    stream = torch.cuda.current_stream(a.device)
    schedule, records, slots = _device_schedule(m, n, k, cu_count, a.device, stream.cuda_stream)
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
        prepare_edges[(sum(triton.cdiv(size, 1024) for size in edge_sizes),)](
            a, b, *edges, M=m, N=n, K=k, BLOCK=1024, num_warps=4, num_stages=1,
        )
    args = (a, b, out, schedule, workspace, flags, *edges, 1)
    grid = (cu_count, 1, 1)
    # These LDS addresses are expressed in inline assembly and are invisible to
    # Triton's allocator. Override only the launch's dynamic-LDS reservation;
    # keep the compiled kernel and its cached metadata unchanged.
    packed = (compiled.metadata.num_warps, compiled.metadata.num_ctas, _SHARED_BYTES)
    run(*grid, stream.cuda_stream, compiled.function, packed,
        compiled.launch_metadata(grid, stream.cuda_stream, *args),
        knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
