"""FP8 E4M3 x packed E2M1 GEMM with post-dot block scales.

Packing follows ROCm/aiter utility/fp4_utils.py: low nibble is even K.
Native mixed-format MMAC consumes FP8 activations and packed E2M1 weights.
Each K=32 dot produces FP32 and is scaled before the FP32 outer accumulation.
"""

import torch
import triton
import triton.language as tl

from boltops.utility.fp8_scale import decode_fp8_scale
from ..descriptors import KernelConfig


@triton.jit
def _gemm_a8wfp4_kernel(
    X, W, XS, WS, Y, OFFSETS,
    T: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    E: tl.constexpr, RAGGED: tl.constexpr, GROUP_K: tl.constexpr,
    XE: tl.constexpr, XM: tl.constexpr, XK: tl.constexpr,
    WE: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
    SE: tl.constexpr, SM: tl.constexpr, SK: tl.constexpr,
    VE: tl.constexpr, VN: tl.constexpr, VK: tl.constexpr,
    YE: tl.constexpr, YM: tl.constexpr, YN: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    tile = tl.program_id(0)
    if RAGGED:
        experts = tl.arange(0, triton.next_power_of_2(E))
        starts = tl.load(OFFSETS + experts, experts < E, other=0)
        ends = tl.load(OFFSETS + experts + 1, experts < E, other=0)
        tiles = tl.where(experts < E, tl.cdiv(ends - starts, BM), 0)
        prefix = tl.cumsum(tiles)
        if tile >= tl.sum(tiles, 0):
            return
        expert = tl.sum((tile >= prefix).to(tl.int32), 0)
        start = tl.sum(tl.where(experts == expert, starts, 0), 0)
        rows = tl.sum(tl.where(experts == expert, ends - starts, 0), 0)
        previous = tl.sum(tl.where(experts < expert, tiles, 0), 0)
        tile -= previous
    else:
        expert = tl.program_id(2)
        start = 0
        rows = M

    rm = tile * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, 32)):
        kk = block * 32 + rk
        a = tl.load(X + expert * XE + (start + rm[:, None]) * XM
                    + kk[None, :] * XK,
                    (rm[:, None] < rows) & (kk[None, :] < K), other=0.0)
        kp = block * 16 + tl.arange(0, 16)
        b = tl.load(W + expert * WE + rn[None, :] * WN
                    + kp[:, None] * WK,
                    (rn[None, :] < N) & (kp[:, None] * 2 < K), other=0)
        sa = decode_fp8_scale(tl.load(
            XS + expert * SE + (start + rm) * SM + (block * 32 // GROUP_K) * SK,
            rm < rows, other=0), True).to(tl.float32)
        sb = decode_fp8_scale(tl.load(
            WS + expert * VE + rn * VN + block * VK,
            rn < N, other=0), True).to(tl.float32)
        acc += tl.dot_scaled(a, None, "e4m3", b, None, "e2m1") * sa[:, None] * sb[None, :]
    tl.store(Y + expert * YE + (start + rm[:, None]) * YM + rn[None, :] * YN,
             acc, (rm[:, None] < rows) & (rn[None, :] < N))


def grouped_gemm_a8wfp4(
    a, b, a_scale, b_scale, *, group_offsets=None,
    activation_group_size=128, dtype=torch.bfloat16, out=None, config=None, problem=None, bias=None,):
    """Grouped FP8 E4M3 x packed FP4 E2M1 -> BF16/FP16/FP32.

    Dense grouped input: x[E,M,K], w[E,N,ceil(K/2)],
    x_scales[E,M,ceil(K/G)], w_scales[E,N,ceil(K/32)].
    Ragged input: x[T,K], x_scales[T,ceil(K/G)], expert_offsets[E+1]
    (contiguous int32/int64 CUDA tensor, nondecreasing from 0 to T).
    Inputs must already be grouped by expert; empty experts are allowed.
    Scale dtype uint8 means raw E8M0; FP32/FP16/BF16 means a numeric scale.
    G is 32 or 128. Weight scales are always per 32 logical K elements.
    Packed weights use low nibble for even K, high nibble for odd K.
    All tensors may have noncontiguous positive strides; y must not overlap
    inputs. No host dequantization is used; native dot_scaled lowering is required.
    """
    if problem is not None:
        activation_group_size = problem.a_scale.block[1]
        config = {"BLOCK_SIZE_M": config["BLOCK_SIZE_M"], "BLOCK_SIZE_N": config["BLOCK_SIZE_N"], "num_warps": 4}
    if out is not None:
        dtype = out.dtype
    if activation_group_size not in (32, 128):
        raise ValueError("activation_group_size must be 32 or 128")
    if a.dtype != torch.float8_e4m3fn:
        raise TypeError("x must contain FP8 E4M3 FN values")
    if b.dtype != torch.uint8 or b.ndim != 3:
        raise TypeError("w must be packed uint8 with shape [E,N,ceil(K/2)]")
    ragged = group_offsets is not None
    if a.ndim != (2 if ragged else 3):
        raise ValueError("x must be [T,K] with offsets, otherwise [E,M,K]")
    e, n, packed_k = b.shape
    k = a.shape[-1]
    if e < 1 or k < 1 or packed_k != triton.cdiv(k, 2):
        raise ValueError("inconsistent expert count or packed K dimension")
    if ragged:
        t = a.shape[0]
        m = 0
        if group_offsets.shape != (e + 1,) or group_offsets.dtype not in (torch.int32, torch.int64) or not group_offsets.is_contiguous():
            raise ValueError("expert_offsets must be contiguous int32/int64 [E+1]")
        expected_xs = (t, triton.cdiv(k, activation_group_size))
        out_shape = (t, n)
    else:
        if a.shape[0] != e:
            raise ValueError("x and w must have the same expert count")
        m = a.shape[1]
        t = e * m
        expected_xs = (e, m, triton.cdiv(k, activation_group_size))
        out_shape = (e, m, n)
    if a_scale.shape != expected_xs or b_scale.shape != (e, n, triton.cdiv(k, 32)):
        raise ValueError("scale shapes must match the specified K groups")
    for scale in (a_scale, b_scale):
        if scale.dtype not in (torch.uint8, torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("scales must be uint8 E8M0 or floating point")
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be BF16, FP16 or FP32")
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=a.device)
    if out.shape != out_shape or out.dtype != dtype:
        raise ValueError("y has incorrect shape or dtype")
    tensors = [a, b, a_scale, b_scale, out]
    if ragged:
        tensors.append(group_offsets)
    if any(not tensor.is_cuda or tensor.device != a.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA/HIP device")
    if any(a.numel() and any(s <= 0 for s in a.stride()) for a in tensors):
        raise ValueError("all tensors must have positive strides")
    cfg = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, num_warps=4)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unsupported config keys: {sorted(unknown)}")
        cfg.update(config)
    bm, bn = cfg['BLOCK_SIZE_M'], cfg['BLOCK_SIZE_N']
    if bm < 16 or bn < 16 or bm & (bm - 1) or bn & (bn - 1):
        raise ValueError("M/N blocks must be powers of two >=16")
    if t == 0 or n == 0:
        return out
    grid = ((triton.cdiv(t, bm) + e - 1) if ragged else triton.cdiv(m, bm),
            triton.cdiv(n, bn), 1 if ragged else e)
    def strides(a):
        return (0, *a.stride()) if a.ndim == 2 else a.stride()
    _gemm_a8wfp4_kernel[grid](
        a, b, a_scale, b_scale, out, group_offsets,
        t, m, n, k, e, ragged, activation_group_size,
        *strides(a), *b.stride(), *strides(a_scale), *b_scale.stride(),
        *strides(out), bm, bn, num_warps=cfg['num_warps'], num_stages=1,
        enable_fp_fusion=False,
    )
    return out


def gemm_a8wfp4(a, b, a_scale, b_scale, *, activation_group_size=128,
                dtype=torch.bfloat16, out=None, config=None, problem=None,):
    """Dense Y=X@W.T; x[M,K], packed w[N,ceil(K/2)], matching block scales."""
    if problem is not None:
        activation_group_size = problem.a_scale.block[1]
        config = {"BLOCK_SIZE_M": config["BLOCK_SIZE_M"], "BLOCK_SIZE_N": config["BLOCK_SIZE_N"], "num_warps": 4}
    if out is not None:
        dtype = out.dtype
    if any(tensor.ndim != 2 for tensor in (a, b, a_scale, b_scale)):
        raise ValueError("dense GEMM inputs and scales must be 2-dimensional")
    result = grouped_gemm_a8wfp4(
        a.unsqueeze(0), b.unsqueeze(0), a_scale.unsqueeze(0), b_scale.unsqueeze(0),
        activation_group_size=activation_group_size, dtype=dtype,
        out=None if out is None else out.unsqueeze(0), config=config,
    )
    return result.squeeze(0) if out is None else out


def _fp4_types(p):
    scales = (p.a_scale, p.b_scale)
    return (p.a_dtype == torch.float8_e4m3fn and p.b_dtype == torch.uint8 and
            p.a.format == 'native' and p.b.format == 'fp4' and p.b.packing == 'e2m1_low_high' and
            all(s is not None and ((s.encoding == 'e8m0' and s.dtype == torch.uint8) or
                (s.encoding == 'numeric' and s.dtype in (torch.float16, torch.bfloat16, torch.float32))) for s in scales) and
            p.a_scale.block in ((1, 32), (1, 128)) and p.b_scale.block == (1, 32) and
            p.out_dtype in (torch.float16, torch.bfloat16, torch.float32))


def _fp4(p):
    return p.kind == 'grouped' and p.groups.mode in ('equal', 'offsets') and _fp4_types(p)


def grouped_configs(p):
    return (KernelConfig(16, 32, 32),) if _fp4(p) else ()


def supports_mixed_fp4(p, config):
    return (not p.has_bias and p.kind == "dense" and _fp4_types(p) and config == KernelConfig(16, 32, 32)
            and all(strides is not None and all(s > 0 for s in strides)
                    for strides in (p.a.strides, p.b.strides)))


def mixed_fp4_configs(p):
    c = KernelConfig(16, 32, 32)
    return (c,) if supports_mixed_fp4(p, c) else ()
