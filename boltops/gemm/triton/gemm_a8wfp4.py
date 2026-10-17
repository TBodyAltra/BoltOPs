"""FP8 E4M3 x packed E2M1 GEMM with post-dot block scales.

Packing follows ROCm/aiter utility/fp4_utils.py: low nibble is even K.
Unlike aiter's native dot_scaled kernel, this implementation decodes E2M1
exactly to E4M3 and applies scales after each ordinary K=32 dot product.
"""

import torch
import triton
import triton.language as tl

from boltops.utility.fp8_scale import decode_fp8_scale


@triton.jit
def _decode_e2m1(code, FP8_T: tl.constexpr):
    mag = code & 7
    # Every E2M1 value has an exact E4M3 representation. Construct those bits
    # directly, avoiding float arithmetic and scaled float-to-FP8 conversion.
    bits = tl.where(mag == 0, 0,
                    tl.where(mag == 1, 0x30,
                             ((mag // 2 + 6) << 3) + ((mag & 1) << 2)))
    bits = bits | ((code & 8) << 4)
    return bits.to(tl.uint8).to(FP8_T, bitcast=True)


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
        packed = tl.load(W + expert * WE + rn[None, :] * WN
                         + (kk[:, None] // 2) * WK,
                         (rn[None, :] < N) & (kk[:, None] < K), other=0)
        code = (packed >> ((kk[:, None] & 1) * 4)) & 15
        b = _decode_e2m1(code, X.dtype.element_ty)
        sa = decode_fp8_scale(tl.load(
            XS + expert * SE + (start + rm) * SM + (block * 32 // GROUP_K) * SK,
            rm < rows, other=0), True).to(tl.float32)
        sb = decode_fp8_scale(tl.load(
            WS + expert * VE + rn * VN + block * VK,
            rn < N, other=0), True).to(tl.float32)
        acc += tl.dot(a, b).to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(Y + expert * YE + (start + rm[:, None]) * YM + rn[None, :] * YN,
             acc, (rm[:, None] < rows) & (rn[None, :] < N))


def grouped_gemm_a8wfp4(
    x, w, x_scales, w_scales, *, expert_offsets=None,
    activation_group_size=128, dtype=torch.bfloat16, y=None, config=None,
):
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
    inputs. No host dequantization or dot_scaled dependency is used.
    """
    if activation_group_size not in (32, 128):
        raise ValueError("activation_group_size must be 32 or 128")
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("x must contain FP8 E4M3 FN values")
    if w.dtype != torch.uint8 or w.ndim != 3:
        raise TypeError("w must be packed uint8 with shape [E,N,ceil(K/2)]")
    ragged = expert_offsets is not None
    if x.ndim != (2 if ragged else 3):
        raise ValueError("x must be [T,K] with offsets, otherwise [E,M,K]")
    e, n, packed_k = w.shape
    k = x.shape[-1]
    if e < 1 or k < 1 or packed_k != triton.cdiv(k, 2):
        raise ValueError("inconsistent expert count or packed K dimension")
    if ragged:
        t = x.shape[0]
        m = 0
        if expert_offsets.shape != (e + 1,) or expert_offsets.dtype not in (torch.int32, torch.int64) or not expert_offsets.is_contiguous():
            raise ValueError("expert_offsets must be contiguous int32/int64 [E+1]")
        expected_xs = (t, triton.cdiv(k, activation_group_size))
        out_shape = (t, n)
    else:
        if x.shape[0] != e:
            raise ValueError("x and w must have the same expert count")
        m = x.shape[1]
        t = e * m
        expected_xs = (e, m, triton.cdiv(k, activation_group_size))
        out_shape = (e, m, n)
    if x_scales.shape != expected_xs or w_scales.shape != (e, n, triton.cdiv(k, 32)):
        raise ValueError("scale shapes must match the specified K groups")
    for scale in (x_scales, w_scales):
        if scale.dtype not in (torch.uint8, torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("scales must be uint8 E8M0 or floating point")
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be BF16, FP16 or FP32")
    if y is None:
        y = torch.empty(out_shape, dtype=dtype, device=x.device)
    if y.shape != out_shape or y.dtype != dtype:
        raise ValueError("y has incorrect shape or dtype")
    tensors = [x, w, x_scales, w_scales, y]
    if ragged:
        tensors.append(expert_offsets)
    if any(not a.is_cuda or a.device != x.device for a in tensors):
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
        return y
    grid = ((triton.cdiv(t, bm) + e - 1) if ragged else triton.cdiv(m, bm),
            triton.cdiv(n, bn), 1 if ragged else e)
    def strides(a):
        return (0, *a.stride()) if a.ndim == 2 else a.stride()
    _gemm_a8wfp4_kernel[grid](
        x, w, x_scales, w_scales, y, expert_offsets,
        t, m, n, k, e, ragged, activation_group_size,
        *strides(x), *w.stride(), *strides(x_scales), *w_scales.stride(),
        *strides(y), bm, bn, num_warps=cfg['num_warps'], num_stages=1,
        enable_fp_fusion=False,
    )
    return y


def gemm_a8wfp4(x, w, x_scales, w_scales, *, activation_group_size=128,
                dtype=torch.bfloat16, y=None, config=None):
    """Dense Y=X@W.T; x[M,K], packed w[N,ceil(K/2)], matching block scales."""
    if any(a.ndim != 2 for a in (x, w, x_scales, w_scales)):
        raise ValueError("dense GEMM inputs and scales must be 2-dimensional")
    out = grouped_gemm_a8wfp4(
        x.unsqueeze(0), w.unsqueeze(0), x_scales.unsqueeze(0), w_scales.unsqueeze(0),
        activation_group_size=activation_group_size, dtype=dtype,
        y=None if y is None else y.unsqueeze(0), config=config,
    )
    return out.squeeze(0) if y is None else y
