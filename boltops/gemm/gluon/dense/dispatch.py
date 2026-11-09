"""Fixed Gluon pipeline capabilities; measured winners are selected centrally."""
from ...registry import Implementation, KernelConfig


def _configurations(p):
    c = KernelConfig(128, 256, 128)
    return (c,) if _supports(p, c) else ()


def _supports(p, config):
    # Legality is independent of measured winners so explicit tuning can try
    # the fixed pipeline even when this exact shape has no performance record.
    return (config == KernelConfig(128, 256, 128) and p.kind == "dense" and
            p.native_operands and not p.has_bias and p.arch == "gfx946" and p.num_cus == 48 and
            [str(t) for t in (p.a_dtype, p.b_dtype, p.out_dtype)] ==
                ["torch.float8_e4m3fn", "torch.float8_e4m3fn", "torch.bfloat16"] and
            p.m > 0 and p.n > 0 and p.m % 128 == 0 and p.n % 256 == 0 and
            256 <= p.k <= 6144 and p.k % 256 == 0 and
            p.a.strides == p.b.strides == (p.k, 1) and p.a_aligned_16 and p.b_aligned_16 and
            p.a_scale is not None and p.b_scale is not None and
            [str(s.dtype) for s in (p.a_scale, p.b_scale)] == ["torch.float32", "torch.float32"] and
            p.a_scale.block == (1, 128) and p.b_scale.block == (128, 128) and
            p.a_scale.encoding == p.b_scale.encoding == "numeric")


IMPLEMENTATIONS = (
    Implementation("gluon.fp8e4m3fn.fp32scale.a1x128_b128x128.to_bf16", "dense",
                   "boltops.gemm.gluon.dense.kernels.gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16",
                   "launch_gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16_bm128_bn256_bk128",
                   _configurations, _supports, backend="gluon"),
)
