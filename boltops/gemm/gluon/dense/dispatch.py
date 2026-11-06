"""Only measured Gluon specializations participate in automatic dispatch."""
from ...registry import Implementation, KernelConfig
from ...performance_map import tuned_configs


def _configurations(problem):
    return tuned_configs(problem, "gluon.fp8e4m3fn.fp32scale.a1x128_b128x128.to_bf16")


def _supports(problem, config):
    # This device implementation has fixed allocation and indexing geometry.
    return config == KernelConfig(128, 256, 128) and config in _configurations(problem)


IMPLEMENTATIONS = (
    Implementation("gluon.fp8e4m3fn.fp32scale.a1x128_b128x128.to_bf16", "dense",
                   "boltops.gemm.gluon.dense.kernels.gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16",
                   "launch_gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16_bm128_bn256_bk128",
                   _configurations, _supports, backend="gluon"),
)
