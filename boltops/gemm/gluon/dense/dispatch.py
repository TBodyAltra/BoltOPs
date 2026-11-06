"""Only measured Gluon specializations participate in automatic dispatch."""
from ...registry import Implementation, KernelConfig
from ...performance_map import tuned_configs


def _configurations(problem):
    return tuned_configs(problem, "gluon.block_scaled.128x256x128")


def _supports(problem, config):
    # This device implementation has fixed allocation and indexing geometry.
    return config == KernelConfig(128, 256, 128) and config in _configurations(problem)


IMPLEMENTATIONS = (
    Implementation("gluon.block_scaled.128x256x128", "dense",
                   "boltops.gemm.gluon.dense.kernels.mls_wdra",
                   "launch_gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16_bm128_bn256_bk128",
                   _configurations, _supports, backend="gluon"),
)
