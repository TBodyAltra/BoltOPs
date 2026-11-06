"""General Triton implementation under the shared block-scaled GEMM contract."""
import math

from ..registry import Implementation, KernelConfig


def _configurations(p):
    if (p.has_bias or p.kind != "dense" or not p.native_operands or
            p.a_scale is None or p.b_scale is None):
        return ()
    q = math.gcd(p.a_scale.block[1], p.b_scale.block[1])
    return (KernelConfig(32, 64, max(32, min(128, (1 << (q - 1).bit_length())))),)


def _supports(p, config):
    return (config in _configurations(p) and
            p.a.strides == (p.k, 1) and p.b.strides == (p.k, 1) and
            all(s.encoding in ("numeric", "e8m0") and
                (s.encoding != "e8m0" or str(s.dtype) == "torch.uint8")
                for s in (p.a_scale, p.b_scale)))


def _fp4(p, config):
    from .grouped_dispatch import _fp4_types
    return (not p.has_bias and p.kind == "dense" and _fp4_types(p) and config == KernelConfig(16, 32, 32)
            and all(strides is not None and all(s > 0 for s in strides)
                    for strides in (p.a.strides, p.b.strides)))


def launch_scaled_fp4(config, problem, a, b, a_scale, b_scale, out):
    from .gemm_a8wfp4 import gemm_a8wfp4
    gemm_a8wfp4(a, b, a_scale, b_scale, activation_group_size=problem.a_scale.block[1],
                dtype=out.dtype, y=out,
                config={"BLOCK_SIZE_M": config.block_m, "BLOCK_SIZE_N": config.block_n, "num_warps": 4})


IMPLEMENTATIONS = (
    Implementation("triton.scaled.fp8_fp4", "dense", __name__, "launch_scaled_fp4",
                   lambda p: (KernelConfig(16, 32, 32),) if _fp4(p, KernelConfig(16, 32, 32)) else (),
                   _fp4, backend="triton"),
    Implementation("triton.block_scaled", "dense",
                   "boltops.gemm.triton.gemm_block_scaled", "launch_gemm_block_scaled",
                   _configurations, _supports, backend="triton"),
)
