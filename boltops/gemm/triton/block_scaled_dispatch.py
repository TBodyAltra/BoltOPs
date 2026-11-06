"""General Triton implementation under the shared block-scaled GEMM contract."""
import math

import triton

from ..registry import Implementation, KernelConfig


def _configurations(p):
    if (p.kind != "dense" or not p.native_operands or
            p.a_scale is None or p.b_scale is None):
        return ()
    q = math.gcd(p.a_scale.block[1], p.b_scale.block[1])
    return (KernelConfig(32, 64, max(32, min(128, triton.next_power_of_2(q)))),)


def _supports(p, config):
    return (config in _configurations(p) and
            p.a.strides == (p.k, 1) and p.b.strides == (p.k, 1) and
            all(s.encoding in ("numeric", "e8m0") and
                (s.encoding != "e8m0" or str(s.dtype) == "torch.uint8")
                for s in (p.a_scale, p.b_scale)))


IMPLEMENTATIONS = (
    Implementation("triton.block_scaled", "dense",
                   "boltops.gemm.triton.gemm_block_scaled", "launch_gemm_block_scaled",
                   _configurations, _supports, backend="triton"),
)
