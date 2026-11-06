"""Select a native implementation and its compatible tile configuration.

A configuration identifies a pipeline, not arbitrary tile substitutions in a
fixed pipeline. New tile/pipeline implementations register another launcher.
No architecture tuning or external tuning-file loading is implied here.
"""
from dataclasses import dataclass
from importlib import import_module
import math

import torch
import triton


@dataclass(frozen=True)
class BlockScaledConfig:
    pipeline: str
    block_m: int
    block_n: int
    block_k: int


_MLS = BlockScaledConfig("mls_wdra48_128x256x128", 128, 256, 128)
_MASKED_FP8 = BlockScaledConfig("masked_fp8_64x128x128", 64, 128, 128)
_PIPELINES = {
    _MLS.pipeline: ("block_scaled_mls_wdra48_128x256x128", "launch"),
    _MASKED_FP8.pipeline: ("block_scaled_masked", "launch_fp8"),
    "masked_general_32x64": ("block_scaled_masked", "launch_general"),
}


def select_config(a, b, a_scale, b_scale, recipe, out_dtype):
    """Preserve the existing recipe, shape, alignment and LDS-capacity gates."""
    m, k = a.shape
    n = b.shape[0]
    tuned = (recipe == ((1, 128), (128, 128)) and
             a.dtype == b.dtype == torch.float8_e4m3fn and
             a_scale.dtype == b_scale.dtype == torch.float32 and
             out_dtype == torch.bfloat16)
    if tuned:
        # A/B double buffering uses 96 KiB; immutable scales grow with K.
        # K6144 uses 120.375 KiB. This implementation requires full K pairs.
        if (m % _MLS.block_m == 0 and n % _MLS.block_n == 0 and
                256 <= k <= 6144 and k % 256 == 0 and
                a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0):
            return _MLS
        return _MASKED_FP8
    q = math.gcd(recipe[0][1], recipe[1][1])
    dot_bk = max(32, min(128, triton.next_power_of_2(q)))
    return BlockScaledConfig("masked_general_32x64", 32, 64, dot_bk)


def launch(config, a, b, a_scale, b_scale, out, *, recipe):
    """Load only the selected implementation; launchers own layouts/resources."""
    module, entry = _PIPELINES[config.pipeline]
    implementation = import_module(".pipelines." + module, __package__)
    getattr(implementation, entry)(config, a, b, a_scale, b_scale, out, recipe=recipe)
