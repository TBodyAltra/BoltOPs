"""Capability-driven dense kernel selection; tile choices belong to implementations."""
from dataclasses import dataclass
from importlib import import_module
import math
from typing import Callable

import torch
import triton

from .descriptors import DenseProblem


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int


@dataclass(frozen=True)
class Implementation:
    name: str
    module: str
    entry: str
    configurations: Callable[[DenseProblem], tuple[KernelConfig, ...]]
    supports: Callable[[DenseProblem, KernelConfig], bool]


@dataclass(frozen=True)
class ExecutionPlan:
    implementation: Implementation
    config: KernelConfig


def _fp8_fp32_k128(p):
    return (p.a_dtype == p.b_dtype == torch.float8_e4m3fn and
            p.out_dtype == torch.bfloat16 and
            p.a_scale.dtype == p.b_scale.dtype == torch.float32 and
            p.a_scale.encoding == p.b_scale.encoding == "numeric" and
            p.a_scale.block == (1, 128) and p.b_scale.block == (128, 128))


_MLS_CONFIG = KernelConfig(128, 256, 128)
_FP8_CONFIG = KernelConfig(64, 128, 128)


def _supports_mls(p, config):
    # This resource/geometry contract is specific to the measured implementation.
    return (config == _MLS_CONFIG and p.arch == "gfx946" and _fp8_fp32_k128(p) and
            p.m % config.block_m == 0 and p.n % config.block_n == 0 and
            256 <= p.k <= 6144 and p.k % 256 == 0 and
            p.a_aligned_16 and p.b_aligned_16)


def _supports_masked_fp8(p, config):
    return config == _FP8_CONFIG and _fp8_fp32_k128(p)


def _general_configs(p):
    q = math.gcd(p.a_scale.block[1], p.b_scale.block[1])
    return (KernelConfig(32, 64, max(32, min(128, triton.next_power_of_2(q)))),)


def _supports_general(p, config):
    # Native input/accumulator legality remains in the target MMAC database.
    return (config in _general_configs(p) and
            all(scale.encoding in ("numeric", "e8m0") and
                (scale.encoding != "e8m0" or scale.dtype == torch.uint8)
                for scale in (p.a_scale, p.b_scale)))


IMPLEMENTATIONS = (
    Implementation("mls_wdra", "mls_wdra", "launch",
                   lambda p: (_MLS_CONFIG,), _supports_mls),
    Implementation("masked_fp8", "masked", "launch_fp8",
                   lambda p: (_FP8_CONFIG,), _supports_masked_fp8),
    Implementation("masked", "masked", "launch_general",
                   _general_configs, _supports_general),
)


def select_plan(problem, implementations=IMPLEMENTATIONS):
    """First eligible implementation wins; no operator-specific branches here."""
    for implementation in implementations:
        for config in implementation.configurations(problem):
            if implementation.supports(problem, config):
                return ExecutionPlan(implementation, config)
    raise NotImplementedError(f"no dense kernel implements {problem}")


def launch(plan, problem, a, b, a_scale, b_scale, out):
    implementation = plan.implementation
    if not implementation.supports(problem, plan.config):
        raise ValueError(f"configuration {plan.config} is incompatible with {implementation.name}")
    module = import_module(".kernels." + implementation.module, __package__)
    getattr(module, implementation.entry)(plan.config, problem, a, b, a_scale, b_scale, out)
