# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Look up an interface configuration, then call the selected implementation."""
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec

from .descriptors import KernelConfig

BACKENDS = ("triton", "gluon", "tilelang")

# A new pipeline adds one row here. Missing backends need no empty registries.
IMPLEMENTATIONS = {
    'triton.dense.fp16': {
        'backend': 'triton',
        'interface': 'dense',
        'module': 'boltops.gemm.triton.dense',
        'launcher': 'launch_dense_fp16',
        'configs': '_tuned_configurations',
    },
    'triton.dense.native': {
        'backend': 'triton',
        'interface': 'dense',
        'module': 'boltops.gemm.triton.dense',
        'launcher': 'launch_dense',
        'configs': '_general_configurations',
    },
    'triton.scaled.block_fp8': {
        'backend': 'triton',
        'interface': 'scaled',
        'module': 'boltops.gemm.triton.scaled',
        'launcher': 'launch_block_scaled',
        'configs': '_block_configs',
    },
    'triton.scaled.row_fp8': {
        'backend': 'triton',
        'interface': 'scaled',
        'module': 'boltops.gemm.triton.scaled',
        'launcher': 'launch_row_scaled',
        'configs': '_row_configs',
    },
    'triton.batched.native16': {
        'backend': 'triton',
        'interface': 'batched',
        'module': 'boltops.gemm.triton.batched',
        'launcher': 'launch_native',
        'default_config': {
            'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 2,
            'BATCH_LAST_GRID': False,
        },
        'capabilities': {
            'input_dtype': ('float16', 'bfloat16'),
            'out_dtype': ('float16', 'bfloat16'),
            'native_operands': (True,), 'unscaled': (True,),
            'a_layout': ('k_contiguous', 'm_contiguous'),
            'b_layout': ('k_contiguous', 'n_contiguous'),
        },
        'validate_config': 'validate_config',
    },
    'triton.grouped.native16': {
        'backend': 'triton',
        'interface': 'grouped',
        'module': 'boltops.gemm.triton.grouped',
        'launcher': 'launch_grouped_native',
        'configs': '_native_config',
    },
    'triton.scaled.awq': {
        'backend': 'triton',
        'interface': 'scaled',
        'module': 'boltops.gemm.triton.quantized',
        'launcher': 'launch_awq',
        'configs': '_awq_configs',
    },
    'triton.scaled.mxfp4': {
        'backend': 'triton',
        'interface': 'scaled',
        'module': 'boltops.gemm.triton.quantized',
        'launcher': 'launch_mxfp4',
        'configs': '_fp4_configs',
    },
    'triton.batched_a16wfp4_online_quant': {
        'backend': 'triton',
        'interface': 'batched_a16wfp4_online_quant',
        'module': 'boltops.gemm.triton.quantized',
        'launcher': 'launch_a16wfp4',
        'configs': '_fp4_configs',
    },
    'triton.block_scaled': {
        'backend': 'triton',
        'interface': 'scaled',
        'module': 'boltops.gemm.triton.scaled',
        'launcher': 'launch_generic',
        'configs': 'generic_configs',
        'supports': 'supports_generic',
    },
    'triton.scaled.fp8_fp4': {
        'backend': 'triton',
        'interface': 'scaled',
        'module': 'boltops.gemm.triton.scaled',
        'launcher': 'launch_scaled_fp4',
        'configs': 'mixed_fp4_configs',
    },
    'triton.grouped.fp8_fp4': {
        'backend': 'triton',
        'interface': 'grouped',
        'module': 'boltops.gemm.triton.grouped',
        'launcher': 'launch_grouped_fp4',
        'configs': 'mixed_fp4_configs',
    },
    'gluon.fp8e4m3fn.fp32scale.a1x128_b128x128.to_bf16': {
        'backend': 'gluon',
        'interface': 'scaled',
        'module': 'boltops.gemm.gluon.dense.launchers',
        'launcher': 'launch',
        'configs': '_configurations',
        'supports': '_supports',
    },
}


@dataclass(frozen=True)
class ExecutionPlan:
    implementation: str
    config: KernelConfig

    @property
    def backend(self):
        return IMPLEMENTATIONS[self.implementation]["backend"]


def validate_backend(backend):
    if backend not in ("auto", *BACKENDS):
        raise ValueError(f"unknown GEMM backend {backend!r}; expected auto, triton, gluon or tilelang")


def available(backend):
    return find_spec("triton" if backend == "gluon" else backend) is not None


def _names(problem, names, backend):
    validate_backend(backend)
    if names is not None:
        unknown = set(names) - IMPLEMENTATIONS.keys()
        if unknown:
            raise ValueError(f"unknown GEMM implementations: {sorted(unknown)}")
    choices = [name for name, row in IMPLEMENTATIONS.items()
               if row["interface"] == problem.interface and
               (names is None or name in names) and (backend == "auto" or row["backend"] == backend)]
    if backend != "auto" and choices and not available(backend):
        raise ImportError(f"GEMM backend {backend} is not installed")
    return tuple(name for name in choices if available(IMPLEMENTATIONS[name]["backend"]))


def _configs(name, problem, tuning=False):
    row = IMPLEMENTATIONS[name]
    if "default_config" in row and not (tuning and "tuning_configs" in row):
        return (KernelConfig.from_params(row["default_config"]),)
    function = row["tuning_configs"] if tuning and "tuning_configs" in row else row["configs"]
    return getattr(import_module(row["module"]), function)(problem)


def supports(name, problem, config):
    row = IMPLEMENTATIONS[name]
    if row["interface"] != problem.interface:
        return False
    for field, accepted in row.get("capabilities", {}).items():
        value = getattr(problem, field)
        if str(value).startswith("torch."):
            value = str(value).removeprefix("torch.")
        if value not in accepted:
            return False
    if "validate_config" in row:
        return getattr(import_module(row["module"]), row["validate_config"])(config.params)
    if "supports" in row:
        return getattr(import_module(row["module"]), row["supports"])(problem, config)
    return config in _configs(name, problem)


def candidate_plans(problem, names=None, *, backend="auto", tuning=False):
    from .performance_map import default_backends
    order = default_backends(problem.interface)
    choices = sorted(_names(problem, names, backend), key=lambda name: (
        order.index(IMPLEMENTATIONS[name]["backend"]),
        -IMPLEMENTATIONS[name].get("default_priority", 0), name))
    for name in choices:
        seen = set()
        for config in _configs(name, problem, tuning):
            if config not in seen and supports(name, problem, config):
                seen.add(config)
                yield ExecutionPlan(name, config)


def select_plan(problem, names=None, *, backend="auto"):
    from .performance_map import matching_entries, entry_config
    choices = _names(problem, names, backend)
    owners = {IMPLEMENTATIONS[name]["backend"] for name in choices}
    records = [e for e in matching_entries(problem, backend=backend) if e["config"]["backend"] in owners]
    if records:
        priority = max(e.get("priority", 0) for e in records)
        winners = [e for e in records if e.get("priority", 0) == priority]
        if len(winners) != 1:
            labels = [f'{e["config"]["backend"]}:{e["config"]["implementation"]}' for e in winners]
            raise ValueError(f"conflicting GEMM winners at priority {priority}: {labels}")
        c = winners[0]["config"]
        name = c["implementation"]
        if name not in choices or IMPLEMENTATIONS[name]["backend"] != c["backend"]:
            raise ValueError(f"GEMM winner must identify one registered implementation: {c}")
        config = entry_config(winners[0])
        if not supports(name, problem, config):
            raise ValueError(f"recorded GEMM winner is unsupported: {c}")
        return ExecutionPlan(name, config)
    plan = next(candidate_plans(problem, choices, backend=backend), None)
    if plan is not None:
        return plan
    raise NotImplementedError(
        f"no registered {problem.kind} GEMM implements {problem} (interface={problem.interface}, backend={backend})")


def launch(plan, problem, **tensors):
    if not supports(plan.implementation, problem, plan.config):
        raise ValueError(f"configuration {plan.config} is incompatible with {plan.implementation}")
    row = IMPLEMENTATIONS[plan.implementation]
    getattr(import_module(row["module"]), row["launcher"])(plan.config, problem, **tensors)
