# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Implementation registry independent of operator family, dtype and side tensors."""
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from .descriptors import GemmProblem


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    source_config: str | None = None

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.block_m, self.block_n, self.block_k)):
            raise ValueError("kernel block sizes must be positive integers")


@dataclass(frozen=True)
class Implementation:
    name: str
    family: str
    module: str
    entry: str
    configurations: Callable[[GemmProblem], tuple[KernelConfig, ...]]
    supports: Callable[[GemmProblem, KernelConfig], bool]
    backend: str
    default_priority: int = 0
    tuning_configurations: Callable[[GemmProblem], tuple[KernelConfig, ...]] | None = None

    def matches(self, problem, config):
        return self.family == problem.kind and self.supports(problem, config)


@dataclass(frozen=True)
class ExecutionPlan:
    implementation: Implementation
    config: KernelConfig


def _candidates(problem, implementations, backend):
    from .catalog import BACKENDS, available, implementations as load, validate_backend
    validate_backend(backend)
    if implementations is not None:
        return tuple(i for i in implementations if backend == "auto" or i.backend == backend)
    result = []
    for owner in BACKENDS if backend == "auto" else (backend,):
        if backend == "auto" and not available(owner):
            continue
        registered = load(owner, problem.interface)
        if not registered:
            continue
        if not available(owner):
            if backend != "auto":
                raise ImportError(f"GEMM backend {owner} is not installed")
            continue
        result.extend(registered)
    return tuple(result)


def candidate_plans(problem, implementations=None, *, backend="auto", tuning=False):
    """Enumerate supported plans without compiling, timing or launching them."""
    from .performance_map import default_backends
    order = default_backends(problem.interface)
    candidates = sorted(_candidates(problem, implementations, backend),
                        key=lambda i: (order.index(i.backend), -i.default_priority, i.name))
    seen = set()
    for implementation in candidates:
        if implementation.family != problem.kind:
            continue
        configs = (implementation.tuning_configurations if tuning and
                   implementation.tuning_configurations is not None else implementation.configurations)
        for config in configs(problem):
            key = (implementation.backend, implementation.name, config)
            if key not in seen and implementation.matches(problem, config):
                seen.add(key)
                yield ExecutionPlan(implementation, config)


def select_plan(problem, implementations=None, *, backend="auto"):
    from .performance_map import matching_entries, entry_config
    candidates = _candidates(problem, implementations, backend)
    owners = {i.backend for i in candidates}
    # Resolve the complete measured set, before examining provider order.
    records = [e for e in matching_entries(problem, backend=backend) if e["config"]["backend"] in owners]
    if records:
        priority = max(e.get("priority", 0) for e in records)
        winners = [e for e in records if e.get("priority", 0) == priority]
        if len(winners) != 1:
            names = [f'{e["config"]["backend"]}:{e["config"]["implementation"]}' for e in winners]
            raise ValueError(f"conflicting GEMM winners at priority {priority}: {names}")
        entry = winners[0]
        c = entry["config"]
        registered = [i for i in candidates if i.backend == c["backend"] and i.name == c["implementation"]]
        if len(registered) != 1:
            raise ValueError(f"GEMM winner must identify one registered implementation: {c}")
        config = entry_config(entry)
        if not registered[0].matches(problem, config):
            raise ValueError(f"recorded GEMM winner is unsupported: {c}")
        return ExecutionPlan(registered[0], config)
    # Default policy is explicit per interface, independent of provider order.
    plan = next(candidate_plans(problem, candidates, backend=backend), None)
    if plan is not None:
        return plan
    raise NotImplementedError(
        f"no registered {problem.kind} GEMM implements {problem} (interface={problem.interface}, backend={backend})")


def launch(plan, problem, **tensors):
    implementation = plan.implementation
    if not implementation.matches(problem, plan.config):
        raise ValueError(f"configuration {plan.config} is incompatible with {implementation.name}")
    module = import_module(implementation.module)
    getattr(module, implementation.entry)(plan.config, problem, **tensors)
