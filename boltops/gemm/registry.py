"""Implementation registry independent of operator family, dtype and side tensors."""
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from .descriptors import GemmProblem


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int

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
    backend: str = "gluon"

    def matches(self, problem, config):
        return self.family == problem.kind and self.supports(problem, config)


@dataclass(frozen=True)
class ExecutionPlan:
    implementation: Implementation
    config: KernelConfig


def select_plan(problem, implementations=None):
    if implementations is None:
        from .catalog import IMPLEMENTATIONS
        implementations = IMPLEMENTATIONS
    for implementation in implementations:
        if implementation.family != problem.kind:
            continue
        for config in implementation.configurations(problem):
            if implementation.matches(problem, config):
                return ExecutionPlan(implementation, config)
    raise NotImplementedError(f"no registered {problem.kind} GEMM implements {problem}")


def launch(plan, problem, **tensors):
    implementation = plan.implementation
    if not implementation.matches(problem, plan.config):
        raise ValueError(f"configuration {plan.config} is incompatible with {implementation.name}")
    module = import_module(implementation.module)
    getattr(module, implementation.entry)(plan.config, problem, **tensors)
