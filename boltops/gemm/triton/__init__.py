"""Triton backend for the GEMM family; kernels are imported on demand."""
from importlib import import_module

__all__ = ["gemm_a16w16", "matmul_persistent"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{name}"), name)
