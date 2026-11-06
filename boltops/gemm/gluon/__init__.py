"""Gluon GEMM framework; block-scaled implementations only."""
from .dense.block_scaled import gemm_block_scaled


__all__ = ["gemm_block_scaled"]
