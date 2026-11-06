"""Shared registry for scaled, unscaled dense and grouped GEMM."""
from .gluon.dense.dispatch import IMPLEMENTATIONS as GLUON_IMPLEMENTATIONS
from .triton.block_scaled_dispatch import IMPLEMENTATIONS as TRITON_IMPLEMENTATIONS

from .triton.dense_dispatch import IMPLEMENTATIONS as DENSE_IMPLEMENTATIONS
from .triton.grouped_dispatch import IMPLEMENTATIONS as GROUPED_IMPLEMENTATIONS

IMPLEMENTATIONS = (GLUON_IMPLEMENTATIONS + TRITON_IMPLEMENTATIONS +
                   DENSE_IMPLEMENTATIONS + GROUPED_IMPLEMENTATIONS)
