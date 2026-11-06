"""One semantic interface: tuned Gluon contracts, then general Triton."""
from .gluon.dense.dispatch import IMPLEMENTATIONS as GLUON_IMPLEMENTATIONS
from .triton.block_scaled_dispatch import IMPLEMENTATIONS as TRITON_IMPLEMENTATIONS

IMPLEMENTATIONS = GLUON_IMPLEMENTATIONS + TRITON_IMPLEMENTATIONS
