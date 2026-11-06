"""Triton registrations; device code is imported only by selected adapters."""
from .block_scaled_dispatch import IMPLEMENTATIONS as SCALED
from .dense_dispatch import IMPLEMENTATIONS as DENSE
from .grouped_dispatch import IMPLEMENTATIONS as GROUPED

INTERFACES = {"scaled": SCALED, "dense": DENSE, "grouped": GROUPED}
