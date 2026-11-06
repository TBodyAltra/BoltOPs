"""Triton registrations; device code is imported only by selected adapters."""
from .block_scaled_dispatch import IMPLEMENTATIONS as SCALED
from .scaled_dispatch import IMPLEMENTATIONS as LEGACY_SCALED
from .batched_dispatch import IMPLEMENTATIONS as BATCHED
from .specialized_dispatch import IMPLEMENTATIONS as SPECIALIZED
from .dense_dispatch import IMPLEMENTATIONS as DENSE
from .grouped_dispatch import IMPLEMENTATIONS as GROUPED

INTERFACES = {"scaled": SCALED + LEGACY_SCALED, "batched": BATCHED, "dense": DENSE, "grouped": GROUPED}
INTERFACES.update(SPECIALIZED)
