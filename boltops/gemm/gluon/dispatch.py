"""Gluon capabilities independent of Triton's GEMM provider."""
from .dense.dispatch import IMPLEMENTATIONS as SCALED

INTERFACES = {"scaled": SCALED, "dense": (), "batched": (), "batched_a16wfp4_online_quant": (), "grouped": ()}
