"""Gluon capabilities independent of Triton's GEMM provider."""
from .dense.dispatch import IMPLEMENTATIONS as SCALED

INTERFACES = {"scaled": SCALED, "dense": (), "batched": (), "awq": (), "mxfp4": (), "batched_a16wfp4": (), "grouped": ()}
