"""TileLang extension point. No GEMM implementation is registered yet.

Populate the appropriate interface with TileLang-owned Implementations when
native kernels are implemented and validated. An empty tuple is unsupported,
not a request to delegate to another backend.
"""
INTERFACES = {"scaled": (), "dense": (), "batched": (), "awq": (), "mxfp4": (), "batched_a16wfp4": (), "grouped": ()}
