"""TileLang extension point. No GEMM implementation is registered yet.

Populate the appropriate interface with TileLang-owned Implementations when
native kernels are implemented and validated. An empty tuple is unsupported,
not a request to delegate to another backend.
"""
INTERFACES = {"scaled": (), "dense": (), "grouped": ()}
