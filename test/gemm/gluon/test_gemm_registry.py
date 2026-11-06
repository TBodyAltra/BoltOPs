"""Framework contracts do not imply additional executable GEMM implementations."""
import pytest
import torch

from boltops.gemm.gluon.descriptors import GemmProblem, GroupSpec, OperandSpec, ScaleSpec
from boltops.gemm.gluon.registry import KernelConfig, select_plan


@pytest.mark.parametrize("kind", ["dense", "batched"])
def test_unscaled_families_have_no_registered_implementation(kind):
    p = GemmProblem(kind, 32, 64, 64, OperandSpec(torch.bfloat16),
                    OperandSpec(torch.bfloat16), torch.float32, "gfx946",
                    batch=2 if kind == "batched" else 1)
    assert p.unscaled
    with pytest.raises(NotImplementedError, match=f"no registered {kind} GEMM"):
        select_plan(p)


@pytest.mark.parametrize("mode", ["equal", "offsets", "masked", "indexed"])
def test_unimplemented_group_addressing_never_uses_dense(mode):
    p = GemmProblem("grouped", 32, 64, 64, OperandSpec(torch.bfloat16),
                    OperandSpec(torch.bfloat16), torch.bfloat16, "gfx946", groups=GroupSpec(mode, 2))
    with pytest.raises(NotImplementedError, match="no registered grouped GEMM"):
        select_plan(p)


@pytest.mark.parametrize("format,packing,dtype,factor", [
    ("int4", "awq", torch.int32, 8), ("fp4", "e2m1_low_high", torch.uint8, 2),
])
def test_packed_storage_never_uses_native_unpacked_kernel(format, packing, dtype, factor):
    p = GemmProblem("dense", 32, 64, 64, OperandSpec(torch.bfloat16),
                    OperandSpec(dtype, format, packing, -1, factor), torch.bfloat16, "gfx946",
                    b_scale=ScaleSpec(torch.bfloat16, (1, 32)))
    with pytest.raises(NotImplementedError, match="no registered dense GEMM"):
        select_plan(p)


def test_packing_must_be_explicit():
    with pytest.raises(ValueError, match="packed operands require"):
        OperandSpec(torch.uint8, format="fp4")


@pytest.mark.parametrize("m", [0, 32.0])
def test_tile_configuration_requires_integer_geometry(m):
    with pytest.raises(ValueError, match="positive integers"):
        KernelConfig(m, 64, 64)


def test_group_count_requires_an_integer():
    with pytest.raises(ValueError, match="grouped GEMM addressing contract"):
        GroupSpec("equal", 2.5)


def test_public_gluon_exports_only_block_scaled():
    import boltops.gemm.gluon as gluon

    assert gluon.__all__ == ["gemm_block_scaled"]
    assert not hasattr(gluon, "gemm") and not hasattr(gluon, "bmm")
