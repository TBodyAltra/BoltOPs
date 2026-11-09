"""Selection depends on semantic descriptors and each implementation's contract."""
from dataclasses import replace

import pytest
import torch

from boltops.gemm.descriptors import GemmProblem, OperandSpec, ScaleSpec
from boltops.gemm.dispatch import (
    KernelConfig, launch, select_plan,
)


def problem(shape=(128, 256, 256), *, aligned=True, dtype=torch.float8_e4m3fn,
            scale_dtype=torch.float32, out_dtype=torch.bfloat16,
            recipe=((1, 128), (128, 128)), arch="gfx946", encodings=("numeric", "numeric"), num_cus=48):
    m, n, k = shape
    return GemmProblem("dense", m, n, k,
                       OperandSpec(dtype, strides=(k, 1)), OperandSpec(dtype, strides=(k, 1)),
                       out_dtype, arch, ScaleSpec(scale_dtype, recipe[0], encodings[0]),
                       ScaleSpec(scale_dtype, recipe[1], encodings[1]),
                       a_aligned_16=aligned, num_cus=num_cus)



@pytest.mark.parametrize("shape", [(128, 256, 256), (256, 512, 6144)])
def test_complete_tiles_select_fixed_mls_pipeline(shape):
    plan = select_plan(problem(shape))
    assert plan.implementation == "gluon.fp8e4m3fn.fp32scale.a1x128_b128x128.to_bf16"
    assert plan.config == KernelConfig(128, 256, 128, num_warps=4, num_stages=2,
                                       wasp_partition_warps=[4, 4, 4, 4], wasp_wdra=True)


@pytest.mark.parametrize("shape", [
    (127, 256, 256), (128, 255, 256), (128, 256, 128),
    (128, 256, 257), (128, 256, 6400),
])
def test_incomplete_or_over_capacity_tiles_use_triton(shape):
    assert select_plan(problem(shape)).implementation == "triton.block_scaled"


def test_unaligned_inputs_reject_the_recorded_mls_configuration():
    with pytest.raises(ValueError, match="winner is unsupported"):
        select_plan(problem(aligned=False))


@pytest.mark.parametrize("kwargs", [
    {"dtype": torch.float16}, {"scale_dtype": torch.bfloat16},
    {"out_dtype": torch.float32}, {"recipe": ((1, 64), (128, 64))},
    {"scale_dtype": torch.uint8, "encodings": ("numeric", "e8m0")},
])
def test_other_recipes_and_dtypes_keep_general_native_path(kwargs):
    assert select_plan(problem(**kwargs)).implementation == "triton.block_scaled"


@pytest.mark.parametrize("recipe,bk", [
    (((1, 24), (2, 40)), 32), (((1, 64), (2, 128)), 64),
    (((1, 256), (2, 512)), 128),
])
def test_general_dot_chunk_respects_common_scale_boundaries(recipe, bk):
    assert select_plan(problem(recipe=recipe, out_dtype=torch.float32)).config == KernelConfig(32, 64, bk)


def test_optimized_pipeline_is_target_specific():
    assert select_plan(problem(arch="gfx938")).implementation != "gluon.fp8e4m3fn.fp32scale.a1x128_b128x128.to_bf16"


def test_invalid_tile_cannot_launch_a_fixed_pipeline():
    p = problem()
    plan = replace(select_plan(p), config=KernelConfig(256, 256, 64))
    with pytest.raises(ValueError, match="incompatible"):
        launch(plan, p, a=None, b=None, a_scale=None, b_scale=None, out=None)


def test_a_new_implementation_needs_no_dispatch_branches(add_impl):
    config = KernelConfig(64, 64, 64)
    name = add_impl('test', backend='gluon', configs=lambda p: (config,),
                    supports=lambda p, c: c == config)
    plan = select_plan(replace(problem(), n=64), names=(name,))
    assert plan.implementation == name and plan.config == config


def test_missing_implementation_is_reported():
    with pytest.raises(NotImplementedError, match="no registered dense GEMM"):
        select_plan(problem(), names=())


def test_registry_rejects_e8m0_with_nonbyte_storage():
    with pytest.raises(NotImplementedError, match="no registered dense GEMM"):
        select_plan(problem(encodings=("numeric", "e8m0")))
