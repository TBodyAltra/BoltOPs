"""Selection depends on semantic descriptors and each implementation's contract."""
from dataclasses import replace

import pytest
import torch

from boltops.gemm.gluon.dense.descriptors import DenseProblem, ScaleSpec
from boltops.gemm.gluon.dense.dispatch import (
    Implementation, KernelConfig, launch, select_plan,
)


def problem(shape=(128, 256, 256), *, aligned=True, dtype=torch.float8_e4m3fn,
            scale_dtype=torch.float32, out_dtype=torch.bfloat16,
            recipe=((1, 128), (128, 128)), arch="gfx946", encodings=("numeric", "numeric")):
    m, n, k = shape
    return DenseProblem(m, n, k, dtype, dtype, out_dtype,
                        ScaleSpec(scale_dtype, recipe[0], encodings[0]),
                        ScaleSpec(scale_dtype, recipe[1], encodings[1]), arch, aligned, True)


@pytest.mark.parametrize("shape", [(128, 256, 256), (256, 512, 6144)])
def test_complete_tiles_select_fixed_mls_pipeline(shape):
    plan = select_plan(problem(shape))
    assert plan.implementation.name == "mls_wdra"
    assert plan.config == KernelConfig(128, 256, 128)


@pytest.mark.parametrize("shape", [
    (127, 256, 256), (128, 255, 256), (128, 256, 128),
    (128, 256, 257), (128, 256, 6400),
])
def test_incomplete_or_over_capacity_tiles_use_masked_fp8(shape):
    assert select_plan(problem(shape)).implementation.name == "masked_fp8"


def test_unaligned_inputs_do_not_select_mls():
    assert select_plan(problem(aligned=False)).implementation.name == "masked_fp8"


@pytest.mark.parametrize("kwargs", [
    {"dtype": torch.float16}, {"scale_dtype": torch.bfloat16},
    {"out_dtype": torch.float32}, {"recipe": ((1, 64), (128, 64))},
    {"scale_dtype": torch.uint8, "encodings": ("numeric", "e8m0")},
])
def test_other_recipes_and_dtypes_keep_general_native_path(kwargs):
    assert select_plan(problem(**kwargs)).implementation.name == "masked"


@pytest.mark.parametrize("recipe,bk", [
    (((1, 24), (2, 40)), 32), (((1, 64), (2, 128)), 64),
    (((1, 256), (2, 512)), 128),
])
def test_general_dot_chunk_respects_common_scale_boundaries(recipe, bk):
    assert select_plan(problem(recipe=recipe, out_dtype=torch.float32)).config == KernelConfig(32, 64, bk)


def test_optimized_pipeline_is_target_specific():
    assert select_plan(problem(arch="gfx938")).implementation.name != "mls_wdra"


def test_invalid_tile_cannot_launch_a_fixed_pipeline():
    p = problem()
    plan = replace(select_plan(p), config=KernelConfig(256, 256, 64))
    with pytest.raises(ValueError, match="incompatible"):
        launch(plan, p, None, None, None, None, None)


def test_a_new_implementation_needs_no_dispatch_branches():
    config = KernelConfig(64, 64, 64)
    implementation = Implementation("test", "not_imported", "launch",
                                    lambda p: (config,), lambda p, c: c == config)
    plan = select_plan(problem(), implementations=(implementation,))
    assert plan.implementation is implementation and plan.config == config


def test_missing_implementation_is_reported():
    with pytest.raises(NotImplementedError, match="no dense kernel"):
        select_plan(problem(), implementations=())


def test_registry_rejects_e8m0_with_nonbyte_storage():
    with pytest.raises(NotImplementedError, match="no dense kernel"):
        select_plan(problem(encodings=("numeric", "e8m0")))
