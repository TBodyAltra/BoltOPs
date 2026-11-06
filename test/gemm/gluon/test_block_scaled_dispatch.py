"""Host selection keeps fixed-pipeline geometry and recipe gates together."""
from types import SimpleNamespace

import pytest
import torch

from boltops.gemm.gluon._block_scaled_dispatch import select_config


def select(shape=(128, 256, 256), *, address=16, dtype=torch.float8_e4m3fn,
           scale_dtype=torch.float32, out_dtype=torch.bfloat16,
           recipe=((1, 128), (128, 128))):
    m, n, k = shape
    a = SimpleNamespace(shape=(m, k), dtype=dtype, data_ptr=lambda: address)
    b = SimpleNamespace(shape=(n, k), dtype=dtype, data_ptr=lambda: 16)
    scale = SimpleNamespace(dtype=scale_dtype)
    return select_config(a, b, scale, scale, recipe, out_dtype)


@pytest.mark.parametrize("shape", [(128, 256, 256), (256, 512, 6144)])
def test_complete_tiles_select_fixed_mls_pipeline(shape):
    config = select(shape)
    assert config.pipeline == "mls_wdra48_128x256x128"
    assert (config.block_m, config.block_n, config.block_k) == (128, 256, 128)


@pytest.mark.parametrize("shape", [
    (127, 256, 256), (128, 255, 256), (128, 256, 128),
    (128, 256, 257), (128, 256, 6400),
])
def test_incomplete_or_over_capacity_tiles_use_masked_fp8(shape):
    assert select(shape).pipeline == "masked_fp8_64x128x128"


def test_unaligned_inputs_do_not_select_mls():
    assert select(address=17).pipeline == "masked_fp8_64x128x128"


@pytest.mark.parametrize("kwargs", [
    {"dtype": torch.float16}, {"scale_dtype": torch.bfloat16},
    {"out_dtype": torch.float32}, {"recipe": ((1, 64), (128, 64))},
])
def test_other_recipes_and_dtypes_keep_general_native_path(kwargs):
    assert select(**kwargs).pipeline == "masked_general_32x64"


@pytest.mark.parametrize("recipe,bk", [
    (((1, 24), (2, 40)), 32), (((1, 64), (2, 128)), 64),
    (((1, 256), (2, 512)), 128),
])
def test_general_dot_chunk_respects_common_scale_boundaries(recipe, bk):
    config = select(recipe=recipe, out_dtype=torch.float32)
    assert (config.block_m, config.block_n, config.block_k) == (32, 64, bk)


def test_legacy_module_is_only_a_forwarder(monkeypatch):
    from importlib import import_module

    public = import_module("boltops.gemm.gluon.gemm_block_scaled")
    legacy = import_module(
        "boltops.gemm.gluon.gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16")
    sentinel = object()
    args = (object(), object(), object(), object())
    recipe = ((2, 32), (4, 64))

    def implementation(*actual_args, **kwargs):
        assert actual_args == args
        assert kwargs == {"scale_recipe": recipe, "out_dtype": torch.float32}
        return sentinel

    monkeypatch.setattr(public, "gemm_block_scaled", implementation)
    assert legacy.gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16(
        *args, scale_recipe=recipe, out_dtype=torch.float32) is sentinel
