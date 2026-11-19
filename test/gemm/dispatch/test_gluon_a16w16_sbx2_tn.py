# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""The recorded R001 TN pipeline is selected only for gfx946/96 CU BF16."""
from copy import deepcopy
from importlib import import_module

import pytest
import torch

from boltops.gemm import gemm_a16w16
from boltops.gemm.config import get_gemm_config

KERNEL = "_gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel"
PROFILE = "DTYPE=bfloat16-OUT=bfloat16"


def config_for(**overrides):
    options = dict(M=64, N=64, K=2048, profile=PROFILE, layout="TN",
                   backend="gluon", arch="gfx946", cu_num=96)
    options.update(overrides)
    return get_gemm_config("GEMM-A16W16", **options)


def test_sbx2_tn_config_is_shape_precision_and_cu_specific():
    cfg, tuned = config_for()
    assert tuned and cfg["kernel"] == KERNEL
    assert (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]) == (16, 16, 128)
    assert (cfg["CHUNKS"], cfg["SLOTS"]) == (2, 3)
    assert cfg["wasp_partition_warps"] == [1, 1] and cfg["wasp_wdra"] is False
    for overrides in [dict(M=63), dict(N=128), dict(K=1024), dict(layout="TT"),
                      dict(profile="DTYPE=float16-OUT=float16")]:
        try:
            other, _ = config_for(**overrides)
        except KeyError:
            continue
        assert other.get("kernel") != KERNEL
    old, tuned = config_for(cu_num=48)
    assert tuned and old["kernel"] == KERNEL
    assert (old.get("CHUNKS", 4), old.get("SLOTS", 2)) == (4, 2)


def require_sbx2():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info
    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 96:
        pytest.skip("requires gfx946 with 96 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")


class Capture:
    def __init__(self, label, calls):
        self.label, self.calls = label, calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((self.label, grid(kwargs) if callable(grid) else grid, args, kwargs))
        return launch


@pytest.mark.parametrize("backend", ["auto", "gluon"])
def test_sbx2_tn_public_dispatch_preserves_tensors_and_config(monkeypatch, backend):
    require_sbx2()
    calls = []
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, KERNEL, Capture("gluon", calls))
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty_like(a)
    out = torch.empty((64, 64), dtype=a.dtype, device=a.device)
    cfg, _ = config_for()
    cfg["B_K_CONTIGUOUS"] = False  # Physical strides own the launch value.
    before = deepcopy(cfg)
    monkeypatch.setattr(module, "get_gemm_config", lambda *args, **kwargs: (deepcopy(cfg), True))
    assert gemm_a16w16(a, w, out=out, backend=backend) is out
    assert cfg == before and len(calls) == 1
    _, grid, args, options = calls[0]
    assert grid == (16,) and args[0] is a and args[1] is w and args[2] is out
    assert options["B_K_CONTIGUOUS"] is True
    assert options["wasp_partition_warps"] == (1, 1)
    assert "kernel" not in options
