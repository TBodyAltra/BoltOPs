# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Dense Gluon dispatch uses the public GEMM contract and precision profile."""
from copy import deepcopy
from importlib import import_module

import pytest
import torch

from boltops.gemm import gemm_a16w16
from boltops.gemm.config import get_gemm_config

PROFILE = "DTYPE=bfloat16-OUT=bfloat16"
KERNEL = "_gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel"
PERSISTENT = "2048x2048x4096"
PERSISTENT_KERNEL = "_gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel"


def config_for(m=64, n=64, k=2048, profile=PROFILE, layout="TN", cu=48):
    return get_gemm_config("GEMM-A16W16", m, n, k, profile=profile,
                           layout=layout, backend="gluon", arch="gfx946", cu_num=cu)


def require_device():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info
    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")


def test_config_has_precision_profile_and_kernel_selector():
    cfg, tuned = config_for()
    assert tuned and cfg['kernel'] == KERNEL
    assert (cfg['BLOCK_SIZE_M'], cfg['BLOCK_SIZE_N'], cfg['BLOCK_SIZE_K']) == (16, 16, 128)
    for kwargs in [dict(m=63), dict(n=128), dict(k=1024), dict(layout="TT"),
                   dict(profile="DTYPE=float16-OUT=bfloat16")]:
        with pytest.raises(KeyError):
            config_for(**kwargs)
    with pytest.raises(FileNotFoundError):
        config_for(cu=192)


class Capture:
    def __init__(self, label, calls):
        self.label, self.calls = label, calls

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((self.label, grid(kwargs) if callable(grid) else grid, args, kwargs))
        return run


def install_captures(monkeypatch):
    require_device()
    calls = []
    gluon = import_module("boltops.gemm.gluon.gemm_a16w16")
    triton = import_module("boltops.gemm.triton.gemm_a16w16")
    monkeypatch.setattr(gluon, KERNEL, Capture("gluon", calls))
    monkeypatch.setattr(triton, "_gemm_a16_w16_kernel", Capture("triton", calls))
    return calls


@pytest.mark.parametrize("case,expected", [
    ("recorded", "gluon"), ("different_m", "triton"),
    ("fp16", "triton"), ("column_major", "triton"),
    ("weight_column_major", "triton"),
    ("fp32_output", "triton"), ("bias", "triton"),
    ("activation", "triton"), ("explicit_triton_config", "triton"),
])
def test_auto_selects_only_recorded_contract(monkeypatch, case, expected):
    calls = install_captures(monkeypatch)
    m = 63 if case == "different_m" else 64
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    a = torch.empty((m, 2048), dtype=dtype, device="cuda")
    w = torch.empty((64, 2048), dtype=dtype, device="cuda")
    if case == "column_major":
        a = a.as_strided(a.shape, (1, m))
    if case == "weight_column_major":
        w = w.as_strided(w.shape, (1, 64))
    kw = {}
    if case == "fp32_output":
        kw['dtype'] = torch.float32
    if case == "bias":
        kw['bias'] = torch.empty((64,), dtype=dtype, device=a.device)
    if case == "activation":
        kw['activation'] = "relu"
    if case == "explicit_triton_config":
        kw['config'] = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=128,
                            GROUP_SIZE_M=1, NUM_KSPLIT=1, SPLITK_BLOCK_SIZE=2048,
                            num_warps=1, num_stages=2)
    out = gemm_a16w16(a, w, **kw)
    assert out.shape == (m, 64)
    assert len(calls) == 1 and calls[0][0] == expected
    if expected == "gluon":
        _, grid, args, params = calls[0]
        assert grid == (16,) and args[0] is a and args[1] is w and args[2] is out
        assert (params['M'], params['N'], params['K']) == (64, 64, 2048)


def test_explicit_gluon_selects_kernel_without_mutating_config(monkeypatch):
    calls = install_captures(monkeypatch)
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, "kernel_from_config", Capture("custom", calls), raising=False)
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty_like(a)
    out = torch.empty((64, 64), dtype=a.dtype, device=a.device)
    cfg, _ = config_for()
    cfg['kernel'] = "kernel_from_config"
    cfg['BLOCK_SIZE_M'] = cfg['BLOCK_SIZE_N'] = 32
    before = deepcopy(cfg)
    result = gemm_a16w16(a, w, out=out, backend="gluon", config=cfg)
    assert result is out and cfg == before and calls[0][0] == "custom"
    assert "kernel" not in calls[0][3] and calls[0][1] == (4,)


@pytest.mark.parametrize("case", ["shape", "dtype", "layout", "weight_layout", "output", "bias", "activation"])
def test_explicit_gluon_rejects_unsupported_contract(monkeypatch, case):
    calls = install_captures(monkeypatch)
    a = torch.empty((63 if case == "shape" else 64, 2048),
                    dtype=torch.float16 if case == "dtype" else torch.bfloat16, device="cuda")
    w = torch.empty((64, 2048), dtype=a.dtype, device=a.device)
    if case == "layout":
        a = a.as_strided(a.shape, (1, a.shape[0]))
    if case == "weight_layout":
        w = w.as_strided(w.shape, (1, 64))
    kw = dict(backend="gluon")
    if case == "output":
        kw['dtype'] = torch.float32
    if case == "bias":
        kw['bias'] = torch.empty((64,), dtype=a.dtype, device=a.device)
    if case == "activation":
        kw['activation'] = "relu"
    with pytest.raises(NotImplementedError, match="no Gluon"):
        gemm_a16w16(a, w, **kw)
    assert not calls


@pytest.mark.parametrize("backend", ["auto", "gluon"])
def test_public_gluon_native_correctness(backend):
    require_device()
    gen = torch.Generator().manual_seed(851)
    a = (torch.rand((64, 2048), generator=gen)-.5).bfloat16()
    w = (torch.rand((64, 2048), generator=gen)-.5).bfloat16()
    gold = (a.float() @ w.float().T).bfloat16()
    result = gemm_a16w16(a.cuda(), w.cuda(), backend=backend)
    torch.testing.assert_close(result.cpu(), gold, atol=.001, rtol=.01)


@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_persistent_config_carries_tile_sms_and_kernel(layout):
    cfg, tuned = config_for(m=2048, n=2048, k=4096, layout=layout)
    assert tuned and cfg["kernel"] == PERSISTENT_KERNEL
    assert (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]) == (256, 256, 64)
    assert cfg["NUM_SMS"] == 48
    assert cfg["wasp_partition_warps"] == [4, 4, 4]
    assert cfg["wasp_partition_regs"] == [16, 244, 244]


@pytest.mark.parametrize("kwargs", [
    dict(m=1024), dict(m=2048, n=1024), dict(m=2048, k=2048),
    dict(m=1024, layout="TT"), dict(m=2048, profile="DTYPE=float16-OUT=bfloat16"),
])
def test_persistent_config_is_shape_and_layout_specific(kwargs):
    # Only the recorded complete-tile shape may select the persistent entry;
    # everything else must miss so dispatch falls back to another backend.
    defaults = dict(m=2048, n=2048, k=4096)
    defaults.update(kwargs)
    with pytest.raises(KeyError):
        config_for(**defaults)


@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_persistent_dispatch_launches_one_tg_per_sm(monkeypatch, layout):
    calls = install_captures(monkeypatch)
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, PERSISTENT_KERNEL, Capture("persistent", calls))
    a = torch.empty((2048, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.empty((2048, 4096), dtype=a.dtype, device=a.device)
    if layout == "TT":
        w = w.as_strided(w.shape, (1, 2048))
    result = gemm_a16w16(a, w)
    assert result.shape == (2048, 2048)
    label, grid, args, params = calls[0]
    assert label == "persistent" and grid == (48,)
    assert args[0] is a and args[1] is w
    assert (params["M"], params["N"], params["K"]) == (2048, 2048, 4096)
    assert params["NUM_SMS"] == 48 and "kernel" not in params
    assert params["B_K_CONTIGUOUS"] is (layout == "TN")


@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_persistent_native_correctness(layout):
    require_device()
    gen = torch.Generator().manual_seed(21)
    a = (torch.randn((2048, 4096), generator=gen) * .5).bfloat16()
    w = (torch.randn((2048, 4096), generator=gen) * .5).bfloat16()
    gold = (a.float() @ w.float().T).bfloat16()
    a_dev = a.cuda()
    w_dev = w.cuda() if layout == "TN" else w.T.contiguous().cuda().T
    result = gemm_a16w16(a_dev, w_dev)
    torch.testing.assert_close(result.cpu(), gold, atol=.001, rtol=.01)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_persistent_native_tt_tail_correctness(dtype):
    require_device()
    m, n, k = 17, 33, 65
    gen = torch.Generator().manual_seed(21)
    a = (torch.randn((m, k), generator=gen) * .5).to(dtype)
    b = (torch.randn((k, n), generator=gen) * .5).to(dtype)
    gold = (a.float() @ b.float()).to(dtype)
    a_dev, b_dev = a.cuda(), b.cuda()
    storage = torch.full((m*n + 128,), 123., dtype=dtype, device=a_dev.device)
    out = storage[64:-64].view(m, n)
    cfg, _ = config_for(m=2048, n=2048, k=4096, layout="TT")
    result = gemm_a16w16(a_dev, b_dev.T, dtype=dtype, out=out,
                         config=cfg, backend="gluon")
    assert result is out
    torch.testing.assert_close(result.cpu(), gold, atol=.001, rtol=.01)
    assert bool(torch.all(storage[:64].cpu() == 123))
    assert bool(torch.all(storage[-64:].cpu() == 123))
    assert torch.equal(a_dev.cpu(), a) and torch.equal(b_dev.cpu(), b)


@pytest.mark.parametrize("selector", [None, "missing_kernel"])
def test_explicit_gluon_config_requires_selector(monkeypatch, selector):
    calls = install_captures(monkeypatch)
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty_like(a)
    cfg, _ = config_for()
    if selector is None:
        cfg.pop("kernel")
    else:
        cfg["kernel"] = selector
    with pytest.raises(ValueError, match="kernel selector"):
        gemm_a16w16(a, w, config=cfg, backend="gluon")
    assert not calls
