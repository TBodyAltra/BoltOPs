# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Complete-tile MLS ring layouts, dispatch and pointer alignment."""
from importlib import import_module

import pytest
import torch

from boltops.gemm import gemm_a16w16
from boltops.gemm.config import get_gemm_config

KERNEL = "_gemm_a16w16_kernel"
PROFILE = "DTYPE=bfloat16-OUT=bfloat16"


def config_for(m=2560, n=64, k=2048, profile=PROFILE, layout="TN"):
    return get_gemm_config("GEMM-A16W16", m, n, k, backend="gluon",
                           arch="gfx946", cu_num=96, layout=layout, profile=profile)


def test_exact_precision_and_shape_profile():
    cfg, tuned = config_for()
    assert tuned and cfg["kernel"] == KERNEL
    for kwargs in [dict(m=63), dict(n=128), dict(k=1024),
                   dict(profile="DTYPE=float16-OUT=float16")]:
        with pytest.raises(KeyError):
            config_for(**kwargs)


class Capture:
    def __init__(self, label, calls):
        self.label, self.calls = label, calls

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            resolved = grid(kwargs) if callable(grid) else grid
            self.calls.append((self.label, resolved, args, kwargs))
        return run


def tensors_and_captures(monkeypatch, unaligned=None):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info
    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 96:
        pytest.skip("requires gfx946 / 96 CU")
    calls = []
    gluon = import_module("boltops.gemm.gluon.gemm_a16w16")
    triton = import_module("boltops.gemm.triton.gemm_a16w16")
    monkeypatch.setattr(gluon, KERNEL, Capture("gluon_r003", calls))
    monkeypatch.setattr(triton, "_gemm_a16_w16_kernel", Capture("triton", calls))

    def make(shape, label):
        count = shape[0] * shape[1]
        offset = int(unaligned == label)
        buf = torch.empty((count + offset,), dtype=torch.bfloat16, device="cuda")
        return buf[offset:].view(shape)

    return make((2560, 2048), "a"), make((64, 2048), "w"), make((2560, 64), "out"), calls


def test_auto_selects_recorded_full_grid(monkeypatch):
    a, w, out, calls = tensors_and_captures(monkeypatch)
    assert gemm_a16w16(a, w, out=out, backend="auto") is out
    assert len(calls) == 1
    label, grid, args, meta = calls[0]
    assert label == "gluon_r003" and grid == (80,)
    assert args[0] is a and args[1] is w and args[2] is out
    assert meta["B_K_CONTIGUOUS"] is True
    assert meta["wasp_partition_warps"] == (2, 8)


@pytest.mark.parametrize("unaligned", ["a", "w", "out"])
def test_auto_unaligned_pointer_falls_back(monkeypatch, unaligned):
    a, w, out, calls = tensors_and_captures(monkeypatch, unaligned)
    assert gemm_a16w16(a, w, out=out, backend="auto") is out
    assert len(calls) == 1 and calls[0][0] == "triton"


@pytest.mark.parametrize("unaligned", ["a", "w", "out"])
def test_explicit_gluon_unaligned_pointer_is_rejected(monkeypatch, unaligned):
    a, w, out, calls = tensors_and_captures(monkeypatch, unaligned)
    with pytest.raises(ValueError, match="16-byte-aligned"):
        gemm_a16w16(a, w, out=out, backend="gluon")
    assert not calls


@pytest.mark.parametrize("tn,prefetch,defer,bm,bn,pw,cw", [
    (True, False, False, 16, 32, 1, 2),
    (True, False, True, 32, 64, 2, 8),
    (False, False, False, 32, 64, 1, 2),
    (False, True, False, 32, 64, 1, 2),
    (False, True, False, 64, 256, 2, 4),
])
def test_mls_ring_reuses_layouts_and_tiles(tn, prefetch, defer, bm, bn, pw, cw):
    if not torch.cuda.is_available():
        pytest.skip("requires HCU GPU")
    from boltops.utility.arch_info import get_device_info
    if get_device_info().arch != "gfx946":
        pytest.skip("requires gfx946")
    from boltops.gemm.gluon.gemm_a16w16 import _gemm_a16w16_kernel

    m, n, k = 64, max(128, bn * 2), 256
    generator = torch.Generator().manual_seed(59)
    a_host = (torch.rand((m, k), generator=generator) - .5).bfloat16()
    w_host = (torch.rand((n, k), generator=generator) - .5).bfloat16()
    expected = (a_host.float() @ w_host.float().T).bfloat16()
    a = a_host.cuda()
    w = w_host.cuda() if tn else w_host.T.contiguous().cuda().T
    # Guard both ends of a complete output allocation.
    storage = torch.full((m * n + 512,), 123, dtype=torch.bfloat16, device="cuda")
    out = storage[256:-256].view(m, n)
    out.fill_(float("nan"))
    handle = _gemm_a16w16_kernel[(m // bm * (n // bn),)](
        a, w, out, M=m, N=n, K=k, BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=64, CHUNKS=1 if bn == 256 else 2,
        SLOTS=3 if bn == 256 else 2, PW=pw, CW=cw,
        LTS=not tn, B_K_CONTIGUOUS=tn, PREFETCH=prefetch,
        DEFER_RELEASE=defer, num_warps=1, num_stages=2,
        wasp_partition_warps=(pw, cw), wasp_wdra=False,
        mmac_layout_force=2, enable_v_mmac_cluster=2,
        hcu_use_gfx946_sched_model_v2=True,
    )
    torch.cuda.synchronize()
    assert handle.n_spills == 0
    host = storage.cpu()
    actual = host[256:-256].view(m, n)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=.001, rtol=.01)
    assert (host[:256] == 123).all() and (host[-256:] == 123).all()
