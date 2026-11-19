# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Dense Gluon dispatch uses the public GEMM contract and precision profile."""
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from boltops.gemm import gemm_a16w16
from boltops.gemm.config import get_gemm_config

PROFILE = "DTYPE=bfloat16-OUT=bfloat16"
KERNEL = "_gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel"
TT_KERNEL = "_gemm_a16w16_bf16_tt_bm16_bn32_bk128_kernel"
PERSISTENT = "2048x2048x4096"
PERSISTENT_KERNEL = "_gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel"
STREAMK_KERNEL = "_gemm_a16w16_bf16_tt_bm256_bn128_bk64_persistent_kernel"


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
    assert cfg['wasp_partition_warps'] == [1, 1] and cfg['wasp_wdra'] is False
    assert (cfg['BLOCK_SIZE_M'], cfg['BLOCK_SIZE_N'], cfg['BLOCK_SIZE_K']) == (16, 16, 128)
    for kwargs in [dict(m=63), dict(n=128), dict(k=1024),
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
    monkeypatch.setattr(gluon, TT_KERNEL, Capture("gluon_tt", calls))
    monkeypatch.setattr(triton, "_gemm_a16_w16_kernel", Capture("triton", calls))
    return calls


def provide_gluon_config(monkeypatch, config):
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, "get_gemm_config", lambda *args, **kwargs: (deepcopy(config), True))


def provide_triton_config(monkeypatch, config):
    module = import_module("boltops.gemm.triton.gemm_a16w16")
    monkeypatch.setattr(module, "get_gemm_config", lambda *args, **kwargs: (deepcopy(config), True))


@pytest.mark.parametrize("case,expected", [
    ("recorded", "gluon"), ("different_m", "triton"),
    ("fp16", "triton"), ("column_major", "triton"),
    ("weight_column_major", "gluon_tt"),
    ("fp32_output", "triton"), ("bias", "triton"),
    ("activation", "triton"),
    ("other_arch_with_config", "gluon"),
])
def test_auto_selects_only_recorded_contract(monkeypatch, case, expected):
    calls = install_captures(monkeypatch)
    gluon = import_module("boltops.gemm.gluon.gemm_a16w16")
    dispatch_gluon = gluon.gemm_a16w16
    gluon_calls = []

    def observe_gluon(*args, **kwargs):
        gluon_calls.append(kwargs["config"])
        return dispatch_gluon(*args, **kwargs)

    monkeypatch.setattr(gluon, "gemm_a16w16", observe_gluon)
    if case == "other_arch_with_config":
        public = import_module("boltops.gemm.gemm_a16w16")
        monkeypatch.setattr(public, "get_device_info",
                            lambda device: SimpleNamespace(arch="gfx950", cu_count=48))

        def lookup(*args, **kwargs):
            assert kwargs["arch"] == "gfx950"
            return config_for()

        monkeypatch.setattr(gluon, "get_gemm_config", lookup)
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
    out = gemm_a16w16(a, w, **kw)
    assert out.shape == (m, 64)
    assert len(calls) == 1 and calls[0][0] == expected
    assert len(gluon_calls) == (expected in ("gluon", "gluon_tt"))
    if expected in ("gluon", "gluon_tt"):
        _, grid, args, params = calls[0]
        assert grid == ((8,) if expected == "gluon_tt" else (16,))
        assert args[0] is a and args[1] is w and args[2] is out
        assert (params['M'], params['N'], params['K']) == (64, 64, 2048)
        assert params['wasp_partition_warps'] == (1, 1)
        assert params['wasp_wdra'] is False
        assert params["B_K_CONTIGUOUS"] is (expected == "gluon")


@pytest.mark.parametrize("backend", ["auto", "gluon"])
def test_gluon_backend_rejects_strided_output(monkeypatch, backend):
    calls = install_captures(monkeypatch)
    config, _ = config_for()
    provide_gluon_config(monkeypatch, config)
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty_like(a)
    out = torch.empty((64, 64), dtype=a.dtype, device=a.device).T

    with pytest.raises(NotImplementedError, match="Gluon A16W16 kernels require a contiguous output"):
        gemm_a16w16(a, w, out=out, backend=backend)
    assert not calls


def test_unknown_backend_uses_triton(monkeypatch):
    calls = install_captures(monkeypatch)
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty_like(a)
    gemm_a16w16(a, w, backend="unknown")
    assert len(calls) == 1 and calls[0][0] == "triton"


@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_explicit_gluon_selects_kernel_without_mutating_config(monkeypatch, layout):
    calls = install_captures(monkeypatch)
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, "kernel_from_config", Capture("custom", calls), raising=False)
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty_like(a)
    out = torch.empty((64, 64), dtype=a.dtype, device=a.device)
    if layout == "TT":
        w = w.as_strided(w.shape, (1, 64))
    cfg, _ = config_for(layout=layout)
    cfg['kernel'] = "kernel_from_config"
    cfg['BLOCK_SIZE_M'] = cfg['BLOCK_SIZE_N'] = 32
    cfg["B_K_CONTIGUOUS"] = layout != "TN"
    before = deepcopy(cfg)
    provide_gluon_config(monkeypatch, cfg)
    result = gemm_a16w16(a, w, out=out, backend="gluon")
    assert result is out and cfg == before and calls[0][0] == "custom"
    assert "kernel" not in calls[0][3] and calls[0][1] == (4,)
    assert calls[0][3]["B_K_CONTIGUOUS"] is (layout == "TN")


@pytest.mark.parametrize("case", ["shape", "dtype", "layout", "output", "bias", "activation"])
def test_explicit_gluon_rejects_unsupported_contract(monkeypatch, case):
    calls = install_captures(monkeypatch)
    a = torch.empty((63 if case == "shape" else 64, 2048),
                    dtype=torch.float16 if case == "dtype" else torch.bfloat16, device="cuda")
    w = torch.empty((64, 2048), dtype=a.dtype, device=a.device)
    if case == "layout":
        a = a.as_strided(a.shape, (1, a.shape[0]))
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


@pytest.mark.parametrize("layout", ["TN", "TT"])
@pytest.mark.parametrize("backend", ["auto", "gluon"])
def test_public_gluon_native_correctness(backend, layout):
    require_device()
    gen = torch.Generator().manual_seed(851)
    a = (torch.rand((64, 2048), generator=gen)-.5).bfloat16()
    w = (torch.rand((64, 2048), generator=gen)-.5).bfloat16()
    gold = (a.float() @ w.float().T).bfloat16()
    weight = w.cuda() if layout == "TN" else w.T.contiguous().cuda().T
    result = gemm_a16w16(a.cuda(), weight, backend=backend)
    torch.testing.assert_close(result.cpu(), gold, atol=.001, rtol=.01)


@pytest.mark.parametrize("layout,profile", [
    ("TN", PROFILE),
    ("TT", "DTYPE=float16-OUT=float16"),
])
def test_persistent_config_carries_tile_sms_and_kernel(layout, profile):
    cfg, tuned = config_for(m=2048, n=2048, k=4096, layout=layout, profile=profile)
    assert tuned and cfg["kernel"] == PERSISTENT_KERNEL
    assert (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]) == (256, 256, 64)
    assert cfg["NUM_SMS"] == 48
    assert cfg["wasp_partition_warps"] == [4, 4, 4]
    assert cfg["wasp_partition_regs"] == [16, 244, 244]


@pytest.mark.parametrize("m,n,k,cu", [
    pytest.param(2048, 2048, 4096, 48, id="R020-48cu"),
    pytest.param(2048, 2048, 4096, 96, id="R020-96cu"),
])
def test_streamk_recorded_config_matches_shape_reuse_compiler(m, n, k, cu):
    cfg, tuned = config_for(m=m, n=n, k=k, layout="TT", cu=cu)
    assert tuned
    assert cfg == dict(
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, NUM_SMS=cu,
        num_warps=2, num_stages=2, wasp_partition_warps=[2, 2, 4, 4],
        wasp_wdra=False, use_async_copy=False, optimize_epilogue=False,
        enable_v_mmac_cluster=1, hcu_use_gfx946_sched_model_v2=True,
        amdgpu_enable_max_ilp_scheduling_strategy=False, kernel=STREAMK_KERNEL,
    )
    # The BF16 inline assembly must not replace the existing FP16 pipeline.
    fp16_cfg, tuned = config_for(
        m=m, n=n, k=k, layout="TT", profile="DTYPE=float16-OUT=float16",
    )
    assert tuned and fp16_cfg["kernel"] == PERSISTENT_KERNEL


@pytest.mark.parametrize("cu", [48, 96])
def test_streamk_schedule_covers_each_tile_and_k_interval_once(cu):
    runtime = import_module("boltops.gemm.gluon.gemm_a16w16_streamk")
    m, n, k = 2048, 2048, 4096
    table, records, slots = runtime._make_schedule(m, n, k, cu)
    tiles = ((m + 255) // 256) * (n // 128)
    assert len(table) // records == cu
    coverage, workload = {}, []
    for start in range(0, len(table), records):
        tasks = [item for item in table[start:start + records] if item[2]]
        assert tasks and table[start + len(tasks)] == (0, 0, 0, 0)
        workload.append(sum(item[2] for item in tasks))
        for tile, offset, length, tag in tasks:
            coverage.setdefault(tile, []).append((offset, offset + length, tag))
    assert set(coverage) == set(range(tiles))
    split_slots = []
    for intervals in coverage.values():
        intervals.sort()
        assert intervals[0][0] == 0 and intervals[-1][1] == k
        assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
        if len(intervals) == 1:
            assert intervals[0][2] == 0
        else:
            assert len(intervals) == 2 and intervals[0][2] < 0
            assert intervals[0][2] == -intervals[1][2]
            split_slots.append(intervals[1][2])
    assert sorted(split_slots) == list(range(1, slots + 1))
    assert max(workload) - min(workload) <= 64


def test_streamk_compiled_kernel_cache_is_device_specific(monkeypatch):
    from triton.backends.compiler import GPUTarget

    runtime = import_module("boltops.gemm.gluon.gemm_a16w16_streamk")
    device = [0]
    compiled = []

    def compile_kernel(*args, **kwargs):
        handle = SimpleNamespace(device_index=device[0], function=object())
        compiled.append(handle)
        return handle

    monkeypatch.setattr(torch.cuda, "current_device", lambda: device[0])
    monkeypatch.setattr(runtime, "GluonASTSource", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime.triton, "compile", compile_kernel)
    args = (object(), 2048, 2048, 4096, 129, (), GPUTarget("hip", "gfx946", 64))
    runtime._compile_kernel_for_device.cache_clear()
    try:
        first = runtime._compile_kernel(*args)
        device[0] = 1
        second = runtime._compile_kernel(*args)
        assert first.device_index == 0 and second.device_index == 1
        assert first.function is not second.function
        for index, expected in ((0, first), (1, second)):
            device[0] = index
            assert runtime._compile_kernel(*args) is expected
        assert compiled == [first, second]
    finally:
        runtime._compile_kernel_for_device.cache_clear()


def test_streamk_native_sbx2_split_k(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info
    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 96:
        pytest.skip("requires gfx946 with 96 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    # Exercise R020's cross-die split/whole-tile schedule with a short K.
    m = n = 2048
    cfg, _ = config_for(m=m, n=n, k=4096, layout="TT", cu=96)
    generator = torch.Generator().manual_seed(96)
    a = (torch.rand((m, 128), generator=generator) - .5).bfloat16()
    b = (torch.rand((128, n), generator=generator) - .5).bfloat16()
    expected = (a.float() @ b.float()).bfloat16()
    ai, bi = a.cuda(), b.cuda()
    storage = torch.full((m * n + 128,), 123., dtype=torch.bfloat16).cuda()
    out = storage[64:-64].view(m, n)
    provide_gluon_config(monkeypatch, cfg)
    assert gemm_a16w16(ai, bi.T, out=out, backend="gluon") is out
    host = storage.cpu()
    torch.testing.assert_close(host[64:-64].view(m, n), expected, atol=.001, rtol=.01)
    assert bool((host[:64] == 123).all() and (host[-64:] == 123).all())
    assert torch.equal(ai.cpu(), a) and torch.equal(bi.cpu(), b)


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


@pytest.mark.parametrize("layout,dtype", [
    ("TN", torch.bfloat16), ("TN", torch.float16), ("TT", torch.float16),
])
@pytest.mark.parametrize("backend", ["auto", "gluon"])
def test_persistent_dispatch_launches_one_tg_per_sm(monkeypatch, layout, dtype, backend):
    calls = install_captures(monkeypatch)
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, PERSISTENT_KERNEL, Capture("persistent", calls))
    a = torch.empty((2048, 4096), dtype=dtype, device="cuda")
    w = torch.empty((2048, 4096), dtype=a.dtype, device=a.device)
    if layout == "TT":
        w = w.as_strided(w.shape, (1, 2048))
    result = gemm_a16w16(a, w, dtype=dtype, backend=backend)
    assert result.shape == (2048, 2048)
    label, grid, args, params = calls[0]
    assert label == "persistent" and grid == (48,)
    assert args[0] is a and args[1] is w
    assert (params["M"], params["N"], params["K"]) == (2048, 2048, 4096)
    assert params["NUM_SMS"] == 48 and "kernel" not in params
    assert params["B_K_CONTIGUOUS"] is (layout == "TN")


@pytest.mark.parametrize("m,n,k,cu", [
    pytest.param(2048, 2048, 4096, 48, id="R020-48cu"),
    pytest.param(2048, 2048, 4096, 96, id="R020-96cu"),
])
@pytest.mark.parametrize("backend", ["auto", "gluon"])
def test_streamk_dispatch_uses_dynamic_lds_launcher(monkeypatch, m, n, k, backend, cu):
    install_captures(monkeypatch)
    public = import_module("boltops.gemm.gemm_a16w16")
    monkeypatch.setattr(public, "get_device_info", lambda device: SimpleNamespace(arch="gfx946", cu_count=cu))
    kernels = import_module("boltops.gemm.gluon.gemm_a16w16")
    calls = []

    def launch(kernel, a, b, out, *, cu_count, config, layout):
        calls.append((kernel, a, b, out, cu_count, config, layout))

    monkeypatch.setitem(kernels._SPECIAL_LAUNCHERS, STREAMK_KERNEL, launch)
    a = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.empty((k, n), dtype=a.dtype, device=a.device).T
    out = gemm_a16w16(a, w, backend=backend)
    assert len(calls) == 1
    kernel, actual_a, actual_w, actual_out, cu_count, config, layout = calls[0]
    assert kernel is getattr(kernels, STREAMK_KERNEL)
    assert layout == "TT"
    assert actual_a is a and actual_w is w and actual_out is out
    assert cu_count == cu and config["NUM_SMS"] == cu and "kernel" not in config
    assert config["wasp_partition_warps"] == [2, 2, 4, 4]


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
def test_persistent_native_tt_tail_correctness(monkeypatch, dtype):
    require_device()
    m, n, k = 17, 33, 65
    gen = torch.Generator().manual_seed(21)
    a = (torch.randn((m, k), generator=gen) * .5).to(dtype)
    b = (torch.randn((k, n), generator=gen) * .5).to(dtype)
    gold = (a.float() @ b.float()).to(dtype)
    a_dev, b_dev = a.cuda(), b.cuda()
    storage = torch.full((m*n + 128,), 123., dtype=dtype, device=a_dev.device)
    out = storage[64:-64].view(m, n)
    # Select the general pipeline explicitly; the recorded BF16 TT shape now
    # dispatches to Stream-K, whose alignment contract excludes this tail case.
    cfg, _ = config_for(m=2048, n=2048, k=4096, layout="TN")
    provide_gluon_config(monkeypatch, cfg)
    result = gemm_a16w16(a_dev, b_dev.T, dtype=dtype, out=out, backend="gluon")
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
    provide_gluon_config(monkeypatch, cfg)
    with pytest.raises(ValueError, match="kernel selector"):
        gemm_a16w16(a, w, backend="gluon")
    assert not calls


def test_fixed_tt_config_has_matching_layout_and_precision():
    cfg, tuned = config_for(layout="TT")
    assert tuned and cfg["kernel"] == TT_KERNEL
    assert (cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]) == (16, 32, 128)
    assert cfg["wasp_partition_warps"] == [1, 1] and cfg["wasp_wdra"] is False


@pytest.mark.parametrize("case", ["layout", "dtype", "output", "shape"])
@pytest.mark.parametrize("kernel_layout", ["TN", "TT"])
def test_explicit_fixed_rejects_mismatched_contract(monkeypatch, case, kernel_layout):
    require_device()
    from triton.compiler.errors import CompilationError
    dtype = torch.float16 if case == "dtype" else torch.bfloat16
    m = 63 if case == "shape" else 64
    a = torch.empty((m, 2048), dtype=dtype, device="cuda")
    w = torch.empty((64, 2048), dtype=dtype, device="cuda")
    actual_layout = ("TN" if kernel_layout == "TT" else "TT") if case == "layout" else kernel_layout
    if actual_layout == "TT":
        w = w.as_strided(w.shape, (1, 64))
    cfg, _ = config_for(layout=kernel_layout)
    out_dtype = torch.float32 if case == "output" else dtype
    provide_gluon_config(monkeypatch, cfg)
    with pytest.raises(CompilationError):
        gemm_a16w16(a, w, dtype=out_dtype, backend="gluon")


NARROW_TT_KERNEL = "_gemm_a16w16_kernel"


@pytest.mark.parametrize("m", [2560, 6656])
def test_narrow_tt_public_dispatch(monkeypatch, m):
    calls = install_captures(monkeypatch)
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, NARROW_TT_KERNEL, Capture("narrow_tt", calls))
    a = torch.empty((m, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty((2048, 64), dtype=a.dtype, device=a.device).T
    out = torch.empty((m, 64), dtype=a.dtype, device=a.device)
    assert gemm_a16w16(a, w, out=out) is out
    label, grid, args, params = calls[0]
    assert label == "narrow_tt" and grid == (m // 64,)
    assert args[0] is a and args[1] is w and args[2] is out
    assert params["wasp_partition_warps"] == (1, 4)
    assert (params["CHUNKS"], params["SLOTS"]) == (2, 3)
    assert params["B_K_CONTIGUOUS"] is False


@pytest.mark.parametrize("case", ["tail", "layout", "dtype", "output"])
def test_narrow_tt_explicit_contract_rejects_unsafe_inputs(monkeypatch, case):
    require_device()
    from triton.compiler.errors import CompilationError
    cfg, _ = config_for(m=2560, layout="TT")
    dtype = torch.float16 if case == "dtype" else torch.bfloat16
    a = torch.empty((2559 if case == "tail" else 2560, 2048), dtype=dtype, device="cuda")
    w = torch.empty((64, 2048), dtype=dtype, device=a.device)
    if case != "layout":
        w = w.as_strided(w.shape, (1, 64))
    output = torch.float32 if case == "output" else dtype
    provide_gluon_config(monkeypatch, cfg)
    with pytest.raises(CompilationError):
        gemm_a16w16(a, w, dtype=output, backend="gluon")


@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_triton_no_splitk_passes_positive_unused_stride(monkeypatch, layout):
    calls = install_captures(monkeypatch)
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty_like(a)
    if layout == "TT":
        w = w.as_strided(w.shape, (1, 64))
    cfg = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
               GROUP_SIZE_M=1, NUM_KSPLIT=1, SPLITK_BLOCK_SIZE=2048,
               num_warps=4, num_stages=2, cache_modifier=".cg")
    provide_triton_config(monkeypatch, cfg)
    result = gemm_a16w16(a, w, backend="triton")
    assert len(calls) == 1 and calls[0][0] == "triton"
    assert calls[0][2][11] == result.numel()


def test_triton_no_splitk_native_correctness(monkeypatch):
    require_device()
    gen = torch.Generator().manual_seed(59)
    a = (torch.rand((64, 2048), generator=gen) - .5).bfloat16()
    b = (torch.rand((2048, 64), generator=gen) - .5).bfloat16()
    expected = (a.float() @ b.float()).bfloat16()
    cfg = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
               GROUP_SIZE_M=1, NUM_KSPLIT=1, SPLITK_BLOCK_SIZE=2048,
               num_warps=4, num_stages=2, cache_modifier=".cg")
    provide_triton_config(monkeypatch, cfg)
    result = gemm_a16w16(a.cuda(), b.cuda().T, backend="triton")
    torch.testing.assert_close(result.cpu(), expected, atol=.001, rtol=.01)


@pytest.mark.parametrize("cu_count", [48, 96])
def test_wide_tt_public_dispatch(monkeypatch, cu_count):
    calls = install_captures(monkeypatch)
    from boltops.utility.arch_info import DeviceInfo
    public = import_module("boltops.gemm.gemm_a16w16")
    monkeypatch.setattr(public, "get_device_info",
                        lambda device: DeviceInfo(device.index, "gfx946", cu_count))
    module = import_module("boltops.gemm.gluon.gemm_a16w16")
    monkeypatch.setattr(module, NARROW_TT_KERNEL, Capture("wide_tt", calls))
    a = torch.empty((64, 2048), dtype=torch.bfloat16, device="cuda")
    w = torch.empty((2048, 248320), dtype=a.dtype, device=a.device).T
    out = torch.empty((64, 248320), dtype=a.dtype, device=a.device)
    assert gemm_a16w16(a, w, out=out) is out
    assert len(calls) == 1
    label, grid, args, params = calls[0]
    assert label == "wide_tt" and grid == (970,)
    assert args[0] is a and args[1] is w and args[2] is out
    assert params["wasp_partition_warps"] == (2, 4)
    assert (params["BLOCK_SIZE_M"], params["BLOCK_SIZE_N"], params["BLOCK_SIZE_K"]) == (64, 256, 64)
    assert (params["CHUNKS"], params["SLOTS"]) == (1, 3)


    assert params["PREFETCH"] is True

@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_fixed_bf16_config_is_not_reused_for_fp16(layout):
    with pytest.raises(KeyError):
        config_for(profile="DTYPE=float16-OUT=float16", layout=layout)
