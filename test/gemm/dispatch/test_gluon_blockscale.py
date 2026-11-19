# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Recorded Gluon selection and its explicit native input contract."""

from importlib import import_module

import pytest
import torch

from boltops.gemm import gemm_a8w8_blockscale
from boltops.gemm.config import get_gemm_config

PROFILE = "DTYPE=float8_e4m3fn-AS=float32-BS=float32-OUT=bfloat16-RECIPE_A=1x128-RECIPE_B=128x128"
MODULE = "gemm_a8w8_blockscale"
KERNEL = "_gemm_a8w8_blockscale_as_fp32_bs_fp32_recipe_a_1x128_recipe_b_128x128_bm128_bn256_bk128_kernel"


def test_48tg_config_is_exact_and_cu_specific():
    params, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE",
        768,
        2048,
        6144,
        profile=PROFILE,
        backend="gluon",
        arch="gfx946",
        cu_num=48,
    )
    assert tuned
    assert (768 // params["BLOCK_SIZE_M"]) * (2048 // params["BLOCK_SIZE_N"]) == 48
    _, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE",
        640,
        2048,
        6144,
        profile=PROFILE,
        backend="gluon",
        arch="gfx946",
        cu_num=48,
    )
    assert not tuned
    with pytest.raises(FileNotFoundError):
        get_gemm_config(
            "GEMM-A8W8_BLOCKSCALE",
            768,
            2048,
            6144,
            backend="gluon",
            arch="gfx946",
            cu_num=192,
        )


@pytest.mark.parametrize(
    "m,scale_dtype,expected_backend,strided_out",
    [
        (768, torch.float32, "gluon", False),
        (640, torch.float32, "triton", False),
        (768, torch.float16, "triton", False),
        (768, torch.float32, "gluon", True),
    ],
)
@pytest.mark.parametrize("a_column_major", [False, True])
def test_auto_selects_only_the_recorded_gluon_shape(
    monkeypatch, m, scale_dtype, expected_backend, a_column_major, strided_out
):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info

    if get_device_info().arch != "gfx946" or get_device_info().cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    gluon = import_module(f"boltops.gemm.gluon.{MODULE}")
    triton = import_module("boltops.gemm.triton.gemm_a8w8_blockscale_fp8")
    calls = []

    class CaptureLaunch:
        def __init__(self, backend):
            self.backend = backend

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(
                    (self.backend, grid(kwargs) if callable(grid) else grid, kwargs)
                )

            return launch

    monkeypatch.setattr(
        gluon,
        KERNEL,
        CaptureLaunch("gluon"),
    )
    monkeypatch.setattr(triton, "_w8a8_triton_block_scaled_mm", CaptureLaunch("triton"))
    a = torch.empty((m, 6144), dtype=torch.float8_e4m3fn, device="cuda")
    if a_column_major:
        a = a.as_strided(a.shape, (1, m))
        expected_backend = "triton"
    b = torch.empty((2048, 6144), dtype=a.dtype, device=a.device)
    sa = torch.empty((m, 48), dtype=scale_dtype, device=a.device)
    sb = torch.empty((16, 48), dtype=torch.float32, device=a.device)
    out = None
    if strided_out:
        out = torch.empty((2048, m), dtype=torch.bfloat16, device=a.device).T
    if strided_out and expected_backend == "gluon":
        with pytest.raises(ValueError, match="output must be contiguous"):
            gemm_a8w8_blockscale(a, b, sa, sb, out=out)
        assert not calls
        return
    result = gemm_a8w8_blockscale(a, b, sa, sb, out=out)
    if strided_out:
        assert result is out
    assert result.shape == (m, 2048) and result.dtype == torch.bfloat16
    assert len(calls) == 1 and calls[0][0] == expected_backend
    if expected_backend == "gluon":
        assert calls[0][1] == (6, 8)
        assert calls[0][2]["wasp_partition_warps"] == (4, 4, 4, 4)


@pytest.mark.parametrize(
    "k,scale_dtype,out_dtype",
    [
        (128, torch.float32, torch.bfloat16),
        (256, torch.float16, torch.bfloat16),
        (256, torch.float32, torch.float32),
    ],
)
def test_explicit_gluon_rejects_unsupported_combinations(k, scale_dtype, out_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info
    from triton.compiler.errors import CompilationError

    if get_device_info().arch != "gfx946" or get_device_info().cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    a = torch.empty((128, k), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((256, k), dtype=a.dtype, device=a.device)
    sa = torch.empty((128, k // 128), dtype=scale_dtype, device=a.device)
    sb = torch.empty((2, k // 128), dtype=scale_dtype, device=a.device)
    with pytest.raises(
        (CompilationError, NotImplementedError),
        match="pipeline requires|no Gluon configuration",
    ):
        gemm_a8w8_blockscale(a, b, sa, sb, dtype=out_dtype, backend="gluon")


def test_native_gluon_blockscale_small():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from boltops.utility.arch_info import get_device_info

    if get_device_info().arch != "gfx946" or get_device_info().cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    torch.manual_seed(713)
    a = (torch.randint(-4, 5, (128, 256)).float() / 4).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (256, 256)).float() / 4).to(torch.float8_e4m3fn)
    sa = torch.randint(1, 5, (128, 2)).float() / 8
    sb = torch.randint(1, 5, (2, 2)).float() / 8
    golden = torch.zeros((128, 256), dtype=torch.float32)
    for group in range(2):
        start, end = group * 128, (group + 1) * 128
        golden += (a[:, start:end].float() @ b[:, start:end].float().T) * (
            sa[:, group, None] * sb[:, group].repeat_interleave(128)[None, :]
        )
    out = gemm_a8w8_blockscale(
        a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), backend="gluon"
    )
    torch.testing.assert_close(out.cpu(), golden.bfloat16(), atol=0, rtol=0)


def test_gluon_kernel_and_tile_come_from_config(monkeypatch, override_gemm_config):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    seen = {}

    class CaptureKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.update(grid=grid(kwargs), params=kwargs)

            return launch

    pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
    kernels = import_module("boltops.gemm.gluon.gemm_a8w8_blockscale")
    monkeypatch.setattr(kernels, "kernel_another_tile", CaptureKernel(), raising=False)
    config = dict(
        kernel="kernel_another_tile",
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
        num_warps=4,
        num_stages=1,
    )
    before = config.copy()
    a = torch.empty((128, 256), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((256, 256), dtype=a.dtype, device=a.device)
    sa = torch.empty((128, 2), dtype=torch.float32, device=a.device)
    sb = torch.empty((2, 2), dtype=torch.float32, device=a.device)
    override_gemm_config("GEMM-A8W8_BLOCKSCALE", config, backend="gluon")
    gemm_a8w8_blockscale(a, b, sa, sb, backend="gluon")
    assert config == before
    assert seen["grid"] == (2, 2)
    assert tuple(seen["params"]["BLOCK_SIZE_" + dim] for dim in ("M", "N", "K")) == (
        64,
        128,
        64,
    )
    assert seen["params"]["GROUP_N"] == seen["params"]["GROUP_K"] == 128


@pytest.mark.parametrize("n,tiles", [(9216, 720), (12288, 960)])
def test_persistent_grid_comes_from_record(monkeypatch, n, tiles):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    kernels = import_module(f"boltops.gemm.gluon.{MODULE}")
    config, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", 2560, n, 2048, profile=PROFILE,
        backend="gluon", arch=device.arch, cu_num=device.cu_count,
    )
    assert tuned
    calls = []

    class CaptureKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid(kwargs), kwargs))
            return launch

    monkeypatch.setattr(kernels, config["kernel"], CaptureKernel())
    a = torch.empty((2560, 2048), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((n, 2048), dtype=a.dtype, device=a.device)
    sa = torch.empty((2560, 16), dtype=torch.float32, device=a.device)
    sb = torch.empty((n // 128, 16), dtype=torch.float32, device=a.device)
    out = gemm_a8w8_blockscale(a, b, sa, sb)
    assert out.shape == (2560, n)
    assert len(calls) == 1
    grid, options = calls[0]
    assert grid == (48,)
    assert options["NUM_SMS"] == 48 and options["TILE_LIMIT"] == tiles


def test_fp32_scale_64x64_config_is_shape_specific():
    profile = "DTYPE=float8_e4m3fn-AS=float32-BS=float32-OUT=bfloat16-RECIPE_A=1x128-RECIPE_B=128x128"
    cfg, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", 64, 2048, 4096, profile=profile,
        backend="gluon", arch="gfx946", cu_num=48)
    assert tuned
    assert [cfg["BLOCK_SIZE_" + d] for d in "MNK"] == [64, 64, 128]
    assert cfg["wasp_partition_warps"] == [1, 4]
    assert cfg["num_warps"] == 1 and cfg["num_stages"] == 2
    assert cfg["wasp_wdra"] is False


@pytest.mark.parametrize("k", [128, 256, 384])
def test_native_gluon_fp32_scales_64x64(override_gemm_config, k):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from boltops.utility.arch_info import get_device_info
    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    profile = "DTYPE=float8_e4m3fn-AS=float32-BS=float32-OUT=bfloat16-RECIPE_A=1x128-RECIPE_B=128x128"
    config, _ = get_gemm_config("GEMM-A8W8_BLOCKSCALE", 64, 2048, 4096,
        profile=profile, backend="gluon", arch=device.arch, cu_num=device.cu_count)
    torch.manual_seed(761)
    m, n = 128, 192
    x = (torch.randint(-4, 5, (m, k)).float() / 4).to(torch.float8_e4m3fn)
    w = (torch.randint(-4, 5, (n, k)).float() / 4).to(torch.float8_e4m3fn)
    sa = torch.randint(1, 5, (m, k // 128)).float() / 8
    # Match R009: retain BF16 scale values in an FP32 kernel buffer.
    sb = (torch.randint(1, 5, (2, k // 128)).float() / 8).bfloat16().float()
    golden = torch.zeros((m, n))
    for group in range(k // 128):
        start, end = group * 128, (group + 1) * 128
        golden += (x[:, start:end].float() @ w[:, start:end].float().T) * (
            sa[:, group, None] * sb[:, group].float().repeat_interleave(128)[:n][None, :])
    override_gemm_config("GEMM-A8W8_BLOCKSCALE", config, backend="gluon")
    out = gemm_a8w8_blockscale(x.cuda(), w.cuda(), sa.cuda(), sb.cuda(),
                              backend="gluon")
    torch.testing.assert_close(out.cpu(), golden.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("m,n,k,tgs", [(64, 12288, 2048, 96), (64, 9216, 2048, 144), (2560, 2048, 4096, 48)])
def test_qwen_recorded_pipeline_grid(monkeypatch, m, n, k, tgs):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    config, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", m, n, k, profile=PROFILE,
        backend="gluon", arch=device.arch, cu_num=device.cu_count,
    )
    assert tuned
    calls = []

    class CaptureKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid(kwargs), kwargs))
            return launch

    kernels = import_module(f"boltops.gemm.gluon.{MODULE}")
    monkeypatch.setattr(kernels, config["kernel"], CaptureKernel())
    a = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((n, k), dtype=a.dtype, device=a.device)
    sa = torch.empty((m, k // 128), dtype=torch.float32, device=a.device)
    sb = torch.empty((n // 128, k // 128), dtype=torch.float32, device=a.device)
    out = gemm_a8w8_blockscale(a, b, sa, sb)
    assert out.shape == (m, n)
    assert len(calls) == 1
    assert calls[0][0] == ((tgs,) if "NUM_SMS" in config else (1, tgs))
    if "TILE_LIMIT" in config:
        assert calls[0][1]["TILE_LIMIT"] == (m // config["BLOCK_SIZE_M"]) * (n // config["BLOCK_SIZE_N"])


@pytest.mark.parametrize(
    "record_shape,shape,tgs",
    [
        ((64, 12288, 2048), (64, 384, 512), 1),
        ((64, 12288, 2048), (64, 256, 2048), 2),
        ((64, 9216, 2048), (64, 192, 512), 3),
        ((64, 9216, 2048), (128, 192, 2048), 6),
        ((2560, 2048, 4096), (256, 768, 2048), 4),
        ((2560, 2048, 4096), (384, 256, 4096), 1),
        ((2560, 2048, 4096), (384, 256, 6144), 1),
        ((2560, 2048, 4096), (256, 768, 4096), 4),
    ],
)
def test_qwen_native_scale_pipelines(override_gemm_config, record_shape, shape, tgs):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    config, _ = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", *record_shape, profile=PROFILE,
        backend="gluon", arch=device.arch, cu_num=device.cu_count,
    )
    m, n, k = shape
    if "NUM_SMS" in config:
        config["NUM_SMS"] = tgs
        config["TILE_LIMIT"] = (m // config["BLOCK_SIZE_M"]) * (n // config["BLOCK_SIZE_N"])
    generator = torch.Generator().manual_seed(721)
    a = (torch.randint(-4, 5, (m, k), generator=generator).float() / 4).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (n, k), generator=generator).float() / 4).to(torch.float8_e4m3fn)
    sa = torch.randint(1, 5, (m, k // 128), generator=generator).float() / 8
    sb = (torch.randint(1, 5, ((n + 127) // 128, k // 128), generator=generator).float() / 8).bfloat16().float()
    golden = torch.zeros((m, n))
    for group in range(k // 128):
        part = slice(group * 128, (group + 1) * 128)
        golden += (a[:, part].float() @ b[:, part].float().T) * (
            sa[:, group, None] * sb[:, group].repeat_interleave(128)[:n][None, :]
        )
    storage = torch.full((m * n + 512,), -4096., dtype=torch.bfloat16, device="cuda")
    output = storage[256:-256].view(m, n)
    override_gemm_config("GEMM-A8W8_BLOCKSCALE", config, backend="gluon")
    result = gemm_a8w8_blockscale(
        a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), out=output,
        backend="gluon",
    )
    assert result is output
    torch.testing.assert_close(result.cpu(), golden.bfloat16(), atol=0, rtol=0)
    assert bool((storage[:256].cpu() == -4096).all())
    assert bool((storage[-256:].cpu() == -4096).all())


def test_chunked_scale_pipeline_rejects_partial_chunks(override_gemm_config):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info
    from triton.compiler.errors import CompilationError

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 48:
        pytest.skip("requires gfx946 with 48 CU")
    config, _ = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", 2560, 2048, 4096, profile=PROFILE,
        backend="gluon", arch=device.arch, cu_num=device.cu_count,
    )
    config.update(NUM_SMS=1, TILE_LIMIT=1)
    a = torch.empty((128, 3072), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((256, 3072), dtype=a.dtype, device=a.device)
    sa = torch.empty((128, 24), dtype=torch.float32, device=a.device)
    sb = torch.empty((2, 24), dtype=torch.float32, device=a.device)
    override_gemm_config("GEMM-A8W8_BLOCKSCALE", config, backend="gluon")
    with pytest.raises(CompilationError, match="complete 16-group chunks"):
        gemm_a8w8_blockscale(a, b, sa, sb, backend="gluon")


@pytest.mark.parametrize("shape,cu_count,layout", [
    ((2560, 2048, 4096), 48, "TN"),
    ((2560, 2048, 4096), 96, "TN"),
    ((2560, 12288, 2048), 96, "TN"),
    ((2560, 12288, 2048), 96, "TT"),
])
def test_mmac_turn_encloses_complete_instruction_group(shape, cu_count, layout):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    import re
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != cu_count:
        pytest.skip(f"requires gfx946 with {cu_count} CU")
    config, _ = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", *shape, profile=PROFILE,
        layout="TT" if layout == "TT" else None,
        backend="gluon", arch=device.arch, cu_num=device.cu_count,
    )
    kernels = import_module(f"boltops.gemm.gluon.{MODULE}")
    kernel = getattr(kernels, config.pop("kernel"))
    config = {key: tuple(value) if isinstance(value, list) else value
              for key, value in config.items()}
    m, n, k = shape
    a = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((n, k), dtype=a.dtype, device=a.device)
    if layout == "TT":
        b = b.as_strided((n, k), (1, n))
    if kernel.fn.__name__.endswith("_persistent_kernel"):
        config["B_K_CONTIGUOUS"] = layout == "TN"
    sa = torch.empty((m, k // 128), dtype=torch.float32, device=a.device)
    sb = torch.empty((n // 128, k // 128), dtype=torch.float32, device=a.device)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    compiled = kernel.warmup(
        a, b, sa, sb, out, M=m, N=n, K=k, GROUP_N=128, GROUP_K=128,
        grid=(config["NUM_SMS"],), **config,
    )
    waiting = None
    mmacs = 0
    groups = 0
    for line in compiled.asm["amdgcn"].splitlines():
        barrier = re.search(r"s_ebarrier_(sync|arrive)\s+([89]),\s*8", line)
        if barrier:
            kind, barrier_id = barrier.group(1), int(barrier.group(2))
            if kind == "sync":
                assert waiting is None, "MMAC turn contains a nested handoff"
                waiting, mmacs = barrier_id, 0
            elif waiting is not None:
                assert barrier_id == 17 - waiting
                assert mmacs == 64, "turn must enclose one complete BK128 MMAC group"
                waiting = None
                groups += 1
            else:
                assert barrier_id == 8, "only the initial turn may be seeded"
        elif "v_mmac_" in line:
            assert waiting is not None, "MMAC escaped its execution-barrier boundaries"
            mmacs += 1
    assert waiting is None and groups == 6


@pytest.mark.parametrize("n,k,tiles", [(12288, 2048, 960), (2048, 4096, 160)])
@pytest.mark.parametrize("layout,m", [("TN", 2560), ("TT", 2560), ("TT", 128), ("sliced", 2560)])
def test_sbx2_layout_dispatch(monkeypatch, layout, m, n, k, tiles):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU allocations")
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 96:
        pytest.skip("requires gfx946 with 96 CU")
    kernels = import_module(f"boltops.gemm.gluon.{MODULE}")
    triton = import_module("boltops.gemm.triton.gemm_a8w8_blockscale_fp8")
    config, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", 2560, n, k, profile=PROFILE,
        layout="TT" if layout == "TT" and n == 12288 else None,
        backend="gluon", arch=device.arch, cu_num=device.cu_count,
    )
    assert tuned
    calls = []

    class CaptureKernel:
        def __init__(self, backend):
            self.backend = backend

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((self.backend, grid(kwargs), kwargs))
            return launch

    monkeypatch.setattr(kernels, config["kernel"], CaptureKernel("gluon"))
    monkeypatch.setattr(triton, "_w8a8_triton_block_scaled_mm", CaptureKernel("triton"))
    a = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="cuda")
    if layout == "TT":
        b = torch.empty((k, n), dtype=a.dtype, device=a.device).T
    elif layout == "sliced":
        b = torch.empty((n, k * 2), dtype=a.dtype, device=a.device)[:, ::2]
    else:
        b = torch.empty((n, k), dtype=a.dtype, device=a.device)
    sa = torch.empty((m, k // 128), dtype=torch.float32, device=a.device)
    sb = torch.empty((n // 128, k // 128), dtype=torch.float32, device=a.device)
    out = gemm_a8w8_blockscale(a, b, sa, sb)
    assert out.shape == (m, n)
    assert len(calls) == 1
    backend, grid, options = calls[0]
    if m == 2560 and (layout == "TN" or (layout == "TT" and n == 12288)):
        assert backend == "gluon" and grid == (96,)
        if n == 12288:
            assert options["B_K_CONTIGUOUS"] == (layout == "TN")
        assert options["TILE_LIMIT"] == tiles
    else:
        assert backend == "triton"


def test_64x32_config_is_shape_and_cu_specific():
    cfg, tuned = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", 64, 2048, 4096, profile=PROFILE,
        backend="gluon", arch="gfx946", cu_num=96,
    )
    assert tuned and [cfg["BLOCK_SIZE_" + d] for d in "MNK"] == [64, 32, 128]
    assert cfg["wasp_partition_warps"] == [1, 2]
    for m in (128, 256):
        with pytest.raises(KeyError, match="no M bucket"):
            get_gemm_config(
                "GEMM-A8W8_BLOCKSCALE", m, 2048, 4096, profile=PROFILE,
                backend="gluon", arch="gfx946", cu_num=96,
            )
    cfg48, tuned48 = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", 64, 2048, 4096, profile=PROFILE,
        backend="gluon", arch="gfx946", cu_num=48,
    )
    assert tuned48 and cfg48["BLOCK_SIZE_N"] == 64


@pytest.mark.parametrize("m,n,k", [(64, 96, 128), (64, 192, 256), (128, 160, 384)])
def test_64x32_scale_groups_and_output_guards(override_gemm_config, m, n, k):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    from boltops.utility.arch_info import get_device_info

    device = get_device_info()
    if device.arch != "gfx946" or device.cu_count != 96:
        pytest.skip("requires gfx946 / 96 CU")
    cfg, _ = get_gemm_config(
        "GEMM-A8W8_BLOCKSCALE", 64, 2048, 4096, profile=PROFILE,
        backend="gluon", arch="gfx946", cu_num=96,
    )
    original = {key: value[:] if isinstance(value, list) else value for key, value in cfg.items()}
    gen = torch.Generator().manual_seed(936)
    a = (torch.randint(-4, 5, (m, k), generator=gen).float() / 4).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (n, k), generator=gen).float() / 4).to(torch.float8_e4m3fn)
    sa = torch.randint(1, 5, (m, k // 128), generator=gen).float() / 8
    sb = (torch.randint(1, 5, ((n + 127) // 128, k // 128), generator=gen).float() / 8).bfloat16().float()
    golden = torch.zeros((m, n))
    for i in range(k // 128):
        tile = slice(i * 128, (i + 1) * 128)
        golden += (a[:, tile].float() @ b[:, tile].float().T) * sa[:, i, None] * sb[:, i].repeat_interleave(128)[:n][None, :]
    storage = torch.full((m * n + 512,), 123., device="cuda", dtype=torch.bfloat16)
    out = storage[256:-256].view(m, n)
    override_gemm_config("GEMM-A8W8_BLOCKSCALE", cfg, backend="gluon")
    result = gemm_a8w8_blockscale(
        a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), out=out, backend="gluon",
    )
    assert result is out and cfg == original
    torch.testing.assert_close(result.cpu(), golden.bfloat16(), atol=0, rtol=0)
    assert bool((storage[:256].cpu() == 123).all())
    assert bool((storage[-256:].cpu() == 123).all())
