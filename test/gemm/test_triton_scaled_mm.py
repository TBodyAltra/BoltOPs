"""Tensor and channel-scale coverage for the unified gemm_block_scaled API."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from boltops.gemm import gemm_block_scaled  # noqa: E402

DEVICE = "cuda"
GPU_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


@GPU_REQUIRED
@pytest.mark.parametrize("M,N,K", [(64, 128, 64)])
def test_triton_scaled_mm_fp16(M: int, N: int, K: int):
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16)
    b = torch.randn((K, N), dtype=torch.float16)
    scale_a = torch.tensor([[1.5]], dtype=torch.float32)
    scale_b = torch.tensor([[0.5]], dtype=torch.float32)
    golden = (a.float() @ b.float()) * scale_a * scale_b.T

    out = gemm_block_scaled(
        a.to(DEVICE),
        b.T.contiguous().to(DEVICE),
        scale_a.to(DEVICE),
        scale_b.to(DEVICE),
        out_dtype=torch.float16,
        scale_recipe=((M, K), (N, K)),
    )
    torch.testing.assert_close(out.cpu().float(), golden, atol=2e-2, rtol=2e-2)


@GPU_REQUIRED
def test_triton_scaled_mm_int8_channel_scales():
    torch.manual_seed(1)
    M, N, K = 32, 64, 64
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    scale_a = torch.rand((M, 1), dtype=torch.float32) * 0.1 + 0.01
    scale_b = torch.rand((N, 1), dtype=torch.float32) * 0.1 + 0.01
    golden = (a.float() @ b.float()) * scale_a * scale_b.T

    out = gemm_block_scaled(
        a.to(DEVICE),
        b.T.contiguous().to(DEVICE),
        scale_a.to(DEVICE),
        scale_b.to(DEVICE),
        out_dtype=torch.float16,
        scale_recipe=((1, K), (1, K)),
    )
    torch.testing.assert_close(out.cpu().float(), golden, atol=2e-2, rtol=2e-2)
