"""Dense shape/scale addressing coverage for the gfx946 Gluon interface."""
import pytest
import torch
import triton

pytest.importorskip("triton.experimental.gluon.language.amd.hcu")
from boltops.gemm import gemm_block_scaled


@pytest.mark.parametrize("shape", [
    (1, 1, 1), (65, 129, 129), (129, 257, 257), (64, 128, 128),
    (128, 256, 256), (256, 256, 256), (128, 512, 256), (256, 512, 256),
    (128, 256, 768), (128, 256, 6144), (8, 17, 8192),
    (3, 5, 0), (0, 17, 129), (9, 0, 128),
])
def test_dense_shapes(shape):
    if (not torch.cuda.is_available() or
        triton.runtime.driver.active.get_current_target().arch != "gfx946"):
        pytest.skip("requires gfx946")
    m, n, k = shape
    groups = triton.cdiv(k, 128)
    torch.manual_seed(20)
    # Exact dyadic inputs/scales give a strict oracle even for fused FP32 FMA.
    a = (torch.randint(-4, 5, (m, k)).float() / 4).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (n, k)).float() / 4).to(torch.float8_e4m3fn)
    sa = torch.randint(1, 5, (m, groups)).float() / 8
    sb = torch.randint(1, 5, (triton.cdiv(n, 128), groups)).float() / 8
    ref = torch.zeros((m, n))
    for i in range(groups):
        lo, hi = i * 128, min((i + 1) * 128, k)
        ref += ((a[:, lo:hi].float() @ b[:, lo:hi].float().T)
                * sa[:, i, None] * sb[:, i].repeat_interleave(128)[:n][None, :])
    out = gemm_block_scaled(*(x.cuda() for x in (a, b, sa, sb)))
    torch.testing.assert_close(out.cpu(), ref.bfloat16(), atol=0, rtol=0)


def test_reject_mismatched_k():
    a = torch.empty((2, 3), dtype=torch.float8_e4m3fn)
    b = torch.empty((4, 5), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="matching K"):
        gemm_block_scaled(a, b, a, b)


@pytest.mark.parametrize("recipe", [((1, 0), (128, 32)), ((-1, 128), (128, 128)),
                                   ((1, 128),), None, (128, 128)])
def test_reject_invalid_scale_recipe(recipe):
    with pytest.raises(ValueError, match="scale_recipe must be"):
        gemm_block_scaled(None, None, None, None, scale_recipe=recipe)


@pytest.mark.parametrize("index,shape", [(2, (1,)), (2, (1, 2)), (2, (2, 8)),
                                        (3, (129, 2)), (3, (2, 8))])
def test_reject_scale_shape(index, shape):
    args = [torch.empty((2, 256), dtype=torch.float8_e4m3fn),
            torch.empty((129, 256), dtype=torch.float8_e4m3fn),
            torch.empty((2, 2), dtype=torch.float32), torch.empty((2, 2), dtype=torch.float32)]
    args[index] = torch.empty(shape, dtype=torch.float32)
    with pytest.raises(ValueError, match="must have shape.*scale_recipe"):
        gemm_block_scaled(*args)
