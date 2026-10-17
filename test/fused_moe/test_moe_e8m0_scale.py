"""FP8 post-dot E8M0 scales in grouped MoE kernels."""
import pytest
import torch
import triton.language as tl
from boltops.fused_moe.triton import moe_op
from boltops.utility import arch_info

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("kind", ["regular", "combined", "persistent"])
@pytest.mark.parametrize("which", ["activation", "weight", "both"])
def test_grouped_fp8_e8m0_scales(kind, which, monkeypatch):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 dot is unavailable")
    torch.manual_seed(283)
    e, m, n, k, bm = 2, 3, 64, 256, 16
    a = (torch.randn(e*m, k) * .5).to(torch.float8_e4m3fn)
    b = (torch.randn(e, n, k) * .5).to(torch.float8_e4m3fn)
    ac = torch.randint(125, 130, (e*m, 2), dtype=torch.uint8)
    bc = torch.randint(125, 130, (e, n, 2), dtype=torch.uint8)
    af = torch.ldexp(torch.ones(ac.shape), ac.int()-127)
    bf = torch.ldexp(torch.ones(bc.shape), bc.int()-127)
    asc = ac if which in ("activation", "both") else af
    bsc = bc if which in ("weight", "both") else bf
    expected = torch.zeros(e*m, n)
    for expert in range(e):
        for start in range(0, k, 128):
            g = start // 128
            aa = a[expert*m:(expert+1)*m, start:start+128].float()
            aa = aa * af[expert*m:(expert+1)*m, g:g+1]
            bb = b[expert, :, start:start+128].float() * bf[expert, :, g:g+1]
            expected[expert*m:(expert+1)*m] += aa @ bb.T
    expected = expected.to(torch.bfloat16)
    ids = torch.tensor([*range(m), *([e*m]*(bm-m)),
                        *range(m, e*m), *([e*m]*(bm-m))], dtype=torch.int32)
    experts = torch.tensor([0, 1], dtype=torch.int32)
    count = torch.tensor([2*bm], dtype=torch.int32)
    topk_ids = torch.arange(e, dtype=torch.int32).repeat_interleave(m).view(-1, 1)
    output = torch.full((e*m, n), float("nan"), dtype=torch.bfloat16).cuda()
    config = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128,
                  GROUP_SIZE_M=1, COMBINE_SCALE_LOAD=(kind == "combined"),
                  num_warps=4, num_stages=2, SPLIT_K=0)
    # Select existing wrapper dispatch paths; no kernel is replaced.
    monkeypatch.setattr(moe_op, "_USE_MOE_PERSISTENT_KERNEL", kind == "persistent")
    moe_op.fused_moe(a.cuda(), b.cuda(), output, asc.cuda(), bsc.cuda(), None,
                     None, topk_ids.cuda(), ids.cuda(), None, experts.cuda(), count.cuda(),
                     False, 1, tl.bfloat16, use_fp8_w8a8=True, block_shape=[1, 128],
                     config=config)
    torch.testing.assert_close(output.cpu(), expected, atol=.01, rtol=.01)
