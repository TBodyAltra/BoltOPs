import pytest
import torch

from boltops.gemm.triton.gemm_a8wfp4 import gemm_a8wfp4, grouped_gemm_a8wfp4


VALUES = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                       -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def scale(shape, encoded, generator):
    exps = torch.randint(-3, 3, shape, generator=generator)
    values = torch.ldexp(torch.ones(shape), exps)
    # Numeric scales deliberately include values that are not powers of 2.
    if not encoded:
        values *= 1.125
    return (exps + 127).to(torch.uint8) if encoded else values, values


def inputs(e=3, m=5, n=35, k=130, group=128, a8=True, b8=True):
    g = torch.Generator().manual_seed(17)
    a = (torch.randn(e, m, k, generator=g) * 2).to(torch.float8_e4m3fn)
    # Includes all 16 E2M1 codes and both nibble positions.
    packed = torch.randint(0, 256, (e, n, (k + 1)//2), generator=g, dtype=torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)[..., :k]
    b = VALUES[codes.long()]
    sa, fa = scale((e, m, (k + group - 1)//group), a8, g)
    sb, fb = scale((e, n, (k + 31)//32), b8, g)
    # Independent reference dequantizes each operand before CPU FP32 matmul.
    da = a.float() * fa.repeat_interleave(group, -1)[..., :k]
    db = b * fb.repeat_interleave(32, -1)[..., :k]
    ref = da @ db.transpose(-1, -2)
    return [z.cuda() for z in (a, packed, sa, sb)], ref


@pytest.mark.parametrize('group', [32, 128])
@pytest.mark.parametrize('a8,b8', [(True, True), (False, True), (True, False), (False, False)])
def test_grouped(group, a8, b8):
    args, ref = inputs(group=group, a8=a8, b8=b8)
    out = grouped_gemm_a8wfp4(*args, activation_group_size=group)
    torch.testing.assert_close(out.cpu(), ref.bfloat16(), atol=.05, rtol=.01)


def test_dense_strides_and_output():
    args, ref = inputs(e=1, k=65, group=32)
    # Preserve values but vary strides for weights, activations and scales.
    args = [a.transpose(-1, -2).contiguous().transpose(-1, -2) for a in args]
    args = [a[0] for a in args]
    y = torch.empty((35, 5), dtype=torch.float32, device='cuda').T
    out = gemm_a8wfp4(*args, activation_group_size=32, dtype=torch.float32, y=y)
    assert out is y
    torch.testing.assert_close(out.cpu(), ref[0], atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize('group,a8', [(32, True), (128, False)])
def test_ragged_empty_experts(group, a8):
    args, ref = inputs(e=4, m=37, k=97, group=group, a8=a8)
    a, w, sa, sb = args
    lengths = [0, 37, 1, 19]
    x = torch.cat([a[i, :m] for i, m in enumerate(lengths)])
    xs = torch.cat([sa[i, :m] for i, m in enumerate(lengths)])
    expected = torch.cat([ref[i, :m] for i, m in enumerate(lengths)])
    offsets = torch.tensor([0, 0, 37, 38, 57], device='cuda', dtype=torch.int32)
    out = grouped_gemm_a8wfp4(x, w, xs, sb, expert_offsets=offsets,
                             activation_group_size=group)
    torch.testing.assert_close(out.cpu(), expected.bfloat16(), atol=.05, rtol=.01)


def test_invalid_scale_shape():
    args, _ = inputs()
    args[2] = args[2][..., :1]
    with pytest.raises(ValueError, match='scale shapes'):
        grouped_gemm_a8wfp4(*args)


def test_all_e2m1_codes_native_gemm():
    # Every column contains one FP4 code in both nibble positions. Native MMAC
    # must interpret all 16 codes, rather than treating raw codes as FP8 bytes.
    a = torch.ones((16, 32)).to(torch.float8_e4m3fn).cuda()
    codes = torch.arange(16, dtype=torch.uint8)
    packed = (codes | (codes << 4))[:, None].expand(16, 16).contiguous().cuda()
    sa = torch.ones((16, 1), device='cuda')
    sb = torch.full((16, 1), 127, dtype=torch.uint8, device='cuda')
    got = gemm_a8wfp4(a, packed, sa, sb, activation_group_size=32,
                      dtype=torch.float32)
    expected = (VALUES * 32)[None, :].expand(16, 16)
    torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('group', [32, 128])
@pytest.mark.parametrize('a8', [False, True])
def test_each_k32_scale_applied_before_accumulation(group, a8):
    # Four weight scales inside one K128 span must each scale their own K32 dot.
    # A single scale after a K128 dot gives a different answer for every column.
    m, n, k = 5, 19, 128
    a = torch.ones((m, k)).to(torch.float8_e4m3fn).cuda()
    w = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device='cuda')
    kg = k // group
    if a8:
        sa_cpu = (torch.arange(kg)[None, :].expand(m, kg) + 126).to(torch.uint8)
        fa = torch.pow(2., sa_cpu.float() - 127)
    else:
        sa_cpu = (torch.arange(kg)[None, :].expand(m, kg).float() + 2) / 4
        fa = sa_cpu
    sb_cpu = (torch.arange(4)[None, :].expand(n, 4) + 123).to(torch.uint8)
    fb = torch.pow(2., sb_cpu.float() - 127)
    got = gemm_a8wfp4(a, w, sa_cpu.cuda(), sb_cpu.cuda(), activation_group_size=group)
    expected = (32 * (fa.repeat_interleave(group // 32, -1)[:, None, :]
                      * fb[None, :, :]).sum(-1)).bfloat16()
    torch.testing.assert_close(got.cpu(), expected, atol=0, rtol=0)


def test_empty_input():
    args, _ = inputs(e=2, m=0, n=17, k=33)
    got = grouped_gemm_a8wfp4(*args)
    assert got.shape == (2, 0, 17)


def test_fp16_output():
    args, ref = inputs(e=1, m=3, n=19, k=96, group=32, a8=False)
    got = gemm_a8wfp4(*[a[0] for a in args], activation_group_size=32,
                      dtype=torch.float16)
    torch.testing.assert_close(got.cpu(), ref[0].half(), atol=.01, rtol=.01)
