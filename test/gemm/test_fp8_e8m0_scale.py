"""E8M0 decoding in ordinary dot-then-scale GEMMs."""
import pytest
import torch
import triton
import triton.language as tl

from boltops.utility.fp8_scale import decode_fp8_scale

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@triton.jit
def _decode_kernel(X, Y, N: tl.constexpr, FP8_INPUT: tl.constexpr):
    i = tl.arange(0, 256)
    value = tl.load(X + i, i < N, other=0)
    decoded = decode_fp8_scale(value, FP8_INPUT)
    tl.store(Y + i, decoded, i < N)


def test_all_e8m0_codes():
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    out = torch.empty(256, dtype=torch.float32).cuda()
    _decode_kernel[(1,)](codes.cuda(), out, 256, True)
    actual = out.cpu()
    expected = torch.ldexp(torch.ones(256), codes.int() - 127)
    expected[255] = float("nan")
    # Check subnormals by their bits; an absolute tolerance could hide code 0.
    assert torch.equal(actual[:255].view(torch.int32), expected[:255].view(torch.int32))
    assert torch.isnan(actual[255])


@pytest.mark.parametrize("fp8_input", [False, True])
def test_floating_scales_are_bitwise_unchanged(fp8_input):
    values = torch.tensor([0., -0., .5, 1.25, float("inf"), float("nan")])
    out = torch.empty_like(values).cuda()
    _decode_kernel[(1,)](values.cuda(), out, values.numel(), fp8_input)
    assert torch.equal(out.cpu().view(torch.int32), values.view(torch.int32))


def test_non_fp8_uint8_scales_remain_numeric():
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    out = torch.empty(256).cuda()
    _decode_kernel[(1,)](codes.cuda(), out, 256, False)
    torch.testing.assert_close(out.cpu(), codes.float(), rtol=0, atol=0)


def _scale(shape, encoded):
    if encoded:
        codes = torch.randint(125, 130, shape, dtype=torch.uint8)
        return codes, torch.ldexp(torch.ones(shape), codes.int() - 127)
    values = torch.rand(shape) * .2 + .05
    return values, values


def _golden(a, w, sa, sw):
    # Independently dequantize operands before matmul.
    result = torch.zeros(a.shape[0], w.shape[0])
    for start in range(0, a.shape[1], 128):
        g = start // 128
        aa = a[:, start:start + 128].float() * sa[:, g:g + 1]
        ww = w[:, start:start + 128].float() * sw[:, g:g + 1]
        result += aa @ ww.T
    return result.to(torch.bfloat16)


@pytest.mark.parametrize("splits", [1, 2])
@pytest.mark.parametrize("a_encoded,b_encoded", [(True, False), (False, True),
                                                (True, True), (False, False)])
def test_fp8_gemm_post_dot_e8m0_scales(a_encoded, b_encoded, splits):
    from boltops.gemm import gemm_a8w8_blockscale
    from boltops.utility import arch_info
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 dot is unavailable")
    torch.manual_seed(281)
    m, n, k = 17, 96, 256
    a = (torch.randn(m, k) * .5).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k) * .5).to(torch.float8_e4m3fn)
    asc, af = _scale((m, 2), a_encoded)
    wsc, wf = _scale((n, 2), b_encoded)
    expected = _golden(a, w, af, wf)
    config = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128,
                  GROUP_SIZE_M=1, NUM_KSPLIT=splits, num_warps=4, num_stages=2,
                  waves_per_eu=0, matrix_instr_nonkdim=16, cache_modifier=".cg")
    out = gemm_a8w8_blockscale(a.cuda(), w.cuda(), asc.cuda(), wsc.cuda(),
                              dtype=torch.bfloat16, config=config, backend="triton")
    torch.testing.assert_close(out.cpu(), expected, atol=.01, rtol=.01)


def test_vllm_block_scaled_fp8_e8m0_scales():
    from boltops.gemm import w8a8_triton_block_scaled_mm
    from boltops.utility import arch_info
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 dot is unavailable")
    torch.manual_seed(282)
    m, n, k = 17, 128, 256
    a = (torch.randn(m, k) * .5).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k) * .5).to(torch.float8_e4m3fn)
    asc, af = _scale((m, 2), True)
    wsc, wf = _scale((1, 2), True)
    expected = _golden(a, w, af, wf.expand(n, 2))
    out = w8a8_triton_block_scaled_mm(a.cuda(), w.cuda(), asc.cuda(), wsc.cuda(),
                                     [128, 128], output_dtype=torch.bfloat16)
    torch.testing.assert_close(out.cpu(), expected, atol=.01, rtol=.01)
