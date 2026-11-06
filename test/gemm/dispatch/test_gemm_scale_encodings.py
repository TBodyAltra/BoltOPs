"""Requirement scale recipes and explicit numeric/E8M0 storage semantics."""
import math

import pytest
import torch
import triton
import triton.language as tl

from boltops.gemm import gemm_scaled
from boltops.gemm.triton.gemm_block_scaled import _decode_block_scale


def require_hcu():
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != "gfx946":
        pytest.skip("requires gfx946")


@triton.jit
def decode_kernel(X, Y):
    i = tl.arange(0, 256)
    value = _decode_block_scale(tl.load(X + i), True)
    tl.store(Y + i, value.to(tl.int32, bitcast=True))


def test_all_e8m0_codes_preserve_exact_fp32_bits():
    require_hcu()
    encoded = torch.arange(256).to(torch.uint8).cuda()
    output = torch.empty(256, dtype=torch.int32, device="cuda")
    decode_kernel[(1,)](encoded, output, num_warps=4)
    reference = torch.tensor([math.ldexp(1., i - 127) for i in range(255)] + [float("nan")])
    assert torch.equal(output.cpu(), reference.view(torch.int32))


@pytest.mark.parametrize("scale_dtypes,encodings,recipe,k", [
    ((torch.float32, torch.bfloat16), ("numeric", "numeric"), ((1, 128), (128, 128)), 256),
    ((torch.float32, torch.uint8), ("numeric", "e8m0"), ((1, 128), (128, 128)), 256),
    ((torch.float32, torch.float32), ("numeric", "numeric"), ((1, 128), (128, 128)), 256),
    ((torch.float32, torch.float32), ("numeric", "numeric"), ((1, 129), (1, 129)), 129),
    ((torch.uint8, torch.float32), ("e8m0", "numeric"), ((1, 32), (2, 64)), 65),
    ((torch.uint8, torch.uint8), ("e8m0", "e8m0"), ((1, 32), (2, 64)), 65),
])
def test_requirement_scale_combinations(scale_dtypes, encodings, recipe, k):
    require_hcu()
    torch.manual_seed(76)
    m, n = 5, 7
    a = (torch.randint(-4, 5, (m, k)).float() / 4).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (n, k)).float() / 4).to(torch.float8_e4m3fn)
    scales, decoded = [], []
    for rows, block, dtype, encoding in zip((m, n), recipe, scale_dtypes, encodings):
        shape = (triton.cdiv(rows, block[0]), triton.cdiv(k, block[1]))
        if encoding == "e8m0":
            value = torch.randint(125, 130, shape).to(torch.uint8)
            golden = torch.pow(2., value.int() - 127)
        else:
            value = (torch.randint(1, 5, shape).float() / 8).to(dtype)
            golden = value.float()
        scales.append(value)
        decoded.append(golden)
    q = math.gcd(recipe[0][1], recipe[1][1])
    ref = torch.zeros(m, n)
    for lo in range(0, k, q):
        aa = decoded[0][torch.arange(m) // recipe[0][0], lo // recipe[0][1]]
        bb = decoded[1][torch.arange(n) // recipe[1][0], lo // recipe[1][1]]
        ref += (a[:, lo:lo+q].float() @ b[:, lo:lo+q].float().T) * aa[:, None] * bb[None, :]
    out = gemm_scaled(*(x.cuda() for x in (a, b, *scales)),
                            scale_recipe=recipe, scale_encodings=encodings)
    torch.testing.assert_close(out.cpu(), ref.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("encodings", ["e8m0", ("numeric",), ("numeric", "unknown")])
def test_invalid_encoding_is_rejected(encodings):
    with pytest.raises(ValueError, match="scale_encodings"):
        gemm_scaled(None, None, None, None, scale_encodings=encodings)


def test_e8m0_requires_byte_storage():
    x = torch.ones(1)
    with pytest.raises(ValueError, match="uint8 storage"):
        gemm_scaled(x, x, x, x, scale_encodings=("numeric", "e8m0"))
