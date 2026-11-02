"""Numerical coverage of native dtype/scale-block dispatch, with small shapes."""
import math
import pytest
import torch
import triton
from boltops.gemm import gemm_block_scaled, gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16


CASES = [
    (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2, torch.float32, (3, 5, 33), ((1, 32), (2, 64))),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float16, torch.bfloat16, torch.float32, (5, 9, 129), ((1, 32), (1, 64))),
    (torch.float8_e5m2, torch.float8_e5m2, torch.float32, torch.bfloat16, torch.float16, (7, 11, 65), ((2, 16), (4, 32))),
    (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float16, torch.bfloat16, (4, 6, 63), ((2, 24), (3, 40))),
    (torch.float8_e5m2, torch.float8_e4m3fn, torch.float16, torch.float32, torch.float32, (3, 7, 31), ((1, 32), (3, 64))),
    (torch.float16, torch.float16, torch.float16, torch.bfloat16, torch.float32, (17, 19, 33), ((4, 3), (5, 5))),
    (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.float32, torch.bfloat16, (9, 13, 97), ((8, 64), (16, 96))),
    (torch.float32, torch.float32, torch.float32, torch.float32, torch.float32, (5, 7, 64), ((1, 128), (1, 128))),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float32, torch.float32, torch.float32, (5, 7, 128), ((1, 128), (128, 128))),
    (torch.float16, torch.float16, torch.float32, torch.float32, torch.float16, (3, 5, 300), ((1, 512), (128, 512))),
    (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.uint8, torch.uint8, torch.bfloat16, (3, 5, 17), ((3, 7), (4, 11))),
]


def require_hcu():
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().arch != "gfx946":
        pytest.skip("requires gfx946")


def data(case):
    ad, bd, sd, td, out, (m, n, k), recipe = case
    (am, ak), (bn, bk) = recipe
    torch.manual_seed(61)
    a = (torch.randint(-4, 5, (m, k)).float() / 4).to(ad)
    b = (torch.randint(-4, 5, (n, k)).float() / 4).to(bd)
    sa = (torch.randint(1, 5, (triton.cdiv(m, am), triton.cdiv(k, ak))).float()
          / (1 if sd == torch.uint8 else 8)).to(sd)
    sb = (torch.randint(1, 5, (triton.cdiv(n, bn), triton.cdiv(k, bk))).float()
          / (1 if td == torch.uint8 else 8)).to(td)
    ref = torch.zeros((m, n))
    q = math.gcd(ak, bk)
    for lo in range(0, k, q):
        aa = sa.float()[torch.arange(m) // am, lo // ak]
        bb = sb.float()[torch.arange(n) // bn, lo // bk]
        ref += ((a[:, lo:lo+q].float() @ b[:, lo:lo+q].float().T) * aa[:, None]) * bb[None, :]
    return (a, b, sa, sb), ref.to(out)


@pytest.mark.parametrize("case", CASES)
def test_numeric_dtypes_and_recipes(case):
    require_hcu()
    args, ref = data(case)
    actual = gemm_block_scaled(*(x.cuda() for x in args), scale_recipe=case[-1], out_dtype=case[4])
    assert actual.dtype == case[4]
    torch.testing.assert_close(actual.cpu(), ref, atol=0, rtol=0)


def test_legacy_name_forwards_nondefault_recipe():
    require_hcu()
    case = next(case for case in CASES if case[0] == torch.float16)
    args, ref = data(case)
    actual = gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16(
        *(x.cuda() for x in args), scale_recipe=case[-1], out_dtype=case[4])
    torch.testing.assert_close(actual.cpu(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(0, 7, 17), (5, 0, 17), (3, 5, 0)])
def test_general_empty(shape):
    require_hcu()
    m,n,k=shape
    args=[torch.empty((m,k),dtype=torch.float16),torch.empty((n,k),dtype=torch.float16),
          torch.empty((triton.cdiv(m,2),triton.cdiv(k,3))),
          torch.empty((triton.cdiv(n,4),triton.cdiv(k,5)))]
    out=gemm_block_scaled(*(x.cuda() for x in args),scale_recipe=((2,3),(4,5)),out_dtype=torch.float32)
    assert out.shape==(m,n) and out.dtype==torch.float32
    if k==0:assert torch.equal(out.cpu(),torch.zeros((m,n)))


@pytest.mark.parametrize("dtypes", [(torch.float16,torch.bfloat16),
                                     (torch.float8_e4m3fnuz,torch.float8_e4m3fnuz)])
def test_unsupported_native_input_reports_capability(dtypes,capfd):
    require_hcu()
    args=[torch.empty((2,32),dtype=dtypes[0],device='cuda'),
          torch.empty((3,32),dtype=dtypes[1],device='cuda'),
          torch.ones((2,1),device='cuda'),torch.ones((1,1),device='cuda')]
    with pytest.raises((triton.CompilationError, ValueError, RuntimeError)) as error:
        gemm_block_scaled(*args,out_dtype=torch.float32)
    captured=capfd.readouterr()
    message=str(error.value)+captured.err+captured.out
    assert 'native MMAC' in message or 'not supported in this architecture' in message
