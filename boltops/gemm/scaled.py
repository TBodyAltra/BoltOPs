"""Native block-scaled dense GEMM with independent A/B scale blocks."""
import operator

import torch
import triton

from .registry import launch, select_plan
from .descriptors import GemmProblem, OperandSpec, ScaleSpec


def _parse_recipe(scale_recipe):
    try:
        recipe = tuple(tuple(operator.index(x) for x in block) for block in scale_recipe)
        if len(recipe) != 2 or any(len(block) != 2 or any(x <= 0 for x in block) for block in recipe):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("scale_recipe must be two positive integer pairs: ((A_M,A_K),(B_N,B_K))") from None
    return recipe


def gemm_scaled(a, b, a_scale, b_scale, *,
                      scale_recipe=((1, 128), (128, 128)),
                      scale_encodings=("numeric", "numeric"), out_dtype=None, b_format="native"):
    """Compute block-scaled A[M,K] @ B[N,K].T using native dot operations.

    Scale shapes are [ceil(M/A_M),ceil(K/A_K)] and
    [ceil(N/B_N),ceil(K/B_K)]. Blocks may differ; K-group is contiguous.
    Scale encoding is explicit per input: "numeric" (default) or "e8m0".
    E8M0 requires uint8 storage; uint8 remains numeric by default.
    Input/accumulator legality is checked by the selected target compiler;
    unsupported combinations raise compiler errors, without input upcasting.
    Output dtype defaults to BF16 and may be selected explicitly. Scale
    arithmetic uses FP32, promoted to FP64 for FP64 inputs or scales.
    A versioned performance map selects measured Gluon configurations.
    Requests outside those contracts use the general Triton implementation.
    """
    if b_format == "fp4_e2m1":
        from .grouped import gemm_grouped
        if any(t.ndim != 2 for t in (a, b, a_scale, b_scale)):
            raise ValueError("scaled FP4 expects rank-two inputs and scales")
        return gemm_grouped(a.unsqueeze(0), b.unsqueeze(0), a_scale=a_scale.unsqueeze(0),
                            b_scale=b_scale.unsqueeze(0), scale_recipe=scale_recipe,
                            scale_encodings=scale_encodings, b_format=b_format,
                            out_dtype=out_dtype).squeeze(0)
    if b_format != "native":
        raise ValueError("b_format must be native or fp4_e2m1")
    recipe = _parse_recipe(scale_recipe)
    if (not isinstance(scale_encodings, (tuple, list)) or len(scale_encodings) != 2 or
            any(e not in ("numeric", "e8m0") for e in scale_encodings)):
        raise ValueError('scale_encodings must be a pair of "numeric" or "e8m0"')
    for tensor, encoding in zip((a_scale, b_scale), scale_encodings):
        if encoding == "e8m0" and tensor.dtype != torch.uint8:
            raise ValueError("e8m0 scale encoding requires uint8 storage")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected A[M,K] and B[N,K] with matching K")
    m, k = a.shape
    n = b.shape[0]
    (am, ak), (bn, bk) = recipe
    expected = (("a", a, (m, k)), ("b", b, (n, k)),
                ("a_scale", a_scale, (triton.cdiv(m, am), triton.cdiv(k, ak))),
                ("b_scale", b_scale, (triton.cdiv(n, bn), triton.cdiv(k, bk))))
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape} for scale_recipe={recipe}, got {tuple(tensor.shape)}")
    for _, tensor, _ in expected:
        if tensor.device != a.device or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous on the same GPU")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    with torch.cuda.device(a.device):
        out = torch.empty((m, n), dtype=out_dtype, device=a.device)
        if m == 0 or n == 0:
            return out
        if k == 0:
            out.zero_()
        else:
            problem = GemmProblem(
                kind="dense", m=m, n=n, k=k, a=OperandSpec(a.dtype, strides=tuple(a.stride())), b=OperandSpec(b.dtype, strides=tuple(b.stride())),
                out_dtype=out_dtype, arch=triton.runtime.driver.active.get_current_target().arch,
                num_cus=torch.cuda.get_device_properties(a.device).multi_processor_count,
                a_scale=ScaleSpec(a_scale.dtype, recipe[0], scale_encodings[0]),
                b_scale=ScaleSpec(b_scale.dtype, recipe[1], scale_encodings[1]),
                a_aligned_16=a.data_ptr() % 16 == 0, b_aligned_16=b.data_ptr() % 16 == 0,
            )
            launch(select_plan(problem), problem, a=a, b=b, a_scale=a_scale, b_scale=b_scale, out=out)
    return out
