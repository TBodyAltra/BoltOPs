"""Native block-scaled dense GEMM with independent A/B scale blocks."""
import operator

import torch
import triton

from ._block_scaled_dispatch import launch, select_config


def _parse_recipe(scale_recipe):
    try:
        recipe = tuple(tuple(operator.index(x) for x in block) for block in scale_recipe)
        if len(recipe) != 2 or any(len(block) != 2 or any(x <= 0 for x in block) for block in recipe):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("scale_recipe must be two positive integer pairs: ((A_M,A_K),(B_N,B_K))") from None
    return recipe


def gemm_block_scaled(a, b, a_scale, b_scale, *,
                      scale_recipe=((1, 128), (128, 128)), out_dtype=None):
    """Compute block-scaled A[M,K] @ B[N,K].T using native MMAC.

    Scale shapes are [ceil(M/A_M),ceil(K/A_K)] and
    [ceil(N/B_N),ceil(K/B_K)]. Blocks may differ; K-group is contiguous.
    Scales contain numeric values (uint8 is numeric, not implicitly E8M0).
    Input/accumulator legality is queried from the target MMAC database;
    unsupported combinations raise compiler errors, without input upcasting.
    Output dtype defaults to BF16 and may be selected explicitly. Scale
    arithmetic uses FP32, promoted to FP64 for FP64 inputs or scales.
    The existing tuned recipe uses its MLS/masked kernels; other combinations
    use a general masked native-MMAC kernel with no performance claim.
    """
    recipe = _parse_recipe(scale_recipe)
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
            config = select_config(a, b, a_scale, b_scale, recipe, out_dtype)
            launch(config, a, b, a_scale, b_scale, out, recipe=recipe)
    return out
