"""Compatibility entry point; implementations live under gluon.pipelines."""


def gemm_fp8e4m3fn_fp32scale_a1x128_b128x128_to_bf16(
    a, b, a_scale, b_scale, *, scale_recipe=((1, 128), (128, 128)), out_dtype=None
):
    """Compatibility name for gemm_block_scaled; defaults preserve the old recipe.

    Other dtypes, scale blocks and output dtypes use the general native path.
    """
    from .gemm_block_scaled import gemm_block_scaled
    return gemm_block_scaled(a, b, a_scale, b_scale, scale_recipe=scale_recipe, out_dtype=out_dtype)
