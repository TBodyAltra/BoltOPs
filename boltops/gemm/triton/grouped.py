"""Reuse equal 16-bit batched GEMM and existing equal/ragged FP8 x FP4 GEMM."""
import torch

from ..descriptors import KernelConfig


def _native(p):
    return (p.kind == 'grouped' and p.groups.mode == 'equal' and p.native_operands and p.unscaled and
            p.a_dtype == p.b_dtype and p.a_dtype in (torch.float16, torch.bfloat16) and
            p.out_dtype in (torch.float16, torch.bfloat16))


def _native_config(p):
    if not _native(p):
        return ()
    if p.b.strides is None:
        return ()
    layout = 'tn' if p.b.strides[-1] == 1 else 'tt' if p.b.strides[-2] == 1 else None
    if layout is None:
        return ()
    from .batched_gemm_bf16 import _get_config
    cfg = _get_config(p.m // p.groups.count, p.n, p.k, p.out_dtype, B=p.groups.count, layout=layout)
    return (KernelConfig(*(cfg['BLOCK_SIZE_' + d] for d in ('M', 'N', 'K'))),)


def _fp4_types(p):
    scales = (p.a_scale, p.b_scale)
    return (p.a_dtype == torch.float8_e4m3fn and p.b_dtype == torch.uint8 and
            p.a.format == 'native' and p.b.format == 'fp4' and p.b.packing == 'e2m1_low_high' and
            all(s is not None and ((s.encoding == 'e8m0' and s.dtype == torch.uint8) or
                (s.encoding == 'numeric' and s.dtype in (torch.float16, torch.bfloat16, torch.float32))) for s in scales) and
            p.a_scale.block in ((1, 32), (1, 128)) and p.b_scale.block == (1, 32) and
            p.out_dtype in (torch.float16, torch.bfloat16, torch.float32))


def _fp4(p):
    return p.kind == 'grouped' and p.groups.mode in ('equal', 'offsets') and _fp4_types(p)


def launch_grouped_native(config, problem, a, b, a_scale, b_scale, group_offsets, bias, out):
    from .batched_gemm_bf16 import batched_gemm_bf16, _get_config, _infer_weight_layout
    cfg = _get_config(a.shape[1], problem.n, problem.k, out.dtype,
                      B=problem.groups.count, layout=_infer_weight_layout(a, b))
    if tuple(cfg['BLOCK_SIZE_' + d] for d in ('M', 'N', 'K')) != (config.block_m, config.block_n, config.block_k):
        raise ValueError('selected grouped tile disagrees with backend configuration')
    batched_gemm_bf16(a, b, bias, dtype=out.dtype, y=out, config=cfg)


def launch_grouped_fp4(config, problem, a, b, a_scale, b_scale, group_offsets, bias, out):
    from .gemm_a8wfp4 import grouped_gemm_a8wfp4
    grouped_gemm_a8wfp4(a, b, a_scale, b_scale, expert_offsets=group_offsets,
                        activation_group_size=problem.a_scale.block[1], dtype=out.dtype, y=out,
                        config={'BLOCK_SIZE_M': config.block_m, 'BLOCK_SIZE_N': config.block_n, 'num_warps': 4})



def mixed_fp4_configs(p):
    return (KernelConfig(16, 32, 32),) if _fp4(p) else ()
