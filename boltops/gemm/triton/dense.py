"""Adapters reuse the existing dense kernels and gfx946 FP16 recipes."""
import torch

from ..performance_map import tuned_configs
from ..descriptors import KernelConfig

_TUNED = "triton.dense.fp16"


def _native_dense(p):
    return p.kind == "dense" and p.native_operands and p.unscaled and p.a_dtype == p.b_dtype


def _tuned_configurations(p):
    return tuned_configs(p, _TUNED, backend="triton") if _native_dense(p) else ()


def _general_config(p):
    if p.a_dtype in (torch.float16, torch.bfloat16):
        from .gemm_a16w16_op import _get_config
        from .gemm_config_utils import compute_splitk_params
        cfg = dict(_get_config(p.m, p.n, p.k)[0])
        cfg["NUM_KSPLIT"] = 1
        return compute_splitk_params(cfg, p.k)
    from .matmul_persistent import _launch_config
    return dict(_launch_config(p.a_dtype))


def _general_configurations(p):
    if not _native_dense(p) or p.a_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return ()
    if p.out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return ()
    if p.a_dtype == torch.float32 and (p.out_dtype != torch.float32 or not p.num_cus):
        return ()
    cfg = _general_config(p)
    return (KernelConfig.from_params(cfg),)


def launch_dense(config, problem, a, b, bias, out):
    if a.dtype in (torch.float16, torch.bfloat16):
        from .gemm_a16w16 import gemm_a16w16
        gemm_a16w16(a, b, bias, dtype=out.dtype, y=out, config=config.params)
    else:
        from .matmul_persistent import matmul_kernel_persistent
        cfg = config.params
        grid = (min(problem.num_cus, ((problem.m + config.block_m - 1) // config.block_m) * ((problem.n + config.block_n - 1) // config.block_n)),)
        matmul_kernel_persistent[grid](
            a, b.T, out, bias, problem.m, problem.n, problem.k,
            *a.stride(), *b.T.stride(), *out.stride(), NUM_SMS=problem.num_cus,
            A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31,
            C_LARGE=out.numel() > 2**31, HAS_BIAS=bias is not None, **cfg)


def launch_dense_fp16(config, problem, a, b, bias, out):
    from .gemm_fp16 import gemm_fp16
    params = config.params
    flags = {key: params.pop(key) for key in ("wasp", "wdra", "use_mls", "use_matrix_store")}
    launch = {key: tuple(value) if isinstance(value, list) else value
              for key, value in params.items()}
    gemm_fp16(a, b.T, bias, out=out, autotune=False, config=launch, **flags)
