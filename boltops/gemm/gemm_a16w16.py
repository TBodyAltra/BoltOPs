# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""FP16/BF16 dense GEMM with backend selection."""

import os

import torch
from boltops.utility.arch_info import get_device_info
from .config import get_gemm_config, validate_backend


_BF16_MLS_PROFILE = "DTYPE=bfloat16-OUT=bfloat16-WASP-MLS-EXPERIMENTAL"
_BF16_MLS_REQUIRED_ENV = {
    "AMDGCN_USE_BUFFER_OPS": "1",
    "TRITON_BUFFER_CACHE_SWIZZLE": "1",
    "TRITON_MLS_B_INTERLEAVE": "2",
}
# Interleave is not a launch option (or a cache-key component on the archived
# compiler). Require process-start configuration, never switch it during calls.
_BF16_MLS_IMPORT_ENV = {key: os.environ.get(key) for key in _BF16_MLS_REQUIRED_ENV}


def _validate_bf16_mls_environment(config):
    for key, expected in _BF16_MLS_REQUIRED_ENV.items():
        if os.environ.get(key) != expected or _BF16_MLS_IMPORT_ENV[key] != expected:
            raise ValueError(
                f"BF16 MLS requires {key}={expected} before importing boltops; "
                "start a fresh process with the recorded environment and a compatible "
                "Triton cache; changing compiler environment during calls is unsupported"
            )
    if config.get("buffer_cache_swizzle") is not True:
        raise ValueError("BF16 MLS requires buffer_cache_swizzle=True")
    # Python knob overrides can disagree with os.environ. Reject those rather
    # than silently compiling a different pipeline or mutating global knobs.
    from triton import knobs

    if not knobs.amd.use_buffer_ops or not knobs.amd.buffer_cache_swizzle:
        raise ValueError("BF16 MLS requires active Triton buffer-ops and BCS knobs")


def _get_gluon_config(m, n, k, *, layout, profile, arch, cu_num):
    """Share same-precision persistent launch geometry between FP16 and BF16."""
    try:
        return get_gemm_config(
            "GEMM-A16W16", m, n, k, backend="gluon", layout=layout,
            profile=profile, arch=arch, cu_num=cu_num,
        )
    except (FileNotFoundError, KeyError):
        if profile != "DTYPE=float16-OUT=float16":
            raise
        config, tuned = get_gemm_config(
            "GEMM-A16W16", m, n, k, backend="gluon", layout=layout,
            profile="DTYPE=bfloat16-OUT=bfloat16", arch=arch, cu_num=cu_num,
        )
        if config.get("kernel") != "_gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel":
            raise KeyError("FP16 requires a persistent Gluon configuration")
        return config, tuned


def gemm_a16w16(
    x,
    w,
    bias=None,
    dtype=torch.bfloat16,
    out=None,
    config=None,
    activation=None,
    backend=None,
):
    """Compute dense FP16/BF16 GEMM, optional bias and fused activation.

    Args:
        x (torch.Tensor): GPU activation [M,K], float16 or bfloat16. The general Triton kernel
            accepts positive-stride views: row-major (K,1) makes K contiguous;
            column-major (1,M) makes M contiguous. Padded positive strides are
            also allowed.
        w (torch.Tensor): GPU weights [N,K], same dtype/device as x. Row-major (K,1)
            makes K contiguous; column-major (1,N) makes N contiguous. The general
            Triton kernel accepts positive-stride views. The recorded MLS/WDRA
            pipeline specifically requires x.stride()=(K,1) and w.stride()=(1,N).
            The logical transpose w.T is internal.
        bias (torch.Tensor | None): Optional [N] bias in x.dtype on the same device,
            with unit stride. Broadcast across M.
        dtype (torch.dtype | None): Output float16, bfloat16 or float32. Defaults to
            bfloat16; None uses x.dtype.
        out (torch.Tensor | None): Optional contiguous output [M,N], on the input
            device with the selected output dtype. Reused and returned directly;
            None allocates a new tensor. Inputs and out must not overlap.
        config (dict[str, object] | None): Complete backend kernel launch options, such
            as
            BLOCK_SIZE_M/N/K, num_warps and num_stages. Overrides recorded/default
            configuration lookup and is copied before use; the caller dict is unchanged.
            None selects cached configuration files for the device and shape. No tuning
            is performed during an ordinary call.
        activation (str | None): Optional "gelu", "gelu_tanh", "silu",
            "silu_exp2", "relu" or "relu6", applied to FP32 results after bias
            and the complete K reduction, before output conversion. None is
            identity. Explicit MLS/WDRA configurations do not support activation.
        backend (str | None): "auto", "triton", "gluon" or "tilelang"; None means
            "auto".
            Auto selects recorded Gluon for matching same-precision BF16/FP16
            TN/TT inputs on gfx946 with a tuned CU-specific config and no explicit
            config; otherwise Triton. Explicit selection never retries another
            backend. Gluon supports the fixed BF16 MNK=(64,64,2048) TN entry and
            a 48-CU persistent FP16/BF16 TN/TT entry with automatic tail handling,
            K-contiguous x, matching input/output dtype, no bias or activation.
            FP16 can reuse the persistent BF16 launch geometry. TileLang is not
            implemented for nonempty computation. After Gluon selection, exact experimental BF16
            TT profiles may select WASP/MLS on gfx946/48 CU without bias or activation.
            BF16 MLS requires complete M/N/K tiles, BF16 output, and the recorded
            compiler environment set before importing boltops:
            AMDGCN_USE_BUFFER_OPS=1, TRITON_BUFFER_CACHE_SWIZZLE=1,
            TRITON_MLS_B_INTERLEAVE=2. Keep interleave fixed for the process and
            use a Triton cache compiled with that setting.

    Returns:
        torch.Tensor: Contiguous [M,N] result in dtype.
        Returns the exact out object when supplied. All GPU tensors must
        be on the same device.

    Raises:
        ValueError: Incompatible tensor shapes, devices, layouts or configuration.
        TypeError: Unsupported input, scale or output types.
        NotImplementedError: Requested backend or operation option is unavailable.
    """
    a, b, out_dtype = (x, w, x.dtype if dtype is None else dtype)
    user_config = config
    backend = "auto" if backend is None else backend
    validate_backend(backend)
    activation_fn = ""
    if activation is not None:
        from .triton.activation import _get_activation_from_str

        try:
            activation_fn = _get_activation_from_str(activation)
        except (KeyError, TypeError):
            raise ValueError(f"unsupported activation: {activation!r}") from None
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("dense expects A[M,K] and B[N,K] with matching K")
    if a.dtype != b.dtype:
        raise TypeError("dense A/B must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("gemm_a16w16 received an unsupported input dtype")
    m, k = a.shape
    n = b.shape[0]
    if bias is not None and (bias.shape != (n,) or bias.dtype != a.dtype):
        raise ValueError("bias must have shape [N] and the input dtype")
    for tensor in (a, b, *(() if bias is None else (bias,))):
        if not tensor.is_cuda or tensor.device != a.device:
            raise ValueError("all tensors must be on the same GPU")
        if tensor.numel() and any((s <= 0 for s in tensor.stride())):
            raise ValueError("dense tensors must have positive strides")
    if bias is not None and bias.stride(-1) != 1:
        raise ValueError("bias must have unit stride along N")
    out_dtype = a.dtype if out_dtype is None else out_dtype
    if out_dtype is not None and out_dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError("gemm_a16w16 received an unsupported output dtype")
    with torch.cuda.device(a.device):
        out = (
            out
            if out is not None
            else torch.empty((m, n), dtype=out_dtype, device=a.device)
        )
        if (
            tuple(out.shape) != (m, n)
            or out.dtype != out_dtype
            or out.device != a.device
            or (not out.is_contiguous())
        ):
            raise ValueError(
                "preallocated output has incompatible shape, dtype, device or layout"
            )
        if not m or not n:
            return out
        if not k:
            out.zero_() if bias is None else out.copy_(bias.expand(m, n))
            if activation is not None:
                import triton
                from .triton.activation import _apply_activation_kernel

                _apply_activation_kernel[(triton.cdiv(out.numel(), 256),)](
                    out, out.numel(), activation_fn, BLOCK_SIZE=256
                )
            return out
        device = get_device_info(a.device)
        gluon_layout = "TN" if b.stride() == (k, 1) else "TT"
        gluon_inputs_supported = (
            device.arch == "gfx946"
            and bias is None
            and activation is None
            and a.stride() == (k, 1)
            and b.stride() in ((k, 1), (1, n))
        )
        gluon_profile = (
            f"DTYPE={str(a.dtype).removeprefix('torch.')}-"
            f"OUT={str(out_dtype).removeprefix('torch.')}"
        )
        gluon_config = None
        if backend == "auto":
            backend = "triton"
            # Existing explicit Triton configs keep their established backend.
            if user_config is None and gluon_inputs_supported:
                try:
                    gluon_config, tuned = _get_gluon_config(
                        m, n, k, layout=gluon_layout,
                        profile=gluon_profile, arch=device.arch, cu_num=device.cu_count,
                    )
                except (FileNotFoundError, KeyError):
                    pass
                else:
                    if tuned:
                        backend = "gluon"
        if backend == "triton":
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            else:
                use_tuned_shape = (
                    a.dtype == out_dtype == torch.float16
                    and bias is None
                    and activation is None
                    and (a.stride() == (k, 1))
                    and (b.stride() == (1, n))
                )
                config = None
                if (
                    device.arch == "gfx946"
                    and device.cu_count == 48
                    and a.dtype == out_dtype == torch.bfloat16
                    and bias is None
                    and activation is None
                    and a.stride() == (k, 1)
                    and b.stride() == (1, n)
                ):
                    try:
                        candidate, tuned = get_gemm_config(
                            "GEMM-A16W16", m, n, k, backend="triton", layout="TT",
                            profile=_BF16_MLS_PROFILE, arch=device.arch,
                            cu_num=device.cu_count,
                        )
                    except (FileNotFoundError, KeyError):
                        pass
                    else:
                        if tuned:
                            config = candidate
                if config is None:
                    config, _ = get_gemm_config(
                        "GEMM-A16W16",
                        m,
                        n if use_tuned_shape else None,
                        k if use_tuned_shape else None,
                        backend=backend,
                        arch=device.arch,
                        cu_num=device.cu_count,
                    )
            import triton

            if "use_mls" in config:
                from .triton import gemm_a16w16_wasp as kernels

                bf16_mls = a.dtype == out_dtype == torch.bfloat16
                if (
                    device.arch != "gfx946"
                    or a.dtype != out_dtype
                    or (a.dtype != torch.float16 and not bf16_mls)
                    or (bf16_mls and (device.cu_count != 48 or not config["use_mls"]))
                    or (bias is not None)
                    or activation is not None
                    or (a.stride() != (k, 1))
                    or (b.stride() != (1, n))
                ):
                    raise ValueError(
                        "recorded pipeline requires gfx946, same-precision FP16 or "
                        "BF16 output and TT operands without bias or activation; "
                        "BF16 requires MLS and 48 CU"
                    )
                if bf16_mls:
                    for dim, size in zip("MNK", (m, n, k)):
                        block = config.get(f"BLOCK_SIZE_{dim}")
                        if type(block) is not int or block <= 0 or size % block:
                            raise ValueError(
                                "BF16 MLS requires M/N/K divisible by positive "
                                "BLOCK_SIZE_M/N/K (unmasked matrix-load tails are unsafe)"
                            )
                    _validate_bf16_mls_environment(config)
                wasp, wdra, use_mls, use_matrix_store = (
                    config.pop(key)
                    for key in ("wasp", "wdra", "use_mls", "use_matrix_store")
                )
                if wdra and (not wasp) or (use_matrix_store and (not use_mls)):
                    raise ValueError("invalid WDRA/MLS configuration")
                if (wasp or wdra) and (not kernels.supports_wasp_wdra(device.arch)):
                    raise ValueError("target does not support WASP/WDRA")
                if use_mls and (not kernels.arch_info.is_mls_avail()):
                    raise ValueError("target does not support MLS")
                load_warps, mma_warps = kernels._resolve_wasp_topology(
                    wasp=wasp, wdra=wdra, wasp_num_load_warps=None, wasp_num_mma_warps=None
                )
                kernels._mode_launch_kwargs(
                    wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
                )
                os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")
                config = {
                    key: tuple(value) if isinstance(value, list) else value
                    for (key, value) in config.items()
                }
                config["HAS_BIAS"] = False
                if use_mls:
                    config["USE_MATRIX_STORE"] = bool(use_matrix_store)
                kernel = (
                    kernels._gemm_fp16_mls_kernel
                    if use_mls
                    else kernels._gemm_fp16_load_kernel
                )
                b = b.T
                grid = lambda meta: (
                    triton.cdiv(m, meta["BLOCK_SIZE_M"])
                    * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
                )
                kernel[grid](
                    a, b, out, b, m, n, k, *a.stride(), *b.stride(), *out.stride(), **config
                )
            else:
                from .triton.gemm_a16w16 import _gemm_a16_w16_kernel
                from .triton.splitk_reduce import _gemm_splitk_reduce_kernel
                from .triton.gemm_config_utils import compute_splitk_params

                if user_config is None:
                    config.setdefault("NUM_KSPLIT", 1)
                    config = compute_splitk_params(config, k)
                M, N, K = (m, n, k)
                w_t = b.T
                if config["NUM_KSPLIT"] > 1:
                    y_pp = torch.empty(
                        (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=out.device
                    )
                else:
                    y_pp = None
                grid = lambda META: (
                    META["NUM_KSPLIT"]
                    * triton.cdiv(M, META["BLOCK_SIZE_M"])
                    * triton.cdiv(N, META["BLOCK_SIZE_N"]),
                )
                _gemm_a16_w16_kernel[grid](
                    a,
                    w_t,
                    bias,
                    out if config["NUM_KSPLIT"] == 1 else y_pp,
                    M,
                    N,
                    K,
                    a.stride(0),
                    a.stride(1),
                    w_t.stride(0),
                    w_t.stride(1),
                    0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
                    out.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
                    out.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
                    activation=activation_fn,
                    use_activation=activation is not None,
                    ADD_BIAS=bias is not None,
                    SKIP_REDUCE=False,
                    **config,
                )
                if config["NUM_KSPLIT"] > 1:
                    REDUCE_BLOCK_SIZE_M = 32
                    REDUCE_BLOCK_SIZE_N = 32
                    ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])
                    grid_reduce = (
                        triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
                        triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
                    )
                    _gemm_splitk_reduce_kernel[grid_reduce](
                        y_pp,
                        out,
                        bias,
                        M,
                        N,
                        y_pp.stride(0),
                        y_pp.stride(1),
                        y_pp.stride(2),
                        out.stride(0),
                        out.stride(1),
                        REDUCE_BLOCK_SIZE_M,
                        REDUCE_BLOCK_SIZE_N,
                        ACTUAL_KSPLIT,
                        triton.next_power_of_2(config["NUM_KSPLIT"]),
                        ADD_BIAS=bias is not None,
                        activation=activation_fn,
                        use_activation=activation is not None,
                        KERNEL_NAME="_gemm_a16w16_reduce_kernel",
                    )
                return out
        elif backend == "gluon":
            if not gluon_inputs_supported:
                raise NotImplementedError(
                    "gemm_a16w16: no Gluon kernel for the requested layout, bias or "
                    "activation; requires gfx946 K-contiguous x and K- or N-contiguous w "
                    "without bias or activation"
                )
            if user_config is not None:
                from copy import deepcopy

                config = deepcopy(user_config)
            elif gluon_config is not None:
                config = gluon_config
            else:
                try:
                    config, _ = _get_gluon_config(
                        m, n, k, layout=gluon_layout,
                        profile=gluon_profile, arch=device.arch, cu_num=device.cu_count,
                    )
                except (FileNotFoundError, KeyError):
                    raise NotImplementedError(
                        "gemm_a16w16: no Gluon configuration for this precision, shape and device"
                    ) from None
            import triton
            from .gluon import gemm_a16w16 as kernels

            selector = config.pop("kernel", None)
            if not isinstance(selector, str) or not hasattr(kernels, selector):
                raise ValueError("Gluon config requires a valid kernel selector")
            kernel = getattr(kernels, selector)
            if selector == "_gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel":
                # Physical weight strides, rather than a caller's config, own
                # the B MLS descriptor and shared-memory order.
                if a.dtype != out_dtype or device.cu_count != 48:
                    raise ValueError(
                        "persistent Gluon requires 48 CU and matching FP16 or BF16 input/output"
                    )
                config["B_K_CONTIGUOUS"] = gluon_layout == "TN"
            elif gluon_layout != "TN":
                raise NotImplementedError(
                    "gemm_a16w16: no Gluon TT kernel for the requested selector"
                )
            config = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in config.items()
            }
            # Match the kernel's active persistent stride, including tiny grids.
            grid = (
                lambda meta: (min(meta["NUM_SMS"],
                                  triton.cdiv(m, meta["BLOCK_SIZE_M"])
                                  * triton.cdiv(n, meta["BLOCK_SIZE_N"])),)
                if "NUM_SMS" in meta
                else (
                    triton.cdiv(m, meta["BLOCK_SIZE_M"])
                    * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
                )
            )
            kernel[grid](a, b, out, M=m, N=n, K=k, **config)
            return out
        elif backend == "tilelang":
            raise NotImplementedError("gemm_a16w16: no TileLang kernel is implemented")
    return out
