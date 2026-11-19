# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Gluon backend for dense FP16/BF16 GEMM."""

from functools import partial
from importlib import import_module
import sys

import torch

from boltops.utility.arch_info import DeviceInfo
from ..backend_interfaces import A16W16Backend
from ..config import get_gemm_config


# Resolve kernels only after the backend accepts a call. Existing config and
# benchmark code can still look up the original symbols on this module.
_KERNEL_MODULES = {
    "_gemm_a16w16_bf16_tn_bm16_bn16_bk128_kernel": ".gemm_a16w16_mls",
    "_gemm_a16w16_bf16_tt_bm16_bn32_bk128_kernel": ".gemm_a16w16_mls",
    "_gemm_a16w16_tt_kernel": ".gemm_a16w16_mls",
    "_gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel": ".gemm_a16w16_persistent",
    "_gemm_a16w16_bf16_tt_bm256_bn128_bk64_persistent_kernel": ".gemm_a16w16_streamk",
    "_gemm_a16w16_bf16_tt_prepare_edges_kernel": ".gemm_a16w16_streamk",
}


def __getattr__(name):
    module = _KERNEL_MODULES.get(name)
    if module is None:
        raise AttributeError(name)
    kernel = getattr(import_module(module, __package__), name)
    globals()[name] = kernel
    return kernel


# Host launch policies. Regular tile kernels use the default launch path.
def _launch_tiles(kernel, a, b, out, *, cu_count, config, layout):
    import triton

    m, k = a.shape
    n = b.shape[0]

    def grid(meta):
        tiles = triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"])
        return (min(meta.get("NUM_SMS", tiles), tiles),)

    config["B_K_CONTIGUOUS"] = layout == "TN"
    options = {key: tuple(value) if isinstance(value, list) else value
               for key, value in config.items()}
    kernel[grid](a, b, out, M=m, N=n, K=k, **options)


def _launch_persistent(kernel, a, b, out, *, cu_count, config, layout):
    if a.dtype != out.dtype:
        raise ValueError("persistent Gluon requires matching FP16 or BF16 input/output")
    # Launch grid and persistent stride both use the input device's CU count.
    config["NUM_SMS"] = cu_count
    _launch_tiles(kernel, a, b, out, cu_count=cu_count, config=config, layout=layout)


def _launch_streamk(kernel, a, b, out, *, cu_count, config, layout):
    from .gemm_a16w16_streamk import _launch_gemm_a16w16_streamk

    _launch_gemm_a16w16_streamk(
        kernel, a, b, out, cu_count=cu_count, config=config, layout=layout,
    )


_SPECIAL_LAUNCHERS = {
    "_gemm_a16w16_tn_bm256_bn256_bk64_persistent_kernel": _launch_persistent,
    "_gemm_a16w16_bf16_tt_bm256_bn128_bk64_persistent_kernel": _launch_streamk,
}


def _supports_fp16_config(config):
    """Return whether the BF16 entry can reuse its geometry for FP16."""
    selector = config.get("kernel")
    return isinstance(selector, str) and _SPECIAL_LAUNCHERS.get(selector) is _launch_persistent


def _launch_configured(a, b, out, *, config, cu_count, layout):
    """Resolve the configured entry and invoke its launch policy."""
    selector = config.pop("kernel", None)
    kernel = getattr(sys.modules[__name__], selector, None) if isinstance(selector, str) else None
    if not hasattr(kernel, "__getitem__") or not callable(kernel.__getitem__):
        raise ValueError("Gluon config requires a valid kernel selector")
    launcher = _SPECIAL_LAUNCHERS.get(selector, _launch_tiles)
    launcher(kernel, a, b, out, cu_count=cu_count, config=config, layout=layout)


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
        if not _supports_fp16_config(config):
            raise KeyError("FP16 requires a persistent Gluon configuration")
        return config, tuned


def resolve_backend(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, *,
    bias: torch.Tensor | None, activation: str | None,
    device: DeviceInfo, automatic: bool,
) -> A16W16Backend | None:
    """Bind launch options for supported inputs. Auto requires a tuned match."""
    m, k = a.shape
    n = b.shape[0]
    if (
        bias is not None or activation is not None
        or a.stride() != (k, 1) or b.stride() not in ((k, 1), (1, n))
    ):
        if automatic:
            return None
        raise NotImplementedError(
            "gemm_a16w16: no Gluon kernel for the requested layout, bias or "
            "activation; requires K-contiguous x and K- or N-contiguous w "
            "without bias or activation"
        )
    layout = "TN" if b.stride() == (k, 1) else "TT"
    profile = (
        f"DTYPE={str(a.dtype).removeprefix('torch.')}-"
        f"OUT={str(out.dtype).removeprefix('torch.')}"
    )
    try:
        config, tuned = _get_gluon_config(
            m, n, k, layout=layout, profile=profile,
            arch=device.arch, cu_num=device.cu_count,
        )
    except (FileNotFoundError, KeyError):
        if automatic:
            return None
        raise NotImplementedError(
            "gemm_a16w16: no Gluon configuration for this precision, shape and device"
        ) from None
    if automatic and not tuned:
        return None
    return partial(gemm_a16w16, config=config)


def gemm_a16w16(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, *,
    bias: torch.Tensor | None, activation: str | None,
    config: dict[str, object],
    device: DeviceInfo,
) -> None:
    """Execute a Gluon call with the configuration resolved by the selector."""
    if not out.is_contiguous():
        raise NotImplementedError("Gluon A16W16 kernels require a contiguous output")
    layout = "TN" if b.stride() == (a.shape[1], 1) else "TT"
    _launch_configured(a, b, out, config=config, cu_count=device.cu_count, layout=layout)
