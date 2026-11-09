# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Interface configuration documents and explicit measured-winner matching."""
import json
import os
from functools import lru_cache
from pathlib import Path

from .descriptors import KernelConfig
from .dispatch import BACKENDS as _BACKENDS

_CONFIG_DIR = Path(__file__).with_name("configs")
_CONFIG_NAMES = ("scaled", "dense", "batched", "grouped", "batched_a16wfp4_online_quant")


def config_path(interface):
    if interface not in _CONFIG_NAMES:
        raise ValueError(f"unknown GEMM configuration interface: {interface}")
    name = f"gemm_{interface}.json"
    override = os.environ.get("BOLTOPS_GEMM_CONFIG_DIR")
    if override:
        path = Path(override) / name
        if path.is_file():
            return path
    return _CONFIG_DIR / name


def _validate_document(data, interface):
    if data["schema_version"] != 4:
        raise ValueError("unsupported GEMM performance-map schema")
    if data["interface"] != f"gemm_{interface}":
        raise ValueError(f"configuration file must belong to gemm_{interface}")
    order = data["default_backends"]
    if not isinstance(order, list) or len(order) != len(_BACKENDS) or set(order) != set(_BACKENDS):
        raise ValueError("default_backends must list triton, gluon and tilelang exactly once")
    if not isinstance(data["entries"], list):
        raise ValueError("GEMM entries must be a list")
    for entry in data["entries"]:
        if type(entry.get("priority", 0)) is not int:
            raise ValueError("GEMM winner priority must be an integer")
        if entry["config"]["backend"] not in _BACKENDS:
            raise ValueError("unknown backend in GEMM winner configuration")
    return data


@lru_cache(maxsize=32)
def _document_at(path, interface):
    return _validate_document(json.loads(Path(path).read_text()), interface)


def document(interface):
    return _document_at(str(config_path(interface).resolve()), interface)


def clear_cache():
    _document_at.cache_clear()


def _entries(interface):
    return document(interface)["entries"]


def default_backends(interface):
    return tuple(document(interface)["default_backends"])


def _scale(s):
    return None if s is None else {"dtype": str(s.dtype).removeprefix("torch."),
                                   "block": list(s.block), "encoding": s.encoding}


def _operand(s):
    return {"format": s.format, "packing": s.packing, "pack_axis": s.pack_axis,
            "values_per_storage": s.values_per_storage,
            "zero_point_dtype": None if s.zero_point_dtype is None else
                str(s.zero_point_dtype).removeprefix("torch.")}


def exact_match(p):
    """Persist semantic/storage metadata, without addresses or measurements."""
    if p.a.strides is None or p.b.strides is None or p.num_cus is None:
        raise ValueError("recording a winner requires operand strides and num_cus")
    return {"kind": p.kind, "arch": p.arch, "num_cus": p.num_cus,
            "dtypes": [str(t).removeprefix("torch.") for t in (p.a_dtype, p.b_dtype, p.out_dtype)],
            "scales": [_scale(p.a_scale), _scale(p.b_scale)],
            "operands": [_operand(p.a), _operand(p.b)],
            "strides": {"a": list(p.a.strides), "b": list(p.b.strides)},
            "batch": p.batch,
            "groups": None if p.groups is None else {"mode": p.groups.mode, "count": p.groups.count},
            "has_bias": p.has_bias, "activation_quantization": p.activation_quantization,
            "aligned_16": [p.a_aligned_16, p.b_aligned_16],
            "shape_contract": {"m": p.m, "n": p.n, "k": p.k}}


def _matches(p, match):
    if min(p.m, p.n, p.k) <= 0 or p.kind != match.get("kind", "dense"):
        return False
    if p.arch != match["arch"] or p.num_cus != match["num_cus"]:
        return False
    if p.has_bias != match.get("has_bias", False) or p.batch != match.get("batch", 1):
        return False
    groups = None if p.groups is None else {"mode": p.groups.mode, "count": p.groups.count}
    if groups != match.get("groups") or p.activation_quantization != match.get("activation_quantization"):
        return False
    if [str(t).removeprefix("torch.") for t in (p.a_dtype, p.b_dtype, p.out_dtype)] != match["dtypes"]:
        return False
    scales = match["scales"] or [None, None]
    if [_scale(p.a_scale), _scale(p.b_scale)] != scales:
        return False
    operands = [_operand(p.a), _operand(p.b)]
    if "operands" in match:
        if operands != match["operands"]:
            return False
    elif not p.native_operands:
        return False
    if "strides" in match:
        if p.a.strides != tuple(match["strides"]["a"]) or p.b.strides != tuple(match["strides"]["b"]):
            return False
    else:
        b_layout = match.get("b_layout", "k_contiguous")
        if b_layout not in ("k_contiguous", "n_contiguous"):
            raise ValueError(f"unknown GEMM B layout: {b_layout}")
        expected_b = (p.k, 1) if b_layout == "k_contiguous" else (1, p.n)
        if p.a.strides != (p.k, 1) or p.b.strides != expected_b:
            return False
    if "aligned_16" in match:
        if [p.a_aligned_16, p.b_aligned_16] != match["aligned_16"]:
            return False
    elif not p.a_aligned_16 or not p.b_aligned_16:
        return False
    c = match["shape_contract"]
    if any(getattr(p, d) != c[d] for d in ("m", "n", "k") if d in c):
        return False
    return not (p.m % c.get("m_multiple", 1) or p.n % c.get("n_multiple", 1) or
                p.k % c.get("k_multiple", 1) or
                not c.get("k_min", 0) <= p.k <= c.get("k_max", p.k))


def matching_entries(problem, *, backend="auto"):
    return tuple(e for e in _entries(problem.interface)
                 if (backend == "auto" or e["config"]["backend"] == backend)
                 and _matches(problem, e["match"]))


def entry_config(entry):
    c = entry["config"]
    return KernelConfig(*c["block_size"], source_config=c.get("source_config"))


def tuned_configs(problem, implementation, *, backend=None):
    """Capabilities may query records; winner resolution belongs to dispatch."""
    return tuple(entry_config(e) for e in matching_entries(problem, backend=backend or "auto")
                 if e["config"]["implementation"] == implementation)
