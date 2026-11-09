# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Interface configuration documents and explicit measured-winner matching."""
import json
import math
import os
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import get_args, get_type_hints

import torch

from .descriptors import GemmProblem, KernelConfig
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
    if data["schema_version"] != 7:
        raise ValueError("unsupported GEMM performance-map schema")
    if data["interface"] != f"gemm_{interface}":
        raise ValueError(f"configuration file must belong to gemm_{interface}")
    if set(data) != {"schema_version", "interface", "entries"}:
        raise ValueError("GEMM document requires schema_version, interface and entries only")
    if not isinstance(data["entries"], list):
        raise ValueError("GEMM entries must be a list")
    for entry in data["entries"]:
        if set(entry) != {"match", "config"}:
            raise ValueError("GEMM entry requires match and config only")
        if set(entry["config"]) != {"backend", "implementation", "params"}:
            raise ValueError("configuration requires backend, implementation and params only")
        entry_config(entry)
        _validate_condition(entry["match"], _PROBLEM_SCHEMA)
        if not {"arch", "num_cus"}.issubset(entry["match"]):
            raise ValueError("GEMM match requires explicit arch and num_cus")
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


def _json_value(value):
    """Expose dataclass fields using JSON values, including future fields."""
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return value


def problem_attributes(problem):
    return _json_value(problem)


def _schema(cls):
    result = {}
    for name, annotation in get_type_hints(cls).items():
        args = get_args(annotation)
        if type(None) in args:
            annotation = next(t for t in args if t is not type(None))
        result[name] = _schema(annotation) if is_dataclass(annotation) else annotation
    return result


_PROBLEM_SCHEMA = _schema(GemmProblem)
_COMPARISONS = {"min", "max", "multiple_of"}


def _validate_condition(condition, schema, path="match"):
    if isinstance(schema, dict):
        if condition is None and path != "match":
            return
        if not isinstance(condition, dict):
            raise ValueError(f"{path} must be an object or null")
        for key, value in condition.items():
            if key not in schema:
                raise ValueError(f"unknown GEMM match field: {path}.{key}")
            _validate_condition(value, schema[key], f"{path}.{key}")
    elif isinstance(condition, dict):
        if schema not in (int, float) or not condition:
            raise ValueError(f"{path} requires a literal value or numeric comparisons")
        for op, value in condition.items():
            if op not in _COMPARISONS:
                raise ValueError(f"unknown GEMM comparison: {path}.{op}")
            if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
                raise ValueError(f"{path}.{op} requires a finite number")
            if op == "multiple_of" and (type(value) is not int or value <= 0):
                raise ValueError(f"{path}.multiple_of requires a positive integer")
        if condition.get("min", -math.inf) > condition.get("max", math.inf):
            raise ValueError(f"{path} has min greater than max")


def _compare(actual, condition):
    if isinstance(condition, dict):
        if isinstance(actual, dict):
            return all(_compare(actual[key], value) for key, value in condition.items())
        if type(actual) not in (int, float):
            return False
        return all(actual >= value if op == "min" else
                   actual <= value if op == "max" else actual % value == 0
                   for op, value in condition.items())
    # JSON booleans are not numeric equality conditions (True != 1 here).
    if isinstance(actual, bool) or isinstance(condition, bool):
        return type(actual) is type(condition) and actual == condition
    return actual == condition


def exact_match(p):
    """Record interface-local semantic/storage attributes, without measurements."""
    if p.a.strides is None or p.b.strides is None or p.num_cus is None:
        raise ValueError("recording a winner requires operand strides and num_cus")
    match = problem_attributes(p)
    # Interface identity already fixes kind and online-quantization semantics.
    match.pop("kind")
    match.pop("activation_quantization")
    match.pop("a_aligned_16")
    match.pop("b_aligned_16")
    if p.kind != "batched":
        match.pop("batch")
    if p.kind != "grouped":
        match.pop("groups")
    if p.interface == "dense":
        # The dense entry accepts native, unscaled A/B with the same dtype.
        match.pop("a_scale")
        match.pop("b_scale")
        for name in ("a", "b"):
            for key in ("format", "packing", "pack_axis", "values_per_storage", "zero_point_dtype"):
                match[name].pop(key)
            match[name].pop("storage_dtype")
    else:
        match.pop("input_dtype")
    # Exact strides already encode the derived contiguity labels.
    match.pop("a_layout")
    match.pop("b_layout")
    return match


def _matches(p, match):
    # Validate the entire rule before comparing: a mismatching arch must not
    # hide an unknown field or malformed comparison later in the same rule.
    _validate_condition(match, _PROBLEM_SCHEMA)
    return min(p.m, p.n, p.k) > 0 and _compare(problem_attributes(p), match)


def matching_entries(problem, *, backend="auto"):
    return tuple(e for e in _entries(problem.interface)
                 if (backend == "auto" or e["config"]["backend"] == backend)
                 and _matches(problem, e["match"]))


def entry_config(entry):
    c = entry["config"]
    return KernelConfig.from_params(c["params"])


def tuned_configs(problem, implementation, *, backend=None):
    """Capabilities may query records; winner resolution belongs to dispatch."""
    return tuple(entry_config(e) for e in matching_entries(problem, backend=backend or "auto")
                 if e["config"]["implementation"] == implementation)
