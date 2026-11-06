"""Read measured configuration contracts; never infer support from timing alone."""
import json
from functools import lru_cache
from pathlib import Path

from .registry import KernelConfig


@lru_cache(maxsize=1)
def _entries():
    path = Path(__file__).with_name("configs") / "block_scaled_dispatch.json"
    data = json.loads(path.read_text())
    if data["schema_version"] != 1:
        raise ValueError("unsupported block-scaled performance-map schema")
    return data["entries"]


def tuned_configs(problem, implementation):
    p = problem
    if (p.kind != "dense" or not p.native_operands or
            p.a_scale is None or p.b_scale is None or min(p.m, p.n, p.k) <= 0):
        return ()
    if p.a.strides != (p.k, 1) or p.b.strides != (p.k, 1):
        return ()
    configs = []
    for entry in _entries():
        if entry["implementation"] != implementation or p.arch != entry["arch"]:
            continue
        dtype_names = [str(t).removeprefix("torch.") for t in (p.a_dtype, p.b_dtype, p.out_dtype)]
        if dtype_names != entry["dtypes"]:
            continue
        scales = [{"dtype": str(s.dtype).removeprefix("torch."),
                   "block": list(s.block), "encoding": s.encoding}
                  for s in (p.a_scale, p.b_scale)]
        if scales != entry["scales"]:
            continue
        c = entry["shape_contract"]
        if (p.m % c["m_multiple"] or p.n % c["n_multiple"] or
                p.k % c["k_multiple"] or not c["k_min"] <= p.k <= c["k_max"] or
                not p.a_aligned_16 or not p.b_aligned_16):
            continue
        configs.append(KernelConfig(*entry["block_size"]))
    return tuple(configs)
