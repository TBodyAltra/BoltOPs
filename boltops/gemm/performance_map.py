"""Read measured configuration contracts; never infer support from timing alone."""
import json
from functools import lru_cache
from pathlib import Path

from .registry import KernelConfig


@lru_cache(maxsize=1)
def _entries():
    path = Path(__file__).with_name("configs") / "gemm_dispatch.json"
    data = json.loads(path.read_text())
    if data["schema_version"] != 3:
        raise ValueError("unsupported GEMM performance-map schema")
    return data["entries"]


def tuned_configs(problem, implementation, *, backend=None):
    p = problem
    if (p.kind != "dense" or not p.native_operands or
            min(p.m, p.n, p.k) <= 0):
        return ()
    configs = []
    for entry in _entries():
        if backend is not None and entry["config"]["backend"] != backend:
            continue
        match = entry["match"]
        if p.arch != match["arch"] or p.num_cus != match["num_cus"]:
            continue
        if p.a.strides != (p.k, 1):
            continue
        b_layout = match.get("b_layout", "k_contiguous")
        expected_b = (p.k, 1) if b_layout == "k_contiguous" else (1, p.n)
        if p.b.strides != expected_b:
            continue
        dtype_names = [str(t).removeprefix("torch.") for t in (p.a_dtype, p.b_dtype, p.out_dtype)]
        if dtype_names != match["dtypes"]:
            continue
        scales = [{"dtype": str(s.dtype).removeprefix("torch."),
                   "block": list(s.block), "encoding": s.encoding}
                  for s in (p.a_scale, p.b_scale) if s is not None]
        if scales != match["scales"]:
            continue
        c = match["shape_contract"]
        if any(getattr(p, dim) != c[dim] for dim in ("m", "n", "k") if dim in c):
            continue
        if (p.m % c.get("m_multiple", 1) or p.n % c.get("n_multiple", 1) or
                p.k % c.get("k_multiple", 1) or not c.get("k_min", 0) <= p.k <= c.get("k_max", p.k) or
                not p.a_aligned_16 or not p.b_aligned_16):
            continue
        # Tile geometry is a selected implementation parameter, never a
        # property used to match the caller's GEMM problem.
        config = entry["config"]
        if config["implementation"] == implementation:
            configs.append(KernelConfig(*config["block_size"], source_config=config.get("source_config")))
    return tuple(configs)
