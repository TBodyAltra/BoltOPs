"""The map selects tuned Gluon only; all other scale contracts use Triton."""
from dataclasses import replace

import pytest
import torch

from boltops.gemm import performance_map
from boltops.gemm.registry import select_plan
from test_block_scaled_dispatch import problem


def test_empty_performance_map_uses_triton(monkeypatch):
    monkeypatch.setattr(performance_map, "_entries", lambda: [])
    assert select_plan(problem()).implementation.backend == "triton"


@pytest.mark.parametrize("changes", [
    {"arch": "gfx90a"}, {"out_dtype": torch.float32},
    {"a_aligned_16": False}, {"m": 127}, {"n": 255}, {"k": 6400},
])
def test_unmeasured_contracts_never_select_gluon(changes):
    base = problem()
    if "k" in changes:
        changes = dict(changes, a=replace(base.a, strides=(changes["k"], 1)),
                       b=replace(base.b, strides=(changes["k"], 1)))
    assert select_plan(replace(base, **changes)).implementation.backend == "triton"


def test_bad_map_tile_cannot_reconfigure_fixed_gluon_pipeline(monkeypatch):
    import copy
    entries = copy.deepcopy(performance_map._entries())
    entries[0]["block_size"] = [256, 256, 64]
    monkeypatch.setattr(performance_map, "_entries", lambda: entries)
    assert select_plan(problem()).implementation.backend == "triton"
