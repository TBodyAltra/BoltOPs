"""The map selects tuned Gluon only; all other scale contracts use Triton."""
from dataclasses import replace

import pytest
import torch

from boltops.gemm import performance_map
from boltops.gemm.dispatch import KernelConfig, select_plan
from test_block_scaled_dispatch import problem


def test_empty_performance_map_uses_triton(monkeypatch):
    monkeypatch.setattr(performance_map, "_entries", lambda interface: [])
    assert select_plan(problem()).backend == "triton"


@pytest.mark.parametrize("changes", [
    {"arch": "gfx90a"}, {"out_dtype": torch.float32},
    {"num_cus": 96}, {"num_cus": 192}, {"num_cus": None},
    {"a_aligned_16": False}, {"m": 127}, {"n": 255}, {"k": 6400},
])
def test_unmeasured_contracts_never_select_gluon(changes):
    base = problem()
    if "k" in changes:
        changes = dict(changes, a=replace(base.a, strides=(changes["k"], 1)),
                       b=replace(base.b, strides=(changes["k"], 1)))
    if changes.get("a_aligned_16") is False:
        with pytest.raises(ValueError, match="winner is unsupported"):
            select_plan(replace(base, **changes))
    else:
        assert select_plan(replace(base, **changes)).backend == "triton"


def test_tile_is_selected_then_checked_by_implementation(monkeypatch):
    import copy
    entries = copy.deepcopy(performance_map._entries("scaled"))
    matching_conditions = copy.deepcopy(entries[0]["match"])
    entries[0]["config"]["params"].update(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64)
    monkeypatch.setattr(performance_map, "_entries", lambda interface: entries if interface == "scaled" else [])
    assert entries[0]["match"] == matching_conditions
    assert performance_map.tuned_configs(
        problem(), entries[0]["config"]["implementation"]
    ) == (KernelConfig(256, 256, 64, num_warps=4, num_stages=2,
                       wasp_partition_warps=[4, 4, 4, 4], wasp_wdra=True),)
    # A stale measured winner must be diagnosed, not silently replaced.
    with pytest.raises(ValueError, match="recorded GEMM winner is unsupported"):
        select_plan(problem())


def test_gluon_capability_does_not_require_an_existing_winner(monkeypatch):
    from boltops.gemm.dispatch import candidate_plans
    monkeypatch.setattr(performance_map, "_entries", lambda interface: [])
    plans = tuple(candidate_plans(problem(), tuning=True))
    assert {p.backend for p in plans} == {"triton", "gluon"}
    assert select_plan(problem()).backend == "triton"
