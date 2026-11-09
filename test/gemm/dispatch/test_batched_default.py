"""Batched defaults, layout capabilities and direct stride-preserving launch."""
from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest
import torch
from boltops.gemm import dispatch
from boltops.gemm.descriptors import GemmProblem, OperandSpec, KernelConfig
from boltops.gemm import performance_map


def problem(a_layout='k', b_layout='k'):
    return GemmProblem('batched', 3, 5, 32,
        OperandSpec(torch.float16, strides=(96, 32, 1) if a_layout=='k' else (96, 1, 3)),
        OperandSpec(torch.float16, strides=(160, 32, 1) if b_layout=='k' else (160, 1, 5)),
        torch.float16, 'gfx946', batch=2, num_cus=48)


@pytest.mark.parametrize('a_layout', ['k', 'm'])
@pytest.mark.parametrize('b_layout', ['k', 'n'])
def test_layouts_and_fixed_default(a_layout, b_layout):
    p = problem(a_layout, b_layout)
    assert p.a_layout == a_layout+'_contiguous'
    assert p.b_layout == b_layout+'_contiguous'
    plan = dispatch.select_plan(p)
    row = dispatch.IMPLEMENTATIONS[plan.implementation]
    assert 'configs' not in row
    assert plan.config.params == row['default_config']


def test_recorded_parameters_override_default(monkeypatch):
    p = problem('m', 'n')
    params = dict(dispatch.IMPLEMENTATIONS['triton.batched.native16']['default_config'],
                  BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, num_warps=4, num_stages=1,
                  BATCH_LAST_GRID=True)
    record = {'match': performance_map.exact_match(p), 'config': {
        'backend': 'triton', 'implementation': 'triton.batched.native16', 'params': params}}
    monkeypatch.setattr(performance_map, '_entries', lambda interface: (record,))
    assert dispatch.select_plan(p).config.params == params


def test_dtype_and_layout_are_implementation_capabilities():
    p=problem()
    for unsupported in (replace(p, out_dtype=torch.float32),
                        replace(p, a=replace(p.a, strides=(192, 64, 2)))):
        with pytest.raises(NotImplementedError):
            dispatch.select_plan(unsupported)


@pytest.mark.parametrize('batch_last', [False, True])
def test_launcher_preserves_both_operands_and_selected_options(monkeypatch, batch_last):
    from boltops.gemm.triton.batched import launch_native
    p = problem('m','n')
    class Tensor:
        def __init__(self, strides): self.strides=strides
        def stride(self, dim=None): return self.strides if dim is None else self.strides[dim]
        def contiguous(self): raise AssertionError('unexpected input copy')
    a,b,out=Tensor(p.a.strides),Tensor(p.b.strides),Tensor((15,5,1))
    calls=[]
    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **params): calls.append((grid(params),args,params))
            return run
    monkeypatch.setitem(sys.modules, 'boltops.gemm.triton.batched_gemm_bf16',
                        SimpleNamespace(_batched_gemm_bf16_kernel=Kernel()))
    params = dict(dispatch.IMPLEMENTATIONS['triton.batched.native16']['default_config'],
                  BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, num_stages=3, BATCH_LAST_GRID=batch_last)
    launch_native(KernelConfig.from_params(params),p,a,b,None,out)
    grid,args,actual=calls[0]
    assert args[0] is a and args[1] is b
    assert args[7:13] == (*a.stride(),b.stride(0),b.stride(2),b.stride(1))
    assert actual == params
    assert grid == ((1,1,2) if batch_last else (2,1))
