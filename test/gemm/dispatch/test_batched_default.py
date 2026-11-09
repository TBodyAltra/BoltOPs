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
    record = {'match': performance_map.exact_match(p), 'auto_backend': 'triton', 'configs': {'triton': ({'implementation': 'triton.batched.native16', 'params': params} if 'triton' == 'triton' else None), 'gluon': ({'implementation': 'triton.batched.native16', 'params': params} if 'triton' == 'gluon' else None), 'tilelang': ({'implementation': 'triton.batched.native16', 'params': params} if 'triton' == 'tilelang' else None)}}
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
    p = problem('m','n')
    class Tensor:
        def __init__(self, shape, strides, storage=None):
            self.shape, self.strides, self.storage = shape, strides, storage or object()
            self.ndim, self.dtype = len(shape), torch.float16
        def transpose(self, x, y):
            shape, strides = list(self.shape), list(self.strides)
            shape[x], shape[y] = shape[y], shape[x]
            strides[x], strides[y] = strides[y], strides[x]
            return Tensor(tuple(shape), tuple(strides), self.storage)
        def stride(self, dim=None): return self.strides if dim is None else self.strides[dim]
        def contiguous(self): raise AssertionError('unexpected input copy')
    a,b,out=Tensor((2,3,32),p.a.strides),Tensor((2,5,32),p.b.strides),Tensor((2,3,5),(15,5,1))
    calls=[]
    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **params): calls.append((grid(params),args,params))
            return run
    from boltops.gemm.triton import batched_gemm_a16w16 as implementation
    monkeypatch.setattr(implementation, '_batched_gemm_a16w16_kernel', Kernel())
    params = dict(dispatch.IMPLEMENTATIONS['triton.batched.native16']['default_config'],
                  BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, num_stages=3, BATCH_LAST_GRID=batch_last)
    plan = dispatch.ExecutionPlan("triton.batched.native16", KernelConfig.from_params(params))
    dispatch.launch(plan, p, a=a, b=b, bias=None, out=out)
    grid,args,actual=calls[0]
    assert args[0] is a and args[1].storage is b.storage
    assert args[7:13] == (*a.stride(),b.stride(0),b.stride(2),b.stride(1))
    assert actual == params
    assert grid == ((1,1,2) if batch_last else (2,1))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_migrated_bw200_record_and_unrecorded_m(dtype):
    def request(m, arch='gfx936', cus=80):
        return GemmProblem('batched', m, 8192, 1024,
            OperandSpec(dtype, strides=(m*1024, 1024, 1)),
            OperandSpec(dtype, strides=(8192*1024, 1024, 1)), dtype, arch,
            batch=16, num_cus=cus)
    tuned = dispatch.select_plan(request(1), backend='triton').config
    assert (tuned.block_m, tuned.block_n, tuned.block_k) == (8, 64, 128)
    default = dispatch.IMPLEMENTATIONS['triton.batched.native16']['default_config']
    # No nearest-M extrapolation, and no BW200 recipe applied on gfx946.
    assert dispatch.select_plan(request(9), backend='triton').config.params == default
    assert dispatch.select_plan(request(1, 'gfx946', 48), backend='triton').config.params == default
