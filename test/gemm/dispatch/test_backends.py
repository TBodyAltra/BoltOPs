"""Backend selection and dependency boundaries of the common interfaces."""
from dataclasses import replace
import os
import subprocess
import sys

import pytest
import torch

from boltops.gemm import dispatch
from boltops.gemm.descriptors import GemmProblem, OperandSpec, ScaleSpec
from boltops.gemm.dispatch import launch, select_plan


def problem():
    return GemmProblem('dense', 128, 256, 256,
                       OperandSpec(torch.float8_e4m3fn, strides=(256, 1)),
                       OperandSpec(torch.float8_e4m3fn, strides=(256, 1)),
                       torch.bfloat16, 'gfx946', num_cus=48,
                       a_scale=ScaleSpec(torch.float32, (1, 128)),
                       b_scale=ScaleSpec(torch.float32, (128, 128)))


def test_auto_prefers_mapped_gluon_over_general_triton():
    assert select_plan(problem()).backend == 'gluon'
    assert select_plan(problem(), backend='triton').implementation == 'triton.block_scaled'
    with pytest.raises(NotImplementedError, match='backend=gluon'):
        select_plan(replace(problem(), num_cus=96), backend='gluon')


@pytest.mark.parametrize('interface', ['scaled', 'dense', 'batched', 'grouped'])
def test_tilelang_has_explicit_empty_capabilities(interface):
    assert not any(row['backend'] == 'tilelang' and row['interface'] == interface
                   for row in dispatch.IMPLEMENTATIONS.values())


def test_unimplemented_tilelang_does_not_delegate():
    with pytest.raises(NotImplementedError, match='backend=tilelang'):
        select_plan(problem(), backend='tilelang')


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match='unknown GEMM backend'):
        select_plan(problem(), backend='gulon')


def test_explicit_missing_runtime_is_reported(monkeypatch):
    monkeypatch.setattr(dispatch, 'available', lambda name: False)
    with pytest.raises(ImportError, match='triton is not installed'):
        select_plan(problem(), backend='triton')
    with pytest.raises(NotImplementedError):
        select_plan(problem())


def test_selected_launch_error_propagates(monkeypatch):
    import boltops.gemm.dispatch as dispatch
    plan = select_plan(problem(), backend='triton')
    def fail(*args, **kwargs):
        raise RuntimeError('native compilation failed')
    from types import SimpleNamespace
    FailedKernel = SimpleNamespace(**{dispatch.IMPLEMENTATIONS[plan.implementation]['launcher']: fail})
    original = dispatch.import_module
    monkeypatch.setattr(dispatch, 'import_module', lambda name: FailedKernel if name == dispatch.IMPLEMENTATIONS[plan.implementation]['module'] else original(name))
    monkeypatch.setattr(dispatch, 'supports', lambda *args: True)
    with pytest.raises(RuntimeError, match='native compilation failed'):
        launch(plan, problem())


def test_frontends_and_table_import_without_any_dsl():
    code = '''
import torch
import importlib.abc
import sys
class BlockDSL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('triton', 'tilelang', 'tvm'):
            raise AssertionError('unexpected DSL import: ' + fullname)
for name in list(sys.modules):
    if name.split('.')[0] in ('triton', 'tilelang', 'tvm'):
        del sys.modules[name]
sys.meta_path.insert(0, BlockDSL())
from boltops.gemm import scaled, dense, batched, grouped, specialized
from boltops.gemm.dispatch import IMPLEMENTATIONS
assert all(row['backend'] != 'tilelang' for row in IMPLEMENTATIONS.values())
# The table is declarative; actual kernel modules load their own DSL on demand.
assert all(row['module'] not in sys.modules for row in IMPLEMENTATIONS.values())
'''
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())


def test_scaled_fp4_is_registered_under_scaled_interface():
    p = replace(problem(), m=16, n=32, k=128,
                a=OperandSpec(torch.float8_e4m3fn, strides=(128, 1)),
                b=OperandSpec(torch.uint8, 'fp4', 'e2m1_low_high', -1, 2, strides=(64, 1)),
                b_scale=ScaleSpec(torch.uint8, (1, 32), 'e8m0'))
    assert select_plan(p, backend='triton').implementation == 'triton.scaled.fp8_fp4'
    strided = replace(p, a=replace(p.a, strides=(256, 2)), b=replace(p.b, strides=(128, 2)))
    assert select_plan(strided, backend='triton').implementation == 'triton.scaled.fp8_fp4'
    with pytest.raises(NotImplementedError, match='backend=gluon'):
        select_plan(p, backend='gluon')


def test_map_configuration_cannot_change_implementation_owner(monkeypatch):
    from boltops.gemm import performance_map
    entry = next(e for e in performance_map._entries("scaled")
                 if e['config']['backend'] == 'gluon')
    wrong = {**entry, 'config': {**entry['config'], 'backend': 'tilelang'}}
    monkeypatch.setattr(performance_map, '_entries', lambda interface: [wrong] if interface == "scaled" else [])
    assert select_plan(problem()).backend == 'triton'
    # The foreign record cannot claim Gluon, but an explicit request can still
    # use Gluon's independently declared legal default configuration.
    assert select_plan(problem(), backend='gluon').backend == 'gluon'


def test_implementation_entries_resolve_directly():
    from importlib import import_module
    for row in dispatch.IMPLEMENTATIONS.values():
        module = import_module(row['module'])
        assert callable(getattr(module, row['launcher']))
        if 'configs' in row:
            assert callable(getattr(module, row['configs']))
        else:
            assert isinstance(row['default_config'], dict)
        if 'supports' in row:
            assert callable(getattr(module, row['supports']))
