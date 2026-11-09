"""Inline launch options survive selection, publication and host adaptation."""
import sys
from types import SimpleNamespace

import pytest

from boltops.gemm.descriptors import KernelConfig
from boltops.gemm import performance_map as maps


def test_parameters_are_immutable_hashable_and_round_trip():
    values = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
                  num_stages=3, wasp_partition_regs=[16, 244, 244])
    config = KernelConfig.from_params(values)
    values['wasp_partition_regs'][0] = 100
    copy = config.params
    copy['wasp_partition_regs'][0] = 200
    assert config.params['wasp_partition_regs'][0] == 16
    assert KernelConfig.from_params(config.params) == config
    assert len({config, KernelConfig.from_params(config.params)}) == 1
    assert config != KernelConfig(32, 64, 32, num_stages=2)


def test_dense_launcher_consumes_inline_options(monkeypatch):
    import torch
    from boltops.gemm.triton import gemm_fp16 as impl
    class Tensor:
        shape = (256, 256)
        dtype = torch.float16
        device = 'cuda'
        is_cuda = True
        @property
        def T(self): return self
        def dim(self): return 2
        def reshape(self, *shape): return self
        def stride(self, dim): return (256, 1)[dim]
    a, b, out = Tensor(), Tensor(), Tensor()
    calls = []
    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs): calls.append((args, kwargs))
            return run
    monkeypatch.setattr(impl, '_gemm_fp16_mls_kernel', Kernel())
    monkeypatch.setattr(impl, '_gemm_fp16_load_kernel', Kernel())
    monkeypatch.setattr(impl, '_prepare_ab', lambda a, b: (a, b, a.shape))
    monkeypatch.setattr(impl, 'supports_wasp_wdra', lambda arch: True)
    monkeypatch.setattr(impl.arch_info, 'is_mls_avail', lambda: True)
    monkeypatch.setattr(impl.arch_info, 'get_arch', lambda: 'gfx946')
    for entry in maps.document('dense')['entries']:
        config = maps.entry_config(entry)
        impl.gemm_fp16(a, b, out=out, config=config.params, problem=object())
        args, actual = calls[-1]
        expected = config.params
        for flag in ('wasp', 'wdra', 'use_mls', 'use_matrix_store'):
            expected.pop(flag)
        expected['HAS_BIAS'] = False
        expected['USE_MATRIX_STORE'] = True
        actual = {k: list(v) if isinstance(v, tuple) else v for k, v in actual.items()}
        assert actual == expected
        assert args[0] is a and args[2] is out


def test_old_config_indirection_is_rejected():
    from copy import deepcopy
    data = deepcopy(maps.document('dense'))
    data['entries'][0]['config']['source_config'] = 'unused.json'
    with pytest.raises(ValueError, match='params only'):
        maps._validate_document(data, 'dense')
