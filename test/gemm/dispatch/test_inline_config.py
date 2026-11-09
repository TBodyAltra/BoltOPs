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
    from boltops.gemm.triton.dense import launch_dense_fp16
    calls = []
    monkeypatch.setitem(sys.modules, 'boltops.gemm.triton.gemm_fp16',
                        SimpleNamespace(gemm_fp16=lambda *args, **kwargs: calls.append(kwargs)))
    for entry in maps.document('dense')['entries']:
        config = maps.entry_config(entry)
        launch_dense_fp16(config, None, object(), SimpleNamespace(T=object()), None, object())
        call = calls[-1]
        assert call['autotune'] is False
        params = config.params
        for flag in ('wasp', 'wdra', 'use_mls', 'use_matrix_store'):
            assert call[flag] == params.pop(flag)
        actual = {k: list(v) if isinstance(v, tuple) else v for k, v in call['config'].items()}
        assert actual == params
        assert actual['wasp_partition_regs'] == [16, 244, 244]
        assert actual['buffer_cache_swizzle'] is True


def test_old_config_indirection_is_rejected():
    from copy import deepcopy
    data = deepcopy(maps.document('dense'))
    data['entries'][0]['config']['source_config'] = 'unused.json'
    with pytest.raises(ValueError, match='params only'):
        maps._validate_document(data, 'dense')
