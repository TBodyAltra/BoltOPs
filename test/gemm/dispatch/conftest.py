# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Host-only implementation rows used to exercise table selection."""
import sys
from types import SimpleNamespace
import pytest
from boltops.gemm import dispatch
from boltops.gemm.descriptors import KernelConfig


@pytest.fixture
def add_impl(monkeypatch):
    monkeypatch.setattr(dispatch, 'available', lambda backend: True)
    def add(name, backend='triton', interface='scaled', configs=None, supports=None, default_priority=0):
        tile = KernelConfig(32, 64, 32)
        module_name = '_gemm_table_test_' + name.replace('.', '_')
        module = SimpleNamespace(configs=configs or (lambda p: (tile,)),
                                 supports=supports or (lambda p, c: c == tile),
                                 launch=lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module_name, module)
        monkeypatch.setitem(dispatch.IMPLEMENTATIONS, name, {
            'backend': backend, 'interface': interface, 'module': module_name,
            'configs': 'configs', 'supports': 'supports', 'launcher': 'launch',
            'default_priority': default_priority,
        })
        return name
    return add
