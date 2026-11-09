# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Backend order cannot override a measured winner or hide ambiguous records."""
from dataclasses import replace
from copy import deepcopy
import itertools
import json

import pytest
import torch

from boltops.gemm import performance_map as maps
from boltops.gemm.autotune import autotune, save_winner
from boltops.gemm.descriptors import GemmProblem, GroupSpec, OperandSpec, ScaleSpec
from boltops.gemm.dispatch import ExecutionPlan, KernelConfig, select_plan
from boltops.gemm import dispatch


@pytest.fixture
def policy(tmp_path, monkeypatch):
    monkeypatch.delenv('BOLTOPS_GEMM_CONFIG_DIR', raising=False)
    monkeypatch.setattr(maps, '_CONFIG_DIR', tmp_path)
    maps.clear_cache()
    for name in ('dense', 'scaled', 'batched', 'grouped', 'batched_a16wfp4_online_quant'):
        (tmp_path / f'gemm_{name}.json').write_text(json.dumps({
            'schema_version': 5, 'interface': f'gemm_{name}',
            'default_backends': ['triton', 'gluon', 'tilelang'], 'entries': []}))
    yield tmp_path
    maps.clear_cache()


def problem():
    return GemmProblem('dense', 32, 64, 64,
                       OperandSpec(torch.bfloat16, strides=(64, 1)),
                       OperandSpec(torch.bfloat16, strides=(64, 1)), torch.bfloat16, 'gfx946',
                       a_scale=ScaleSpec(torch.float32, (1, 32)),
                       b_scale=ScaleSpec(torch.float32, (32, 32)), num_cus=48)


@pytest.fixture
def choices(add_impl):
    return tuple(add_impl(f'{b}.test', backend=b) for b in ('triton', 'gluon', 'tilelang'))


def record(p, impl, *, priority=0):
    return {'match': maps.exact_match(p), 'priority': priority,
            'config': {'backend': dispatch.IMPLEMENTATIONS[impl]['backend'], 'implementation': impl,
                       'block_size': [32, 64, 32]}}


def write(policy, p, entries, order=None):
    path = policy / f'gemm_{p.interface}.json'
    data = json.loads(path.read_text())
    data['entries'] = entries
    if order is not None:
        data['default_backends'] = order
    path.write_text(json.dumps(data))
    maps.clear_cache()
    return path


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_measured_winner_is_independent_of_registration_order(policy, order, choices):
    p = problem()
    write(policy, p, [record(p, choices[2])])
    assert select_plan(p, [choices[i] for i in order]).implementation == choices[2]


@pytest.mark.parametrize('reverse', [False, True])
def test_ambiguous_records_fail_even_when_the_first_one_is_supported(policy, reverse, choices):
    p = problem()
    records = [record(p, choices[0]), record(p, choices[1])]
    write(policy, p, records[::-1] if reverse else records)
    with pytest.raises(ValueError, match='conflicting GEMM winners'):
        select_plan(p, choices)
    assert select_plan(p, choices, backend='gluon').implementation == choices[1]


def test_explicit_record_priority_overrides_broader_record(policy, choices):
    p = problem()
    broad = record(p, choices[0])
    broad['match'].update(m={'multiple_of': 16}, n={'multiple_of': 32}, k={'min': 32})
    write(policy, p, [broad, record(p, choices[2], priority=10)])
    assert select_plan(p, choices).implementation == choices[2]


def test_defaults_follow_interface_policy_not_registration_order(policy, choices):
    p = problem()
    write(policy, p, [], order=['tilelang', 'gluon', 'triton'])
    assert select_plan(p, choices).implementation == choices[2]
    assert select_plan(p, choices[::-1]).implementation == choices[2]


def test_defaults_have_stable_implementation_preference(policy, choices, add_impl):
    p = problem()
    preferred = add_impl('zz', default_priority=10)
    other = add_impl('aa')
    assert select_plan(p, (other, preferred)).implementation == preferred
    assert select_plan(p, (preferred, other)).implementation == preferred


@pytest.mark.parametrize('what', ['unknown_implementation', 'unsupported_tile'])
def test_stale_measured_record_is_not_silently_replaced(policy, what, choices):
    p = problem()
    entry = record(p, choices[1])
    entry['config']['implementation' if what == 'unknown_implementation' else 'block_size'] = (
        'gluon.absent' if what == 'unknown_implementation' else [64, 64, 32])
    write(policy, p, [entry])
    with pytest.raises(ValueError, match='registered implementation|winner is unsupported'):
        select_plan(p, choices)


def test_unavailable_backend_record_uses_available_defaults(policy, choices):
    p = problem()
    write(policy, p, [record(p, choices[2])])
    assert select_plan(p, choices[:2]).implementation == choices[0]


@pytest.mark.parametrize('order', [['triton', 'triton', 'gluon'], ['gluon'], 'triton'])
def test_invalid_default_policy_fails(policy, order, choices):
    p = problem()
    write(policy, p, [], order=order)
    with pytest.raises(ValueError, match='default_backends'):
        select_plan(p, choices)


def test_duplicate_identical_winners_are_an_explicit_configuration_error(policy, choices):
    p = problem()
    entry = record(p, choices[0])
    write(policy, p, [entry, deepcopy(entry)])
    with pytest.raises(ValueError, match='conflicting GEMM winners'):
        select_plan(p, choices)


def test_autotune_requires_validation_and_never_runs_from_auto(policy, choices):
    p = problem()
    assert select_plan(p, choices).implementation == choices[0]
    with pytest.raises(TypeError, match='validation and benchmark'):
        autotune(p, names=choices, benchmark=lambda plan: 1, validate=None)


def test_explicit_tuning_checks_correctness_before_measuring(policy, choices):
    p = problem()
    measured = []
    def validate(plan):
        if plan.backend == 'triton':
            return False
    def benchmark(plan):
        name = plan.backend
        measured.append(name)
        if name == 'tilelang':
            raise RuntimeError('compiler unavailable for this candidate')
        return 5
    result = autotune(p, names=choices, benchmark=benchmark, validate=validate)
    assert result.winner.implementation == choices[1]
    assert measured == ['gluon', 'tilelang']
    assert result.trials[0].score is None and 'validation' in result.trials[0].error
    assert 'compiler unavailable' in result.trials[2].error


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True])
def test_invalid_metric_does_not_write_a_winner(policy, bad, choices):
    p = problem()
    path = policy / 'gemm_scaled.json'
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='no validated'):
        autotune(p, names=choices, validate=lambda plan: True,
                 benchmark=lambda plan: bad, output_path=path)
    assert path.read_bytes() == before


def test_tuning_writes_one_exact_winner_without_measurements(policy, choices):
    p = problem()
    scores = {'triton': 3, 'gluon': 1, 'tilelang': 2}
    path = policy / 'gemm_scaled.json'
    result = autotune(p, names=choices, validate=lambda plan: True,
                      benchmark=lambda plan: scores[plan.backend], output_path=path)
    assert result.winner.implementation == choices[1]
    data = json.loads(path.read_text())
    assert len(data['entries']) == 1
    assert set(data['entries'][0]) == {'match', 'config', 'priority'}
    assert set(data['entries'][0]['config']) == {'backend', 'implementation', 'block_size'}
    assert select_plan(p, choices[::-1]).implementation == choices[1]
    save_winner(p, result.winner, path)
    assert json.loads(path.read_text()) == data


def test_tuned_exact_override_preserves_other_shapes_and_range_record(policy, choices):
    p = problem()
    broad = record(p, choices[0])
    broad['match']['m'] = {'multiple_of': 16}
    unrelated = record(replace(p, n=128), choices[0])
    path = write(policy, p, [broad, unrelated])
    plan = ExecutionPlan(choices[1], KernelConfig(32, 64, 32))
    save_winner(p, plan, path)
    data = json.loads(path.read_text())
    assert data['entries'][:2] == [broad, unrelated]
    assert data['entries'][2]['priority'] == 1
    assert select_plan(p, choices).implementation == choices[1]
    assert select_plan(replace(p, m=48), choices).implementation == choices[0]


def test_explicit_tuning_limits_backend_and_metric_ties_use_policy(policy, choices):
    p = problem()
    result = autotune(p, names=choices[::-1], validate=lambda plan: True,
                      benchmark=lambda plan: 1, backend='gluon')
    assert len(result.trials) == 1 and result.winner.implementation == choices[1]
    write(policy, p, [], order=['tilelang', 'triton', 'gluon'])
    result = autotune(p, names=choices, validate=lambda plan: True, benchmark=lambda plan: 1)
    assert result.winner.implementation == choices[2]


def test_override_directory_is_optional_per_interface(policy, tmp_path, monkeypatch, choices):
    override = tmp_path / 'user-config'
    override.mkdir()
    p = problem()
    data = json.loads((policy / 'gemm_scaled.json').read_text())
    data['default_backends'] = ['gluon', 'tilelang', 'triton']
    (override / 'gemm_scaled.json').write_text(json.dumps(data))
    monkeypatch.setenv('BOLTOPS_GEMM_CONFIG_DIR', str(override))
    assert select_plan(p, choices).backend == 'gluon'
    assert maps.config_path('dense') == policy / 'gemm_dense.json'


@pytest.mark.parametrize('mode', ['batched', 'grouped', 'awq', 'mxfp4', 'online'])
def test_winners_round_trip_all_operation_and_packing_contracts(policy, mode, choices, monkeypatch):
    p = problem()
    if mode in ('batched', 'grouped'):
        p = replace(p, kind=mode, a_scale=None, b_scale=None,
                    a=replace(p.a, strides=(2048, 64, 1)), b=replace(p.b, strides=(4096, 64, 1)),
                    batch=2 if mode == 'batched' else 1,
                    groups=GroupSpec('equal', 2) if mode == 'grouped' else None)
    elif mode == 'awq':
        p = replace(p, a_scale=None, b_scale=ScaleSpec(torch.bfloat16, (1, 32)),
                    b=OperandSpec(torch.int32, 'int4', 'awq', 1, 8,
                                  zero_point_dtype=torch.int32, strides=(8, 1)))
    else:
        p = replace(p, b=OperandSpec(torch.uint8, 'fp4', 'e2m1_low_high', -1, 2, strides=(32, 1)),
                    b_scale=ScaleSpec(torch.uint8, (1, 32), 'e8m0'))
        if mode == 'mxfp4':
            p = replace(p, a=replace(p.b), a_scale=replace(p.b_scale))
        else:
            p = replace(p, kind='batched', batch=2, a_scale=None, activation_quantization='mxfp4')
    impl = choices[1]
    monkeypatch.setitem(dispatch.IMPLEMENTATIONS, impl, {**dispatch.IMPLEMENTATIONS[impl], 'interface': p.interface})
    plan = ExecutionPlan(impl, KernelConfig(32, 64, 32))
    path = policy / f'gemm_{p.interface}.json'
    save_winner(p, plan, path)
    assert select_plan(p, (impl,)) == plan
    assert maps.matching_entries(replace(p, k=128)) == ()
    assert maps.matching_entries(replace(p, has_bias=True)) == ()


def test_atomic_publish_failure_keeps_original_file(policy, monkeypatch, choices):
    import boltops.gemm.autotune as tuning
    p = problem()
    target = policy / 'gemm_scaled.json'
    before = target.read_bytes()
    def fail(*args):
        raise OSError('simulated filesystem failure')
    monkeypatch.setattr(tuning.os, 'replace', fail)
    with pytest.raises(OSError, match='filesystem failure'):
        save_winner(p, ExecutionPlan(choices[0], KernelConfig(32, 64, 32)), target)
    assert target.read_bytes() == before
    assert not list(policy.glob('*.tmp'))


def test_parallel_publications_preserve_other_workers_records(policy, choices):
    from concurrent.futures import ThreadPoolExecutor
    base = problem()
    impl = choices[0]
    plan = ExecutionPlan(impl, KernelConfig(32, 64, 32))
    target = policy / 'gemm_scaled.json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda m: save_winner(replace(base, m=m), plan, target), range(16, 32)))
    data = json.loads(target.read_text())
    assert len(data['entries']) == 16
    assert {e['match']['m'] for e in data['entries']} == set(range(16, 32))


def test_auto_does_not_import_unavailable_backend(policy, monkeypatch, choices):
    rows = {name: dispatch.IMPLEMENTATIONS[name] for name in choices}
    monkeypatch.setattr(dispatch, 'IMPLEMENTATIONS', rows)
    monkeypatch.setattr(dispatch, 'available', lambda backend: backend != 'tilelang')
    original = dispatch.import_module
    def load(module):
        assert module != rows[choices[2]]['module'], 'unavailable implementation must not be imported'
        return original(module)
    monkeypatch.setattr(dispatch, 'import_module', load)
    assert select_plan(problem()).backend == 'triton'


@pytest.mark.parametrize('invalid', [0, 1, 'PASS'])
def test_validation_requires_success_not_an_arbitrary_return_value(policy, invalid, choices):
    with pytest.raises(RuntimeError, match='no validated'):
        autotune(problem(), names=choices, validate=lambda plan: invalid,
                 benchmark=lambda plan: 1)
