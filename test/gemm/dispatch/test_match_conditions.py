# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Configuration conditions share one field-independent comparison engine."""
from dataclasses import replace, fields

import pytest
import torch

from boltops.gemm import performance_map as maps
from boltops.gemm.descriptors import GemmProblem, GroupSpec, OperandSpec, ScaleSpec


def problem():
    return GemmProblem('dense', 128, 256, 1024,
                       OperandSpec(torch.float8_e4m3fn, strides=(1024, 1)),
                       OperandSpec(torch.float8_e4m3fn, strides=(1024, 1)),
                       torch.bfloat16, 'gfx946', num_cus=48,
                       a_scale=ScaleSpec(torch.float32, (1, 128)),
                       b_scale=ScaleSpec(torch.float32, (128, 128)))


def test_all_descriptor_fields_are_exposed_without_a_matcher_whitelist():
    p = problem()
    assert set(maps.problem_attributes(p)) == {f.name for f in fields(p)}
    assert set(maps._PROBLEM_SCHEMA) == {f.name for f in fields(p)}
    assert maps._matches(p, maps.exact_match(p))


@pytest.mark.parametrize('condition,expected', [
    ({'m': 128}, True), ({'m': 64}, False),
    ({'m': {'min': 128, 'max': 128, 'multiple_of': 64}}, True),
    ({'m': {'min': 129}}, False), ({'m': {'max': 127}}, False),
    ({'m': {'multiple_of': 3}}, False),
    ({'num_cus': {'min': 24, 'max': 96, 'multiple_of': 24}}, True),
    ({'a': {'storage_dtype': 'float8_e4m3fn'}}, True),
    ({'a': {'strides': [1024, 1]}}, True),
    ({'a': {'strides': [1, 128]}}, False),
    ({'a_scale': {'block': [1, 128], 'encoding': 'numeric'}}, True),
    ({'a_scale': None}, False), ({'groups': None}, True),
    ({'has_bias': 0}, False), ({'m': True}, False),
])
def test_exact_partial_and_numeric_conditions(condition, expected):
    assert maps._matches(problem(), condition) is expected


def test_omission_is_unconstrained_not_an_implicit_dense_default():
    p = replace(problem(), kind='batched', batch=4, has_bias=True,
                a_aligned_16=False, a=OperandSpec(torch.float8_e4m3fn, strides=(131072, 1024, 1)))
    assert maps._matches(p, {'batch': {'min': 2, 'multiple_of': 2}})
    assert maps._matches(p, {'has_bias': True, 'a_aligned_16': False})


def test_optional_group_fields_use_the_same_numeric_comparisons():
    p = replace(problem(), kind='grouped', groups=GroupSpec('equal', 4))
    assert maps._matches(p, {'groups': {'mode': 'equal', 'count': {'min': 2, 'multiple_of': 2}}})
    assert not maps._matches(problem(), {'groups': {'count': {'min': 2}}})
    assert not maps._matches(replace(problem(), a_scale=None), {'a_scale': {'dtype': 'float32'}})


@pytest.mark.parametrize('condition,error', [
    ({'arch': 'other', 'typo': 1}, 'unknown GEMM match field'),
    ({'arch': 'other', 'a_scale': {'typo': 1}}, 'unknown GEMM match field'),
    ({'groups': {'typo': 1}}, 'unknown GEMM match field'),
    ({'m': {'greater': 1}}, 'unknown GEMM comparison'),
    ({'m': {'multiple_of': 0}}, 'positive integer'),
    ({'m': {'multiple_of': -2}}, 'positive integer'),
    ({'m': {'multiple_of': 1.5}}, 'positive integer'),
    ({'m': {'min': True}}, 'finite number'),
    ({'m': {'min': float('inf')}}, 'finite number'),
    ({'m': {'min': 4, 'max': 2}}, 'min greater than max'),
    ({'arch': {'min': 1}}, 'numeric comparisons'),
    ({'m': {}}, 'numeric comparisons'),
])
def test_invalid_rule_is_not_hidden_by_a_nonmatch(condition, error):
    with pytest.raises(ValueError, match=error):
        maps._matches(problem(), condition)


def test_packaged_rules_are_valid_and_keep_dense_layout_and_scaled_range():
    for interface in maps._CONFIG_NAMES:
        maps._validate_document(maps.document(interface), interface)
    dense = maps.document('dense')['entries'][0]['match']
    p = GemmProblem('dense', 4096, 4608, 512,
                    OperandSpec(torch.float16, strides=(512, 1)),
                    OperandSpec(torch.float16, strides=(1, 4608)),
                    torch.float16, 'gfx946', num_cus=48)
    assert maps._matches(p, dense)
    assert maps._matches(replace(p, a_aligned_16=False, b_aligned_16=False), dense)
    assert not {"a_aligned_16", "b_aligned_16"} & maps.exact_match(p).keys()
    for other in (replace(p, has_bias=True), replace(p, num_cus=96),
                  replace(p, b=OperandSpec(torch.float16, strides=(512, 1)))):
        assert not maps._matches(other, dense)
    scaled = maps.document('scaled')['entries']
    if scaled:
        rule = scaled[0]['match']
        assert maps._matches(problem(), rule)
        assert not maps._matches(replace(problem(), k=6400), rule)
        assert not maps._matches(replace(problem(), n=257), rule)


def test_document_rejects_unknown_rule_even_without_matching_backend():
    d = {'schema_version': 8, 'auto_backend': 'triton', 'interface': 'gemm_dense',
         'entries': [{'match': {'arch': 'other', 'num_cus': 48, 'bad': 1}, 'auto_backend': 'tilelang', 'configs': {'triton': ({'implementation': 'test', 'params': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}} if 'tilelang' == 'triton' else None), 'gluon': ({'implementation': 'test', 'params': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}} if 'tilelang' == 'gluon' else None), 'tilelang': ({'implementation': 'test', 'params': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}} if 'tilelang' == 'tilelang' else None)}}]}
    with pytest.raises(ValueError, match='unknown GEMM match field'):
        maps._validate_document(d, 'dense')
    d['entries'][0]['match'] = {'arch': 'gfx946'}
    with pytest.raises(ValueError, match='explicit arch and num_cus'):
        maps._validate_document(d, 'dense')
    d['schema_version'] = 4
    with pytest.raises(ValueError, match='schema'):
        maps._validate_document(d, 'dense')
